//===- LoadStoreOpToLLVM.cpp - LoadOp/StoreOp to LLVM for Custom Backend --===//
//
// This file implements the conversion of triton::LoadOp and triton::StoreOp
// to LLVM IR for the Custom SIMT backend. This is a simplified implementation
// that uses standard LLVM load/store operations for global memory only.
//
//===----------------------------------------------------------------------===//

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

// Helper function to create a zero value for a given type
Value createZeroValue(RewriterBase &rewriter, Location loc, Type elemTy) {
  if (isa<FloatType>(elemTy)) {
    return arith::ConstantOp::create(rewriter, loc, elemTy,
                                     rewriter.getFloatAttr(elemTy, 0.0));
  } else {
    return arith::ConstantOp::create(rewriter, loc, elemTy,
                                     rewriter.getIntegerAttr(elemTy, 0));
  }
}

/// Custom LoadOp conversion using standard LLVM loads
/// This is a simplified version that handles global memory loads without
/// complex vectorization optimizations.
struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const Custom::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Get the element type
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    
    unsigned numElems = getTotalElemsPerThread(ptr.getType());

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
      assert(otherElems.size() == numElems);
    }

    // Load each element
    SmallVector<Value> loadedVals;
    for (size_t i = 0; i < numElems; ++i) {
      Value pred = mask ? maskElems[i] : nullptr;
      Value ptrElem = ptrElems[i];

      // Create a default value (either from 'other' or zero)
      Value falseVal;
      if (other) {
        falseVal = otherElems[i];
      } else {
        falseVal = createZeroValue(rewriter, loc, valueElemTy);
      }

      // Perform the load
      Value loadedVal;
      if (pred) {
        // Masked load: use if-then-else
        // if (pred) loadedVal = load(ptr) else loadedVal = falseVal
        auto loadOp = LLVM::LoadOp::create(rewriter, loc, valueElemTy, ptrElem);
        loadedVal = arith::SelectOp::create(rewriter, loc, pred, loadOp, falseVal);
      } else {
        // Unconditional load
        loadedVal = LLVM::LoadOp::create(rewriter, loc, valueElemTy, ptrElem);
      }

      loadedVals.push_back(loadedVal);
    }

    // Pack the loaded values into the result struct
    Type llvmResultStructTy = typeConverter->convertType(valueTy);
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }

private:
  const Custom::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

/// Custom StoreOp conversion using standard LLVM stores
struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp> {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const Custom::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();

    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();

    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert store with a tensor pointer into LLVM; "
           "this case should be transformed to normal store before lowering");

    Value llPtr = adaptor.getPtr();
    Value llValue = adaptor.getValue();
    Value llMask = adaptor.getMask();

    unsigned numElems = getTotalElemsPerThread(ptr.getType());

    // Get the LLVM values for pointers, values, and mask
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == numElems);
    assert(valueElems.size() == numElems);

    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Store each element
    // For masked stores, we use LLVM's conditional branch structure
    for (size_t i = 0; i < numElems; ++i) {
      Value pred = mask ? maskElems[i] : nullptr;
      Value ptrElem = ptrElems[i];
      Value valElem = valueElems[i];

      if (pred) {
        // Conditional store using LLVM control flow
        // Create basic blocks for conditional execution
        auto parentOp = op->getParentOp();
        auto *currentBlock = rewriter.getInsertionBlock();
        auto *remainderBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
        auto *storeBlock = rewriter.createBlock(remainderBlock);

        // Insert conditional branch
        rewriter.setInsertionPointToEnd(currentBlock);
        LLVM::CondBrOp::create(rewriter, loc, pred, storeBlock, remainderBlock);

        // Store block
        rewriter.setInsertionPointToStart(storeBlock);
        LLVM::StoreOp::create(rewriter, loc, valElem, ptrElem);
        LLVM::BrOp::create(rewriter, loc, remainderBlock);

        // Continue in remainder block
        rewriter.setInsertionPointToStart(remainderBlock);
      } else {
        // Unconditional store
        LLVM::StoreOp::create(rewriter, loc, valElem, ptrElem);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const Custom::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

} // namespace

void mlir::triton::Custom::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<LoadOpConversion, StoreOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit);
}
