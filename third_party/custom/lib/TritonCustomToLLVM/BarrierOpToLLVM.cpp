//===- BarrierOpToLLVM.cpp - Barrier Ops to LLVM for Custom SIMT ----------===//
//
// Convert barrier operations to custom LLVM intrinsics.
//
//===----------------------------------------------------------------------===//

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace {

/// Convert gpu::BarrierOp to llvm.custom.barrier intrinsic
struct BarrierOpConversion : public ConvertOpToLLVMPattern<gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {});

    // Get or insert the barrier intrinsic declaration
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("llvm.custom.barrier")) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto func = LLVM::LLVMFuncOp::create(rewriter, module.getLoc(),
                                           "llvm.custom.barrier", funcTy,
                                           LLVM::Linkage::External);
      func.setConvergent(true);
      func.setNoUnwind(true);
    }

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{},
                                               "llvm.custom.barrier",
                                               ValueRange{});
    return success();
  }
};

} // namespace

namespace mlir::triton::Custom {

void populateBarrierOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
  patterns.add<BarrierOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::Custom
