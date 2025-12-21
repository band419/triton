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
      // Barrier is convergent: cannot be moved past control flow
      func.setConvergent(true);
      func.setNoUnwind(true);
      // Barrier acts as a full memory fence: reads and writes to all memory
      // This ensures loads/stores cannot be reordered across the barrier
      // Memory effects: inaccessiblemem_or_argmemonly (affects all observable memory)
      func.setMemoryEffects(LLVM::MemoryEffectsAttr::get(
          rewriter.getContext(),
          /*other=*/LLVM::ModRefInfo::ModRef,      // Fence affects "other" memory
          /*argMem=*/LLVM::ModRefInfo::ModRef,     // Fence affects argument memory
          /*inaccessibleMem=*/LLVM::ModRefInfo::ModRef));  // Fence affects inaccessible memory
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
