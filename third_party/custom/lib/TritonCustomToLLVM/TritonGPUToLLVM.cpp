//===- TritonGPUToLLVM.cpp - TritonGPU to LLVM for Custom SIMT backend ----===//
//
// Main conversion pass from TritonGPU dialect to LLVM dialect for the Custom
// SIMT backend. This pass produces NVIDIA-free LLVM IR with custom intrinsics.
//
//===----------------------------------------------------------------------===//

#include "TritonCustomToLLVM/Passes.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONCUSTOMTOLLVM
#include "TritonCustomToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

// ============================================================================
// GPU Dialect to Custom LLVM Conversion Patterns
// ============================================================================

/// Convert mlir::gpu::ThreadIdOp to llvm.riscv.simt.thread.id intrinsic
struct ThreadIdOpConversion
    : public ConvertOpToLLVMPattern<mlir::gpu::ThreadIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();
    auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});

    // Get or insert the thread id intrinsic declaration
    // Maps to CSR_SIMT_TID_X (0xFC0)
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("llvm.riscv.simt.thread.id")) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto func = LLVM::LLVMFuncOp::create(rewriter, module.getLoc(),
                                           "llvm.riscv.simt.thread.id", funcTy,
                                           LLVM::Linkage::External);
      func.setNoUnwind(true);
    }

    // Map dimension to axis value (x=0, y=1, z=2)
    int32_t axis = 0;
    switch (op.getDimension()) {
    case mlir::gpu::Dimension::x:
      axis = 0;
      break;
    case mlir::gpu::Dimension::y:
      axis = 1;
      break;
    case mlir::gpu::Dimension::z:
      axis = 2;
      break;
    }
    Value axisVal = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getI32IntegerAttr(axis));
    auto call = LLVM::CallOp::create(rewriter, loc, i32Ty,
                                     "llvm.riscv.simt.thread.id",
                                     ValueRange{axisVal});
    // Convert i32 to index type
    Value result = arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getIndexType(),
                                              call.getResult());
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert mlir::gpu::BlockIdOp to llvm.riscv.simt.block.id intrinsic
struct BlockIdOpConversion : public ConvertOpToLLVMPattern<mlir::gpu::BlockIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BlockIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();
    auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});

    // Get or insert the block id intrinsic declaration
    // Maps to CSR_SIMT_CTAID_X (0xFD8)
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("llvm.riscv.simt.block.id")) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto func = LLVM::LLVMFuncOp::create(rewriter, module.getLoc(),
                                           "llvm.riscv.simt.block.id", funcTy,
                                           LLVM::Linkage::External);
      func.setNoUnwind(true);
    }

    int32_t axis = 0;
    switch (op.getDimension()) {
    case mlir::gpu::Dimension::x:
      axis = 0;
      break;
    case mlir::gpu::Dimension::y:
      axis = 1;
      break;
    case mlir::gpu::Dimension::z:
      axis = 2;
      break;
    }
    Value axisVal = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getI32IntegerAttr(axis));
    auto call = LLVM::CallOp::create(rewriter, loc, i32Ty,
                                     "llvm.riscv.simt.block.id",
                                     ValueRange{axisVal});
    Value result = arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getIndexType(),
                                              call.getResult());
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert mlir::gpu::BlockDimOp to constant or intrinsic
struct BlockDimOpConversion : public ConvertOpToLLVMPattern<mlir::gpu::BlockDimOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BlockDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();
    auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});

    // Get or insert the block dim intrinsic declaration
    // Maps to CSR_SIMT_NTID_X (0xFCC)
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("llvm.riscv.simt.block.dim")) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto func = LLVM::LLVMFuncOp::create(rewriter, module.getLoc(),
                                           "llvm.riscv.simt.block.dim", funcTy,
                                           LLVM::Linkage::External);
      func.setNoUnwind(true);
    }

    int32_t axis = 0;
    switch (op.getDimension()) {
    case mlir::gpu::Dimension::x:
      axis = 0;
      break;
    case mlir::gpu::Dimension::y:
      axis = 1;
      break;
    case mlir::gpu::Dimension::z:
      axis = 2;
      break;
    }
    Value axisVal = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getI32IntegerAttr(axis));
    auto call = LLVM::CallOp::create(rewriter, loc, i32Ty,
                                     "llvm.riscv.simt.block.dim",
                                     ValueRange{axisVal});
    Value result = arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getIndexType(),
                                              call.getResult());
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert mlir::gpu::GridDimOp to intrinsic
struct GridDimOpConversion : public ConvertOpToLLVMPattern<mlir::gpu::GridDimOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::GridDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();
    auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});

    // Get or insert the grid dim intrinsic declaration
    // Maps to grid_size from kernel descriptor
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("llvm.riscv.simt.grid.dim")) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto func = LLVM::LLVMFuncOp::create(rewriter, module.getLoc(),
                                           "llvm.riscv.simt.grid.dim", funcTy,
                                           LLVM::Linkage::External);
      func.setNoUnwind(true);
    }

    int32_t axis = 0;
    switch (op.getDimension()) {
    case mlir::gpu::Dimension::x:
      axis = 0;
      break;
    case mlir::gpu::Dimension::y:
      axis = 1;
      break;
    case mlir::gpu::Dimension::z:
      axis = 2;
      break;
    }
    Value axisVal = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getI32IntegerAttr(axis));
    auto call = LLVM::CallOp::create(rewriter, loc, i32Ty,
                                     "llvm.riscv.simt.grid.dim",
                                     ValueRange{axisVal});
    Value result = arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getIndexType(),
                                              call.getResult());
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Helper to populate GPU dialect patterns
void populateCustomGpuToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<ThreadIdOpConversion>(typeConverter, benefit);
  patterns.add<BlockIdOpConversion>(typeConverter, benefit);
  patterns.add<BlockDimOpConversion>(typeConverter, benefit);
  patterns.add<GridDimOpConversion>(typeConverter, benefit);
}

// ============================================================================
// Conversion Targets
// ============================================================================

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    // Warp specialization is lowered later if needed
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

struct ConvertTritonCustomToLLVM
    : public triton::impl::ConvertTritonCustomToLLVMBase<
          ConvertTritonCustomToLLVM> {
  
  explicit ConvertTritonCustomToLLVM(int32_t warpSize) {
    this->warpSize = warpSize;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::triton::Custom::TargetInfo targetInfo(this->warpSize);

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Axis info analysis for elementwise operations
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Lower functions
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // Note: Custom backend does NOT use shared memory (global memory only)
    // No initSharedMemory() call needed

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;

    // Populate custom elementwise op patterns (includes floating-point ops)
    mlir::triton::Custom::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

    // Populate conversion patterns
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    // Dot lowering (SIMT FMA fallback).
    mlir::triton::Custom::populateDotOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                  targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                                benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                  targetInfo, benefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                  patterns, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                    patterns, benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                      targetInfo, benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);

    // Custom SIMT-specific patterns
    mlir::triton::Custom::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                       targetInfo, benefit);
    mlir::triton::Custom::populateBarrierOpToLLVMPattern(typeConverter, patterns,
                                                          benefit);
    mlir::triton::Custom::populateLoadStoreOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);

    // GPU dialect to custom LLVM patterns (thread_id, block_id, etc.)
    populateCustomGpuToLLVMPatterns(typeConverter, patterns, benefit);

    // Standard MLIR conversion patterns
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Make all warp group code isolated from above
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonCustomToLLVMPass(int32_t warpSize) {
  return std::make_unique<ConvertTritonCustomToLLVM>(warpSize);
}

} // namespace mlir::triton
