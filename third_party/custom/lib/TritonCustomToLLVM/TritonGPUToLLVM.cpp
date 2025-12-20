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
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
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

    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

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

    // Initialize shared memory (use global memory for Custom backend)
    initSharedMemory(typeConverter);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;

    // Populate conversion patterns
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
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

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    
    // For Custom backend, shared memory is allocated from global memory
    // We still create a placeholder global for compatibility with existing code
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    LLVM::GlobalOp::create(b, loc, arrayTy, /*isConstant=*/false,
                           LLVM::Linkage::External, "global_smem",
                           /*value=*/Attribute(), /*alignment=*/16,
                           /*addrSpace=*/0);
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonCustomToLLVMPass(int32_t warpSize) {
  return std::make_unique<ConvertTritonCustomToLLVM>(warpSize);
}

} // namespace mlir::triton
