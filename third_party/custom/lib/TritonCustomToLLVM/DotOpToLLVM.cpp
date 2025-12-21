#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value D = op.getResult();
    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();

    // Custom SIMT backend: only support the generic FMA-based dot lowering.
    // This requires a BlockedEncoding layout on the accumulator/result.
    if (isa<triton::gpu::BlockedEncodingAttr>(dEncoding))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM for Custom "
        "backend (expected BlockedEncoding for FMA dot)."
    );
  }
};
} // namespace

namespace mlir::triton::Custom {

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 const TargetInfo &targetInfo,
                                 PatternBenefit benefit) {
  (void)axisInfoAnalysis;
  (void)targetInfo;
  patterns.add<DotOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::Custom
