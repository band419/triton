#ifndef TRITON_THIRD_PARTY_CUSTOM_LIB_TRITONCUSTOMTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_
#define TRITON_THIRD_PARTY_CUSTOM_LIB_TRITONCUSTOMTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::Custom {

class TargetInfo;

/// Pattern benefit constants
constexpr int patternBenefitDefault = 1;
constexpr int patternBenefitPrioritizeOverLLVMConversions = 10;

/// Populate SPMD operation patterns
void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfo &targetInfo,
                                 PatternBenefit benefit);

/// Populate barrier operation patterns
void populateBarrierOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit);

} // namespace mlir::triton::Custom

#endif // TRITON_THIRD_PARTY_CUSTOM_LIB_TRITONCUSTOMTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_
