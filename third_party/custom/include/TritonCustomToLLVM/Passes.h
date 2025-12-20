#ifndef TRITON_THIRD_PARTY_CUSTOM_INCLUDE_TRITONCUSTOMTOLLVM_PASSES_H_
#define TRITON_THIRD_PARTY_CUSTOM_INCLUDE_TRITONCUSTOMTOLLVM_PASSES_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

} // namespace mlir

namespace mlir::triton {

#define GEN_PASS_DECL
#include "TritonCustomToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonCustomToLLVMPass(int32_t warpSize = 32);

std::unique_ptr<OperationPass<ModuleOp>>
createAllocateCustomSharedMemoryPass();

#define GEN_PASS_REGISTRATION
#include "TritonCustomToLLVM/Passes.h.inc"

} // namespace mlir::triton

namespace mlir::triton::Custom {

class TargetInfo;

/// Populate patterns for SPMD operations (program_id, num_programs, etc.)
void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfo &targetInfo,
                                 PatternBenefit benefit);

/// Populate patterns for barrier operations
void populateBarrierOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit);

/// Populate patterns for memory operations
void populateMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfo &targetInfo,
                                    PatternBenefit benefit);

/// Populate patterns for load/store operations
void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       const TargetInfo &targetInfo,
                                       PatternBenefit benefit);

} // namespace mlir::triton::Custom

#endif // TRITON_THIRD_PARTY_CUSTOM_INCLUDE_TRITONCUSTOMTOLLVM_PASSES_H_
