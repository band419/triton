#ifndef TRITON_THIRD_PARTY_CUSTOM_LIB_TRITONCUSTOMTOLLVM_TARGETINFO_H_
#define TRITON_THIRD_PARTY_CUSTOM_LIB_TRITONCUSTOMTOLLVM_TARGETINFO_H_

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include <string>

namespace mlir::triton::Custom {

/// TargetInfo for Custom SIMT backend.
///
/// This class implements the TargetInfoBase interface using LLVM intrinsics
/// that will be lowered by a custom LLVM backend or external codegen.
/// All SIMT operations (barrier, shuffle, program_id, etc.) are represented
/// as calls to custom intrinsics (llvm.custom.*).
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  explicit TargetInfo(int32_t warpSize = 32) : warpSize_(warpSize) {}

  int32_t getWarpSize() const { return warpSize_; }

  // ==========================================================================
  // TargetInfoBase interface implementation
  // ==========================================================================

  bool supportMaximumMinimum() const override;

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(RewriterBase &rewriter, Location loc, Type type,
               Value cmp) const override;

  void barrier(Location loc, RewriterBase &rewriter,
               bool isWarpSync = false) const override;

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Value val,
                    Value pred) const override;

  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;

  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override;

  Value permute(RewriterBase &rewriter, Location loc, Value a, Value b,
                Value selector) const override;

  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  ProgramIDDim axis) const override;

  bool warpReduce(RewriterBase &rewriter, Location loc, SmallVector<Value> &acc,
                  triton::ReduceOp op, unsigned numLaneToReduce,
                  unsigned interleave) const override;

  std::string getMulhiFuncName(Type resultElementTy) const override;

  void printf(RewriterBase &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void assertFail(RewriterBase &rewriter, Location loc, StringRef message,
                  StringRef file, StringRef func, int line) const override;

  int getSharedAddressSpace() const override;

  int getAddressSpace(Attribute addressSpace) const override;

  bool supportVectorizedAtomics() const override;

  // ==========================================================================
  // Custom intrinsic helpers
  // ==========================================================================

  /// Get or create the declaration for a custom intrinsic function
  LLVM::LLVMFuncOp getOrInsertIntrinsic(RewriterBase &rewriter, ModuleOp module,
                                        StringRef name,
                                        LLVM::LLVMFunctionType type,
                                        ArrayRef<StringRef> attrs = {}) const;

  /// Get lane ID (0..warp_size-1) via llvm.custom.lane.id intrinsic
  Value getLaneId(RewriterBase &rewriter, Location loc) const;

  /// Get warp size via llvm.custom.warp.size intrinsic
  Value getWarpSizeValue(RewriterBase &rewriter, Location loc) const;

  /// Get number of programs via llvm.custom.num.programs intrinsic
  Value getNumPrograms(RewriterBase &rewriter, Location loc,
                       ProgramIDDim axis) const;

private:
  int32_t warpSize_;
};

} // namespace mlir::triton::Custom

#endif // TRITON_THIRD_PARTY_CUSTOM_LIB_TRITONCUSTOMTOLLVM_TARGETINFO_H_
