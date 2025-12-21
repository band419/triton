//===- TargetInfo.cpp - Custom SIMT TargetInfo Implementation -------------===//
//
// This file implements the TargetInfo class for the Custom SIMT backend.
// All SIMT operations are lowered to custom LLVM intrinsics that will be
// further processed by a custom LLVM backend or external codegen.
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::Custom {

namespace {

/// Helper to get or insert a function declaration
LLVM::LLVMFuncOp getOrInsertFunction(ModuleOp module, RewriterBase &rewriter,
                                     Location loc, StringRef name,
                                     LLVM::LLVMFunctionType type,
                                     ArrayRef<LLVM::MemoryEffectsAttr> memAttrs = {}) {
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    return func;
  }
  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto func = LLVM::LLVMFuncOp::create(rewriter, loc, name, type,
                                        LLVM::Linkage::External);
  return func;
}

} // namespace

// =============================================================================
// Intrinsic helpers
// =============================================================================

LLVM::LLVMFuncOp TargetInfo::getOrInsertIntrinsic(
    RewriterBase &rewriter, ModuleOp module, StringRef name,
    LLVM::LLVMFunctionType type, ArrayRef<StringRef> attrs) const {
  Location loc = module.getLoc();
  auto func = getOrInsertFunction(module, rewriter, loc, name, type);
  
  // Add function attributes if not already present
  for (auto attr : attrs) {
    if (attr == "convergent") {
      // convergent: prevents LLVM from moving this call past control flow
      // Critical for barrier and cross-lane operations
      func.setConvergent(true);
    } else if (attr == "nounwind") {
      // nounwind: function does not throw exceptions
      func.setNoUnwind(true);
    } else if (attr == "willreturn") {
      // willreturn: function will eventually return (not infinite loop)
      func.setWillReturn(true);
    } else if (attr == "nosync") {
      // nosync: function does not synchronize with another thread
      func.setNoSync(true);
    } else if (attr == "readnone") {
      // readnone: function does not access memory
      // In LLVM 16+, this is expressed via memory(none)
      func.setMemoryEffects(LLVM::MemoryEffectsAttr::get(
          rewriter.getContext(),
          /*other=*/LLVM::ModRefInfo::NoModRef,
          /*argMem=*/LLVM::ModRefInfo::NoModRef,
          /*inaccessibleMem=*/LLVM::ModRefInfo::NoModRef));
    } else if (attr == "readonly") {
      // readonly: function only reads memory
      func.setMemoryEffects(LLVM::MemoryEffectsAttr::get(
          rewriter.getContext(),
          /*other=*/LLVM::ModRefInfo::Ref,
          /*argMem=*/LLVM::ModRefInfo::Ref,
          /*inaccessibleMem=*/LLVM::ModRefInfo::Ref));
    }
  }
  return func;
}

Value TargetInfo::getLaneId(RewriterBase &rewriter, Location loc) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = rewriter.getI32Type();
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {});
  
  // lane.id is pure: readnone, nounwind, willreturn
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.custom.lane.id", funcTy,
                       {"readnone", "nounwind", "willreturn"});
  
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{}).getResult();
}

Value TargetInfo::getWarpSizeValue(RewriterBase &rewriter, Location loc) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = rewriter.getI32Type();
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {});
  
  // warp.size is pure: readnone, nounwind, willreturn
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.custom.warp.size", funcTy,
                       {"readnone", "nounwind", "willreturn"});
  
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{}).getResult();
}

Value TargetInfo::getNumPrograms(RewriterBase &rewriter, Location loc,
                                 ProgramIDDim axis) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = rewriter.getI32Type();
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});
  
  // num.programs is pure: readnone, nounwind, willreturn
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.custom.num.programs", funcTy,
                       {"readnone", "nounwind", "willreturn"});
  
  Value axisVal = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(static_cast<int32_t>(axis)));
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{axisVal}).getResult();
}

// =============================================================================
// TargetInfoBase interface implementation
// =============================================================================

bool TargetInfo::supportMaximumMinimum() const {
  // Custom backend doesn't have native maximum/minimum instructions
  return false;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  // Custom backend doesn't support multi-CTA clusters, return 0
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(0));
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  // Ballot: gather predicate from all lanes into a bitmask
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = rewriter.getI32Type();
  auto i1Ty = rewriter.getI1Type();
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i1Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.custom.ballot", funcTy,
                       {"convergent", "nounwind"});
  
  Value result = LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{cmp}).getResult();
  
  // If the result type is different from i32, extend/truncate
  if (type != i32Ty) {
    if (type.getIntOrFloatBitWidth() > 32) {
      result = LLVM::ZExtOp::create(rewriter, loc, type, result);
    } else {
      result = LLVM::TruncOp::create(rewriter, loc, type, result);
    }
  }
  return result;
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         bool isWarpSync) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
  
  StringRef intrinsicName = isWarpSync ? "llvm.custom.warp.barrier"
                                       : "llvm.custom.barrier";
  auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {});
  
  // Barrier intrinsic attributes:
  // - convergent: cannot be moved past control flow
  // - nounwind: does not throw exceptions  
  // Note: memory effects (ModRef on all memory) are set in getOrInsertIntrinsic
  // to act as a full memory fence, preventing load/store reordering
  auto funcOp = getOrInsertIntrinsic(rewriter, module, intrinsicName, funcTy,
                       {"convergent", "nounwind"});
  
  // Set memory effects explicitly for barrier fence semantics
  // This ensures the barrier acts as a full memory fence
  funcOp.setMemoryEffects(LLVM::MemoryEffectsAttr::get(
      rewriter.getContext(),
      /*other=*/LLVM::ModRefInfo::ModRef,
      /*argMem=*/LLVM::ModRefInfo::ModRef,
      /*inaccessibleMem=*/LLVM::ModRefInfo::ModRef));
  
  LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{});
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  // Custom SIMT backend does NOT support shared memory.
  // All memory is global memory only.
  llvm::report_fatal_error(
      "Custom SIMT backend does not support shared memory operations. "
      "Use global memory only.");
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  // Custom SIMT backend does NOT support shared memory.
  // All memory is global memory only.
  llvm::report_fatal_error(
      "Custom SIMT backend does not support shared memory operations. "
      "Use global memory only.");
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto valTy = val.getType();
  auto i32Ty = rewriter.getI32Type();
  
  // Shuffle intrinsic: llvm.custom.shuffle.xor(value, lane_mask) -> value
  auto funcTy = LLVM::LLVMFunctionType::get(valTy, {valTy, i32Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.custom.shuffle.xor", funcTy,
                       {"convergent", "nounwind"});
  
  Value laneMask = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                            rewriter.getI32IntegerAttr(i));
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{val, laneMask}).getResult();
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto valTy = val.getType();
  auto i32Ty = rewriter.getI32Type();
  
  auto funcTy = LLVM::LLVMFunctionType::get(valTy, {valTy, i32Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.custom.shuffle.up", funcTy,
                       {"convergent", "nounwind"});
  
  Value delta = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                         rewriter.getI32IntegerAttr(i));
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{val, delta}).getResult();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto valTy = val.getType();
  auto i32Ty = rewriter.getI32Type();
  
  auto funcTy = LLVM::LLVMFunctionType::get(valTy, {valTy, i32Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.custom.shuffle.idx", funcTy,
                       {"convergent", "nounwind"});
  
  Value laneIdx = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(i));
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{val, laneIdx}).getResult();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto valTy = val.getType();
  auto i32Ty = rewriter.getI32Type();
  
  auto funcTy = LLVM::LLVMFunctionType::get(valTy, {valTy, i32Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.custom.shuffle.idx", funcTy,
                       {"convergent", "nounwind"});
  
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{val, i}).getResult();
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = rewriter.getI32Type();
  
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.custom.permute", funcTy,
                       {"nounwind"});
  
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{a, b, selector}).getResult();
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  auto i32Ty = rewriter.getI32Type();
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});
  
  // program.id is pure: readnone, nounwind, willreturn
  auto funcOp = getOrInsertIntrinsic(rewriter, moduleOp, "llvm.custom.program.id", funcTy,
                       {"readnone", "nounwind", "willreturn"});
  
  Value axisVal = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(static_cast<int32_t>(axis)));
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{axisVal}).getResult();
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  // Custom backend doesn't have specialized warp reduce instructions
  // Return false to use the generic tree reduction
  return false;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  if (resultElementTy.isInteger(32)) {
    return "__mulhi_i32";
  } else if (resultElementTy.isInteger(64)) {
    return "__mulhi_i64";
  }
  return "__mulhi";
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  // Custom backend printf - call a runtime function
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Location loc = formatStrStart.getLoc();
  
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto i32Ty = rewriter.getI32Type();
  
  // Simple printf: int printf(const char* format, ...)
  // For now, just declare and call - actual implementation is runtime-dependent
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {ptrTy}, /*isVarArg=*/true);
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "printf", funcTy, {"nounwind"});
  
  SmallVector<Value> printfArgs;
  printfArgs.push_back(formatStrStart);
  printfArgs.append(args.begin(), args.end());
  
  LLVM::CallOp::create(rewriter, loc, funcOp, printfArgs);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  // Create format string and call printf
  // For debug purposes - implementation can be expanded as needed
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
  auto funcTy = LLVM::LLVMFunctionType::get(voidTy, {});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.custom.assert.fail", funcTy,
                       {"nounwind"});
  
  LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{});
}

int TargetInfo::getSharedAddressSpace() const {
  // Custom backend uses address space 0 for everything (global memory only)
  // If we later add shared memory support, this would return a different value
  return 0;
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  // For Custom backend, all address spaces map to 0 (flat/global)
  return 0;
}

bool TargetInfo::supportVectorizedAtomics() const {
  // Custom backend doesn't support vectorized atomics
  return false;
}

} // namespace mlir::triton::Custom
