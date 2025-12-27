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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
      // Note: nosync is not directly supported in new LLVM MLIR dialect
      // It can be added via passthrough if needed, but we skip it for now
      // as it's mainly an optimization hint
    } else if (attr == "readnone") {
      // readnone: function does not access memory
      // In LLVM 16+, this is expressed via memory(none)
      // Args: other, argMem, inaccessibleMem, errnoMem, targetMem0, targetMem1
      func.setMemoryEffectsAttr(LLVM::MemoryEffectsAttr::get(
          rewriter.getContext(),
          llvm::ArrayRef<LLVM::ModRefInfo>{
              LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef,
              LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef,
              LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef}));
    } else if (attr == "readonly") {
      // readonly: function only reads memory
      // Args: other, argMem, inaccessibleMem, errnoMem, targetMem0, targetMem1
      func.setMemoryEffectsAttr(LLVM::MemoryEffectsAttr::get(
          rewriter.getContext(),
          llvm::ArrayRef<LLVM::ModRefInfo>{
              LLVM::ModRefInfo::Ref, LLVM::ModRefInfo::Ref,
              LLVM::ModRefInfo::Ref, LLVM::ModRefInfo::Ref,
              LLVM::ModRefInfo::Ref, LLVM::ModRefInfo::Ref}));
    }
  }
  return func;
}

Value TargetInfo::getLaneId(RewriterBase &rewriter, Location loc) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = rewriter.getI32Type();
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {});
  
  // lane.id is pure: readnone, nounwind, willreturn
  // Maps to llvm.riscv.simt.lane.id -> CSR_SIMT_LANEID (0xFE4)
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.riscv.simt.lane.id", funcTy,
                       {"readnone", "nounwind", "willreturn"});
  
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{}).getResult();
}

Value TargetInfo::getWarpSizeValue(RewriterBase &rewriter, Location loc) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = rewriter.getI32Type();
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {});
  
  // warp.size is pure: readnone, nounwind, willreturn
  // Maps to llvm.riscv.simt.warp.size -> CSR_SIMT_WARPSIZE (0xFE8)
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.riscv.simt.warp.size", funcTy,
                       {"readnone", "nounwind", "willreturn"});
  
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{}).getResult();
}

Value TargetInfo::getNumPrograms(RewriterBase &rewriter, Location loc,
                                 ProgramIDDim axis) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = rewriter.getI32Type();
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});
  
  // num.programs is pure: readnone, nounwind, willreturn
  // Maps to llvm.riscv.simt.num.programs -> grid_size from kernel descriptor
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.riscv.simt.num.programs", funcTy,
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
  // Maps to llvm.riscv.simt.ballot_mask
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = rewriter.getI32Type();
  auto i1Ty = rewriter.getI1Type();
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i1Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.riscv.simt.ballot.mask", funcTy,
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
  
  // Maps to RISCV SIMT barrier instructions
  StringRef intrinsicName = isWarpSync ? "llvm.riscv.simt.warp.barrier"
                                       : "llvm.riscv.simt.barrier";
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
  // Args: other, argMem, inaccessibleMem, errnoMem, targetMem0, targetMem1
  funcOp.setMemoryEffectsAttr(LLVM::MemoryEffectsAttr::get(
      rewriter.getContext(),
      llvm::ArrayRef<LLVM::ModRefInfo>{
          LLVM::ModRefInfo::ModRef, LLVM::ModRefInfo::ModRef,
          LLVM::ModRefInfo::ModRef, LLVM::ModRefInfo::ModRef,
          LLVM::ModRefInfo::ModRef, LLVM::ModRefInfo::ModRef}));
  
  LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{});
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  // Custom SIMT backend emulates shared memory via global scratch.
  // ctaId is not supported (CTA-internal exchange only).
  assert(!ctaId.has_value() && 
         "Custom backend does not support cross-CTA shared memory access");

  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // Handle non-vector types: wrap in vector
  if (!isa<VectorType>(val.getType())) {
    storeDShared(rewriter, loc, ptr, ctaId, 
                 ::mlir::packLLVector(loc, {val}, rewriter), pred);
    return;
  }

  auto vecTy = cast<VectorType>(val.getType());
  Type elemTy = vecTy.getElementType();
  unsigned vec = vecTy.getNumElements();
  unsigned elemBitwidth = ::mlir::getIntOrFloatOrPtrBitWidth(elemTy);

  // Handle sub-byte elements: extend to 8-bit
  if (elemBitwidth < 8) {
    assert(vec == 1 && 
           "don't know how to store vectors of sub-byte elems");
    SmallVector<Value> vals = ::mlir::unpackLLVector(loc, val, rewriter);
    for (Value &v : vals) {
      v = b.zext(int_ty(8), b.bitcast(v, int_ty(elemBitwidth)));
    }
    storeDShared(rewriter, loc, ptr, ctaId, 
                 ::mlir::packLLVector(loc, vals, rewriter), pred);
    return;
  }

  // Handle non-integer types: convert to integers
  if (!elemTy.isInteger()) {
    SmallVector<Value> vals = ::mlir::unpackLLVector(loc, val, rewriter);
    for (Value &v : vals) {
      if (isa<LLVM::LLVMPointerType>(v.getType())) {
        v = b.ptrtoint(int_ty(elemBitwidth), v);
      } else {
        v = b.bitcast(v, int_ty(elemBitwidth));
      }
    }
    storeDShared(rewriter, loc, ptr, ctaId, 
                 ::mlir::packLLVector(loc, vals, rewriter), pred);
    return;
  }

  // Handle vectors larger than v4 with small elements: pack to b32
  if (vec > 4 && elemBitwidth < 32) {
    assert(llvm::isPowerOf2_32(vec));
    int elemsPerPack = 32 / elemBitwidth;
    SmallVector<Value> oldVals = ::mlir::unpackLLVector(loc, val, rewriter);

    SmallVector<Value> newVals;
    for (unsigned i = 0; i < vec / elemsPerPack; i++) {
      Value v = ::mlir::packLLVector(
          loc, ArrayRef(oldVals).slice(i * elemsPerPack, elemsPerPack),
          rewriter);
      newVals.push_back(b.bitcast(v, i32_ty));
    }
    storeDShared(rewriter, loc, ptr, ctaId,
                 ::mlir::packLLVector(loc, newVals, rewriter), pred);
    return;
  }

  // Handle vectors exceeding 128 bits: split into multiple stores
  if (vec * elemBitwidth > 128) {
    assert(llvm::isPowerOf2_32(vec));
    assert(elemBitwidth == 32 || elemBitwidth == 64);
    int maxVec = 128 / elemBitwidth;

    SmallVector<Value> vals = ::mlir::unpackLLVector(loc, val, rewriter);
    for (unsigned i = 0; i < vec / maxVec; i++) {
      auto newPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i * maxVec),
                          LLVM::GEPNoWrapFlags::inbounds);
      storeDShared(
          rewriter, loc, newPtr, ctaId,
          ::mlir::packLLVector(loc, ArrayRef(vals).slice(i * maxVec, maxVec), rewriter),
          pred);
    }
    return;
  }

  // Final store: use predicated store via scf.if
  assert(elemBitwidth >= 8);
  assert(elemTy.isInteger());
  assert(1 <= vec && vec <= 4);
  assert(vec * elemBitwidth <= 128);

  // Check if predicate is constant true
  bool isConstantTrue = false;
  if (auto constOp = pred.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      isConstantTrue = intAttr.getInt() != 0;
    }
  }

  if (isConstantTrue) {
    // Unconditional store
    unsigned align = vec * elemBitwidth / 8;
    b.store(val, ptr, align);
  } else {
    // Predicated store using scf.if
    scf::IfOp::create(
        rewriter, loc, pred,
        [&](OpBuilder &thenBuilder, Location thenLoc) {
          auto tb = TritonLLVMOpBuilder(thenLoc, thenBuilder);
          unsigned align = vec * elemBitwidth / 8;
          tb.store(val, ptr, align);
          scf::YieldOp::create(thenBuilder, thenLoc);
        });
  }
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type loadTy,
                              Value pred, Operation *localLoadOp) const {
  // Custom SIMT backend emulates shared memory via global scratch.
  // ctaId is not supported (CTA-internal exchange only).
  assert(!ctaId.has_value() && 
         "Custom backend does not support cross-CTA shared memory access");

  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // Handle non-vector types: wrap in vector
  if (!isa<VectorType>(loadTy)) {
    SmallVector<Value> values = ::mlir::unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, vec_ty(loadTy, 1), pred),
        rewriter);
    assert(values.size() == 1);
    return values[0];
  }

  auto vecTy = cast<VectorType>(loadTy);
  Type elemTy = vecTy.getElementType();
  unsigned vec = vecTy.getNumElements();
  unsigned elemBitwidth = ::mlir::getIntOrFloatOrPtrBitWidth(elemTy);

  // Handle sub-byte elements: load as 8-bit
  if (elemBitwidth < 8) {
    assert(vec == 1 && 
           "don't know how to load vectors of sub-byte elems");
    SmallVector<Value> vals = ::mlir::unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, int_ty(8), pred), rewriter);
    assert(vals.size() == 1);
    return b.bitcast(b.trunc(int_ty(elemBitwidth), vals[0]), elemTy);
  }

  // Handle non-integer types: load as integers and convert
  if (!elemTy.isInteger()) {
    Type newLoadTy = vec_ty(int_ty(elemBitwidth), vec);
    SmallVector<Value> vals = ::mlir::unpackLLVector(
        loc, loadDShared(rewriter, loc, ptr, ctaId, newLoadTy, pred), rewriter);
    for (Value &v : vals) {
      v = b.bitcast(v, elemTy);
    }
    return ::mlir::packLLVector(loc, vals, rewriter);
  }

  // Handle vectors larger than v4 with small elements: load as b32
  if (vec > 4 && elemBitwidth < 32) {
    int newVec = vec / (32 / elemBitwidth);
    auto newVecTy = vec_ty(i32_ty, newVec);
    auto res = loadDShared(rewriter, loc, ptr, ctaId, newVecTy, pred);

    // Unpack the b32's into the original vector type
    SmallVector<Value> vals;
    for (Value v : ::mlir::unpackLLVector(loc, res, rewriter)) {
      Value vv = b.bitcast(v, vec_ty(elemTy, 32 / elemBitwidth));
      for (Value vvv : ::mlir::unpackLLVector(loc, vv, rewriter)) {
        vals.push_back(vvv);
      }
    }
    return ::mlir::packLLVector(loc, vals, rewriter);
  }

  // Handle vectors exceeding 128 bits: split into multiple loads
  if (vec * elemBitwidth > 128) {
    assert(elemBitwidth == 32 || elemBitwidth == 64);
    assert(llvm::isPowerOf2_32(vec));
    int maxVec = 128 / elemBitwidth;

    SmallVector<Value> vals;
    for (unsigned i = 0; i < vec / maxVec; i++) {
      auto newPtr = b.gep(ptr.getType(), elemTy, ptr, b.i32_val(i * maxVec),
                          LLVM::GEPNoWrapFlags::inbounds);
      auto newVal = loadDShared(rewriter, loc, newPtr, ctaId,
                                vec_ty(elemTy, maxVec), pred);
      for (Value v : ::mlir::unpackLLVector(loc, newVal, rewriter)) {
        vals.push_back(v);
      }
    }
    return ::mlir::packLLVector(loc, vals, rewriter);
  }

  // Final load: use predicated load via scf.if
  assert(elemBitwidth >= 8);
  assert(elemTy.isInteger());
  assert(1 <= vec && vec <= 4);
  assert(vec * elemBitwidth <= 128);

  Type resultTy = vec == 1 ? Type(int_ty(elemBitwidth))
                           : Type(vec_ty(int_ty(elemBitwidth), vec));

  // Check if predicate is constant true
  bool isConstantTrue = false;
  if (auto constOp = pred.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      isConstantTrue = intAttr.getInt() != 0;
    }
  }

  Value load;
  if (isConstantTrue) {
    // Unconditional load
    unsigned align = vec * elemBitwidth / 8;
    load = b.load(resultTy, ptr, align);
  } else {
    // Predicated load using scf.if with else branch returning undef
    // Create IfOp with withElseRegion=true
    auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{resultTy}, pred,
                                  /*withElseRegion=*/true);
    
    // Fill the then region
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto tb = TritonLLVMOpBuilder(loc, rewriter);
      unsigned align = vec * elemBitwidth / 8;
      Value loaded = tb.load(resultTy, ptr, align);
      scf::YieldOp::create(rewriter, loc, loaded);
    }
    
    // Fill the else region
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Value undef = LLVM::UndefOp::create(rewriter, loc, resultTy);
      scf::YieldOp::create(rewriter, loc, undef);
    }
    
    load = ifOp.getResult(0);
  }

  // Convert to vector format for return
  SmallVector<Value> resultVals;
  if (vec == 1) {
    resultVals.push_back(load);
  } else {
    for (unsigned i = 0; i < vec; i++) {
      resultVals.push_back(b.extract_element(load, b.i32_val(i)));
    }
  }
  return ::mlir::packLLVector(loc, resultVals, rewriter);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto valTy = val.getType();
  auto i32Ty = rewriter.getI32Type();
  
  // Shuffle intrinsic: llvm.riscv.simt.shfl_bfly(value, mask) -> value
  // XOR shuffle is implemented using butterfly shuffle pattern
  auto funcTy = LLVM::LLVMFunctionType::get(valTy, {valTy, i32Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.riscv.simt.shfl.bfly", funcTy,
                       {"convergent", "nounwind"});
  
  Value laneMask = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                            rewriter.getI32IntegerAttr(i));
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{val, laneMask}).getResult();
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  // shuffle.up(val, delta) = get value from lane (lane_id - delta)
  // Implemented using shfl.idx with computed source lane
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto valTy = val.getType();
  auto i32Ty = rewriter.getI32Type();
  
  auto funcTy = LLVM::LLVMFunctionType::get(valTy, {valTy, i32Ty});
  
  // Use llvm.riscv.simt.shfl.idx with lane_id - delta
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.riscv.simt.shfl.idx", funcTy,
                       {"convergent", "nounwind"});
  
  // Compute source lane: lane_id - delta
  Value laneId = getLaneId(rewriter, loc);
  Value delta = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                         rewriter.getI32IntegerAttr(i));
  Value srcLane = LLVM::SubOp::create(rewriter, loc, laneId, delta);
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{val, srcLane}).getResult();
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto valTy = val.getType();
  auto i32Ty = rewriter.getI32Type();
  
  // Maps to llvm.riscv.simt.shfl.idx(value, src_lane)
  auto funcTy = LLVM::LLVMFunctionType::get(valTy, {valTy, i32Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.riscv.simt.shfl.idx", funcTy,
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
  
  // Maps to llvm.riscv.simt.shfl.idx(value, src_lane)
  auto funcTy = LLVM::LLVMFunctionType::get(valTy, {valTy, i32Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.riscv.simt.shfl.idx", funcTy,
                       {"convergent", "nounwind"});
  
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{val, i}).getResult();
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  auto module = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto i32Ty = rewriter.getI32Type();
  
  // Maps to llvm.riscv.simt.permute for byte permutation
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty});
  
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.riscv.simt.permute", funcTy,
                       {"nounwind"});
  
  return LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{a, b, selector}).getResult();
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  auto i32Ty = rewriter.getI32Type();
  auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty});
  
  // program.id is pure: readnone, nounwind, willreturn
  // Maps to llvm.riscv.simt.program.id -> CSR_SIMT_CTAID_X (0xFD8)
  auto funcOp = getOrInsertIntrinsic(rewriter, moduleOp, "llvm.riscv.simt.program.id", funcTy,
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
  
  // Maps to llvm.riscv.simt.assert.fail
  auto funcOp = getOrInsertIntrinsic(rewriter, module, "llvm.riscv.simt.assert.fail", funcTy,
                       {"nounwind"});
  
  LLVM::CallOp::create(rewriter, loc, funcOp, ValueRange{});
}

int TargetInfo::getSharedAddressSpace() const {
  // Custom backend emulates shared memory via global scratch (address space 1).
  // This must match the address space used by getGlobalScratchPtr in Utility.cpp.
  return 1;
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
