# Custom SIMT LLVM Backend Implementation - Changes Summary

Date: 2025-12-22

This document summarizes the changes made to implement SIMT support in the LLVM RISC-V backend for the Custom Vector core.

## Overview

The implementation adds support for the Custom SIMT Vector Core extension to the LLVM RISC-V backend. This enables Triton to generate efficient code for the SIMT execution model using custom opcode `0x0B` instructions.

## Files Added

### Intrinsics Definition
- **`llvm/include/llvm/IR/IntrinsicsCustomSIMT.td`**
  - Defines all `llvm.custom.*` intrinsics including:
    - Thread/Block ID: `tid.x/y/z`, `ctaid.x/y/z`, `ntid.x/y/z`
    - Warp primitives: `lane.id`, `warp.size`, `activemask`
    - Synchronization: `barrier`, `warp.barrier`
    - Shuffle: `shuffle.xor/up/down/idx` for i32, i64, f32, f64
    - Vote: `ballot`, `all`, `any`
    - Lane mask: `lmask.push`

### RISC-V Target Files
- **`llvm/lib/Target/RISCV/RISCVSystemOperandsSIMT.td`**
  - Defines SIMT CSR addresses (0xFC0-0xFEC)
  - Thread ID, Block dimension, CTA ID, Lane ID, Warp size, Active mask CSRs

- **`llvm/lib/Target/RISCV/RISCVInstrFormatsSIMT.td`**
  - Defines instruction format classes for custom opcode 0x0B:
    - RRTR, RRPTR, RRTP, PPTP, PPTR, RRRP, CTRL types
  - Defines funct7 values for all SIMT instructions

- **`llvm/lib/Target/RISCV/RISCVInstrInfoSIMT.td`**
  - Defines predicate register class (p0-p7)
  - Defines all SIMT instructions:
    - Shuffle: SHFL.IDX, SHFL.BFLY
    - Barrier: BAR.SYNC
    - Lane mask: LMASK.PUSH
    - Predicate compare: PCMP.EQ/NE/LT/LTU/LE, PCMP.F.EQ/LT/LE
    - Predicate select: PSEL, PSEL.F
    - Predicate logic: PAND, POR, PXOR, PNOT, PMOV
    - Predicate transfer: PMOV.TO.X, PMOV.FROM.X
    - Predicated load: PLB/PLBU/PLH/PLHU/PLW/PFLW
    - Predicated store: PSB/PSH/PSW/PFSW
  - Defines pseudo instructions for CSR reads and barrier
  - Defines instruction selection patterns

- **`llvm/lib/Target/RISCV/RISCVExpandSIMTPseudoInsts.cpp`**
  - Machine pass to expand SIMT pseudo instructions:
    - PseudoReadSIMT_* → csrr with correct CSR address
    - PseudoSIMT_BARRIER → fence rw,rw + bar.sync
    - PseudoSIMT_LMASK_PUSH_LABEL → auipc+addi + lmask.push

- **`llvm/lib/Target/RISCV/RISCVInsertLMaskPush.cpp`**
  - Machine pass to insert LMASK.PUSH before divergent branches
  - Uses post-dominator tree to find reconvergence points

### Tests
- **`llvm/test/CodeGen/RISCV/custom-simt-csrr.ll`**
  - Tests CSR reads for all ID intrinsics
  
- **`llvm/test/CodeGen/RISCV/custom-simt-barrier.ll`**
  - Tests barrier lowering to fence + bar.sync
  
- **`llvm/test/CodeGen/RISCV/custom-simt-shuffle.ll`**
  - Tests shuffle intrinsic lowering
  
- **`llvm/test/CodeGen/RISCV/custom-simt-predicate.ll`**
  - Tests predicate instructions (PCMP, PSEL, PL*, PS*)
  
- **`llvm/test/CodeGen/RISCV/custom-simt-kernel.ll`**
  - Tests complete kernel examples

## Files Modified

- **`llvm/include/llvm/IR/Intrinsics.td`**
  - Added include for IntrinsicsCustomSIMT.td

- **`llvm/lib/Target/RISCV/RISCV.td`**
  - Added include for RISCVInstrInfoSIMT.td

- **`llvm/lib/Target/RISCV/RISCV.h`**
  - Declared new passes: createRISCVExpandSIMTPseudoPass, createRISCVInsertLMaskPushPass
  - Declared pass initialization functions

- **`llvm/lib/Target/RISCV/RISCVTargetMachine.cpp`**
  - Added pass initialization in LLVMInitializeRISCVTarget()
  - Added RISCVInsertLMaskPushPass to addPreRegAlloc() pipeline stage
  - Added RISCVExpandSIMTPseudoPass to addPreEmitPass2() pipeline stage

- **`llvm/lib/Target/RISCV/CMakeLists.txt`**
  - Added new source files to build

- **`llvm/lib/Target/RISCV/RISCVFeatures.td`**
  - Contains FeatureVendorXCustomSIMT definition (pre-existing)

## Feature Flag

Enable SIMT extension with:
```
-mattr=+xcustomsimt
```

Or in LLVM IR:
```llvm
target datalayout = "e-m:e-p:32:32-..."
target triple = "riscv32-unknown-unknown"
```

## Usage Example

```llvm
define void @add_kernel(ptr %x, ptr %y, ptr %out, i32 %n) #0 {
  %tid = call i32 @llvm.custom.tid.x()
  %ctaid = call i32 @llvm.custom.ctaid.x()
  %ntid = call i32 @llvm.custom.ntid.x()
  %gid = add i32 (mul i32 %ctaid, %ntid), %tid
  
  %cmp = icmp ult i32 %gid, %n
  br i1 %cmp, label %body, label %exit

body:
  ; ... load, compute, store ...
  br label %exit

exit:
  ret void
}

attributes #0 = { "kernel" }
```

## Next Steps

1. ✅ ~~Integrate passes into RISCVTargetMachine pipeline~~ (Done)
2. Add GlobalISel support (optional)
3. Implement ballot instruction (pending ISA encoding)
4. Performance optimizations:
   - Skip LMASK.PUSH for uniform branches
   - Better predicate register allocation
   - Optimize shuffle patterns
