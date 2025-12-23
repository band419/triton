# Custom SIMT on RISC-V LLVM Backend: Implementation Plan

> Date: 2025-12-22
>
> Goal: modify/extend the LLVM **RISC-V target backend** to codegen for the custom SIMT Vector core, consuming the custom LLIR contract produced by Triton custom pipeline.
>
> Scope of this plan:
> - **Codegen correctness first**, then performance.
> - Keep Triton LLIR stable: the contract remains `llvm.custom.*` + standard LLVM IR control flow/load/store.
> - Lower SIMT semantics to the Vector ISA (custom opcode `0x0B`, CSRs, predicate regs, LMASK, etc.).

---

## 0. Inputs (spec + contract)

### 0.1 Hardware/ISA (Vector SIMT)

Source: `docs/gpnpu_spec/Vector/Vector_Arch_Spec.md`

- Base (v1): **RV32IMF + zicsr** (SIMT custom extension opcode `0x0B`).
  - **Non-goal (v1)**: RVV (RISC-V Vector Extension). Do not assume RVV types, registers, or vsetvl/vscale.
- Execution: **Multi-PC SIMT** / independent lane PC + warp scheduler issuing execution groups (same PC + mask).
- Control flow: compiler inserts `LMASK.PUSH` before divergent branches; hardware auto reconverges via branch-tree + lane stack.
- Predication:
  - Predicate regs `p0..p7` (1-bit per lane; warp-wide view 32-bit mask).
  - Predicate compare/select: `PCMP.*` → predicate, `PSEL` → select.
  - Predicated memory: `PL*` / `PS*`.
- Sync:
  - `BAR.SYNC` (funct7 `0x40`) is **CTA-level sync**.
  - **Decision**: `BAR.SYNC` does **not** imply fence.

### 0.2 Custom LLIR contract

Sources:
- `docs/custom-intrinsic-spec.md`
- `docs/custom-kernel-abi.md`
- `docs/custom-memory-semantics.md`

Key points:
- Triple: `riscv32-unknown-unknown` (subject to future custom triple naming).
- Global memory only: pointers are `addrspace(1)`.
- SIMT semantics via `llvm.custom.*` calls (barrier/program_id/lane_id/shuffle/ballot).
- Barrier has **sync + fence** semantics at the IR contract level.
- Shuffle intrinsics exist in LLIR as `xor/up/down/idx`.

---

## 1. Decisions already made (locked for v1)

1. **CSR addresses follow Vector spec**
   - Use CSR addresses as defined in `docs/gpnpu_spec/Vector/Vector_Arch_Spec.md`.
   - All other docs should be updated to match.

2. **Shuffle lowering strategy (v1)**
   - All `llvm.custom.shuffle.*` lower to **`SHFL.IDX`**.
   - For `xor/up/down`, compute `src_lane` from `lane.id` then emit `SHFL.IDX`.

3. **Barrier lowering strategy (v1)**
   - `llvm.custom.barrier()` lowers to: `fence` + `bar.sync`.
   - Rationale: `bar.sync` is CTA sync only; IR contract requires fence.

---

## 2. High-level architecture

We target a pipeline where Triton emits LLIR with `llvm.custom.*`, and LLVM codegen produces machine code using:
- CSR reads for IDs,
- Custom opcode `0x0B` instructions for cross-lane/sync/LMASK,
- Predicate register class + predicated instructions for if-conversion and masked memory.

We avoid requiring LLIR to encode any target-specific inline asm.

---

## 3. Work breakdown (phases)

### Phase A — “Backend can consume LLIR” (minimal correctness)

#### A1. Make `llvm.custom.*` real LLVM intrinsics (preferred)

**Why**: if `llvm.custom.*` remains a plain external function, codegen will emit `call` and cannot reliably lower to CSR/custom instructions.

Implementation sketch:
- Add new intrinsic IDs in LLVM core (`llvm/include/llvm/IR/Intrinsics.td` + generated headers).
- Define attributes precisely:
  - ID intrinsics: `nounwind`, `willreturn`, `memory(none)`.
  - Barrier/shuffle/ballot: `convergent`, `nounwind`.
  - Barrier must carry memory effects consistent with the contract (see `custom-memory-semantics.md`).

Deliverables:
- IR still prints as `@llvm.custom.*`.
- CodeGen can switch on `Intrinsic::ID`.

#### A2. Lower ID intrinsics to CSR reads

Intrinsics:
- `llvm.custom.program.id(axis)`
- `llvm.custom.thread.id(axis)`
- `llvm.custom.block.dim(axis)`
- `llvm.custom.lane.id()`
- `llvm.custom.warp.size()`

Lowering:
- Map to `csrr` with CSR addresses from Vector spec.

Where to implement:
- RISC-V backend lowering (SelectionDAG or GlobalISel; choose one path and keep minimal in v1).

Acceptance criteria:
- `llc` output contains `csrr` to the correct CSRs.
- No libcalls or unresolved calls are introduced.

#### A3. Lower CTA barrier intrinsic

Intrinsic:
- `llvm.custom.barrier()`

Lowering:
- Emit `fence` (recommended: `fence rw, rw` for v1) then `bar.sync` (`opcode 0x0B`, funct7 `0x40`).
- Ensure barrier is treated as a scheduling barrier (side-effects) and not removed.

Acceptance criteria:
- Assembly has `fence` followed by custom `bar.sync`.
- Optimizer does not move memory operations across the intrinsic.

#### A4. Minimal tests (LLVM lit)

Add tests in `llvm-project/llvm/test/CodeGen/RISCV/`:
- `custom-simt-csrr.ll`: FileCheck CSR reads for IDs.
- `custom-simt-barrier.ll`: FileCheck `fence` + `bar.sync` sequence.

---

### Phase B — Independent Thread Scheduling (ITS) correctness: reconvergence

#### B1. Implement `LMASK.PUSH` insertion for divergent branches

Background:
- Hardware uses lane stacks + branch tree; compiler must insert `LMASK.PUSH(reconv_pc)` before branch.
- This is analogous to NVPTX/SASS reconvergence management.

**ISA encoding**: `LMASK.PUSH` is **CTRL TYPE**, funct7=`0x20`, rs1=reconv_pc address.
```
.insn r 0x0B, 0, 0x20, x0, <gpr_with_addr>, x0
```

Strategy (v1 correctness-first):
- Insert `LMASK.PUSH` before **every conditional branch** in kernels (even if uniform), then refine later.

Implementation details:
- Create a MachineFunction pass (post-ISel, pre-emit) that:
  - Finds conditional branches (`BEQ`, `BNE`, `BLT`, etc.).
  - Computes reconvergence block (initially: immediate postdom/merge block; or conservative choice if unknown).
  - Emits a pseudo `LMASK_PUSH_LABEL %mergeBB`.
- Expand pseudo late:
  - Materialize target address into GPR (e.g. `auipc` + `addi` / `la`-style sequence for PC-relative).
  - Emit actual custom instruction with funct7=0x20.

Acceptance criteria:
- For a simple `if/else`, assembly shows an `LMASK.PUSH` before the branch and uses the merge label address.
- Programs reconverge correctly on the simulator/functional model.

---

### Phase C — Predication: predicate regs + predicated instructions

Goal: exploit `p0..p7` and `PCMP/PSEL/PL*/PS*` to reduce control-flow overhead and match hardware.

#### C1. Add predicate register class

- Define physical predicate regs `p0..p7` (8 registers).
- Predicate encoding: uses 3-bit field (funct3 position for src, or rd position for dst).
- Add a RegisterClass `PReg` for predicates with register allocation.
- Add printing/parsing support (asm printer + asm matcher/tablegen) for predicate operands.

**Key point**: Each predicate register is logically 1-bit per lane (warp-wide = 32-bit mask), but from compiler's view it's a single "predicate" value.

#### C2. Instruction selection patterns (v1)

Start with common patterns produced by Triton custom lowering:

1) `select i1 %cond, %a, %b`:
- If `%cond` is in GPR (0/1): emit `PMOV.FROM.X pd, rs1` (funct7=0x60) to get predicate.
- Then emit `PSEL rd, rs_true, rs_false, pd` (funct7=0x58/0x59).

2) Masked store (control-flow form):
- Recognize `br %mask -> store` shape.
- Lower to `PS* rs3, base, offset, ps` (funct7=0x67-0x6A).
- Address = base + offset (rs2); use x0 for zero offset.

3) Masked load (load + select with 0):
- When `other` is 0, map to `PL* rd, base, offset, ps` (funct7=0x61-0x66).
- Semantic: if `ps[lane] && LMASK_ACT[lane]`, load; else rd=0.

4) Integer/FP compare to predicate:
- `icmp eq/ne/lt/le` → `PCMP.*` (funct7=0x50-0x54).
- `fcmp oeq/olt/ole` → `PCMP.F.*` (funct7=0x55-0x57).

#### C3. Predicate logic operations

- `and/or/xor` of predicates → `PAND/POR/PXOR` (funct7=0x5A-0x5C).
- `not` of predicate → `PNOT` (funct7=0x5D).
- Copy predicate → `PMOV` (funct7=0x5E).
- Convert predicate to GPR → `PMOV.TO.X` (funct7=0x5F).

#### C4. Spill strategy (v1 minimal)

- If predicates spill, use `PMOV.TO.X` to GPR then spill GPR to stack.
- Restore: load GPR from stack, then `PMOV.FROM.X` back to predicate.
- This is not optimal but provides correctness.

Acceptance criteria:
- Some kernels show `PCMP/PSEL/PL/PS` in generated assembly.
- No miscompile due to predicate register allocation.

---

### Phase D — Shuffle/ballot via cross-lane unit

#### D1. Implement shuffle lowering (v1: all to IDX)

Intrinsics:
- `llvm.custom.shuffle.idx.i32(val, src_lane)` → direct emit `SHFL.IDX`
- `llvm.custom.shuffle.xor.i32(val, mask)` → compute `src_lane = lane_id ^ mask`, emit `SHFL.IDX`
- `llvm.custom.shuffle.up.i32(val, delta)` → compute `src_lane = lane_id - delta`, emit `SHFL.IDX`
- `llvm.custom.shuffle.down.i32(val, delta)` → compute `src_lane = lane_id + delta`, emit `SHFL.IDX`

**ISA encoding**: `SHFL.IDX` is **RRTR TYPE**, funct7=`0x31`.
```
.insn r 0x0B, 0, 0x31, rd, rs1_val, rs2_src_lane
```

Implementation:
- For `xor/up/down`, first emit `csrr tmp, CSR_SIMT_LANEID` (0xFE4).
- Compute `src_lane` in GPR using standard RISC-V ALU ops.
- Emit `SHFL.IDX rd, val, src_lane`.

Float versions (`shuffle.*.f32`) work the same but use FPR.

#### D2. Implement ballot lowering

Intrinsic:
- `llvm.custom.ballot(i1 pred)` → returns i32 bitmask

Lowering:
- **ISA encoding TBD**: If ballot has dedicated funct7, emit it; otherwise synthesize via `PMOV.FROM.X` + cross-lane reduction (not ideal).
- v1 plan: if not available, emit error or fallback warning.

Acceptance criteria:
- `shuffle.*` always emits `SHFL.IDX` (with lane calculation if needed).
- `ballot` emits a single custom instruction (once ISA encoding confirmed).

---

## 4. Notes on “NVPTX reference” and mapping

This project should refer to NVPTX backend for conceptual alignment:
- Predicate register modeling (`%p` regs, predicated load/store/select).
- Reconvergence and barrier constraints (`convergent` semantics).
- Volta+ ITS implications: barriers and cross-lane ops must not be incorrectly hoisted/sunk across divergent control flow.

Important difference:
- We are not emitting PTX; we emit RISC-V + custom opcode `0x0B` instructions.

---

## 5. Open questions / follow-ups

1) Warp barrier:
- LLIR has `llvm.custom.warp.barrier()`, but Vector ISA currently only fixes CTA `bar.sync`.
- v1 plan: lower warp barrier to CTA barrier (`fence` + `bar.sync`).

2) Reconvergence PC materialization:
- Confirm whether the hardware expects absolute PC, PC-relative, or an address in a specific space.
- Current assumption: absolute address in GPR (`rs1`).

3) Ballot ISA encoding:
- If ISA encoding is not finalized, keep lowering behind a feature flag and fail with clear error until defined.

4) Predicate semantics vs. LMASK:
- Ensure predicated instructions obey `ps[lane] && LMASK_ACT[lane]` rule from Vector spec.
- This means: even if ps=1, if lane is inactive (LMASK_ACT=0), the operation does not fire.

5) Predicated memory address calculation:
- Address = rs1 + rs2 (base + offset).
- For zero offset, use `x0` as rs2.

6) Multi-dimensional support (Y/Z):
- CSRs for Y/Z dimensions exist; intrinsic lowering should respect `axis` parameter.

---

## 6. ISA encoding reference (for TableGen / asm)

All SIMT custom instructions use **opcode `0x0B`**. The `funct7` field determines specific operation.

### 6.1 Instruction format types

| Type   | [31:25] | [24:20] | [19:15] | [14:12]  | [11:7]  | [6:0]  | Notes                          |
|--------|---------|---------|---------|----------|---------|--------|--------------------------------|
| RRTR   | funct7  | rs2     | rs1     | 0x0      | rd      | 0x0B   | GPR×2 → GPR                    |
| RRPTR  | funct7  | rs2     | rs1     | **ps**   | rd      | 0x0B   | GPR×2 + pred → GPR (PSEL etc.) |
| RRTP   | funct7  | rs2     | rs1     | 0x0      | **pd**  | 0x0B   | GPR×2 → pred (PCMP etc.)       |
| PPTP   | funct7  | ps2     | ps1     | 0x0      | pd      | 0x0B   | pred×2 → pred (PAND/POR etc.)  |
| PPTR   | funct7  | ps2     | ps1     | 0x0      | rd      | 0x0B   | pred×2 → GPR (PMOV.TO.X)       |
| RRRP   | funct7  | rs2     | rs1     | **ps**   | rs3     | 0x0B   | GPR×2 + pred → store (PS*)     |
| CTRL   | funct7  | 0x0     | rs1     | 0x0      | 0x0     | 0x0B   | Control (LMASK.PUSH, BAR.SYNC) |

**Notes**:
- `ps/pd` (predicate src/dst) are encoded in 3-bit field (funct3 position or rd position) → values 0-7 map to `p0..p7`.
- For predicated load (PL*): funct3 = ps, rd = GPR dest.
- For predicated store (PS*): funct3 = ps, rd position = rs3 (store data).

### 6.2 funct7 allocation

| funct7 | Mnemonic       | Type  | Description                            |
|--------|----------------|-------|----------------------------------------|
| 0x20   | LMASK.PUSH     | CTRL  | Push reconv_pc (rs1) to lane stack     |
| 0x30   | SHFL.BFLY      | RRTR  | Butterfly shuffle                      |
| 0x31   | SHFL.IDX       | RRTR  | Index shuffle (rd ← lane[rs2].rs1)     |
| 0x32   | SHFL.WIDTH     | RRTR  | Width-based rotate                     |
| 0x40   | BAR.SYNC       | CTRL  | CTA barrier (rs1=0)                    |
| 0x50   | PCMP.EQ        | RRTP  | pd ← (rs1 == rs2)                      |
| 0x51   | PCMP.NE        | RRTP  | pd ← (rs1 != rs2)                      |
| 0x52   | PCMP.LT        | RRTP  | pd ← (rs1 < rs2) signed                |
| 0x53   | PCMP.LTU       | RRTP  | pd ← (rs1 < rs2) unsigned              |
| 0x54   | PCMP.LE        | RRTP  | pd ← (rs1 <= rs2) signed               |
| 0x55   | PCMP.F.EQ      | RRTP  | pd ← (frs1 == frs2)                    |
| 0x56   | PCMP.F.LT      | RRTP  | pd ← (frs1 < frs2)                     |
| 0x57   | PCMP.F.LE      | RRTP  | pd ← (frs1 <= frs2)                    |
| 0x58   | PSEL           | RRPTR | rd ← ps ? rs1 : rs2                    |
| 0x59   | PSEL.F         | RRPTR | fd ← ps ? frs1 : frs2                  |
| 0x5A   | PAND           | PPTP  | pd ← ps1 & ps2                         |
| 0x5B   | POR            | PPTP  | pd ← ps1 \| ps2                        |
| 0x5C   | PXOR           | PPTP  | pd ← ps1 ^ ps2                         |
| 0x5D   | PNOT           | PPTP  | pd ← ~ps1                              |
| 0x5E   | PMOV           | PPTP  | pd ← ps1                               |
| 0x5F   | PMOV.TO.X      | PPTR  | rd ← ps1 ? 1 : 0                       |
| 0x60   | PMOV.FROM.X    | RRTP  | pd ← (rs1 != 0)                        |
| 0x61   | PLB            | RRPTR | Predicated load byte (sign-ext)        |
| 0x62   | PLBU           | RRPTR | Predicated load byte unsigned          |
| 0x63   | PLH            | RRPTR | Predicated load halfword (sign-ext)    |
| 0x64   | PLHU           | RRPTR | Predicated load halfword unsigned      |
| 0x65   | PLW            | RRPTR | Predicated load word                   |
| 0x66   | PFLW           | RRPTR | Predicated FP load word                |
| 0x67   | PSB            | RRRP  | Predicated store byte                  |
| 0x68   | PSH            | RRRP  | Predicated store halfword              |
| 0x69   | PSW            | RRRP  | Predicated store word                  |
| 0x6A   | PFSW           | RRRP  | Predicated FP store word               |

### 6.3 CSR address map (from Vector spec)

| CSR Name           | Address | Description              |
|--------------------|---------|--------------------------|
| CSR_SIMT_TID_X     | 0xFC0   | threadIdx.x              |
| CSR_SIMT_TID_Y     | 0xFC4   | threadIdx.y              |
| CSR_SIMT_TID_Z     | 0xFC8   | threadIdx.z              |
| CSR_SIMT_NTID_X    | 0xFCC   | blockDim.x               |
| CSR_SIMT_NTID_Y    | 0xFD0   | blockDim.y               |
| CSR_SIMT_NTID_Z    | 0xFD4   | blockDim.z               |
| CSR_SIMT_CTAID_X   | 0xFD8   | blockIdx.x (program_id)  |
| CSR_SIMT_CTAID_Y   | 0xFDC   | blockIdx.y               |
| CSR_SIMT_CTAID_Z   | 0xFE0   | blockIdx.z               |
| CSR_SIMT_LANEID    | 0xFE4   | lane id (0..warp_size-1) |
| CSR_SIMT_WARPSIZE  | 0xFE8   | warp size (32)           |
| CSR_SIMT_LMASK_ACT | 0xFEC   | active lane mask (RO)    |

---

## 7. Tracking checklist

- [x] LLVM core: add `llvm.custom.*` intrinsic definitions
  - File: `llvm/include/llvm/IR/IntrinsicsCustomSIMT.td`
  - Included in: `llvm/include/llvm/IR/Intrinsics.td`
- [x] RISC-V: lower ID intrinsics → `csrr`
  - Files: `RISCVInstrInfoSIMT.td`, `RISCVExpandSIMTPseudoInsts.cpp`
- [x] RISC-V: lower barrier → `fence` + `bar.sync`
  - File: `RISCVExpandSIMTPseudoInsts.cpp`
- [x] RISC-V: add `LMASK.PUSH` insertion pass + pseudo expansion
  - Files: `RISCVInsertLMaskPush.cpp`, `RISCVExpandSIMTPseudoInsts.cpp`
- [x] RISC-V: predicate RegisterClass (`p0..p7`) + asm support
  - File: `RISCVInstrInfoSIMT.td` (PR register class)
- [x] RISC-V: ISel/late pass for `select`/masked ld/st → `PSEL`/`PL*`/`PS*`
  - File: `RISCVInstrInfoSIMT.td` (instruction definitions)
- [x] RISC-V: shuffle.* → `SHFL.IDX`
  - File: `RISCVInstrInfoSIMT.td` (patterns)
- [ ] RISC-V: ballot → cross-lane op (pending ISA encoding)
- [x] LLVM lit tests for each feature
  - Files: `test/CodeGen/RISCV/custom-simt-*.ll`

---

## 8. Implementation Files Summary

### Added Files

| File | Description |
|------|-------------|
| `llvm/include/llvm/IR/IntrinsicsCustomSIMT.td` | Custom SIMT intrinsic definitions |
| `llvm/lib/Target/RISCV/RISCVSystemOperandsSIMT.td` | SIMT CSR definitions |
| `llvm/lib/Target/RISCV/RISCVInstrFormatsSIMT.td` | Custom instruction format classes |
| `llvm/lib/Target/RISCV/RISCVInstrInfoSIMT.td` | SIMT instruction definitions |
| `llvm/lib/Target/RISCV/RISCVExpandSIMTPseudoInsts.cpp` | Pseudo instruction expansion |
| `llvm/lib/Target/RISCV/RISCVInsertLMaskPush.cpp` | LMASK.PUSH insertion pass |
| `llvm/test/CodeGen/RISCV/custom-simt-csrr.ll` | CSR read tests |
| `llvm/test/CodeGen/RISCV/custom-simt-barrier.ll` | Barrier tests |
| `llvm/test/CodeGen/RISCV/custom-simt-shuffle.ll` | Shuffle tests |
| `llvm/test/CodeGen/RISCV/custom-simt-predicate.ll` | Predicate instruction tests |
| `llvm/test/CodeGen/RISCV/custom-simt-kernel.ll` | Complete kernel examples |

### Modified Files

| File | Changes |
|------|---------|
| `llvm/include/llvm/IR/Intrinsics.td` | Include IntrinsicsCustomSIMT.td |
| `llvm/lib/Target/RISCV/RISCV.td` | Include RISCVInstrInfoSIMT.td |
| `llvm/lib/Target/RISCV/RISCV.h` | Declare new passes |
| `llvm/lib/Target/RISCV/CMakeLists.txt` | Add new source files |
