# Custom SIMT Target Spec (Questionnaire)

> Goal: define a complete, implementable spec for a Triton backend targeting a custom SIMT core derived from (but not ABI/ISA-compatible with) RISC-V.
>
> How to use: answer items section by section. If an item is unknown, write **TBD**.

---

## 0. Summary (fill first)

- **Target name (string)**: (e.g. `custom`)  
- **One-line description**:  
- **Primary bring-up path**: (choose one)
  - [ ] simulator
 

---

## 1. Hardware Overview

each lane is a rv32_imf core with custom extension

From `gpnpu_spec/Vector`:

- Vector core is a SIMT engine based on RV32IMF(C) + zicsr (+ optional RVV noted), with a custom SIMT extension (opcode `0x0B`) and a cross-lane execute unit.
- Execution model is **Multi-PC SIMT** (each lane has an independent PC; warp schedules execution groups with same PC+mask).

1. **SIMT width and scheduling**
   - **lane width (warp size)**: 32
   - max warps resident per core:  8
   - scheduling granularity: warp

1. **Registers**
   - scalar register file per lane: 32bit per reg(32 int reg, 32 fp reg)
   - vector registers N
   - predicate/mask registers? y, 1bit per p reg, 8 p reg per lane

From `gpnpu_spec/Vector`:

- Predicate registers: `p0..p7`, per-lane 1-bit, warp-wide view is a 32-bit mask.
- Predicate regs are caller-save; spilling/stacking strategy is TBD in the spec.

1. **Special registers / builtins available** (list)
   - CSR_SIMT_TID_X (0xFC0)
     功能：返回CTA内 lane 的全局线程 ID（threadIdx.x）
     取值范围：0 到 (CTA_size - 1), launch后只读
     使用场景：用于索引 CTA 内的数据，计算全局偏移量
   示例：csrr t0, 0xFC0 读取当前 lane 的线程 ID

   - CSR_SIMT_NTID_X (0xFCC)
     功能：返回 CTA 的维度大小（blockDim.x）
     取值范围：通常为 32 的倍数
     使用场景：用于边界检查，计算循环次数

   - CSR_SIMT_CTAID_X (0xFD8)
     功能：返回GRID内当前 CTA 的 ID（blockIdx.x）
     取值范围：0 到 (grid_size - 1),, launch后只读
     使用场景：用于多 CTA 场景下的数据分区
     注：目前仅有一个维度X，未来会有Y和Z维度

   - CSR_SIMT_LANEID (0xFE4)
     功能：返回当前 lane 在 warpWARP 内的索引
     取值范围：0 到 (warp_size - 1)，通常为 0-31, 只读
     使用场景：用于 lane-local 计算、栈偏移计算
   示例：csrr t0, 0xFE4 读取当前 lane 的 lane ID

   Additional CSRs mentioned in `gpnpu_spec/Vector` (confirm addresses on your RTL/model):

   - `CSR_SIMT_WARPSIZE` (`0xFE8`): RO warp size
   - `CSR_SIMT_LMASK_ACT` (`0x8E0`): RO active lane mask

---

## 2. Execution Model (Triton mapping)

> Triton launches a grid of **programs**. We must map a program to your hardware execution entity.

1. **Program mapping** (choose one)
   - [x] 1 program == 1 CTA

From `gpnpu_spec/Vector`:

- Conceptual hierarchy is `Grid → CTA → Warp → Lane`.
- A `block/CTA` is scheduled onto a vector core; within the core, the CTA is further split into warps for SIMT execution.
 
1. **`program_id(axis)` semantics**
   - supported axes: (x)  
   - maximum grid dims (x):  runtime-defined (grid is split to CTAs by runtime/global engine)
   - how program_id is computed / provided to kernel:  硬件上通过上述的CSR 特别寄存器计算获取

From `gpnpu_spec/Vector`:

- `program_id(0)` likely corresponds to `CSR_SIMT_CTAID_X` (blockIdx.x).
- `tl.program_id(axis)` and `tl.num_programs(axis)` need a clear mapping to descriptor fields (`grid_size`, `block_size`) and/or CSRs.

1. **Control flow**
   - divergence model: both support
   - max nesting / stack constraints: 64 nested  

From `gpnpu_spec/Vector`:

- Multi-PC SIMT: lanes form execution groups when their (PC, mask) match; scheduler issues one execution group at a time.
- Reconvergence: compiler inserts `LMASK.PUSH` before a branch to provide reconvergence PC + active mask; hardware maintains a branch tree and auto reconverges (no software `LMASK.POP`).

1. **Synchronization**
   - intra-warp barrier? (Y/N; instruction name)  
   - inter-warp (within group) barrier? y (CTA barrier)
   - memory fence instructions: fence

From `gpnpu_spec/Vector`:

- CTA-level barrier is mentioned but the exact ISA/semantics are marked “uncertain” in the spec.

---

## 3. Memory Model & Address Spaces

1. **Address spaces**
   只有一个global memory space

From `gpnpu_spec/Vector`:

- No separate `local`/`shared` memory abstraction at ISA level; everything is modeled as global memory.
- Runtime allocates a per-kernel global memory region before launch; the global engine partitions it per block/workset.

1. **Pointers**
   - pointer width: 32  
   - pointer meaning: device VA
   - alignment requirements:  无

1. **Shared/scratchpad**
   runtime分配

From `gpnpu_spec/Vector`:

- Any “scratchpad/cache” is considered an implementation detail of the global memory system, not an exposed address space.


---

## 4. Data Types & Math

1. **Scalar types supported in hardware** (check)
   i32/fp32 (RV32F baseline)

   fp64: TBD (assume not supported in v1 unless emulated)

From `gpnpu_spec/Vector` (needs confirmation for what is truly implemented in v1):

- Base core listed as RV32IMF(C) so FP32 is expected; the doc also lists desired support for FP16/BF16/FP8/FP4 as roadmap-level “supported precisions”.

1. **Vector types / packed ops**
   no vector

1. **Special math**
   - div/sqrt/rsqrt availability: HW  
   - transcendental support: HW SFU  

1. **Tensor core / matrix acceleration** (if any)
   no tensor core

---

## 5. Kernel Runtime ABI (Host ↔ Device)

> This defines how the runtime passes parameters and launch info to your kernel.

1. **Binary format** (choose)
      raw binary

   v1 decision: **Scheme A (flat image, no relocations)**

   - Output is a flat byte blob that the driver copies to an executable memory region.
   - The blob contains **no relocation table**, and the driver does **no fixups/patching**.
   - `start_pc` is provided by the driver at launch time (based on where it copied the blob).

   Implications / constraints on codegen:

   - No references that require link-time / load-time relocation.
   - Control flow should be encoded with PC-relative branch/jump forms (within ISA range).
   - Any “external address” (buffers, parameter block base, constants base if not PC-relative) must be provided via parameter block and/or CSRs.
   - Prefer a single-kernel-per-blob model in v1 (already selected below).

1. **Blob Format v1 (recommended)**

    Goal: allow the driver to load/launch with a single `memcpy` + `start_pc = base + entry_off`.

    Overall layout:

    - `[Header][.text][.rodata][.meta(optional)]`
    - Integer encoding: little-endian
    - Alignment: `.text` 16B, `.rodata` 16B (raise if fetch/LSU requires)

    Header (fixed 64B, little-endian):

    - `magic[8]`: ASCII `GPNPUVEC` (or another project-specific tag)
    - `version_u16`: `1`
    - `flags_u16`:
       - bit0 = 1: `NO_RELOC` (v1 must be 1)
       - bit1 = 1: `HAS_RODATA`
       - bit2 = 1: `HAS_META`
    - `header_size_u32`: `64`
    - `entry_off_u32`: entry PC offset from blob base (usually `64`)
    - `text_off_u32`, `text_size_u32`
    - `rodata_off_u32`, `rodata_size_u32` (0 if absent)
    - `meta_off_u32`, `meta_size_u32` (0 if absent)
    - `reserved[...]`: zero-filled

    Driver responsibilities:

    - Allocate executable memory region `base` (size = blob size).
    - Copy blob bytes into `base`.
    - Compute `start_pc = base + entry_off` and program the kernel descriptor.
    - Provide parameter block address / CTA id / grid config via descriptor + CSRs.
    - No parsing/linking beyond reading the fixed header fields above.

    Codegen constraints implied by this blob format:

    - `.text` must not require relocations (no unresolved symbols).
    - Calls/branches must resolve within `.text` (PC-relative forms preferred).
    - Any external addresses (buffers, param block, optional constants base) must come from parameter block / CSRs.

      Optional `.meta` (recommended for bring-up/debug):

      - Purpose: host/runtime validation and debugging only; hardware execution must not depend on it.
      - Encoding: a sequence of TLV records within the `.meta` segment. All integers little-endian.
      - Layout: repeated `{ type_u16, size_u16, payload[size] }` until `meta_size_u32` is exhausted.
         - `type_u16`: record kind
         - `size_u16`: payload byte length (not including the 4-byte TLV header)
         - `payload`: raw bytes; strings are UTF-8 without NUL terminator

      Suggested record types (v1):

      - `1 (ABI_TAG)`: string, e.g. `ilp32f`
      - `2 (KERNEL_NAME)`: string (optional, for logs)
      - `3 (STACK_PER_WARP_BYTES)`: `u32` (v1: 1024)
      - `4 (USES_CTA_BARRIER)`: `u8` (0/1)
      - `5 (NUM_WARPS_HINT)`: `u16` (0 if unknown)
      - `6 (WARP_SIZE)`: `u16` (v1: 32)

1. **Entry symbol**
   - naming scheme:  plain
   - multiple kernels per binary allowed? N  

1. **Calling convention**
   - [x] parameters in registers / and memory
   following riscv32 ilp32f
   

1. **Kernel parameter layout**
   - layout rule: C struct
   - per-arg alignment:  32bit alignment
   - endianness: little  

From `gpnpu_spec/Vector`:

- Runtime builds a **parameter block** in memory (packed according to ABI rules), and passes its base address via a descriptor/CSR.
- Descriptor fields called out: `start_pc` (code entry), `block_size`, `grid_size`, `param_block_addr`.
- Engine side launch flow: configure descriptor CSRs (e.g. via `tsync.config ...`), then `tsync.launch.dep ...`.

1. **Implicit parameters** (fill all that apply)
   through custom instruction accessing special register,
   config all parameter first then launch to core
   - program_id x/y/z: x via `CSR_SIMT_CTAID_X`
   - num_programs x/y/z: x == `grid_size` provided to driver; runtime/global engine does grid → CTA split; hardware only “sees” CTAs
   - warp_size / num_warps: warp size fixed, num_warps hw split  
   - shared base pointer:  
   - printf buffer pointer:  
   - profiling buffer pointer:  

1. **Resource metadata required by runtime**
   - max registers / spills info needed?  
   - shared memory size needed?  
   - stack size needed?  
   - other constraints:  

   Filled:
   - stack model: stack_size per warp = 1KB; recursion supported; dynamic alloca not supported

From `gpnpu_spec/Vector` (confirm):

- Vector core context setup mentions a per-task stack region: set `SP = stack_ptr + stack_size` and keep `stack_limit = stack_ptr`.

---

## 6. Toolchain & Codegen Strategy

1. **Backend strategy** (choose one)
   - [x] LLVM-based codegen (custom LLVM target or forked RISC-V)
   - [ ] custom assembler/codegen (no LLVM ISel)
   - [ ] hybrid (LLVM to MIR-ish, then custom)

1. If **LLVM-based**:
   - desired LLVM triple:  
   - do you already have a working LLVM fork? (Y/N; repo/path)  
   - required target features flags:  
   - object file format: flat image (raw bytes), no relocations

From `gpnpu_spec/Vector`:

- ISA includes a custom SIMT extension (opcode `0x0B`) with lane-mask/shuffle/predicate-related funct7 allocations.
- If you want LLVM-based codegen, we need a clear mapping: (a) custom LLVM target OR (b) RISC-V backend with custom extension + SIMT lowering strategy.

1. If **custom codegen**:
   - input to codegen: (LLVM IR text / MLIR LLVM dialect / custom IR)  
   - assembler availability: (Y)  
   - relocation/linking support: (n)  

1. **Debug info**
   - DWARF supported? (Y/N)  
   - source-level debugging expectations:  

---

## 7. Runtime / Driver Interface

1. **Device management API**
   - how to enumerate devices:  
   - context model:  

1. **Memory management**
   - alloc/free API:  
   - memcpy H2D/D2H/D2D:  

1. **Module loading**
   - load API: (ioctl? library call? simulator API?)  
   - code upload constraints: (alignment, max size)  

1. **Kernel launch**
   - launch API signature:  
   - stream/queue model:  
   - supported grid dims:  

From `gpnpu_spec/Vector`:

- Launch is described as descriptor/CSR configuration + submit to global engine; engine splits kernel descriptor to CTA descriptors, then to warp descriptors, then issues ready warps.

1. **Error model**
   - synchronous vs async errors:  
   - error codes / exceptions mapping:  

---

## 8. Performance & Limits (needed for compiler heuristics)

- max registers per lane:  32 int 32 fp
- max shared per core:  no shared
- max warps resident:  8
- max instructions per kernel (if any):  no
- preferred memory transaction sizes:  
- alignment constraints:  

---

## 9. Minimal Bring-up Kernels (acceptance tests)

Check the smallest set you want as Phase-1 success:

- [x] vector add (axpy)
- [ ] elementwise (relu/sigmoid)
- [ ] reduction (sum)
- [x] softmax
- [ ] matmul (small)

For each selected kernel, provide expected dtype(s) and sizes:

---

## 10. Open Questions / TBD

List anything not covered above:

- LLVM triple / target-features list is still TBD (even with Scheme A chosen).

