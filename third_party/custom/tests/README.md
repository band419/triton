# Custom SIMT Backend Tests

This directory contains test kernels and scripts for debugging the custom SIMT backend compilation pipeline.

## Quick Start

```bash
cd /Volumes/t7/prj/triton

# Run simple kernel test
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/tests/simple_matmul.py

# Run full pipeline test
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/tests/test_matmul_pipeline.py --all

# Test with ASM generation (requires LLVM with RISCV target)
TRITON_CUSTOM_LLIR_MODE=1 TRITON_CUSTOM_CODEGEN_MODE=asm python third_party/custom/tests/test_matmul_pipeline.py --stage asm
```

## Test Files

| File | Description |
|------|-------------|
| `simple_matmul.py` | Minimal matmul and vector add kernels |
| `test_matmul_pipeline.py` | Full pipeline test with all stages |

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `TRITON_CUSTOM_LLIR_MODE` | `1` | Enable custom LLIR generation |
| `TRITON_CUSTOM_CODEGEN_MODE` | `asm/obj/full/blob` | Select output format |
| `TRITON_CUSTOM_DUMP_LLIR` | `1` | Print generated LLVM IR |
| `TRITON_CUSTOM_DUMP_ASM` | `1` | Print generated assembly |
| `TRITON_CUSTOM_LLVM_TRIPLE` | e.g. `riscv32-unknown-elf` | Target triple |
| `TRITON_CUSTOM_LLVM_CPU` | e.g. `generic-rv32` | Target CPU |
| `TRITON_CUSTOM_LLVM_FEATURES` | e.g. `+m,+f` | CPU features |

## Pipeline Stages

```
TTIR (Triton IR)
    ↓
TTGIR (Triton GPU IR)
    ↓
LLIR (LLVM IR with llvm.riscv.simt.* intrinsics)
    ↓
ASM (RISCV Assembly) or OBJ (Object file)
```

## Debugging Tips

1. **Check LLVM IR generation:**
   ```bash
   TRITON_CUSTOM_LLIR_MODE=1 TRITON_CUSTOM_DUMP_LLIR=1 python simple_matmul.py
   ```

2. **Check for correct intrinsics:**
   ```bash
   # Should see llvm.riscv.simt.* not llvm.custom.*
   grep "llvm.riscv.simt" generated_ir.ll
   ```

3. **Test RISCV assembly generation:**
   ```bash
   TRITON_CUSTOM_CODEGEN_MODE=asm python test_matmul_pipeline.py --stage asm --dump
   ```

## Expected Intrinsics

The custom backend should generate LLVM IR with these intrinsics:

- `llvm.riscv.simt.program.id` - Get program/block ID
- `llvm.riscv.simt.thread.id` - Get thread ID
- `llvm.riscv.simt.lane.id` - Get lane ID within warp
- `llvm.riscv.simt.warp.size` - Get warp size
- `llvm.riscv.simt.barrier` - Thread barrier
- `llvm.riscv.simt.shfl.bfly` - Warp shuffle butterfly
- `llvm.riscv.simt.shfl.idx` - Warp shuffle indexed
- `llvm.riscv.simt.ballot.mask` - Warp ballot

## Kernel Descriptions

### Vector Add
Simple element-wise addition: `Z = X + Y`
- Uses `tl.program_id()` for block indexing
- Uses `tl.load()/tl.store()` with masks

### Matrix Multiplication
Tiled matrix multiplication: `C = A @ B`
- Uses 2D block indexing
- Uses `tl.dot()` for tile computation
- Demonstrates loop-based accumulation
