# GPNPU SIMT CSRs for RISC-V

This document describes the custom Control and Status Registers (CSRs) added to the RISC-V LLVM backend for GPNPU SIMT (Single Instruction Multiple Thread) support.

## Overview

The GPNPU SIMT extension adds CSRs that mirror the programming model of modern GPU architectures like CUDA/PTX. These CSRs provide thread identification and SIMT execution control capabilities on a RISC-V based GPNPU.

## CSR Address Map

| CSR Name | Address | Description | Access |
|----------|---------|-------------|--------|
| `gpnpu.tid_x` | 0xFC0 | Thread ID in X dimension | RO |
| `gpnpu.tid_y` | 0xFC1 | Thread ID in Y dimension | RO |
| `gpnpu.tid_z` | 0xFC2 | Thread ID in Z dimension | RO |
| `gpnpu.ntid_x` | 0xFC3 | Number of threads in X dimension | RO |
| `gpnpu.ntid_y` | 0xFC4 | Number of threads in Y dimension | RO |
| `gpnpu.ntid_z` | 0xFC5 | Number of threads in Z dimension | RO |
| `gpnpu.ctaid_x` | 0xFC6 | CTA (block) ID in X dimension | RO |
| `gpnpu.ctaid_y` | 0xFC7 | CTA (block) ID in Y dimension | RO |
| `gpnpu.ctaid_z` | 0xFC8 | CTA (block) ID in Z dimension | RO |
| `gpnpu.nctaid_x` | 0xFC9 | Number of CTAs in X dimension | RO |
| `gpnpu.nctaid_y` | 0xFCA | Number of CTAs in Y dimension | RO |
| `gpnpu.nctaid_z` | 0xFCB | Number of CTAs in Z dimension | RO |
| `gpnpu.laneid` | 0xFCD | Lane ID within warp | RO |
| `gpnpu.warpid` | 0xFCE | Warp ID within block | RO |
| `gpnpu.nwarpid` | 0xFCF | Number of warps per block | RO |
| `gpnpu.warpsize` | 0xFD0 | Number of threads per warp | RO |
| `gpnpu.gridid` | 0xFD1 | Grid ID for multi-grid support | RO |
| `gpnpu.active_mask` | 0xFE0 | Active lane mask for current warp | RW |
| `gpnpu.barrier_cnt` | 0xFE1 | Barrier synchronization count | RW |
| `gpnpu.predicate` | 0xFE2 | Predicate register for conditional execution | RW |
| `gpnpu.conv_mask` | 0xFE3 | Mask for warp convergence tracking | RW |

## Usage Examples

### Assembly

```asm
# Read thread IDs (equivalent to CUDA threadIdx)
csrr a0, gpnpu.tid_x      # a0 = threadIdx.x
csrr a1, gpnpu.tid_y      # a1 = threadIdx.y
csrr a2, gpnpu.tid_z      # a2 = threadIdx.z

# Read block dimensions (equivalent to CUDA blockDim)
csrr a3, gpnpu.ntid_x     # a3 = blockDim.x
csrr a4, gpnpu.ntid_y     # a4 = blockDim.y
csrr a5, gpnpu.ntid_z     # a5 = blockDim.z

# Read block IDs (equivalent to CUDA blockIdx)
csrr a6, gpnpu.ctaid_x    # a6 = blockIdx.x
csrr a7, gpnpu.ctaid_y    # a7 = blockIdx.y
csrr t0, gpnpu.ctaid_z    # t0 = blockIdx.z

# Read grid dimensions (equivalent to CUDA gridDim)
csrr t1, gpnpu.nctaid_x   # t1 = gridDim.x
csrr t2, gpnpu.nctaid_y   # t2 = gridDim.y
csrr t3, gpnpu.nctaid_z   # t3 = gridDim.z

# Warp-level information
csrr t4, gpnpu.laneid     # Lane ID within warp (0..warpsize-1)
csrr t5, gpnpu.warpsize   # Warp size (e.g., 32)
```

### Computing Global Thread ID

```asm
# Compute global thread ID: globalID = blockIdx.x * blockDim.x + threadIdx.x
csrr    a0, gpnpu.ctaid_x    # a0 = blockIdx.x
csrr    a1, gpnpu.ntid_x     # a1 = blockDim.x
mul     a0, a0, a1           # a0 = blockIdx.x * blockDim.x
csrr    a1, gpnpu.tid_x      # a1 = threadIdx.x
add     a0, a0, a1           # a0 = globalID
```

## CUDA/PTX Equivalent Mapping

| CUDA/PTX | GPNPU CSR |
|----------|-----------|
| `threadIdx.x` | `gpnpu.tid_x` |
| `threadIdx.y` | `gpnpu.tid_y` |
| `threadIdx.z` | `gpnpu.tid_z` |
| `blockDim.x` | `gpnpu.ntid_x` |
| `blockDim.y` | `gpnpu.ntid_y` |
| `blockDim.z` | `gpnpu.ntid_z` |
| `blockIdx.x` | `gpnpu.ctaid_x` |
| `blockIdx.y` | `gpnpu.ctaid_y` |
| `blockIdx.z` | `gpnpu.ctaid_z` |
| `gridDim.x` | `gpnpu.nctaid_x` |
| `gridDim.y` | `gpnpu.nctaid_y` |
| `gridDim.z` | `gpnpu.nctaid_z` |
| `%laneid` | `gpnpu.laneid` |
| `%warpid` | `gpnpu.warpid` |
| `WARP_SZ` | `gpnpu.warpsize` |

## Build Instructions

The GPNPU SIMT CSRs are integrated into the RISC-V LLVM backend. To build:

```bash
mkdir build-riscv && cd build-riscv
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="RISCV"
ninja llc llvm-mc
```

## Testing

```bash
# Assemble and show encoding
llvm-mc -triple=riscv32 -show-encoding test.s

# Compile to object file
llvm-mc -triple=riscv32 -filetype=obj -o test.o test.s
```

## Implementation Notes

1. CSR addresses are in the Machine Custom range (0xFC0-0xFFF) to avoid conflicts with standard RISC-V CSRs
2. Most CSRs are read-only (thread/block/grid IDs)
3. Execution control CSRs (active_mask, barrier_cnt, predicate, conv_mask) are read-write
4. The implementation is in `llvm/lib/Target/RISCV/RISCVSystemOperands.td`
