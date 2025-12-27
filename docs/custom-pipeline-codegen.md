# Custom SIMT Backend - Pipeline & Codegen 配置

本文档描述 Custom SIMT backend 的编译 pipeline 和代码生成配置。

---

## 1. Pipeline 概述

Custom backend 的编译流程：

```
Triton DSL -> TTIR -> TTGIR -> LLVM IR -> [ASM/OBJ/BLOB]
```

### 1.1 Pipeline Stages

| Stage | 输入 | 输出 | 说明 |
|-------|------|------|------|
| `ttir` | Triton AST | TTIR | Triton IR 生成 |
| `ttgir` | TTIR | TTGIR | TritonGPU IR 生成 |
| `llir` | TTGIR | LLVM IR | LLVM IR 生成 (使用 `llvm.riscv.simt.*` intrinsics) |
| `asm` | LLVM IR | RISCV 汇编 | 汇编代码生成 (可选) |
| `obj` | LLVM IR | ELF 目标文件 | 目标文件生成 (可选) |
| `blob` | LLVM IR | Blob 二进制 | 最终可执行 blob (默认) |

---

## 2. 环境变量配置

### 2.1 Codegen 模式选择

```bash
# 选择代码生成模式
export TRITON_CUSTOM_CODEGEN_MODE=blob   # 默认: 使用外部 codegen 工具
export TRITON_CUSTOM_CODEGEN_MODE=asm    # 生成 RISCV 汇编 (调试用)
export TRITON_CUSTOM_CODEGEN_MODE=obj    # 生成 ELF 目标文件
export TRITON_CUSTOM_CODEGEN_MODE=full   # 完整 pipeline: llir -> asm -> obj -> blob
```

### 2.2 LLVM IR 生成配置

```bash
# 启用 Custom LLIR 模式 (使用 llvm.riscv.simt.* intrinsics)
export TRITON_CUSTOM_LLIR_MODE=1

# TTGIR pipeline 模式
export TRITON_CUSTOM_TTGIR_MODE=custom   # 默认: NVIDIA-free path
export TRITON_CUSTOM_TTGIR_MODE=nvidia   # 对比: 使用 NVIDIA TTGIR passes
```

### 2.3 RISCV SIMT 后端配置

```bash
# Target Triple
export TRITON_CUSTOM_LLVM_TRIPLE=riscv32-unknown-unknown

# CPU 型号
export TRITON_CUSTOM_LLVM_CPU=generic-rv32

# CPU Features
export TRITON_CUSTOM_LLVM_FEATURES=+f,+m

# 额外 LLC 选项
export TRITON_CUSTOM_LLC_FLAGS=""

# Data Layout
export TRITON_CUSTOM_LLVM_DATALAYOUT="e-m:e-p:32:32-i64:64-n32-S128"
```

### 2.4 调试选项

```bash
# 打印生成的汇编代码
export TRITON_CUSTOM_DUMP_ASM=1

# 打印目标文件大小
export TRITON_CUSTOM_DUMP_OBJ_SIZE=1

# LLVM 优化级别 (O0/O3)
export TRITON_CUSTOM_LLVM_OPT=O3
```

---

## 3. 使用示例

### 3.1 生成 RISCV 汇编 (调试)

```bash
export TRITON_CUSTOM_LLIR_MODE=1
export TRITON_CUSTOM_CODEGEN_MODE=asm
export TRITON_CUSTOM_DUMP_ASM=1

python my_triton_kernel.py
```

### 3.2 生成 ELF 目标文件

```bash
export TRITON_CUSTOM_LLIR_MODE=1
export TRITON_CUSTOM_CODEGEN_MODE=obj

python my_triton_kernel.py
```

### 3.3 使用外部 Codegen (默认)

```bash
export TRITON_CUSTOM_CODEGEN=/path/to/my-codegen-tool
export TRITON_CUSTOM_CODEGEN_MODE=blob

python my_triton_kernel.py
```

---

## 4. 架构对比

### 4.1 NVIDIA Pipeline

```
TTGIR -> LLVM IR (NVPTX) -> PTX -> CUBIN
         ↓
    @llvm.nvvm.*
    nvptx64-nvidia-cuda
```

### 4.2 AMD Pipeline

```
TTGIR -> LLVM IR (AMDGPU) -> AMDGCN -> HSACO
         ↓
    amdgpu_kernel
    amdgcn-amd-amdhsa
```

### 4.3 Custom SIMT Pipeline

```
TTGIR -> LLVM IR (RISCV) -> ASM/OBJ -> BLOB
         ↓
    @llvm.riscv.simt.*
    riscv32-unknown-unknown
```

---

## 5. Intrinsic 映射

Custom backend 生成的 LLVM IR 使用 `llvm.riscv.simt.*` intrinsics，这些 intrinsics 会被 LLVM RISCV SIMT 后端识别并降低为机器指令。

| Triton 操作 | LLVM Intrinsic | RISCV 指令 |
|-------------|----------------|------------|
| `tl.program_id(0)` | `@llvm.riscv.simt.program.id(i32 0)` | `csrr t0, CSR_SIMT_CTAID_X` |
| `__syncthreads()` | `@llvm.riscv.simt.barrier()` | `fence + bar.sync` |
| `tl.xor_shuffle(val, mask)` | `@llvm.riscv.simt.shfl.bfly(val, mask)` | `SHFL.BFLY` |

---

## 6. 构建 LLVM

确保 LLVM 编译时包含 RISCV 目标：

```bash
# 设置 LLVM 目标包含 RISCV
export LLVM_TARGETS="Native;NVPTX;AMDGPU;RISCV"

# 或者直接使用脚本 (已自动包含 RISCV)
./scripts/build-llvm-project.sh
```

验证 RISCV 支持：

```bash
./llvm-project-simt/build/bin/llc --version | grep -i riscv
```

---

## 7. 故障排除

### 7.1 "RISCV target not found"

LLVM 未编译 RISCV 目标。重新编译：

```bash
LLVM_CLEAN=1 LLVM_TARGETS="Native;NVPTX;AMDGPU;RISCV" ./scripts/build-llvm-project.sh
```

### 7.2 "Unknown intrinsic: llvm.riscv.simt.*"

LLVM RISCV SIMT 后端未正确实现。确保使用 `llvm-project-simt` 分支：

```bash
cd llvm-project-simt
git checkout simt-main
git pull origin simt-main
```

### 7.3 "Failed to translate LLVM IR to RISCV assembly"

检查 LLVM IR 是否包含正确的 target triple：

```bash
grep "target triple" output.llir
# 应该是: target triple = "riscv32-unknown-unknown"
```
