# Custom SIMT Backend - Kernel ABI Specification

本文档定义 Custom SIMT backend 的 Kernel ABI 规范，确保 LLIR 层的参数布局与 runtime/driver launcher 一致。

---

## 1. 概述

Custom backend 采用 **展开参数 ABI (Expanded Arguments)**，即 kernel 函数在 LLVM IR 层直接接收各个独立参数，而不是通过单一参数块指针。

### 目标架构

- **Triple**: `riscv32-unknown-unknown`
- **ABI**: `ilp32f` (RV32F with soft-float ABI)
- **指针宽度**: 32 位
- **对齐**: 按 C 语言规则，最小 4 字节对齐

---

## 2. Kernel Function Signature

### LLVM IR 层签名格式

```llvm
define void @kernel_name(
    ptr addrspace(1) %arg0,    ; 指针参数（global memory）
    ptr addrspace(1) %arg1,    ; 指针参数（global memory）
    i32 %arg2,                  ; 32位整型标量
    float %arg3,                ; 32位浮点标量
    ...
) local_unnamed_addr #0 {
    ...
}
```

### 参数类型映射

| Triton 类型 | LLVM IR 类型 | 说明 |
|-------------|--------------|------|
| `*fp32` | `ptr addrspace(1)` | 全局内存指针 (f32) |
| `*fp16` | `ptr addrspace(1)` | 全局内存指针 (f16) |
| `*i32` | `ptr addrspace(1)` | 全局内存指针 (i32) |
| `i32` | `i32` | 32位整型标量 |
| `i64` | `i64` | 64位整型标量 |
| `fp32` | `float` | 32位浮点标量 |
| `fp16` | `half` | 16位浮点标量 |

### Address Space 约定

| Address Space | 含义 | 说明 |
|---------------|------|------|
| 0 | Generic/Default | 不使用 |
| 1 | Global Memory | 所有设备内存指针 |
| 3 | ~~Shared Memory~~ | **不支持** (global-only) |
| 4 | ~~Constant~~ | 作为 global 处理 |

---

## 3. SIMT 语义 Intrinsics

以下 intrinsic 在 LLIR 中用于获取 SIMT 执行上下文：

### 3.1 Program/Block ID

```llvm
; 获取当前 CTA 的 ID (对应 Triton 的 program_id)
declare i32 @llvm.custom.program.id(i32 %axis) #readnone

; axis: 0=x, 1=y, 2=z
; 映射: CSR_SIMT_CTAID_X (0x7C6)
```

### 3.2 Thread ID

```llvm
; 获取 CTA 内线程 ID (threadIdx)
declare i32 @llvm.custom.thread.id(i32 %axis) #readnone

; axis: 0=x, 1=y, 2=z
; 映射: CSR_SIMT_TID_X (0x7C0)
```

### 3.3 Block/Grid Dimensions

```llvm
; 获取 block 维度 (blockDim)
declare i32 @llvm.custom.block.dim(i32 %axis) #readnone
; 映射: CSR_SIMT_NTID_X (0x7C3)

; 获取 grid 维度 (gridDim)
declare i32 @llvm.custom.grid.dim(i32 %axis) #readnone
; 映射: descriptor.grid_size
```

### 3.4 Lane/Warp 信息

```llvm
; 获取 lane ID (0..warp_size-1)
declare i32 @llvm.custom.lane.id() #readnone
; 映射: CSR_SIMT_LANEID (0x7CD)

; 获取 warp size
declare i32 @llvm.custom.warp.size() #readnone
; 映射: CSR_SIMT_WARPSIZE (0x7CC)
```

### 3.5 同步

```llvm
; CTA barrier（所有 warp 同步）
declare void @llvm.custom.barrier() #convergent #nounwind

; 映射: bar.sync 或 fence + custom barrier instruction
```

### 3.6 Cross-lane 操作

```llvm
; Warp shuffle (lane 间数据交换)
declare i32 @llvm.custom.shuffle.idx.i32(i32 %val, i32 %src_lane) #convergent

; Ballot (warp-wide predicate collection)
declare i32 @llvm.custom.ballot(i1 %pred) #convergent

; 映射: SIMT extension opcode 0x0B 的 cross-lane execute unit
```

---

## 4. Memory Operations

### 4.1 Load 操作

Custom backend 的 load 操作转换为标准 LLVM load：

```llvm
; 无条件加载
%val = load float, ptr addrspace(1) %ptr

; 带 mask 的加载（转换为 select）
%loaded = load float, ptr addrspace(1) %ptr
%val = select i1 %mask, float %loaded, float %default
```

### 4.2 Store 操作

```llvm
; 无条件存储
store float %val, ptr addrspace(1) %ptr

; 带 mask 的存储（转换为条件分支）
br i1 %mask, label %do_store, label %skip_store
do_store:
  store float %val, ptr addrspace(1) %ptr
  br label %skip_store
skip_store:
  ...
```

### 4.3 内存模型

- **一致性**: 依赖 CTA barrier 保证可见性
- **原子操作**: 通过 LLVM atomic 指令实现
- **Fence**: 使用 `fence` 指令

---

## 5. Runtime Launcher Contract

### 5.1 参数传递流程

```
Host (Python/C++)          Runtime                    Device
      |                        |                         |
      | kernel(arg0,arg1,...)  |                         |
      |----------------------->|                         |
      |                        | pack_params(args)       |
      |                        |------------------------>|
      |                        | set_descriptor(...)     |
      |                        |------------------------>|
      |                        | launch_kernel()         |
      |                        |------------------------>|
      |                        |                         | execute
```

### 5.2 参数打包

Runtime 按以下规则打包参数：

1. 所有参数按顺序排列
2. 指针参数：32位地址值
3. 标量参数：按类型宽度存储，4字节对齐
4. 参数块基址通过 descriptor 传递给硬件

### 5.3 Driver API

```python
# TRITON_CUSTOM_RUNTIME_MODULE 需要实现：

def load_binary(name: str, blob: bytes, shared: int, device: int):
    """加载 kernel blob 到设备内存
    
    Returns:
        (module_handle, function_handle, n_regs, n_spills, n_max_threads)
    """
    pass

def launch(gridX: int, gridY: int, gridZ: int,
           stream: int, function: Any,
           packed_metadata: bytes, launch_metadata: Any,
           enter_hook: Any, exit_hook: Any,
           *args):
    """启动 kernel 执行
    
    Args:
        gridX/Y/Z: grid 维度
        stream: 执行流 handle
        function: kernel function handle
        args: 展开的 kernel 参数
    """
    pass
```

---

## 6. 验证清单

- [x] Kernel signature 使用展开参数
- [x] 指针使用 `addrspace(1)` (global memory)
- [x] 标量参数直接传递
- [x] SIMT intrinsics 正确映射
- [x] Load/Store 使用标准 LLVM 操作
- [ ] Runtime launcher 按 ABI 打包参数
- [ ] 端到端执行验证

---

## 7. 示例

### 输入 Triton Kernel

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 生成的 LLVM IR Signature

```llvm
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

define void @add_kernel(
    ptr addrspace(1) %x_ptr,
    ptr addrspace(1) %y_ptr,
    ptr addrspace(1) %output_ptr,
    i32 %n_elements
) {
    ; BLOCK_SIZE 是 constexpr，已内联
    %pid = call i32 @llvm.custom.program.id(i32 0)
    ; ... 后续计算
}
```
