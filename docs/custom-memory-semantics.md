# Custom SIMT Backend: Memory and Synchronization Semantics

本文档定义 Custom SIMT 后端的内存模型与同步语义规范。

> 版本: 1.0  
> 日期: 2025-12-21  
> 阶段: Phase B3

---

## 1. 内存模型概述

### 1.1 Address Spaces

Custom SIMT 后端采用简化的内存模型：

| Address Space | 值 | 用途 | 备注 |
|---------------|-----|------|------|
| Generic | 0 | 通用指针 | 很少使用 |
| **Global** | 1 | 全局内存 | **主要使用** |
| Constant | 4 | 常量内存 | 只读 |

**关键决策**：Custom 后端 **不使用 Shared Memory (addrspace 3)**。所有数据通信通过全局内存完成。

### 1.2 内存访问特性

```
Global Memory (addrspace 1):
├── Visible to all threads across all CTAs
├── Coherent after barrier synchronization
├── Uses standard LLVM load/store
└── No special volatile/atomic requirements for basic ops
```

---

## 2. Load/Store 语义

### 2.1 Load 操作

**Triton `tt.load` → LLVM `load`**

```llvm
; 无条件加载
%val = load float, ptr addrspace(1) %ptr

; 有条件加载 (带 mask)
%loaded = load float, ptr addrspace(1) %ptr
%result = select i1 %mask, float %loaded, float %other
```

属性：
- **无特殊 volatile 标记**（除非有原子性需求）
- **无特殊 ordering**（使用默认 unordered/monotonic）
- LLVM 优化器可自由重排（受 barrier 约束）

### 2.2 Store 操作

**Triton `tt.store` → LLVM `store`**

```llvm
; 无条件存储
store float %val, ptr addrspace(1) %ptr

; 有条件存储 (带 mask) - 使用控制流
br i1 %mask, label %store_bb, label %skip_bb
store_bb:
  store float %val, ptr addrspace(1) %ptr
  br label %skip_bb
skip_bb:
  ; continue
```

属性：
- **无特殊 volatile 标记**
- **无特殊 ordering**
- 支持条件存储（通过 branch）

---

## 3. 同步语义

### 3.1 CTA Barrier

**Intrinsic**: `llvm.custom.barrier()`

```llvm
declare void @llvm.custom.barrier() #barrier_attrs

attributes #barrier_attrs = {
  convergent      ; Cannot be moved past control flow
  nounwind        ; Does not throw
  memory(readwrite, argmem: readwrite, inaccessiblemem: readwrite)
}
```

**语义**：
1. **执行同步**: CTA 内所有线程在此点等待
2. **内存 Fence**: 所有之前的内存写入对所有线程可见
3. **编译器约束**: `convergent` 属性防止跨控制流重排

**内存可见性保证**：

```
Thread 0                    Thread 1
---------                   ---------
store X                     
barrier()                   barrier()
                            load X  ← 保证看到 Thread 0 的写入
```

### 3.2 Warp Barrier

**Intrinsic**: `llvm.custom.warp.barrier()`

```llvm
declare void @llvm.custom.warp.barrier() #barrier_attrs
```

**语义**：
- 仅在 warp 内同步（32 线程）
- 同样具有 fence 语义
- 比 CTA barrier 更轻量

### 3.3 Barrier 内存效果详解

Barrier 的 `memory(readwrite, ...)` 效果意味着：

| 效果 | 说明 |
|------|------|
| **ModRef on Other** | Barrier 读写"其他"内存，阻止 load/store 跨 barrier 移动 |
| **ModRef on ArgMem** | Barrier 影响参数指向的内存 |
| **ModRef on InaccessibleMem** | Barrier 影响不可直接访问的内存（例如其他线程的内存） |

**优化器行为**：
- LLVM 不会将 load 移动到 barrier 之后
- LLVM 不会将 store 移动到 barrier 之前
- 保证内存访问顺序符合程序语义

---

## 4. 内存 Ordering 规范

### 4.1 默认 Ordering

对于非原子操作：

```llvm
; 默认使用 unordered/not atomic
load float, ptr addrspace(1) %ptr
store float %val, ptr addrspace(1) %ptr
```

### 4.2 原子操作 (Future)

如需原子操作，使用 LLVM 原子指令：

```llvm
; 原子加载
%val = load atomic float, ptr addrspace(1) %ptr seq_cst

; 原子存储
store atomic float %val, ptr addrspace(1) %ptr seq_cst

; 原子 RMW
%old = atomicrmw add ptr addrspace(1) %ptr, i32 1 seq_cst
```

> **注意**: Phase B3 暂不实现原子操作，留待后续扩展。

---

## 5. Fence 指令 (Optional)

如需显式 fence，可添加：

**Intrinsic**: `llvm.custom.fence(ordering)`

```llvm
declare void @llvm.custom.fence(i32 %ordering) #fence_attrs

; 使用示例
call void @llvm.custom.fence(i32 4)  ; 4 = seq_cst
```

**Ordering 值**：
| 值 | 名称 | 说明 |
|----|------|------|
| 0 | NotAtomic | 非原子 |
| 1 | Unordered | 无序 |
| 2 | Monotonic | 单调 |
| 3 | Acquire | 获取 |
| 4 | Release | 释放 |
| 5 | AcquireRelease | 获取-释放 |
| 6 | SequentiallyConsistent | 顺序一致 |

> **当前状态**: Custom 后端使用 barrier 作为隐式 fence，暂不需要显式 fence intrinsic。

---

## 6. 内存可见性保证

### 6.1 基本保证

1. **单线程**: 程序顺序保证
2. **跨线程无 barrier**: 无保证（需假设 relaxed）
3. **跨线程有 barrier**: barrier 之前的写入对 barrier 之后的读取可见

### 6.2 示例

```python
# Triton kernel with barrier
@triton.jit
def kernel_with_barrier(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    tid = tl.arange(0, BLOCK_SIZE)
    
    # Phase 1: Each thread writes its data
    tl.store(x_ptr + pid * BLOCK_SIZE + tid, compute_value(tid))
    
    # Barrier: ensure all writes are visible
    tl.barrier()  # → llvm.custom.barrier()
    
    # Phase 2: Read data written by other threads
    val = tl.load(y_ptr + pid * BLOCK_SIZE + tid)
```

对应 LLIR：

```llvm
define void @kernel_with_barrier(ptr addrspace(1) %x_ptr, ptr addrspace(1) %y_ptr) {
entry:
  ; Phase 1: store
  store float %val, ptr addrspace(1) %store_ptr
  
  ; Barrier with full fence semantics
  call void @llvm.custom.barrier()
  
  ; Phase 2: load (guaranteed to see Phase 1 writes)
  %loaded = load float, ptr addrspace(1) %load_ptr
  ret void
}
```

---

## 7. 与 NVIDIA 后端的差异

| 特性 | NVIDIA | Custom SIMT |
|------|--------|-------------|
| Shared Memory | 是 (addrspace 3) | **否** |
| Memory Fence | `fence.acq_rel`, `membar` | barrier 隐式 fence |
| Barrier | `bar.sync`, `__syncthreads` | `llvm.custom.barrier` |
| Atomic Ops | 完整支持 | Phase B3 暂不实现 |
| Volatile | 用于 shared/global | 不需要 |

---

## 8. 验证要点

### 8.1 LLIR 检查

```bash
# 验证 barrier 有正确属性
grep -E "declare.*llvm.custom.barrier.*convergent" kernel.ll

# 验证无 NVIDIA 特定指令
grep -c "nvvm\|@llvm.nvvm" kernel.ll  # 应为 0

# 验证全局内存使用
grep -c "addrspace(1)" kernel.ll  # 应 > 0
grep -c "addrspace(3)" kernel.ll  # 应为 0
```

### 8.2 语义检查

1. **Barrier 不被移动**: 确保 `convergent` 属性生效
2. **Load/Store 不跨 barrier**: 检查 barrier 的 memory effects
3. **无 shared memory**: 确保无 addrspace(3) 使用

---

## 9. 实现状态

| 功能 | 状态 | 说明 |
|------|------|------|
| Global load/store | ✅ 已实现 | `LoadStoreOpToLLVM.cpp` |
| CTA barrier | ✅ 已实现 | `BarrierOpToLLVM.cpp`, `TargetInfo.cpp` |
| Warp barrier | ✅ 已实现 | `TargetInfo.cpp` |
| Barrier fence | ✅ 已实现 | memory(readwrite) 属性 |
| Explicit fence | 🔲 未实现 | 暂不需要 |
| Atomic operations | 🔲 未实现 | 待后续扩展 |

---

## 10. 参考

- LLVM Language Reference: [Memory Model](https://llvm.org/docs/LangRef.html#memory-model)
- LLVM Attributes: [Function Attributes](https://llvm.org/docs/LangRef.html#function-attributes)
- Triton GPU IR: [TritonGPU Dialect](https://triton-lang.org/main/dialects/TritonGPU/index.html)
