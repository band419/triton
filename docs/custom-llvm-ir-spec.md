# Custom LLVM IR Specification

本文档定义了 Triton Custom Backend 生成的 LLVM IR 中的 intrinsics 和约定，供 LLVM 后端 lowering 参考。

---

## 1. 目标架构信息

| 属性 | 值 |
|------|-----|
| Target Triple | `riscv32-unknown-unknown` |
| Data Layout | `e-m:e-p:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-n32` |
| Warp Size | 32 threads |
| 执行模型 | SIMT with Independent Thread Scheduling (ITS) |
| 内存模型 | Relaxed consistency，需显式 barrier/fence |

---

## 2. Kernel ABI

### 2.1 入口函数

```llvm
define void @kernel_name(
    ptr addrspace(1) %arg0,    ; Global memory pointer
    i32 %arg1,                  ; Scalar argument
    ...
) #0 {
    ...
}

attributes #0 = {
    "kernel"                    ; 标记为 kernel 入口
    "nvvm.kernel" = "1"         ; 兼容性属性（可选）
}
```

### 2.2 地址空间

| 地址空间 | 用途 |
|----------|------|
| `addrspace(0)` | Generic/Private |
| `addrspace(1)` | Global memory |
| `addrspace(3)` | Shared memory (未使用，保留) |

---

## 3. Thread/Block ID Intrinsics

### 3.1 Thread ID

```llvm
declare i32 @llvm.custom.tid.x() #readonly
declare i32 @llvm.custom.tid.y() #readonly
declare i32 @llvm.custom.tid.z() #readonly
```

**语义**: 返回当前线程在 block 内的 ID (0 到 blockDim-1)

**属性**:
- `readonly`: 不修改内存
- `nounwind`: 不抛异常
- `speculatable`: 可推测执行

**Lowering 建议**:
```
%tid = call i32 @llvm.custom.tid.x()
=>
读取特殊寄存器 / CSR: threadIdx.x
```

### 3.2 Block ID (CTA ID)

```llvm
declare i32 @llvm.custom.ctaid.x() #readonly
declare i32 @llvm.custom.ctaid.y() #readonly
declare i32 @llvm.custom.ctaid.z() #readonly
```

**语义**: 返回当前 block 在 grid 内的 ID (0 到 gridDim-1)

### 3.3 Block Dimension

```llvm
declare i32 @llvm.custom.ntid.x() #readonly
declare i32 @llvm.custom.ntid.y() #readonly
declare i32 @llvm.custom.ntid.z() #readonly
```

**语义**: 返回 block 的尺寸 (线程数)

### 3.4 Grid Dimension

```llvm
declare i32 @llvm.custom.nctaid.x() #readonly
declare i32 @llvm.custom.nctaid.y() #readonly
declare i32 @llvm.custom.nctaid.z() #readonly
```

**语义**: 返回 grid 的尺寸 (block 数)

### 3.5 Lane ID

```llvm
declare i32 @llvm.custom.lane.id() #readonly
```

**语义**: 返回当前线程在 warp 内的 lane ID (0 到 31)

**计算**: `lane_id = tid.x % 32`

---

## 4. Synchronization Intrinsics

### 4.1 CTA Barrier (Block-level)

```llvm
declare void @llvm.custom.barrier() #barrier_attrs
```

**语义**: 
- 同步 CTA 内所有线程
- 作为完整的内存 fence

**属性**:
```llvm
#barrier_attrs = {
    convergent,      ; 不能跨控制流移动
    nounwind,        ; 不抛异常
    memory(readwrite) ; 完整内存 fence
}
```

**Lowering 建议**:
```
call void @llvm.custom.barrier()
=>
cta.sync        ; 或等效的 CTA 同步指令
membar          ; 如果硬件 barrier 不包含 fence
```

### 4.2 Warp Barrier

```llvm
declare void @llvm.custom.warp.barrier() #barrier_attrs
```

**语义**:
- 同步 warp 内所有 **active** 线程
- 硬件自动检测 active mask，无需软件传递
- 作为完整的内存 fence

**Lowering 建议**:
```
call void @llvm.custom.warp.barrier()
=>
warp.sync       ; 硬件自动使用 activemask
```

**注意**: 与 NVIDIA 的 `bar.warp.sync membermask` 不同，本 intrinsic 无 mask 参数，因为硬件自动处理。

---

## 5. Warp Shuffle Intrinsics

所有 shuffle 操作都是 **同步的**（自带 warp sync 语义）。

### 5.1 Shuffle XOR

```llvm
declare i32 @llvm.custom.shuffle.xor.i32(i32 %val, i32 %lane_mask) #shuffle_attrs
declare i64 @llvm.custom.shuffle.xor.i64(i64 %val, i32 %lane_mask) #shuffle_attrs
declare float @llvm.custom.shuffle.xor.f32(float %val, i32 %lane_mask) #shuffle_attrs
declare double @llvm.custom.shuffle.xor.f64(double %val, i32 %lane_mask) #shuffle_attrs
```

**语义**:
```
source_lane = current_lane XOR lane_mask
return value_from(source_lane)
```

**示例**:
```
lane 0 XOR 1 = 1  → lane 0 读 lane 1 的值
lane 1 XOR 1 = 0  → lane 1 读 lane 0 的值
lane 2 XOR 1 = 3  → lane 2 读 lane 3 的值
```

**属性**:
```llvm
#shuffle_attrs = {
    convergent,      ; 不能跨控制流移动
    nounwind,
    memory(none)     ; 不访问内存（warp 内寄存器交换）
}
```

### 5.2 Shuffle Up

```llvm
declare i32 @llvm.custom.shuffle.up.i32(i32 %val, i32 %delta) #shuffle_attrs
declare i64 @llvm.custom.shuffle.up.i64(i64 %val, i32 %delta) #shuffle_attrs
declare float @llvm.custom.shuffle.up.f32(float %val, i32 %delta) #shuffle_attrs
declare double @llvm.custom.shuffle.up.f64(double %val, i32 %delta) #shuffle_attrs
```

**语义**:
```
source_lane = current_lane - delta
if (source_lane < 0) return own_value  ; 边界处理
return value_from(source_lane)
```

### 5.3 Shuffle Down

```llvm
declare i32 @llvm.custom.shuffle.down.i32(i32 %val, i32 %delta) #shuffle_attrs
declare i64 @llvm.custom.shuffle.down.i64(i64 %val, i32 %delta) #shuffle_attrs
declare float @llvm.custom.shuffle.down.f32(float %val, i32 %delta) #shuffle_attrs
declare double @llvm.custom.shuffle.down.f64(double %val, i32 %delta) #shuffle_attrs
```

**语义**:
```
source_lane = current_lane + delta
if (source_lane >= 32) return own_value  ; 边界处理
return value_from(source_lane)
```

### 5.4 Shuffle Index

```llvm
declare i32 @llvm.custom.shuffle.idx.i32(i32 %val, i32 %src_lane) #shuffle_attrs
declare i64 @llvm.custom.shuffle.idx.i64(i64 %val, i32 %src_lane) #shuffle_attrs
declare float @llvm.custom.shuffle.idx.f32(float %val, i32 %src_lane) #shuffle_attrs
declare double @llvm.custom.shuffle.idx.f64(double %val, i32 %src_lane) #shuffle_attrs
```

**语义**:
```
return value_from(src_lane % 32)
```

---

## 6. Vote/Ballot Intrinsics

### 6.1 Ballot

```llvm
declare i32 @llvm.custom.ballot(i1 %pred) #vote_attrs
```

**语义**:
- 收集 warp 内所有线程的 predicate
- 返回 32-bit mask，bit i = 1 表示 lane i 的 pred 为 true
- 自带 warp sync 语义

**示例**:
```
lane 0: pred=true,  lane 1: pred=false, lane 2: pred=true, ...
result = 0b...101 = 0x00000005
```

**属性**:
```llvm
#vote_attrs = {
    convergent,
    nounwind,
    memory(none)
}
```

---

## 7. Memory Intrinsics

### 7.1 标准 Load/Store

Custom backend 使用标准 LLVM load/store 指令访问全局内存：

```llvm
; Load from global memory
%val = load i32, ptr addrspace(1) %ptr, align 4

; Store to global memory
store i32 %val, ptr addrspace(1) %ptr, align 4
```

### 7.2 Atomic Operations

使用标准 LLVM atomicrmw：

```llvm
%old = atomicrmw add ptr addrspace(1) %ptr, i32 %val seq_cst
%old = atomicrmw xchg ptr addrspace(1) %ptr, i32 %val seq_cst
%old = atomicrmw max ptr addrspace(1) %ptr, i32 %val seq_cst
```

### 7.3 Memory Fence

```llvm
fence seq_cst  ; 标准 LLVM fence
```

---

## 8. Intrinsic 属性总结

| Intrinsic 类别 | convergent | memory | nounwind |
|----------------|------------|--------|----------|
| Thread ID | ❌ | none | ✅ |
| Barrier | ✅ | readwrite | ✅ |
| Shuffle | ✅ | none | ✅ |
| Ballot | ✅ | none | ✅ |

`convergent` 属性含义：
- 不能被 LLVM 优化 pass 移动到控制流之外
- 不能被 loop unswitching 等优化影响
- 确保所有参与的线程在同一执行点调用

---

## 9. Lowering 指南

### 9.1 Intrinsic → 机器指令映射表

| LLVM Intrinsic | 建议指令 | 备注 |
|----------------|----------|------|
| `@llvm.custom.tid.x` | `csrr rd, tidx` | 读特殊寄存器 |
| `@llvm.custom.ctaid.x` | `csrr rd, ctaidx` | 读特殊寄存器 |
| `@llvm.custom.barrier` | `cta.sync` | CTA 同步 + fence |
| `@llvm.custom.warp.barrier` | `warp.sync` | 自动 activemask |
| `@llvm.custom.shuffle.xor.*` | `shfl.xor rd, rs, mask` | |
| `@llvm.custom.shuffle.up.*` | `shfl.up rd, rs, delta` | |
| `@llvm.custom.shuffle.down.*` | `shfl.down rd, rs, delta` | |
| `@llvm.custom.shuffle.idx.*` | `shfl.idx rd, rs, lane` | |
| `@llvm.custom.ballot` | `ballot rd, pred` | |

### 9.2 实现方式选择

在 LLVM 后端中，可以选择以下方式之一实现 intrinsics：

#### 方式 1: Custom Intrinsics (推荐)

在 `llvm/include/llvm/IR/IntrinsicsCustom.td` 中定义：

```tablegen
let TargetPrefix = "custom" in {
  def int_custom_tid_x : Intrinsic<[llvm_i32_ty], [], 
      [IntrNoMem, IntrSpeculatable]>;
  def int_custom_barrier : Intrinsic<[], [], 
      [IntrConvergent, IntrWillReturn]>;
  // ...
}
```

#### 方式 2: 运行时函数调用

保持为外部函数调用，在链接时提供实现：

```llvm
declare i32 @__custom_tid_x()  ; 链接到运行时库
```

#### 方式 3: 内联汇编

直接生成内联汇编（简单但不推荐）：

```llvm
%tid = call i32 asm "csrr $0, tidx", "=r"()
```

---

## 10. 示例 LLVM IR

### 10.1 Vector Add Kernel

```llvm
target datalayout = "e-m:e-p:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-n32"
target triple = "riscv32-unknown-unknown"

define void @add_kernel(
    ptr addrspace(1) %x,
    ptr addrspace(1) %y,
    ptr addrspace(1) %output,
    i32 %n
) #0 {
entry:
    ; 计算全局线程 ID
    %tid = call i32 @llvm.custom.tid.x()
    %ctaid = call i32 @llvm.custom.ctaid.x()
    %ntid = call i32 @llvm.custom.ntid.x()
    %tmp = mul i32 %ctaid, %ntid
    %gid = add i32 %tmp, %tid
    
    ; 边界检查
    %cmp = icmp ult i32 %gid, %n
    br i1 %cmp, label %compute, label %exit

compute:
    ; 计算地址
    %x_ptr = getelementptr float, ptr addrspace(1) %x, i32 %gid
    %y_ptr = getelementptr float, ptr addrspace(1) %y, i32 %gid
    %out_ptr = getelementptr float, ptr addrspace(1) %output, i32 %gid
    
    ; 加载
    %x_val = load float, ptr addrspace(1) %x_ptr, align 4
    %y_val = load float, ptr addrspace(1) %y_ptr, align 4
    
    ; 计算
    %sum = fadd float %x_val, %y_val
    
    ; 存储
    store float %sum, ptr addrspace(1) %out_ptr, align 4
    br label %exit

exit:
    ret void
}

; Intrinsic declarations
declare i32 @llvm.custom.tid.x() #1
declare i32 @llvm.custom.ctaid.x() #1
declare i32 @llvm.custom.ntid.x() #1

attributes #0 = { "kernel" }
attributes #1 = { nounwind readnone speculatable }
```

### 10.2 Warp Reduction

```llvm
define float @warp_reduce_sum(float %val) {
entry:
    ; Butterfly reduction using shuffle xor
    %s1 = call float @llvm.custom.shuffle.xor.f32(float %val, i32 16)
    %r1 = fadd float %val, %s1
    
    %s2 = call float @llvm.custom.shuffle.xor.f32(float %r1, i32 8)
    %r2 = fadd float %r1, %s2
    
    %s3 = call float @llvm.custom.shuffle.xor.f32(float %r2, i32 4)
    %r3 = fadd float %r2, %s3
    
    %s4 = call float @llvm.custom.shuffle.xor.f32(float %r3, i32 2)
    %r4 = fadd float %r3, %s4
    
    %s5 = call float @llvm.custom.shuffle.xor.f32(float %r4, i32 1)
    %r5 = fadd float %r4, %s5
    
    ret float %r5
}

declare float @llvm.custom.shuffle.xor.f32(float, i32) #0
attributes #0 = { convergent nounwind readnone }
```

---

## 11. 版本信息

| 属性 | 值 |
|------|-----|
| 文档版本 | 1.0 |
| 日期 | 2025-12-22 |
| Triton 分支 | custom-simt-backend |
| LLVM 分支 | custom-simt-backend |

---

## 12. 附录：完整 Intrinsic 列表

```llvm
; ============ Thread/Block ID ============
declare i32 @llvm.custom.tid.x()
declare i32 @llvm.custom.tid.y()
declare i32 @llvm.custom.tid.z()
declare i32 @llvm.custom.ctaid.x()
declare i32 @llvm.custom.ctaid.y()
declare i32 @llvm.custom.ctaid.z()
declare i32 @llvm.custom.ntid.x()
declare i32 @llvm.custom.ntid.y()
declare i32 @llvm.custom.ntid.z()
declare i32 @llvm.custom.nctaid.x()
declare i32 @llvm.custom.nctaid.y()
declare i32 @llvm.custom.nctaid.z()
declare i32 @llvm.custom.lane.id()

; ============ Synchronization ============
declare void @llvm.custom.barrier()
declare void @llvm.custom.warp.barrier()

; ============ Shuffle (i32) ============
declare i32 @llvm.custom.shuffle.xor.i32(i32, i32)
declare i32 @llvm.custom.shuffle.up.i32(i32, i32)
declare i32 @llvm.custom.shuffle.down.i32(i32, i32)
declare i32 @llvm.custom.shuffle.idx.i32(i32, i32)

; ============ Shuffle (i64) ============
declare i64 @llvm.custom.shuffle.xor.i64(i64, i32)
declare i64 @llvm.custom.shuffle.up.i64(i64, i32)
declare i64 @llvm.custom.shuffle.down.i64(i64, i32)
declare i64 @llvm.custom.shuffle.idx.i64(i64, i32)

; ============ Shuffle (f32) ============
declare float @llvm.custom.shuffle.xor.f32(float, i32)
declare float @llvm.custom.shuffle.up.f32(float, i32)
declare float @llvm.custom.shuffle.down.f32(float, i32)
declare float @llvm.custom.shuffle.idx.f32(float, i32)

; ============ Shuffle (f64) ============
declare double @llvm.custom.shuffle.xor.f64(double, i32)
declare double @llvm.custom.shuffle.up.f64(double, i32)
declare double @llvm.custom.shuffle.down.f64(double, i32)
declare double @llvm.custom.shuffle.idx.f64(double, i32)

; ============ Vote ============
declare i32 @llvm.custom.ballot(i1)
```
