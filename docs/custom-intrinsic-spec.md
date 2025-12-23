# Custom SIMT Backend - Intrinsic Specification

本文档定义 Custom SIMT backend 的 LLVM intrinsic 规范，包括命名约定、参数格式、返回类型和属性要求。

---

## 1. 命名约定

所有 Custom backend intrinsic 使用 `llvm.custom.` 前缀：

```
llvm.custom.<category>.<operation>
```

类别包括：
- **simt**: SIMT 执行模型相关（program_id, thread_id, lane_id）
- **sync**: 同步操作（barrier, fence）
- **shuffle**: Cross-lane 数据交换
- **ballot**: Warp-wide 断言收集

---

## 2. Intrinsic 完整列表

### 2.1 Program/Thread ID 相关

#### `llvm.custom.program.id`
获取当前 CTA/block 的 ID（对应 Triton 的 `tl.program_id`）。

```llvm
declare i32 @llvm.custom.program.id(i32 %axis) #readnone_attrs
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `axis` | i32 | 维度: 0=x, 1=y, 2=z |
| 返回值 | i32 | 该维度上的 program ID |

**属性**: `readnone`, `nounwind`, `willreturn`

**硬件映射**: `csrr t0, CSR_SIMT_CTAID_X` (0xFD8)

---

#### `llvm.custom.thread.id`
获取 CTA 内的线程 ID（对应 CUDA 的 `threadIdx`）。

```llvm
declare i32 @llvm.custom.thread.id(i32 %axis) #readnone_attrs
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `axis` | i32 | 维度: 0=x, 1=y, 2=z |
| 返回值 | i32 | 该维度上的 thread ID |

**属性**: `readnone`, `nounwind`, `willreturn`

**硬件映射**: `csrr t0, CSR_SIMT_TID_X` (0xFC0)

---

#### `llvm.custom.lane.id`
获取当前 lane 在 warp 内的索引。

```llvm
declare i32 @llvm.custom.lane.id() #readnone_attrs
```

| 返回值 | 类型 | 说明 |
|--------|------|------|
| lane_id | i32 | 0 到 warp_size-1 |

**属性**: `readnone`, `nounwind`, `willreturn`

**硬件映射**: `csrr t0, CSR_SIMT_LANEID` (0xFE4)

---

#### `llvm.custom.warp.size`
获取 warp 大小（通常为常量 32）。

```llvm
declare i32 @llvm.custom.warp.size() #readnone_attrs
```

**属性**: `readnone`, `nounwind`, `willreturn`

**硬件映射**: `csrr t0, CSR_SIMT_WARPSIZE` (0xFE8) 或编译时常量

---

#### `llvm.custom.num.programs`
获取 grid 维度（program 总数）。

```llvm
declare i32 @llvm.custom.num.programs(i32 %axis) #readnone_attrs
```

**属性**: `readnone`, `nounwind`, `willreturn`

**硬件映射**: 从 kernel descriptor 获取 grid_size

---

#### `llvm.custom.block.dim`
获取 block 维度（CTA 内线程数）。

```llvm
declare i32 @llvm.custom.block.dim(i32 %axis) #readnone_attrs
```

**属性**: `readnone`, `nounwind`, `willreturn`

**硬件映射**: `csrr t0, CSR_SIMT_NTID_X` (0xFCC)

---

#### `llvm.custom.block.id`
获取 block ID（同 program.id）。

```llvm
declare i32 @llvm.custom.block.id(i32 %axis) #readnone_attrs
```

**属性**: `readnone`, `nounwind`, `willreturn`

---

#### `llvm.custom.grid.dim`
获取 grid 维度。

```llvm
declare i32 @llvm.custom.grid.dim(i32 %axis) #readnone_attrs
```

**属性**: `readnone`, `nounwind`, `willreturn`

---

### 2.2 同步操作

#### `llvm.custom.barrier`
CTA 级别的 barrier（所有 warp 同步）。

```llvm
declare void @llvm.custom.barrier() #convergent_attrs
```

**属性**: `convergent`, `nounwind`

**关键**: `convergent` 属性防止 LLVM 优化器将此调用移动到条件分支之外。

**硬件映射**: `fence` + `bar.sync`（其中 `bar.sync` 只保证 CTA 级别同步，不保证 fence；若 LLIR barrier 语义要求 sync+fence，则需要显式 `fence`）

**内存语义**: Barrier 同时充当 memory fence，保证 barrier 前后的内存访问可见性。

---

#### `llvm.custom.warp.barrier`
Warp 级别的 barrier（warp 内同步）。

```llvm
declare void @llvm.custom.warp.barrier() #convergent_attrs
```

**属性**: `convergent`, `nounwind`

**硬件映射**: v1 可降级为 CTA barrier（`fence` + `bar.sync`）；若后续 ISA 增加 warp sync 再单独 lower

---

### 2.3 Cross-lane 操作

#### `llvm.custom.shuffle.xor`
XOR shuffle: 与 `lane_id ^ mask` 位置的 lane 交换数据。

```llvm
declare i32 @llvm.custom.shuffle.xor.i32(i32 %val, i32 %mask) #convergent_attrs
declare float @llvm.custom.shuffle.xor.f32(float %val, i32 %mask) #convergent_attrs
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `val` | 任意标量 | 要交换的值 |
| `mask` | i32 | XOR 掩码 |
| 返回值 | 同 val | 来自 `lane_id ^ mask` 的值 |

**属性**: `convergent`, `nounwind`

**注意**: 如果目标 lane 不活跃，结果未定义。

**Lowering 约定（v1）**：后端/编译器统一将 `shuffle.*` 降到 `SHFL.IDX`。

---

#### `llvm.custom.shuffle.up`
向上 shuffle: 获取 `lane_id - delta` 位置的值。

```llvm
declare i32 @llvm.custom.shuffle.up.i32(i32 %val, i32 %delta) #convergent_attrs
```

**属性**: `convergent`, `nounwind`

**Lowering 约定（v1）**：降到 `SHFL.IDX`，其中 `src_lane = lane_id - delta`。

---

#### `llvm.custom.shuffle.down`
向下 shuffle: 获取 `lane_id + delta` 位置的值。

```llvm
declare i32 @llvm.custom.shuffle.down.i32(i32 %val, i32 %delta) #convergent_attrs
```

**属性**: `convergent`, `nounwind`

**Lowering 约定（v1）**：降到 `SHFL.IDX`，其中 `src_lane = lane_id + delta`。

---

#### `llvm.custom.shuffle.idx`
索引 shuffle: 获取指定 lane 的值。

```llvm
declare i32 @llvm.custom.shuffle.idx.i32(i32 %val, i32 %src_lane) #convergent_attrs
```

**属性**: `convergent`, `nounwind`

**Lowering 约定（v1）**：直接 lower 到 `SHFL.IDX`。

---

### 2.4 Ballot/Vote 操作

#### `llvm.custom.ballot`
收集 warp 内所有 lane 的断言到位掩码。

```llvm
declare i32 @llvm.custom.ballot(i1 %pred) #convergent_attrs
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `pred` | i1 | 每个 lane 的断言 |
| 返回值 | i32 | 32 位掩码，bit i = lane i 的断言 |

**属性**: `convergent`, `nounwind`

---

### 2.5 其他

#### `llvm.custom.permute`
字节排列操作。

```llvm
declare i32 @llvm.custom.permute(i32 %a, i32 %b, i32 %selector) #nounwind_attrs
```

**属性**: `nounwind`

---

#### `llvm.custom.assert.fail`
断言失败处理。

```llvm
declare void @llvm.custom.assert.fail() #nounwind_attrs
```

**属性**: `nounwind`, `noreturn`

---

## 3. 属性组定义

在 LLIR 中使用以下属性组：

```llvm
; Pure functions (no side effects, no memory access)
attributes #readnone_attrs = { nounwind willreturn memory(none) }

; Convergent functions (must not be moved past control flow)
attributes #convergent_attrs = { convergent nounwind }

; Basic functions
attributes #nounwind_attrs = { nounwind }
```

---

## 4. 关键约束

### 4.1 Convergent 操作

以下操作 **必须** 标记为 `convergent`：
- `llvm.custom.barrier`
- `llvm.custom.warp.barrier`
- `llvm.custom.shuffle.*`
- `llvm.custom.ballot`

**原因**: 这些操作依赖于所有（或特定）lane 同时执行。如果 LLVM 优化器将这些调用移动到条件分支之外，会导致死锁或数据错误。

### 4.2 优化安全性

- `readnone` 操作可以被 CSE（公共子表达式消除）和 LICM（循环不变量外提）
- `convergent` 操作不能被 LLVM 的控制流优化移动
- `willreturn` 保证函数会正常返回（不会无限循环）

### 4.3 内存模型

- Barrier 操作隐含 fence 语义
- 所有内存操作使用 `addrspace(1)` (global memory)
- 无 shared memory 支持（addrspace 3 禁用）

---

## 5. 示例

### 生成的 LLIR 示例

```llvm
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown"

; Intrinsic declarations
declare i32 @llvm.custom.program.id(i32) #0
declare i32 @llvm.custom.thread.id(i32) #0
declare void @llvm.custom.barrier() #1
declare i32 @llvm.custom.shuffle.xor.i32(i32, i32) #1

; Kernel definition
define void @add_kernel(ptr addrspace(1) %x, ptr addrspace(1) %y, 
                        ptr addrspace(1) %out, i32 %n) #2 {
entry:
  %pid = call i32 @llvm.custom.program.id(i32 0)
  %tid = call i32 @llvm.custom.thread.id(i32 0)
  ; ... computation ...
  call void @llvm.custom.barrier()
  ; ... more computation ...
  ret void
}

; Attribute groups
attributes #0 = { nounwind willreturn memory(none) }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind }
```

---

## 6. 验证清单

- [x] 所有 intrinsic 使用 `llvm.custom.` 前缀
- [x] ID 类操作标记为 `readnone`
- [x] Barrier/shuffle/ballot 标记为 `convergent`
- [x] 所有操作标记为 `nounwind`
- [ ] LLVM 优化后 barrier 位置正确（需要测试验证）
- [ ] CSE 不会错误合并有副作用的调用

---

## 7. 硬件映射表

| Intrinsic | CSR/Instruction | 地址/Opcode |
|-----------|-----------------|-------------|
| program.id | CSR_SIMT_CTAID_X | 0xFD8 |
| thread.id | CSR_SIMT_TID_X | 0xFC0 |
| lane.id | CSR_SIMT_LANEID | 0xFE4 |
| warp.size | CSR_SIMT_WARPSIZE | 0xFE8 |
| block.dim | CSR_SIMT_NTID_X | 0xFCC |
| barrier | bar.sync | SIMT extension |
| shuffle | Cross-lane unit | Opcode 0x0B |
| ballot | Cross-lane unit | Opcode 0x0B |
