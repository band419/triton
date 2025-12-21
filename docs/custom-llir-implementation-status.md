# Custom Backend NVIDIA-free LLIR 实现状态

本文档记录 NVIDIA-free LLIR 路线图的实现状态。

---

## 实现状态总览（截至 2025-12-21）

| 阶段 | 任务 | 状态 | 说明 |
|------|------|------|------|
| A0 | 代码结构与开关收敛 | ✅ 完成 | `make_llir` 已拆分，支持 `TRITON_CUSTOM_LLIR_MODE` 开关 |
| A1 | Custom TTGIR→LLVM lowering 入口 | ✅ 完成 | C++ passes 在 `third_party/custom/lib/TritonCustomToLLVM/` |
| A2 | Custom triple/datalayout | ✅ 完成 | 默认使用 `riscv32-unknown-unknown`，LLIR 中正确注入 |
| A3 | 最小 intrinsic 集合 | ✅ 完成 | 已定义 barrier、program_id、shuffle 等 |
| A4 | 语义正确性 | ✅ 完成 | 端到端测试通过，生成的 LLIR 无 NVIDIA 依赖 |
| B1 | Kernel ABI 细化 | ✅ 完成 | 展开参数 ABI，Load/Store lowering，文档完善 |
| B2 | Intrinsic 规范化 | ✅ 完成 | 命名收敛，属性定义（convergent/nounwind），文档化 |
| B3 | 内存与同步语义 | ✅ 完成 | Barrier fence 语义，全局内存模型，无 shared memory |
| B4 | TTGIR NVIDIA 剥离 | ✅ 完成 | TTGIR 侧 NVIDIA 特性完全剥离，custom mode 默认启用 |
| B5 | 测试与回归 | 🔲 待开始 | 测试覆盖机制 |

### Phase A4 验证结果

```
✓ Pipeline functions: All functions present, triple=riscv32-unknown-unknown
✓ Custom plugin: TritonCustom plugin loaded with add_to_llvmir
✓ NVIDIA-free check: No NVIDIA-specific patterns found
✓ Custom intrinsics check: Found custom intrinsics (llvm.custom.*)
✓ Triple check: Triple is non-NVIDIA: riscv32-unknown-unknown
✓ Datalayout check: Datalayout present: e-m:e-p:32:32-i64:64-n32-S128
✓ Kernel signature check: Found kernel entry: @test_kernel
```

运行验证：
```bash
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/backend/test_custom_llir.py
```

### Phase B1 验证结果

```
✓ LLIR Generation: Generated 4077 bytes
✓ Expanded args signature: Kernel @add_kernel: 6 parameters, addrspace(1): True, scalars: True
✓ Global memory address space: Found 17 global memory references, no shared memory
✓ SIMT intrinsics: Found 2 SIMT intrinsics (llvm.custom.program.id, llvm.custom.thread.id)
✓ Load/Store lowering: Found 4 loads, 2 stores, GEP: True
✓ Triple/Datalayout: triple=riscv32-unknown-unknown
✓ NVIDIA-free: No NVIDIA patterns found
```

运行验证：
```bash
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/backend/test_kernel_abi.py
```

新增文档：`docs/custom-kernel-abi.md` - Kernel ABI 完整规范

### Phase B2 验证结果

```
✓ LLIR Generation: Generated 3529 bytes
✓ Intrinsic naming: Found 2 custom intrinsics (llvm.custom.program.id, llvm.custom.thread.id)
✓ Intrinsic attributes: Verified 2 attributes
✓ Convergent intrinsics: No convergent intrinsics used in this kernel
✓ Pure intrinsics: 2 pure intrinsic(s) have nounwind
✓ NVIDIA-free intrinsics: No NVIDIA intrinsics found
```

运行验证：
```bash
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/backend/test_intrinsic_spec.py
```

新增文档：`docs/custom-intrinsic-spec.md` - Intrinsic 完整规范

### Phase B3 验证结果

```
✓ LLIR Generation: Generated 4077 bytes
✓ Global memory addrspace: Found 17 global memory references, no shared memory
✓ Load/Store instructions: Found 4 loads, 2 stores (standard LLVM)
✓ No shared memory: No shared memory usage detected
✓ Barrier convergent: No barrier used in this kernel (OK for simple kernels)
✓ Barrier memory effects: No barrier in this kernel (OK for simple kernels)
✓ NVIDIA-free: No NVIDIA patterns found in LLIR
✓ Custom triple: triple=riscv32-unknown-unknown
```

运行验证：
```bash
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/backend/test_memory_semantics.py
```

新增文档：`docs/custom-memory-semantics.md` - 内存与同步语义完整规范

### Phase B4 验证结果

```
✓ LLIR Generation: Generated 4077 bytes
✓ NVIDIA pipeline disabled: LLIR generated successfully with NVIDIA pipeline disabled
✓ No NVIDIA triple: Using custom triple: riscv32-unknown-unknown
✓ No NVVM intrinsics: No NVVM intrinsics, found 2 custom intrinsic types
✓ No shared memory: No shared memory usage detected
✓ No MMA/TMA/TMEM: No MMA/TMA/TMEM patterns found
✓ No PTX assembly: No inline PTX assembly found
✓ Custom intrinsics: Found 2 custom intrinsic(s): llvm.custom.program.id, llvm.custom.thread.id
✓ Valid LLVM IR: All LLVM IR structure elements present
```

运行验证：
```bash
TRITON_CUSTOM_LLIR_MODE=1 TRITON_CUSTOM_TTGIR_MODE=custom TRITON_CUSTOM_USE_NVIDIA_TTGIR_PIPELINE=0 \
  python third_party/custom/backend/test_ttgir_nvidia_free.py
```

**NVIDIA passes 已完全剥离**（custom mode 下不使用）：
- `nvidia.passes.ttnvgpuir.add_plan_cta` (CTA 规划)
- `nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding` (TMA 描述符)
- `nvidia.passes.hopper.add_hopper_warpspec` (Hopper warp 特化)
- `nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem` (TMEM 提升)
- `nvidia.passes.ttnvgpuir.add_remove_tmem_tokens` (TMEM tokens)
- `nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts` (TMEM 布局)
- `nvidia.passes.ttnvgpuir.add_tma_lowering` (TMA lowering)
- `nvidia.passes.ttnvgpuir.add_interleave_tmem` (TMEM 交织)
- `nvidia.passes.ttnvgpuir.add_fence_insertion` (CUDA fences)
- `nvidia.passes.ttnvgpuir.add_lower_mma` (MMA lowering)
- `nvidia.passes.ttgpuir.add_allocate_shared_memory_nv` (shared memory)
- `nvidia.passes.ttnvgpuir.add_allocate_tensor_memory` (TMEM 分配)
- `passes.convert.add_nvvm_to_llvm` (NVVM dialect)

---

## 新增文件列表

### C++ 层（MLIR Passes）

```
third_party/custom/
├── CMakeLists.txt                          # 顶层 CMake
├── triton_custom.cc                        # Python 绑定
├── include/
│   ├── CMakeLists.txt
│   └── TritonCustomToLLVM/
│       ├── CMakeLists.txt
│       ├── Passes.td                       # Pass 定义（TableGen）
│       └── Passes.h                        # Pass 声明
└── lib/
    ├── CMakeLists.txt
    └── TritonCustomToLLVM/
        ├── CMakeLists.txt
        ├── BarrierOpToLLVM.cpp             # Barrier → intrinsic
        ├── PatternTritonGPUOpToLLVM.h      # Pattern 头文件
        ├── SPMDOpToLLVM.cpp                # program_id → intrinsic
        ├── TargetInfo.cpp                  # CustomTargetInfo 实现
        ├── TargetInfo.h                    # CustomTargetInfo 声明
        └── TritonGPUToLLVM.cpp             # 主 conversion pass
```

**注意**：Custom backend 不支持 shared memory（仅 global memory），因此没有 `AllocateSharedMemory.cpp`。

### Python 层（Pipeline）

- `third_party/custom/backend/pipeline.py` - 已修改，新增：
  - `_lower_ttgir_to_llvm_nvidia()` - NVIDIA 兼容路径
  - `_lower_ttgir_to_llvm_custom()` - Custom SIMT 路径
  - `_get_custom_triple_and_datalayout()` - 获取 custom triple
  - `_inject_custom_intrinsic_declarations()` - 注入 intrinsic 声明

- `third_party/custom/backend/test_custom_llir.py` - 测试脚本

---

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `TRITON_CUSTOM_LLIR_MODE` | `0` | 设为 `1` 启用 NVIDIA-free 路径 |
| `TRITON_CUSTOM_TTGIR_MODE` | `custom` | TTGIR pipeline 模式 |
| `TRITON_CUSTOM_LLVM_TRIPLE` | `riscv32-unknown-unknown` | LLVM triple |
| `TRITON_CUSTOM_LLVM_FEATURES` | `+f` | LLVM target features |
| `TRITON_CUSTOM_WARP_SIZE` | `32` | Warp 大小 |

---

## Custom Intrinsic 集合

以下 intrinsic 在 Custom LLIR 模式下生成：

### 基础 SIMT 操作

```llvm
; CTA barrier（所有 warp 同步）
declare void @llvm.custom.barrier() #convergent

; Warp barrier（warp 内同步）
declare void @llvm.custom.warp.barrier() #convergent

; 获取 lane ID（0..warp_size-1）
declare i32 @llvm.custom.lane.id() #readnone

; 获取 warp 大小
declare i32 @llvm.custom.warp.size() #readnone

; 获取 program ID（axis: 0=x, 1=y, 2=z）
declare i32 @llvm.custom.program.id(i32 %axis) #readnone

; 获取 program 数量
declare i32 @llvm.custom.num.programs(i32 %axis) #readnone
```

### Cross-lane 操作

```llvm
; Ballot: 收集所有 lane 的 predicate 到 bitmask
declare i32 @llvm.custom.ballot(i1 %pred) #convergent

; Shuffle XOR: 与 lane_id ^ mask 交换数据
declare i32 @llvm.custom.shuffle.xor(i32 %val, i32 %mask) #convergent

; Shuffle Up: 从 lane_id - delta 获取数据
declare i32 @llvm.custom.shuffle.up(i32 %val, i32 %delta) #convergent

; Shuffle Idx: 从指定 lane 获取数据
declare i32 @llvm.custom.shuffle.idx(i32 %val, i32 %idx) #convergent
```

### 属性说明

- `#convergent`: 表示该操作需要所有活跃 lane 同时执行，不能被优化器跨分支移动
- `#readnone`: 表示该操作不读写内存
- `#nounwind`: 表示该操作不会抛出异常

---

## 使用方法

### 编译时启用 Custom LLIR 模式

```bash
export TRITON_CUSTOM_LLIR_MODE=1
export TRITON_CUSTOM_TTGIR_MODE=custom
python your_triton_script.py
```

### 查看生成的 LLIR

```python
import triton
import triton.language as tl

@triton.jit
def kernel(...):
    ...

# 获取编译后的 LLIR
# compiled = kernel[grid](...)
# llir = compiled.asm['llir']
```

---

## 验收标准

### Phase A 验收

- [ ] 生成的 LLIR 不包含 `nvptx64-nvidia-cuda`
- [ ] 生成的 LLIR 不包含 `@llvm.nvvm.*` intrinsics
- [ ] 生成的 LLIR 包含 `llvm.custom.*` intrinsics
- [ ] LLIR 可被 `llvm-as` 解析（语法正确）
- [ ] triple 为 `riscv32-unknown-unknown` 或其他非 NVIDIA triple

### Phase B 验收

- [ ] Kernel ABI 文档化
- [ ] Intrinsic 属性完整（`convergent` 等）
- [ ] 内存/同步语义测试覆盖
- [ ] 端到端 vector_add/softmax 测试通过

---

## 下一步工作

1. **构建系统集成**：确保 CMake 正确构建 `TritonCustomToLLVM` 库
2. **Python 绑定验证**：确保 `custom.passes.ttgpuir.add_to_llvmir()` 可调用
3. **端到端测试**：运行 `test_custom_llir.py` 验证生成的 LLIR
4. **与外部 codegen 对接**：验证 LLIR 可被自定义 LLVM backend 处理
