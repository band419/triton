# Custom Backend NVIDIA-free LLIR 实现状态

本文档记录 NVIDIA-free LLIR 路线图的实现状态。

---

## 实现状态总览（截至 2025-12-20）

| 阶段 | 任务 | 状态 | 说明 |
|------|------|------|------|
| A0 | 代码结构与开关收敛 | ✅ 完成 | `make_llir` 已拆分，支持 `TRITON_CUSTOM_LLIR_MODE` 开关 |
| A1 | Custom TTGIR→LLVM lowering 入口 | ✅ 完成 | C++ passes 在 `third_party/custom/lib/TritonCustomToLLVM/` |
| A2 | Custom triple/datalayout | ✅ 完成 | 默认使用 `riscv32-unknown-unknown` |
| A3 | 最小 intrinsic 集合 | ✅ 完成 | 已定义 barrier、program_id、shuffle 等 |
| A4 | 语义正确性 | 🔲 待验证 | 需要端到端测试 |
| B1-B5 | Phase B 系统化 | 🔲 待开始 | 需要完成 Phase A 验收后进行 |

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
