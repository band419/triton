# Custom Backend：NVIDIA-free LLIR 路线图（Phase A/B）

本文档用于**跟踪**将 Triton `custom` backend 的 `llir` 产物从“复用 NVIDIA lowering（含 NVVM/NVGPU 语义）”逐步演进到“完全不包含 NVIDIA 硬件特性、可对接自定义 SIMT 处理器 LLVM codegen”的实现计划。

> 说明：本文只记录 Phase A/B（工程落地层面）。更偏 ABI/产物格式的规范见：`docs/custom-simt-target-spec.md`。

---

## 背景与目标

### 当前状态（截至 2025-12-20）
- 已有 in-tree `custom` backend，可输出 `.blob`（Scheme A / Blob v1）。
- `custom` 的 pipeline 已独立在 `third_party/custom/backend/pipeline.py`，但 `make_llir` 仍依赖 NVIDIA 的 TTGIR→LLVM lowering / NVVM→LLVM 等路径（因此 LLIR 仍带 NVIDIA 语义）。

### 最终目标（Long-term）
- `llir`（LLVM IR）中不包含 NVIDIA 专用：
  - 不依赖 NVVM/NVGPU dialect lowering
  - 不使用 `nvptx64-nvidia-cuda` triple/datalayout/features
  - 不含 PTX-specific address space/metadata 约定（除非被我们明确复用并文档化为“自定义约定”）
- `llir` 能表达 custom SIMT 语义，并可交给自定义 LLVM backend（或外部 codegen）生成目标机器码。

---

## 已确认决策（来自你最新输入）

### 1) Kernel entry ABI（LLVM IR 层）：展开参数（expanded args）
- Kernel 入口函数在 LLVM IR 层采用“参数展开”方式传参（而不是单一 param-block 指针）。
- 影响：TTGIR→LLVM lowering 必须产出稳定的 LLVM function signature，并在 runtime/driver 层与 launcher 调用约定一致。

### 2) SIMT 语义：先表示成 `llvm.intrinsic`
- 先用 LLVM intrinsic 表达关键 SIMT/运行时语义（例如 barrier、lane id、program id 等），后续再在 custom LLVM backend 或自定义 LLVM pass 中进行 lowering。
- 原则：
  - Phase A：允许先用“declare intrinsic + call”的形式保住语义
  - Phase B：逐步替换/收敛 intrinsic 集合，并形成明确的 ABI + lowering 规范

---

## Phase A：把 LLIR 从 NVIDIA lowering 中“解耦出来”（可用、可迭代）

**目标**：让 `custom` backend 生成的 `llir` 不再依赖 NVPTX/NVVM 的转换链条；即使暂时无法完整 codegen，也要做到 LLIR 的语义来源是“custom contract”，而不是 NVIDIA contract。

### A0. 代码结构与开关收敛（Python 层）
- 位置：`third_party/custom/backend/pipeline.py`
- 工作项：
  - 将 `make_llir` 中与 NVIDIA 强绑定的步骤拆分为独立函数（例如 `lower_ttgir_to_llvmir_nvidia()`、`postprocess_llvm_ir_custom()`）。
  - 默认行为逐步从“复用 NVIDIA”切到“custom 优先”，但保留环境变量回退开关（便于 bisect）。

**验收**：
- 在不开启 NVIDIA 回退时，pipeline 不会调用 `passes.convert.add_nvvm_to_llvm` / 不会设置 NVPTX triple。

### A1. 增加 Custom-specific TTGIR→LLVMIR lowering 入口（C++/MLIR 扩展）
- 位置建议：
  - Pass 定义：`include/triton/Dialect/Triton/Transforms/Passes.td`
  - Pass 实现：`lib/Dialect/Triton*/Transforms/...`（具体放置取决于现有目录结构与 dialect 归属）
- 新增内容：
  - 一个最小可用的 pass（或 pass pipeline）用于将“custom TTGIR（或其子集）”转换到 MLIR LLVM dialect / 或直接导出 LLVM IR。
  - 这条链路**不得**引用 NVGPU/NVVM 专用转换。

**验收**：
- `custom` backend 在不链接 NVIDIA lowering 的情况下，能生成可解析的 LLVM IR 文本（`llvm-as` 可过）。

### A2. 定义并注入 Custom LLVM triple / datalayout / calling conv（最小版本）
- 决策点：
  - triple：例如 `riscv32-unknown-unknown`（后续按你的 SIMT 扩展命名）
  - datalayout：至少要与 RV32F/指针宽度/对齐策略一致
- 位置：`third_party/custom/backend/pipeline.py`（生成 LLIR 后的属性注入/修正）或 C++ lowering 时直接设定。

**验收**：
- `llir` 内含目标 triple/datalayout，且不包含 `nvptx64-nvidia-cuda` 字样。

### A3. Intrinsic：最小集合（先把语义钉住）
先定义一组最小 intrinsic（名字可先占位，后续可替换为 LLVM 官方/Target intrinsic 方案）：
- `llvm.custom.barrier()`：CTA barrier
- `llvm.custom.lane.id()`：lane id（0..warp_size-1）
- `llvm.custom.warp.size()`：warp size（常量/或 target constant）
- `llvm.custom.program.id(axis)`：program_id
- `llvm.custom.num.programs(axis)`：num_programs
- （如需要）`llvm.custom.printf`/debug（可选，不建议 Phase A 绑定）

**验收**：
- lowering 链路能在 LLIR 中产出 `declare` + `call`，且调用点覆盖 Triton 语义需要的位置。

### A4. 暂不追求“性能正确”，先追求“语义正确、可链接/可读”
- Phase A 的成功标准是：生成的 LLIR 不含 NVIDIA 专用语义，并且能作为后续 LLVM backend/codegen 的输入。

---

## Phase B：把“custom SIMT 语义”系统化（可 codegen、可维护）

**目标**：让 LLIR 的语义与 custom SIMT 硬件模型一一对应，并形成稳定 ABI + intrinsic + lowering 规范；为后端 codegen（外部工具或 LLVM target backend）提供确定输入。

### B1. Kernel ABI 细化（expanded args）
- 明确 kernel signature：
  - 标量参数类型映射（i32/f32/pointer 等）
  - 指针 address space 与 memory model（global-only 的情况下更简单）
  - 约定 `program_id`、`lane_id` 等通过 intrinsic 获取，而不是通过隐式参数注入（除非未来需要优化）

**验收**：
- runtime/driver 的 launcher 能严格按“展开参数 ABI”组装调用参数。

### B2. Intrinsic 规范化与命名收敛
- 将 Phase A 的占位 intrinsic：
  - 统一命名空间、参数、返回类型
  - 定义属性（例如 `readnone`, `convergent`, `nounwind`）
  - 明确哪些 intrinsic 必须 `convergent`（例如 barrier）

**验收**：
- LLIR 的 intrinsic 调用在 LLVM 优化后仍保序/不被错误移动（尤其 barrier）。

### B3. 内存与同步语义
- 全局内存 + CTA barrier 的情况下：
  - 明确 fence/ordering（如果硬件只有 barrier，没有更细粒度 fence，需要在语义上约束）
  - 将 Triton 的 memory ops 映射到 LLVM 的 load/store + 合适的 `volatile`/`atomic`/`fence` 或 custom intrinsic

**验收**：
- 一组 lit/unittest 覆盖：barrier 前后的内存可见性符合预期（至少在模型层面）。

### B4. TTGIR 侧：剥离 NVIDIA 特性
- 将 `make_ttgir` 的 NVIDIA-only pass 逐步替换为 custom 对应物：
  - 计划/布局/向量化策略应以 custom ISA/SIMT 模型为准
  - 禁用/替换 MMA/TMA/特定 shared-memory 优化等

**验收**：
- 在关闭 `TRITON_CUSTOM_USE_NVIDIA_TTGIR_PIPELINE` 时仍能产出合法 LLIR。

### B5. 测试与可回归机制
- 增加最小测试：
  - `custom` backend 的 `llir` 快照测试（检查不含 `nvptx`/`nvvm` 字样，且包含预期 triple、intrinsic 声明）
  - 关键 lowering pass 的单测（MLIR 层）

**验收**：
- CI 或本地 `lit` 可跑，至少覆盖“编译出 LLIR + blob”的主路径。

---

## 开放问题（暂不阻塞 Phase A）
- Custom LLVM backend 采用：
  - 方案 1：LLVM out-of-tree target
  - 方案 2：外部 codegen（`TRITON_CUSTOM_CODEGEN`）读取 LLIR
- Address space / calling convention 的最终形式（Phase A 先简化，Phase B 固化）。

---

## 近期行动项（建议按顺序）
1. Phase A1：新增 custom TTGIR→LLVM lowering pass（先最小化）
2. Phase A2：注入 custom triple/datalayout
3. Phase A3：落地最小 intrinsic 集合 + 确认属性（`convergent` 等）
4. Phase B1：把 kernel 展开参数 ABI 文档化并对齐 launcher

