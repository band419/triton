# Custom SIMT Backend 开发环境搭建指南

本文档说明如何在新电脑上编译 Triton + LLVM 并复现 Custom SIMT Backend 的开发环境。

---

## 1. 环境要求

### 1.1 硬件/系统
- **已验证平台**: macOS 26.x (ARM64/Apple Silicon)
- **应该兼容**: macOS 12+, Linux (Ubuntu 20.04+)
- **内存**: 建议 16GB+ (LLVM 编译需要大量内存)
- **磁盘**: 约 50GB 空闲空间

### 1.2 软件依赖

```bash
# macOS (使用 Homebrew)
brew install cmake ninja ccache python@3.11 git

# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y cmake ninja-build ccache python3.11 python3.11-dev git \
    build-essential clang lld zlib1g-dev

# 验证版本
cmake --version   # >= 3.20
ninja --version   # >= 1.11
python3 --version # == 3.11.x (推荐)
```

### 1.3 Conda 环境

```bash
# 安装 Miniconda (如果没有)
# macOS ARM64:
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Linux:
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建开发环境
conda create -n triton-dev python=3.11 -y
conda activate triton-dev

# 安装必要的 Python 包
pip install numpy pybind11 pytest lit filecheck
```

---

## 2. 获取源代码

### 2.1 Clone Triton 仓库

```bash
# 使用 fork 仓库 (包含 custom backend 代码)
git clone git@github.com:band419/triton.git
cd triton

# 切换到 custom-simt-backend 分支
git checkout custom-simt-backend

# 当前提交 (2025-12-21):
# 5c93510643c40cf250f06d46b0b6728b0f82eb73 matmul custom llvmir
```

### 2.2 仓库结构概览

```
triton/
├── cmake/
│   └── llvm-hash.txt          # LLVM commit hash
├── llvm-project/              # LLVM 子目录 (需编译)
├── third_party/
│   └── custom/                # Custom SIMT backend ⭐
│       ├── backend/           # Python pipeline
│       ├── include/           # C++ headers
│       └── lib/               # C++ implementation
├── python/                    # Triton Python 层
└── scripts/
    └── build-llvm-project.sh  # LLVM 编译脚本
```

---

## 3. 编译 LLVM

Custom backend 需要完整的 LLVM + MLIR 支持。

### 3.1 使用官方脚本编译

```bash
cd /path/to/triton

# 设置环境变量 (可选，使用默认值)
export LLVM_BUILD_TYPE=RelWithDebInfo    # 编译类型
export LLVM_TARGETS="Native;NVPTX;AMDGPU" # 目标架构
export LLVM_PROJECTS="mlir;llvm;lld"      # 需要的子项目

# 重要: 使用 fork 仓库以确保版本稳定
export LLVM_PROJECT_URL="https://github.com/band419/llvm-project"
export LLVM_COMMIT_HASH="custom-simt-backend"  # 使用分支名而不是 hash

# 执行编译 (首次约 30-60 分钟)
./scripts/build-llvm-project.sh

# LLVM 将编译到:
# - 源码: llvm-project/
# - 构建: llvm-project/build/
# - 安装: llvm-project/install/
```

### 3.2 手动编译 (高级)

```bash
cd /path/to/triton

# 获取 LLVM hash
LLVM_HASH=$(cat cmake/llvm-hash.txt)
echo "LLVM commit: $LLVM_HASH"
# 当前: a992f29451b9e140424f35ac5e20177db4afbdc0

# 如果 llvm-project 不存在，clone 它
# 使用 fork 仓库以确保版本稳定
if [ ! -d llvm-project ]; then
    git clone https://github.com/band419/llvm-project.git
fi

cd llvm-project

# 切换到 custom-simt-backend 分支
git fetch origin custom-simt-backend
git checkout custom-simt-backend

# 添加上游仓库以便同步更新 (可选)
git remote add upstream https://github.com/llvm/llvm-project.git

# 配置 CMake
mkdir -p build && cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    ../llvm

# 编译 (使用所有核心)
ninja -j$(nproc)
```

### 3.3 验证 LLVM 编译

```bash
# 检查关键工具
./llvm-project/build/bin/llvm-config --version
./llvm-project/build/bin/mlir-opt --version

# 应该输出类似: 21.0.0git
```

---

## 4. 编译 Triton (含 Custom Backend)

### 4.1 设置环境变量

```bash
# 激活 conda 环境
conda activate triton-dev

# 设置 LLVM 路径
export LLVM_BUILD_DIR=/path/to/triton/llvm-project/build
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export MLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir

# 启用 Custom backend
export TRITON_CODEGEN_BACKENDS="custom"
```

### 4.2 开发模式安装 Triton

```bash
cd /path/to/triton

# 开发模式安装 (可编辑，修改后无需重装)
pip install -e python/ --no-build-isolation -v

# 或者使用 setup.py
cd python
python setup.py develop
```

### 4.3 验证安装

```bash
# 检查 Triton 安装
python -c "import triton; print(triton.__version__)"

# 检查 Custom backend
python -c "from triton.backends.custom import pipeline; print('Custom backend OK')"

# 检查 libtriton
python -c "from triton._C import libtriton; print('libtriton OK')"
```

---

## 5. 编译 Custom Backend C++ 插件

Custom backend 的 C++ 代码需要单独编译。

### 5.1 编译步骤

```bash
cd /path/to/triton

# 配置 CMake (包含 custom backend)
mkdir -p build && cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DTRITON_BUILD_PYTHON_MODULE=ON \
    -DTRITON_CODEGEN_BACKENDS="custom" \
    -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
    ..

# 编译
ninja
```

### 5.2 验证 Custom 插件

```bash
python -c "from triton._C.libtriton import custom; print('Custom plugin loaded')"
```

---

## 6. 运行验证测试

### 6.1 环境变量设置

```bash
# 启用 Custom LLIR 模式
export TRITON_CUSTOM_LLIR_MODE=1
export TRITON_CUSTOM_TTGIR_MODE=custom
export TRITON_CUSTOM_USE_NVIDIA_TTGIR_PIPELINE=0
export TRITON_CUSTOM_ACTIVE=1
```

### 6.2 运行 Phase B 验证测试

```bash
cd /path/to/triton

# B1: Kernel ABI 测试
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/backend/test_kernel_abi.py

# B2: Intrinsic 规范测试
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/backend/test_intrinsic_spec.py

# B3: 内存语义测试
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/backend/test_memory_semantics.py

# B4: NVIDIA-free TTGIR 测试
TRITON_CUSTOM_LLIR_MODE=1 TRITON_CUSTOM_TTGIR_MODE=custom \
    TRITON_CUSTOM_USE_NVIDIA_TTGIR_PIPELINE=0 \
    python third_party/custom/backend/test_ttgir_nvidia_free.py
```

### 6.3 预期输出

每个测试都应该显示类似:
```
✓ LLIR Generation: Generated XXXX bytes
✓ ... (各项检查)
🎉 Phase BX PASSED
```

### 6.4 快速验证脚本

```bash
#!/bin/bash
# 保存为 run_all_tests.sh
set -e

export TRITON_CUSTOM_LLIR_MODE=1
export TRITON_CUSTOM_TTGIR_MODE=custom
export TRITON_CUSTOM_USE_NVIDIA_TTGIR_PIPELINE=0

cd /path/to/triton

echo "=== Running Phase B1 Test ==="
python third_party/custom/backend/test_kernel_abi.py

echo "=== Running Phase B2 Test ==="
python third_party/custom/backend/test_intrinsic_spec.py

echo "=== Running Phase B3 Test ==="
python third_party/custom/backend/test_memory_semantics.py

echo "=== Running Phase B4 Test ==="
python third_party/custom/backend/test_ttgir_nvidia_free.py

echo ""
echo "✅ All tests passed!"
```

---

## 7. 开发工作流

### 7.1 日常开发

```bash
# 激活环境
conda activate triton-dev

# 修改 Python 代码后无需重新编译
# 修改 C++ 代码后需要重新编译:
cd build && ninja && cd ..

# 运行测试
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/backend/test_kernel_abi.py
```

### 7.2 关键文件位置

| 文件 | 用途 |
|------|------|
| `third_party/custom/backend/pipeline.py` | Python 编译 pipeline |
| `third_party/custom/backend/driver.py` | Kernel 启动驱动 |
| `third_party/custom/lib/TritonCustomToLLVM/TargetInfo.cpp` | Intrinsic 实现 |
| `third_party/custom/lib/TritonCustomToLLVM/LoadStoreOpToLLVM.cpp` | Load/Store lowering |
| `third_party/custom/lib/TritonCustomToLLVM/BarrierOpToLLVM.cpp` | Barrier lowering |

### 7.3 调试技巧

```bash
# 查看生成的 LLIR
TRITON_CUSTOM_LLIR_MODE=1 python your_kernel.py
# LLIR 保存在 /tmp/custom_*.ll

# 启用 MLIR 调试输出
TRITON_CUSTOM_LLIR_MODE=1 MLIR_ENABLE_DUMP=1 python your_kernel.py

# 使用 LLVM 工具分析 IR
llvm-project/build/bin/opt -S /tmp/custom_test.ll -o - | less
```

---

## 8. 文档参考

| 文档 | 路径 |
|------|------|
| 路线图 | `docs/custom-llir-nvidia-free-plan.md` |
| 实现状态 | `docs/custom-llir-implementation-status.md` |
| 目标规范 | `docs/custom-simt-target-spec.md` |
| Kernel ABI | `docs/custom-kernel-abi.md` |
| Intrinsic 规范 | `docs/custom-intrinsic-spec.md` |
| 内存语义 | `docs/custom-memory-semantics.md` |

---

## 9. 常见问题

### Q1: LLVM 编译失败 (内存不足)
```bash
# 限制并行度
ninja -j4  # 而不是 ninja -j$(nproc)
```

### Q2: Custom plugin 加载失败
```bash
# 确保编译时启用了 custom backend
cmake -DTRITON_CODEGEN_BACKENDS="custom" ...
```

### Q3: Python import 失败
```bash
# 重新安装 Triton
cd python && pip install -e . --no-build-isolation
```

### Q4: 测试找不到 custom backend
```bash
# 确保环境变量正确
export TRITON_CUSTOM_LLIR_MODE=1
export TRITON_CUSTOM_ACTIVE=1
```

---

## 10. 版本信息

| 组件 | 版本/Commit |
|------|-------------|
| Triton 分支 | `custom-simt-backend` |
| Triton Commit | `5c93510643c40cf250f06d46b0b6728b0f82eb73` |
| LLVM Commit | `a992f29451b9e140424f35ac5e20177db4afbdc0` |
| Python | 3.11.x |
| 文档日期 | 2025-12-21 |

---

## 快速开始 (TL;DR)

```bash
# 1. Clone 并切换分支
git clone git@github.com:band419/triton.git && cd triton
git checkout custom-simt-backend

# 2. 创建 conda 环境
conda create -n triton-dev python=3.11 -y
conda activate triton-dev
pip install numpy pybind11 pytest lit filecheck

# 3. 编译 LLVM (~30-60 分钟)
./scripts/build-llvm-project.sh

# 4. 安装 Triton
export LLVM_BUILD_DIR=$(pwd)/llvm-project/build
export TRITON_CODEGEN_BACKENDS="custom"
pip install -e python/ --no-build-isolation

# 5. 运行测试
TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/backend/test_kernel_abi.py
```
