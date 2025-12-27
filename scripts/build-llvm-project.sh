#!/usr/bin/env bash

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Include RISCV target for Custom SIMT backend
LLVM_TARGETS=${LLVM_TARGETS:-Native;NVPTX;AMDGPU;RISCV}
LLVM_PROJECTS=${LLVM_PROJECTS:-mlir;llvm;lld}
LLVM_BUILD_TYPE=${LLVM_BUILD_TYPE:-RelWithDebInfo}
LLVM_BUILD_SHARED_LIBS=${LLVM_BUILD_SHARED_LIBS:-OFF}
# Use local llvm-project-simt directory
LLVM_PROJECT_PATH=${LLVM_PROJECT_PATH:-"$REPO_ROOT/llvm-project-simt"}
LLVM_BUILD_PATH=${LLVM_BUILD_PATH:-"$LLVM_PROJECT_PATH/build"}
LLVM_INSTALL_PATH=${LLVM_INSTALL_PATH:-"$LLVM_PROJECT_PATH/install"}
LLVM_PROJECT_URL=${LLVM_PROJECT_URL:-"https://github.com/3YCArch/llvm-project-simt.git"}
LLVM_BRANCH=${LLVM_BRANCH:-"simt-main"}

if [ -z "$CMAKE_ARGS" ]; then
    if [ "$#" -eq 0 ]; then
        CMAKE_ARGS=(
            -G Ninja
              -DCMAKE_BUILD_TYPE="$LLVM_BUILD_TYPE"
              -DLLVM_CCACHE_BUILD=OFF
              -DLLVM_ENABLE_ASSERTIONS=ON
              -DCMAKE_C_COMPILER=clang
              -DCMAKE_CXX_COMPILER=clang++
              -DLLVM_ENABLE_LLD=ON
              -DBUILD_SHARED_LIBS="$LLVM_BUILD_SHARED_LIBS"
              -DLLVM_OPTIMIZED_TABLEGEN=ON
              -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
              -DLLVM_ENABLE_ZSTD=OFF
              -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS"
              -DCMAKE_EXPORT_COMPILE_COMMANDS=1
              -DLLVM_ENABLE_PROJECTS="$LLVM_PROJECTS"
              -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PATH"
              -B"$LLVM_BUILD_PATH" "$LLVM_PROJECT_PATH/llvm"
        )
    else
        CMAKE_ARGS=("$@")
    fi
fi

if [ -n "$LLVM_CLEAN" ] && [ -e "$LLVM_BUILD_PATH" ]; then
    rm -rf "$LLVM_BUILD_PATH"
fi

# Always pull the latest version from llvm-project-simt
if [ -e "$LLVM_PROJECT_PATH" ]; then
    echo "Pulling latest changes from llvm-project-simt (branch: $LLVM_BRANCH)"
    git -C "$LLVM_PROJECT_PATH" remote set-url origin "$LLVM_PROJECT_URL"
    git -C "$LLVM_PROJECT_PATH" fetch origin
    git -C "$LLVM_PROJECT_PATH" checkout "$LLVM_BRANCH"
    git -C "$LLVM_PROJECT_PATH" pull origin "$LLVM_BRANCH"
else
    echo "Cloning llvm-project-simt from $LLVM_PROJECT_URL (branch: $LLVM_BRANCH)"
    git clone -b "$LLVM_BRANCH" "$LLVM_PROJECT_URL" "$LLVM_PROJECT_PATH"
fi
echo "Using llvm-project-simt at $(git -C "$LLVM_PROJECT_PATH" rev-parse HEAD)"
echo "Configuring with ${CMAKE_ARGS[@]}"
cmake "${CMAKE_ARGS[@]}"
echo "Building LLVM"
ninja -C "$LLVM_BUILD_PATH"
