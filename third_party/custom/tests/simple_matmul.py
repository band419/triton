#!/usr/bin/env python3
"""
Simple Matmul Kernel - Minimal test for custom SIMT backend

This is a minimal kernel definition for testing the compilation pipeline.
Run with:
    cd /Volumes/t7/prj/triton
    TRITON_CUSTOM_LLIR_MODE=1 python third_party/custom/tests/simple_matmul.py
"""

import os
import sys

# Ensure custom backend is active
os.environ.setdefault("TRITON_CUSTOM_LLIR_MODE", "1")

import triton
import triton.language as tl


@triton.jit
def simple_matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 16,
):
    """Simple tiled matrix multiplication C = A @ B
    
    Note: This version uses explicit element-wise ops instead of tl.dot
    to avoid shared memory requirements that the custom backend
    doesn't yet support.
    """
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Starting positions
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Block offsets
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop over K (element-wise, no tl.dot)
    for k in range(K):
        # Load A column: A[offs_m, k]
        a_ptrs = A + offs_m * stride_am + k * stride_ak
        a_mask = offs_m < M
        a_col = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M]
        
        # Load B row: B[k, offs_n]
        b_ptrs = B + k * stride_bk + offs_n * stride_bn
        b_mask = offs_n < N
        b_row = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_N]
        
        # Outer product: acc += a_col[:, None] * b_row[None, :]
        acc += a_col[:, None] * b_row[None, :]
    
    # Store result
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def vector_add_kernel(
    X, Y, Z,
    N,
    BLOCK: tl.constexpr = 256,
):
    """Simple vector addition Z = X + Y"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    z = x + y
    tl.store(Z + offs, z, mask=mask)


def get_kernel_ir(kernel_fn, sig, constants, stage='llir'):
    """Extract IR at a specific stage from a kernel."""
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget
    
    # Use custom backend target
    target = GPUTarget('custom', 70, 32)
    
    src = ASTSource(
        fn=kernel_fn,
        signature=sig,
        constexprs=constants,
        attrs=None,
    )
    
    compiled = triton_compile(
        src=src,
        target=target,
        options={'num_warps': 4, 'num_stages': 2},
    )
    
    return compiled


def test_vector_add():
    """Test vector add kernel compilation."""
    print("\n=== Vector Add Kernel ===")
    
    sig = {
        'X': "*fp32",
        'Y': "*fp32",
        'Z': "*fp32",
        'N': "i32",
    }
    constants = {'BLOCK': 256}
    
    try:
        result = get_kernel_ir(vector_add_kernel, sig, constants)
        print(f"Compilation successful!")
        print(f"  Type: {type(result)}")
        
        # Try to get LLIR
        if hasattr(result, 'asm'):
            asm = result.asm
            if isinstance(asm, dict):
                for key, value in asm.items():
                    size = len(value) if isinstance(value, (str, bytes)) else str(value)
                    print(f"  {key}: {size}")
            else:
                print(f"  asm: {type(asm)}")
                
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_matmul():
    """Test matmul kernel compilation."""
    print("\n=== Matmul Kernel ===")
    
    sig = {
        'A': "*fp32",
        'B': "*fp32",
        'C': "*fp32",
        'M': "i32",
        'N': "i32",
        'K': "i32",
        'stride_am': "i32",
        'stride_ak': "i32",
        'stride_bk': "i32",
        'stride_bn': "i32",
        'stride_cm': "i32",
        'stride_cn': "i32",
    }
    constants = {
        'BLOCK_M': 16,
        'BLOCK_N': 16,
    }
    
    try:
        result = get_kernel_ir(simple_matmul_kernel, sig, constants)
        print(f"Compilation successful!")
        print(f"  Type: {type(result)}")
        
        if hasattr(result, 'asm'):
            asm = result.asm
            if isinstance(asm, dict):
                for key, value in asm.items():
                    size = len(value) if isinstance(value, (str, bytes)) else str(value)
                    print(f"  {key}: {size}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Custom SIMT Backend - Simple Kernel Test")
    print("=" * 60)
    
    # Print environment
    print("\nEnvironment:")
    for var in ["TRITON_CUSTOM_LLIR_MODE", "TRITON_CUSTOM_CODEGEN_MODE"]:
        print(f"  {var} = {os.environ.get(var, '(not set)')}")
    
    # Run tests
    v_ok = test_vector_add()
    m_ok = test_matmul()
    
    print("\n" + "=" * 60)
    print(f"Results: vector_add={'PASS' if v_ok else 'FAIL'}, matmul={'PASS' if m_ok else 'FAIL'}")
    print("=" * 60)
    
    return 0 if (v_ok and m_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
