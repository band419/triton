#!/usr/bin/env python3
"""
Custom SIMT Backend - Matrix Multiplication Kernel Pipeline Test

This script tests the complete compilation pipeline for custom SIMT backend
using a simple matrix multiplication kernel.

Usage:
    # Test LLVM IR generation only
    TRITON_CUSTOM_LLIR_MODE=1 python test_matmul_pipeline.py --stage llir

    # Test assembly generation (requires LLVM with RISCV target)
    TRITON_CUSTOM_LLIR_MODE=1 TRITON_CUSTOM_CODEGEN_MODE=asm python test_matmul_pipeline.py --stage asm

    # Test all stages
    python test_matmul_pipeline.py --all
"""

import os
import sys
import argparse
from pathlib import Path

# Set custom backend environment before importing triton
os.environ.setdefault("TRITON_CUSTOM_LLIR_MODE", "1")
os.environ.setdefault("TRITON_CUSTOM_TTGIR_MODE", "custom")
os.environ.setdefault("TRITON_CUSTOM_ACTIVE", "1")

import triton
import triton.language as tl


# =============================================================================
# Matrix Multiplication Kernel
# =============================================================================

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (compile-time constants)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Compute C = A @ B
    
    A: (M, K)
    B: (K, N)
    C: (M, N)
    """
    # Program ID
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Compute starting row/col for this block
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N
    
    # Create block pointers
    offs_m = row_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = col_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Compute block matmul
        acc += tl.dot(a, b)
    
    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# =============================================================================
# Simple Vector Add Kernel (for basic testing)
# =============================================================================

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple vector addition: output = x + y"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# =============================================================================
# Compilation Pipeline Test
# =============================================================================

def get_custom_target():
    """Get custom backend target."""
    from triton.backends.compiler import GPUTarget
    # Custom target: backend='custom', arch=70, warp_size=32
    return GPUTarget('custom', 70, 32)


def compile_kernel_to_stage(kernel, stage: str, **kwargs):
    """Compile a kernel to a specific stage and return the IR.
    
    Args:
        kernel: Triton JIT kernel
        stage: Target stage ('ttir', 'ttgir', 'llir', 'asm', 'obj', 'blob')
        **kwargs: Additional compile arguments
        
    Returns:
        Compiled IR at the specified stage
    """
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.backends.compiler import GPUTarget
    
    # Default compile options
    default_kwargs = {
        'num_warps': 4,
        'num_stages': 2,
    }
    default_kwargs.update(kwargs)
    
    # Create target
    target = get_custom_target()
    
    # Get kernel source
    src = ASTSource(
        fn=kernel,
        signature={
            # For add_kernel
            0: "*fp32",  # x_ptr
            1: "*fp32",  # y_ptr
            2: "*fp32",  # output_ptr
            3: "i32",    # n_elements
        } if kernel.__name__ == 'add_kernel' else {
            # For matmul_kernel
            0: "*fp32",  # a_ptr
            1: "*fp32",  # b_ptr
            2: "*fp32",  # c_ptr
            3: "i32",    # M
            4: "i32",    # N
            5: "i32",    # K
            6: "i32",    # stride_am
            7: "i32",    # stride_ak
            8: "i32",    # stride_bk
            9: "i32",    # stride_bn
            10: "i32",   # stride_cm
            11: "i32",   # stride_cn
        },
        constants={
            'BLOCK_SIZE': 256,
        } if kernel.__name__ == 'add_kernel' else {
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 32,
        },
        attrs=triton.compiler.AttrsDescriptor.from_dict({
            'tt.divisibility': (0, 1, 2),
        }),
    )
    
    # Compile
    try:
        compiled = triton_compile(
            src=src,
            target=target,
            options=default_kwargs,
        )
        
        # Get the requested stage
        if hasattr(compiled, stage):
            return getattr(compiled, stage)
        elif hasattr(compiled, 'asm') and stage == 'llir':
            # Some versions store LLIR in asm dict
            return compiled.asm.get('llir', compiled.asm)
        else:
            return str(compiled)
            
    except Exception as e:
        return f"Compilation failed at stage '{stage}': {e}"


def test_llir_generation():
    """Test LLVM IR generation for both kernels."""
    print("\n" + "="*70)
    print("Testing LLVM IR Generation")
    print("="*70)
    
    # Test add_kernel
    print("\n--- add_kernel LLIR ---")
    try:
        llir = compile_kernel_to_stage(add_kernel, 'llir')
        if isinstance(llir, str) and 'define' in llir:
            print(f"✓ add_kernel LLIR generated ({len(llir)} bytes)")
            
            # Check for custom intrinsics
            if 'llvm.riscv.simt' in llir:
                print("✓ Contains llvm.riscv.simt.* intrinsics")
            elif 'llvm.custom' in llir:
                print("⚠ Contains llvm.custom.* intrinsics (old naming)")
            
            # Check target triple
            if 'riscv32' in llir:
                print("✓ Target triple: riscv32")
            elif 'nvptx' in llir:
                print("⚠ Target triple: nvptx (not custom)")
                
            # Optionally print LLIR
            if os.environ.get("TRITON_CUSTOM_DUMP_LLIR"):
                print("\n--- LLIR Content ---")
                print(llir[:2000] + "..." if len(llir) > 2000 else llir)
        else:
            print(f"✗ Failed: {llir}")
    except Exception as e:
        print(f"✗ Exception: {e}")
    
    # Test matmul_kernel
    print("\n--- matmul_kernel LLIR ---")
    try:
        llir = compile_kernel_to_stage(matmul_kernel, 'llir')
        if isinstance(llir, str) and 'define' in llir:
            print(f"✓ matmul_kernel LLIR generated ({len(llir)} bytes)")
            
            # Check for dot product
            if 'dot' in llir.lower() or 'fmul' in llir.lower():
                print("✓ Contains matrix operations")
        else:
            print(f"✗ Failed: {llir}")
    except Exception as e:
        print(f"✗ Exception: {e}")


def test_asm_generation():
    """Test RISCV assembly generation."""
    print("\n" + "="*70)
    print("Testing RISCV Assembly Generation")
    print("="*70)
    
    # Set codegen mode
    os.environ["TRITON_CUSTOM_CODEGEN_MODE"] = "asm"
    
    print("\n--- add_kernel ASM ---")
    try:
        asm = compile_kernel_to_stage(add_kernel, 'asm')
        if isinstance(asm, str):
            if '.text' in asm or 'add' in asm.lower():
                print(f"✓ add_kernel ASM generated ({len(asm)} bytes)")
                
                # Check for RISCV instructions
                riscv_instrs = ['lw', 'sw', 'add', 'addi', 'lui', 'jal', 'beq']
                found_instrs = [i for i in riscv_instrs if i in asm.lower()]
                if found_instrs:
                    print(f"✓ RISCV instructions found: {found_instrs[:5]}")
                
                if os.environ.get("TRITON_CUSTOM_DUMP_ASM"):
                    print("\n--- ASM Content ---")
                    print(asm[:2000] + "..." if len(asm) > 2000 else asm)
            else:
                print(f"⚠ ASM generated but may not be RISCV: {asm[:200]}")
        else:
            print(f"✗ Failed: {asm}")
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()


def test_pipeline_stages():
    """Test all pipeline stages in order."""
    print("\n" + "="*70)
    print("Testing Complete Pipeline Stages")
    print("="*70)
    
    stages = ['ttir', 'ttgir', 'llir']
    
    # Add asm stage if RISCV is available
    codegen_mode = os.environ.get("TRITON_CUSTOM_CODEGEN_MODE", "blob")
    if codegen_mode == "asm":
        stages.append('asm')
    
    for stage in stages:
        print(f"\n--- Stage: {stage} ---")
        try:
            result = compile_kernel_to_stage(add_kernel, stage)
            if result:
                if isinstance(result, str):
                    print(f"✓ {stage}: {len(result)} bytes")
                elif isinstance(result, bytes):
                    print(f"✓ {stage}: {len(result)} bytes (binary)")
                else:
                    print(f"✓ {stage}: {type(result)}")
            else:
                print(f"✗ {stage}: No output")
        except Exception as e:
            print(f"✗ {stage}: {e}")


def direct_pipeline_test():
    """Test pipeline functions directly without going through triton.compile."""
    print("\n" + "="*70)
    print("Direct Pipeline Function Test")
    print("="*70)
    
    try:
        from triton._C.libtriton import ir
        from third_party.custom.backend import pipeline
        
        # Create a minimal TTIR module
        print("\n--- Testing make_llir availability ---")
        print(f"✓ pipeline.make_llir: {pipeline.make_llir}")
        print(f"✓ pipeline.make_ttir: {pipeline.make_ttir}")
        print(f"✓ pipeline.make_ttgir: {pipeline.make_ttgir}")
        
        if hasattr(pipeline, 'make_asm'):
            print(f"✓ pipeline.make_asm: {pipeline.make_asm}")
        else:
            print("⚠ pipeline.make_asm: not found")
            
        if hasattr(pipeline, 'make_obj'):
            print(f"✓ pipeline.make_obj: {pipeline.make_obj}")
        else:
            print("⚠ pipeline.make_obj: not found")
            
    except ImportError as e:
        print(f"✗ Import error: {e}")


def print_environment():
    """Print relevant environment variables."""
    print("\n" + "="*70)
    print("Environment Configuration")
    print("="*70)
    
    env_vars = [
        "TRITON_CUSTOM_LLIR_MODE",
        "TRITON_CUSTOM_TTGIR_MODE",
        "TRITON_CUSTOM_CODEGEN_MODE",
        "TRITON_CUSTOM_LLVM_TRIPLE",
        "TRITON_CUSTOM_LLVM_CPU",
        "TRITON_CUSTOM_LLVM_FEATURES",
        "TRITON_CUSTOM_DUMP_ASM",
        "TRITON_CUSTOM_DUMP_LLIR",
        "TRITON_CUSTOM_ACTIVE",
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "(not set)")
        print(f"  {var} = {value}")


def main():
    parser = argparse.ArgumentParser(description="Custom SIMT Backend Pipeline Test")
    parser.add_argument("--stage", choices=['ttir', 'ttgir', 'llir', 'asm', 'obj', 'all'],
                        default='llir', help="Target compilation stage")
    parser.add_argument("--kernel", choices=['add', 'matmul', 'both'],
                        default='both', help="Kernel to test")
    parser.add_argument("--dump", action='store_true', help="Dump IR output")
    parser.add_argument("--all", action='store_true', help="Run all tests")
    parser.add_argument("--env", action='store_true', help="Print environment only")
    
    args = parser.parse_args()
    
    if args.dump:
        os.environ["TRITON_CUSTOM_DUMP_LLIR"] = "1"
        os.environ["TRITON_CUSTOM_DUMP_ASM"] = "1"
    
    print_environment()
    
    if args.env:
        return
    
    if args.all or args.stage == 'all':
        direct_pipeline_test()
        test_llir_generation()
        test_asm_generation()
        test_pipeline_stages()
    elif args.stage == 'llir':
        test_llir_generation()
    elif args.stage == 'asm':
        test_asm_generation()
    else:
        test_pipeline_stages()
    
    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)


if __name__ == "__main__":
    main()
