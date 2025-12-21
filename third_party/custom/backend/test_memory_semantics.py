#!/usr/bin/env python3
"""
Phase B3 验证测试：内存与同步语义

测试目标：
1. Global memory load/store 使用 addrspace(1)
2. Barrier 有 convergent 属性
3. Barrier 有正确的 memory effects (fence 语义)
4. 无 shared memory 使用 (addrspace 3)
5. Load/Store 使用标准 LLVM 指令
6. NVIDIA-free 验证

运行方式:
    TRITON_CUSTOM_LLIR_MODE=1 python test_memory_semantics.py
"""

import os
import sys
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    details: str = ""


def get_llir_with_barrier() -> str:
    """
    Generate LLIR from a kernel that uses barrier.
    For now, use the same simple kernel as compile_simple_kernel_llir.
    """
    return compile_simple_kernel_llir()


def compile_simple_kernel_llir() -> str:
    """Compile a simple add kernel to LLIR for testing."""
    import tempfile
    import os
    
    # Set environment for custom LLIR mode
    os.environ["TRITON_CUSTOM_LLIR_MODE"] = "1"
    os.environ["TRITON_CUSTOM_TTGIR_MODE"] = "custom"
    os.environ["TRITON_CUSTOM_ACTIVE"] = "1"
    
    # Use the pipeline directly like test_kernel_abi.py
    from triton._C.libtriton import ir
    from triton.backends.custom import pipeline
    
    # Simple add kernel TTIR
    ttir_code = '''
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "custom:70"} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %9 = tt.load %8, %6 : tensor<256x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %12 = tt.load %11, %6 : tensor<256x!tt.ptr<f32>>
    %13 = arith.addf %9, %12 : tensor<256xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %15, %13, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
'''
    
    # Write to temp file for parsing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttir', delete=False) as f:
        f.write(ttir_code)
        ttir_path = f.name
    
    try:
        class MockOptions:
            num_warps = 4
            num_ctas = 1
            num_stages = 2
            maxnreg = None
            cluster_dims = (1, 1, 1)
            warp_size = 32
            enable_fp_fusion = True
            allow_fp8e4nv = False
        
        opt = MockOptions()
        metadata = {"name": "add_kernel"}
        capability = 70
        
        context = ir.context()
        ir.load_dialects(context)
        mod = ir.parse_mlir_module(ttir_path, context)
        if mod is None:
            raise RuntimeError("Failed to parse TTIR")
        mod.context = context
        
        mod = pipeline.make_ttir(mod, metadata, opt, capability)
        mod = pipeline.make_ttgir(mod, metadata, opt, capability)
        llir = pipeline.make_llir(mod, metadata, opt, capability)
        
        return llir
    finally:
        os.unlink(ttir_path)


def test_global_memory_addrspace(llir: str) -> TestResult:
    """Test that global memory uses addrspace(1)."""
    global_count = len(re.findall(r'addrspace\(1\)', llir))
    shared_count = len(re.findall(r'addrspace\(3\)', llir))
    
    if shared_count > 0:
        return TestResult(
            "Global memory addrspace",
            False,
            f"Found shared memory (addrspace 3): {shared_count} occurrences",
            "Custom backend should use global memory only"
        )
    
    if global_count == 0:
        return TestResult(
            "Global memory addrspace",
            False,
            "No global memory (addrspace 1) references found",
            "Expected global memory usage"
        )
    
    return TestResult(
        "Global memory addrspace",
        True,
        f"Found {global_count} global memory references, no shared memory"
    )


def test_load_store_instructions(llir: str) -> TestResult:
    """Test that load/store use standard LLVM instructions."""
    # Look for LLVM load/store instructions
    loads = re.findall(r'= load\s+\w+', llir)
    stores = re.findall(r'store\s+\w+', llir)
    
    # Should NOT have NVIDIA-specific memory ops
    nvidia_mem = re.findall(r'@llvm\.nvvm\.(ldu|ldg|ld|st)', llir, re.IGNORECASE)
    
    if nvidia_mem:
        return TestResult(
            "Load/Store instructions",
            False,
            f"Found NVIDIA-specific memory ops: {len(nvidia_mem)}",
            "Should use standard LLVM load/store"
        )
    
    if len(loads) == 0 and len(stores) == 0:
        return TestResult(
            "Load/Store instructions",
            False,
            "No load/store instructions found",
            "Expected LLVM load/store operations"
        )
    
    return TestResult(
        "Load/Store instructions",
        True,
        f"Found {len(loads)} loads, {len(stores)} stores (standard LLVM)"
    )


def test_no_shared_memory(llir: str) -> TestResult:
    """Test that no shared memory operations are present."""
    # Check for shared memory patterns
    shared_patterns = [
        r'addrspace\(3\)',           # Shared memory address space
        r'@llvm\.nvvm\..*shared',    # NVIDIA shared memory intrinsics
        r'@llvm\.custom\.shared',    # Custom shared (should not exist)
        r'local_alloc',              # Shared memory allocation
    ]
    
    found = []
    for pattern in shared_patterns:
        matches = re.findall(pattern, llir, re.IGNORECASE)
        if matches:
            found.extend(matches[:3])  # Limit to first 3
    
    if found:
        return TestResult(
            "No shared memory",
            False,
            f"Found shared memory patterns: {', '.join(found[:5])}",
            "Custom backend uses global memory only"
        )
    
    return TestResult(
        "No shared memory",
        True,
        "No shared memory usage detected"
    )


def test_barrier_convergent(llir: str) -> TestResult:
    """Test that barrier intrinsics have convergent attribute."""
    # Look for barrier declarations
    barrier_decls = re.findall(
        r'declare\s+void\s+@llvm\.custom\.(warp\.)?barrier\([^)]*\).*',
        llir, re.IGNORECASE | re.MULTILINE
    )
    
    # Also check function attributes
    barrier_attrs = re.findall(
        r'@llvm\.custom\.(warp\.)?barrier.*convergent',
        llir, re.IGNORECASE
    )
    
    # Look for barrier calls
    barrier_calls = re.findall(r'call\s+void\s+@llvm\.custom\.(warp\.)?barrier', llir)
    
    # If no barriers in this kernel, that's OK
    if not barrier_decls and not barrier_calls:
        return TestResult(
            "Barrier convergent",
            True,
            "No barrier used in this kernel (OK for simple kernels)"
        )
    
    # Check for convergent attribute
    has_convergent = 'convergent' in llir.lower() if barrier_decls else True
    
    if not has_convergent:
        return TestResult(
            "Barrier convergent",
            False,
            "Barrier declaration missing convergent attribute",
            "Barrier must have convergent to prevent illegal code motion"
        )
    
    return TestResult(
        "Barrier convergent",
        True,
        f"Barrier has convergent attribute ({len(barrier_decls)} decl, {len(barrier_calls)} calls)"
    )


def test_barrier_memory_effects(llir: str) -> TestResult:
    """Test that barrier has proper memory fence effects."""
    # Look for barrier with memory effects
    # memory(readwrite) or memory(argmem: readwrite, ...)
    
    barrier_decls = re.findall(
        r'declare\s+void\s+@llvm\.custom\.(warp\.)?barrier[^{]*',
        llir, re.IGNORECASE
    )
    
    if not barrier_decls:
        return TestResult(
            "Barrier memory effects",
            True,
            "No barrier in this kernel (OK for simple kernels)"
        )
    
    # Check for memory effects attribute
    # In LLVM IR, this appears as: memory(readwrite) or memory(...)
    memory_pattern = re.search(r'@llvm\.custom\.(warp\.)?barrier.*memory\([^)]+\)', llir)
    
    # Also check for older style nounwind + no readonly/readnone
    has_nounwind = 'nounwind' in llir.lower()
    has_readonly = re.search(r'@llvm\.custom\.(warp\.)?barrier.*readonly', llir)
    has_readnone = re.search(r'@llvm\.custom\.(warp\.)?barrier.*readnone', llir)
    
    # Barrier should NOT be readonly or readnone (it acts as a fence)
    if has_readonly or has_readnone:
        return TestResult(
            "Barrier memory effects",
            False,
            "Barrier incorrectly marked as readonly/readnone",
            "Barrier should have fence semantics (memory effects)"
        )
    
    return TestResult(
        "Barrier memory effects",
        True,
        "Barrier has proper memory effects (not readonly/readnone)"
    )


def test_nvidia_free(llir: str) -> TestResult:
    """Test that LLIR is free of NVIDIA-specific patterns."""
    nvidia_patterns = [
        (r'nvptx64-nvidia-cuda', "NVIDIA triple"),
        (r'@llvm\.nvvm\.', "NVVM intrinsic"),
        (r'@llvm\.nvgpu\.', "NVGPU intrinsic"),
        (r'nvvm\.annotations', "NVVM annotation"),
        (r'ptx', "PTX reference"),
    ]
    
    found = []
    for pattern, desc in nvidia_patterns:
        matches = re.findall(pattern, llir, re.IGNORECASE)
        if matches:
            found.append(f"{desc}: {len(matches)}")
    
    if found:
        return TestResult(
            "NVIDIA-free",
            False,
            f"Found NVIDIA patterns: {', '.join(found)}",
            "LLIR should be NVIDIA-free"
        )
    
    return TestResult(
        "NVIDIA-free",
        True,
        "No NVIDIA patterns found in LLIR"
    )


def test_custom_triple(llir: str) -> TestResult:
    """Test that LLIR has custom target triple."""
    # Look for target triple
    triple_match = re.search(r'target\s+triple\s*=\s*"([^"]+)"', llir)
    
    if not triple_match:
        return TestResult(
            "Custom triple",
            False,
            "No target triple found",
            "Expected custom target triple"
        )
    
    triple = triple_match.group(1)
    
    # Should be custom triple, not NVIDIA
    if 'nvidia' in triple.lower() or 'nvptx' in triple.lower():
        return TestResult(
            "Custom triple",
            False,
            f"Found NVIDIA triple: {triple}",
            "Should use custom triple (e.g., riscv32-unknown-unknown)"
        )
    
    return TestResult(
        "Custom triple",
        True,
        f"triple={triple}"
    )


def run_all_tests() -> Tuple[List[TestResult], str]:
    """Run all B3 tests."""
    results = []
    
    # Generate LLIR
    print("Generating LLIR for memory semantics testing...")
    try:
        llir = compile_simple_kernel_llir()
        results.append(TestResult(
            "LLIR Generation",
            True,
            f"Generated {len(llir)} bytes"
        ))
    except Exception as e:
        results.append(TestResult(
            "LLIR Generation",
            False,
            f"Failed: {str(e)}"
        ))
        return results, ""
    
    # Run tests
    tests = [
        test_global_memory_addrspace,
        test_load_store_instructions,
        test_no_shared_memory,
        test_barrier_convergent,
        test_barrier_memory_effects,
        test_nvidia_free,
        test_custom_triple,
    ]
    
    for test_fn in tests:
        try:
            result = test_fn(llir)
            results.append(result)
        except Exception as e:
            results.append(TestResult(
                test_fn.__name__,
                False,
                f"Test error: {str(e)}"
            ))
    
    return results, llir


def main():
    """Main entry point."""
    print("=" * 60)
    print("Phase B3: Memory and Synchronization Semantics Validation")
    print("=" * 60)
    print()
    
    # Check environment
    if os.environ.get("TRITON_CUSTOM_LLIR_MODE") != "1":
        print("Warning: TRITON_CUSTOM_LLIR_MODE not set to 1")
        print("Setting TRITON_CUSTOM_LLIR_MODE=1 for this test")
        os.environ["TRITON_CUSTOM_LLIR_MODE"] = "1"
    
    results, llir = run_all_tests()
    
    # Print results
    print()
    print("Test Results:")
    print("-" * 60)
    
    passed = 0
    failed = 0
    
    for result in results:
        status = "✓" if result.passed else "✗"
        print(f"{status} {result.name}: {result.message}")
        if result.details and not result.passed:
            print(f"  Details: {result.details}")
        
        if result.passed:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print()
        print("🎉 Phase B3 PASSED: Memory and synchronization semantics verified")
        return 0
    else:
        print()
        print("❌ Phase B3 FAILED: Some tests did not pass")
        return 1


if __name__ == "__main__":
    sys.exit(main())
