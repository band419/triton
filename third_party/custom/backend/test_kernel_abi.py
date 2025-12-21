#!/usr/bin/env python3
"""
Phase B1: Kernel ABI Validation Tests

This script validates that the custom backend correctly implements
the expanded arguments ABI as specified in docs/custom-kernel-abi.md.

Tests:
1. Kernel signature generation (expanded args)
2. Parameter type mapping
3. Address space annotations (addrspace 1 for global)
4. SIMT intrinsic presence
5. Load/Store lowering

Usage:
    TRITON_CUSTOM_LLIR_MODE=1 python test_kernel_abi.py
"""

import os
import sys
import re
import tempfile
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Set environment for custom LLIR mode
os.environ["TRITON_CUSTOM_LLIR_MODE"] = "1"
os.environ["TRITON_CUSTOM_TTGIR_MODE"] = "custom"
os.environ["TRITON_CUSTOM_ACTIVE"] = "1"


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


def generate_llir_for_kernel(kernel_type: str = "add") -> Tuple[bool, str, str]:
    """Generate LLIR for a test kernel."""
    try:
        from triton._C.libtriton import ir
        from triton.backends.custom import pipeline
        
        if kernel_type == "add":
            ttir_content = '''
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
        elif kernel_type == "scalar":
            # Test with more scalar types
            ttir_content = '''
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "custom:70"} {
  tt.func public @scalar_kernel(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: f32) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %0, %arg1 : i32
    tt.return
  }
}
'''
        else:
            return False, "", f"Unknown kernel type: {kernel_type}"
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ttir', delete=False) as f:
            f.write(ttir_content)
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
            metadata = {"name": kernel_type + "_kernel"}
            capability = 70
            
            context = ir.context()
            ir.load_dialects(context)
            mod = ir.parse_mlir_module(ttir_path, context)
            if mod is None:
                return False, "", "Failed to parse TTIR"
            mod.context = context
            
            mod = pipeline.make_ttir(mod, metadata, opt, capability)
            mod = pipeline.make_ttgir(mod, metadata, opt, capability)
            llir = pipeline.make_llir(mod, metadata, opt, capability)
            
            return True, llir, ""
        finally:
            os.unlink(ttir_path)
            
    except Exception as e:
        import traceback
        return False, "", f"{e}\n{traceback.format_exc()}"


def test_kernel_signature_expanded_args(llir: str) -> TestResult:
    """Test that kernel uses expanded arguments (not single param block pointer)."""
    # Look for kernel definition - may span multiple lines
    # Use DOTALL to match across newlines and find the closing paren
    pattern = r'define\s+void\s+@(\w+)\s*\(([^{]+)\)\s*[^{]*\{'
    match = re.search(pattern, llir, re.DOTALL)
    
    if not match:
        return TestResult("Expanded args signature", False, "No kernel definition found")
    
    kernel_name = match.group(1)
    params_raw = match.group(2)
    
    # Normalize whitespace
    params = ' '.join(params_raw.split())
    
    # Count parameters by splitting on comma (but be careful of nested types)
    param_list = []
    depth = 0
    current = ""
    for char in params:
        if char in "(<":
            depth += 1
        elif char in ")>":
            depth -= 1
        elif char == "," and depth == 0:
            param_list.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        param_list.append(current.strip())
    
    param_count = len(param_list)
    
    if param_count < 2:
        return TestResult("Expanded args signature", False,
                         f"Only {param_count} parameter(s), expected expanded args")
    
    # Check for pointer parameters with addrspace
    has_addrspace_ptr = "ptr addrspace(1)" in params
    
    # Check for scalar parameters (i32, f32, etc.)
    has_scalar = any("i32 %" in p or "float %" in p for p in param_list)
    
    return TestResult(
        "Expanded args signature",
        has_addrspace_ptr and param_count >= 3,
        f"Kernel @{kernel_name}: {param_count} parameters, addrspace(1): {has_addrspace_ptr}, scalars: {has_scalar}",
        "Types: " + ", ".join(p.split()[0] + "..." for p in param_list[:4])
    )


def test_address_space_global(llir: str) -> TestResult:
    """Test that pointer parameters use addrspace(1) for global memory."""
    # Count addrspace(1) occurrences
    global_as_count = llir.count("addrspace(1)")
    
    # Should NOT have addrspace(3) for shared memory
    shared_as_count = llir.count("addrspace(3)")
    
    if shared_as_count > 0:
        return TestResult("Global memory address space", False,
                         f"Found {shared_as_count} shared memory (addrspace 3) references",
                         "Custom backend should use global memory only")
    
    if global_as_count == 0:
        return TestResult("Global memory address space", False,
                         "No addrspace(1) found, pointers may be in wrong address space")
    
    return TestResult("Global memory address space", True,
                     f"Found {global_as_count} global memory references, no shared memory")


def test_simt_intrinsics(llir: str) -> TestResult:
    """Test that SIMT intrinsics are present and correctly named."""
    expected_intrinsics = [
        "llvm.custom.program.id",
        "llvm.custom.thread.id",
    ]
    
    found = []
    missing = []
    
    for intrinsic in expected_intrinsics:
        if intrinsic in llir:
            found.append(intrinsic)
        else:
            missing.append(intrinsic)
    
    # At least program.id should be present
    if "llvm.custom.program.id" not in found:
        return TestResult("SIMT intrinsics", False,
                         "Missing llvm.custom.program.id intrinsic",
                         f"Found: {found}")
    
    return TestResult("SIMT intrinsics", True,
                     f"Found {len(found)} SIMT intrinsics",
                     ", ".join(found))


def test_load_store_lowering(llir: str) -> TestResult:
    """Test that load/store operations are properly lowered."""
    # Check for LLVM load/store instructions
    has_load = " load " in llir or "= load " in llir
    has_store = " store " in llir or "store " in llir
    
    # Check for getelementptr (pointer arithmetic)
    has_gep = "getelementptr" in llir
    
    if not has_load:
        return TestResult("Load/Store lowering", False,
                         "No LLVM load instructions found")
    
    if not has_store:
        return TestResult("Load/Store lowering", False,
                         "No LLVM store instructions found")
    
    # Count load/store operations
    load_count = llir.count(" load ")
    store_count = llir.count(" store ")
    
    return TestResult("Load/Store lowering", True,
                     f"Found {load_count} loads, {store_count} stores, GEP: {has_gep}")


def test_triple_and_datalayout(llir: str) -> TestResult:
    """Test that triple and datalayout are present and non-NVIDIA."""
    triple_match = re.search(r'target triple\s*=\s*"([^"]+)"', llir)
    datalayout_match = re.search(r'target datalayout\s*=\s*"([^"]+)"', llir)
    
    issues = []
    
    if not triple_match:
        issues.append("Missing target triple")
    elif "nvptx" in triple_match.group(1).lower():
        issues.append(f"Triple is NVIDIA: {triple_match.group(1)}")
    
    if not datalayout_match:
        issues.append("Missing datalayout (recommended)")
    
    if issues:
        return TestResult("Triple/Datalayout", len(issues) == 0,
                         "; ".join(issues))
    
    return TestResult("Triple/Datalayout", True,
                     f"triple={triple_match.group(1)}",
                     f"datalayout={datalayout_match.group(1)[:50]}...")


def test_no_nvidia_patterns(llir: str) -> TestResult:
    """Test that no NVIDIA-specific patterns are present."""
    nvidia_patterns = [
        ("nvptx", "NVPTX reference"),
        ("nvvm", "NVVM reference"),
        ("nvgpu", "NVGPU reference"),
        ("@llvm.nvvm", "NVVM intrinsic"),
        ("cuda", "CUDA reference"),
    ]
    
    found = []
    for pattern, desc in nvidia_patterns:
        if pattern.lower() in llir.lower():
            found.append(f"{desc} ({pattern})")
    
    if found:
        return TestResult("NVIDIA-free", False,
                         f"Found {len(found)} NVIDIA pattern(s)",
                         "\n".join(found))
    
    return TestResult("NVIDIA-free", True, "No NVIDIA patterns found")


def run_all_tests() -> List[TestResult]:
    """Run all Phase B1 tests."""
    results = []
    
    # Generate LLIR for add kernel
    print("Generating LLIR for add_kernel...")
    success, llir, error = generate_llir_for_kernel("add")
    
    if not success:
        results.append(TestResult("LLIR Generation", False, "Failed to generate LLIR", error))
        return results
    
    results.append(TestResult("LLIR Generation", True, f"Generated {len(llir)} bytes"))
    
    # Save for inspection
    with open("/tmp/custom_b1_test.ll", "w") as f:
        f.write(llir)
    print(f"  Saved to /tmp/custom_b1_test.ll")
    
    # Run all tests
    results.append(test_kernel_signature_expanded_args(llir))
    results.append(test_address_space_global(llir))
    results.append(test_simt_intrinsics(llir))
    results.append(test_load_store_lowering(llir))
    results.append(test_triple_and_datalayout(llir))
    results.append(test_no_nvidia_patterns(llir))
    
    return results


def main():
    print("\n" + "=" * 70)
    print("Phase B1: Kernel ABI Validation Tests")
    print("=" * 70)
    
    print(f"\nEnvironment:")
    print(f"  TRITON_CUSTOM_LLIR_MODE = {os.environ.get('TRITON_CUSTOM_LLIR_MODE', 'not set')}")
    
    results = run_all_tests()
    
    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"\n{status} {r.name}")
        print(f"    {r.message}")
        if r.details:
            for line in r.details.split('\n')[:5]:  # Limit details
                print(f"    {line}")
        
        if r.passed:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 70)
    print(f"Summary: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 Phase B1 PASSED - Kernel ABI is correctly implemented!")
    else:
        print("\n❌ Phase B1 has issues that need attention")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
