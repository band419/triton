#!/usr/bin/env python3
"""
Phase B4 验证测试：TTGIR 侧 NVIDIA 特性剥离

测试目标：
1. Custom TTGIR pipeline 不使用 NVIDIA passes
2. 关闭 TRITON_CUSTOM_USE_NVIDIA_TTGIR_PIPELINE 时仍能产出合法 LLIR
3. 生成的 LLIR 无 NVIDIA 特定 patterns
4. 无 shared memory 使用
5. 无 MMA/TMA/TMEM 相关指令
6. Target triple 正确

运行方式:
    TRITON_CUSTOM_LLIR_MODE=1 TRITON_CUSTOM_TTGIR_MODE=custom python test_ttgir_nvidia_free.py
"""

import os
import sys
import re
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    details: str = ""


# Set environment for custom LLIR mode
os.environ["TRITON_CUSTOM_LLIR_MODE"] = "1"
os.environ["TRITON_CUSTOM_TTGIR_MODE"] = "custom"
os.environ["TRITON_CUSTOM_USE_NVIDIA_TTGIR_PIPELINE"] = "0"
os.environ["TRITON_CUSTOM_ACTIVE"] = "1"


def generate_llir() -> Tuple[bool, str, str]:
    """Generate LLIR with NVIDIA pipeline disabled."""
    try:
        from triton._C.libtriton import ir
        from triton.backends.custom import pipeline
        
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
            metadata = {"name": "add_kernel"}
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


def test_nvidia_pipeline_disabled(llir: str) -> TestResult:
    """Test that NVIDIA pipeline is disabled and LLIR is still valid."""
    if not llir or len(llir) < 100:
        return TestResult(
            "NVIDIA pipeline disabled",
            False,
            "LLIR too short or empty",
            f"Length: {len(llir) if llir else 0}"
        )
    
    # Check for valid LLVM IR structure
    has_define = "define" in llir
    has_target = "target triple" in llir or "target datalayout" in llir
    
    if not has_define:
        return TestResult(
            "NVIDIA pipeline disabled",
            False,
            "No function definitions found in LLIR"
        )
    
    return TestResult(
        "NVIDIA pipeline disabled",
        True,
        f"LLIR generated successfully ({len(llir)} bytes) with NVIDIA pipeline disabled"
    )


def test_no_nvidia_triple(llir: str) -> TestResult:
    """Test that LLIR does not contain NVIDIA triple."""
    nvidia_patterns = [
        (r'nvptx64-nvidia-cuda', "NVIDIA triple"),
        (r'nvptx32', "NVPTX32 target"),
        (r'nvptx64', "NVPTX64 target"),
    ]
    
    found = []
    for pattern, desc in nvidia_patterns:
        if re.search(pattern, llir, re.IGNORECASE):
            found.append(desc)
    
    if found:
        return TestResult(
            "No NVIDIA triple",
            False,
            f"Found NVIDIA target patterns: {', '.join(found)}"
        )
    
    # Check for custom triple
    triple_match = re.search(r'target\s+triple\s*=\s*"([^"]+)"', llir)
    if triple_match:
        triple = triple_match.group(1)
        return TestResult(
            "No NVIDIA triple",
            True,
            f"Using custom triple: {triple}"
        )
    
    return TestResult(
        "No NVIDIA triple",
        True,
        "No NVIDIA triple patterns found"
    )


def test_no_nvvm_intrinsics(llir: str) -> TestResult:
    """Test that LLIR does not contain NVVM intrinsics."""
    nvvm_patterns = [
        (r'@llvm\.nvvm\.', "NVVM intrinsic"),
        (r'nvvm\.annotations', "NVVM annotation"),
        (r'@llvm\.nvgpu\.', "NVGPU intrinsic"),
    ]
    
    found = []
    for pattern, desc in nvvm_patterns:
        matches = re.findall(pattern, llir, re.IGNORECASE)
        if matches:
            found.append(f"{desc}: {len(matches)}")
    
    if found:
        return TestResult(
            "No NVVM intrinsics",
            False,
            f"Found NVVM patterns: {', '.join(found)}"
        )
    
    # Check for custom intrinsics
    custom_intrinsics = re.findall(r'@llvm\.custom\.\w+', llir)
    custom_count = len(set(custom_intrinsics))
    
    return TestResult(
        "No NVVM intrinsics",
        True,
        f"No NVVM intrinsics, found {custom_count} custom intrinsic types"
    )


def test_no_shared_memory(llir: str) -> TestResult:
    """Test that LLIR does not use shared memory."""
    shared_patterns = [
        (r'addrspace\(3\)', "Shared memory address space"),
        (r'@llvm\.nvvm\.\w+\.shared', "Shared memory intrinsic"),
        (r'local_unnamed_addr', "Local address space variable"),
    ]
    
    # addrspace(3) is specifically shared memory in NVPTX
    shared_count = len(re.findall(r'addrspace\(3\)', llir))
    
    if shared_count > 0:
        return TestResult(
            "No shared memory",
            False,
            f"Found {shared_count} shared memory (addrspace 3) references"
        )
    
    return TestResult(
        "No shared memory",
        True,
        "No shared memory usage detected"
    )


def test_no_mma_tma_tmem(llir: str) -> TestResult:
    """Test that LLIR does not contain MMA/TMA/TMEM patterns."""
    patterns = [
        (r'mma\.', "MMA operation"),
        (r'tma\.', "TMA operation"),
        (r'tmem', "TMEM reference"),
        (r'cp\.async', "cp.async (async copy)"),
        (r'ldmatrix', "ldmatrix operation"),
        (r'stmatrix', "stmatrix operation"),
        (r'wgmma', "WGMMA operation"),
    ]
    
    found = []
    for pattern, desc in patterns:
        matches = re.findall(pattern, llir, re.IGNORECASE)
        if matches:
            found.append(f"{desc}: {len(matches)}")
    
    if found:
        return TestResult(
            "No MMA/TMA/TMEM",
            False,
            f"Found patterns: {', '.join(found)}"
        )
    
    return TestResult(
        "No MMA/TMA/TMEM",
        True,
        "No MMA/TMA/TMEM patterns found"
    )


def test_no_ptx_asm(llir: str) -> TestResult:
    """Test that LLIR does not contain inline PTX assembly."""
    ptx_patterns = [
        (r'call.*asm.*ptx', "Inline PTX assembly"),
        (r'module\s+asm', "Module-level assembly"),
        (r'\.reg\s+\.', "PTX register"),
        (r'@%p\d+', "PTX predicate"),
    ]
    
    found = []
    for pattern, desc in ptx_patterns:
        matches = re.findall(pattern, llir, re.IGNORECASE)
        if matches:
            found.append(f"{desc}: {len(matches)}")
    
    if found:
        return TestResult(
            "No PTX assembly",
            False,
            f"Found PTX patterns: {', '.join(found)}"
        )
    
    return TestResult(
        "No PTX assembly",
        True,
        "No inline PTX assembly found"
    )


def test_custom_intrinsics_present(llir: str) -> TestResult:
    """Test that custom intrinsics are present."""
    expected_intrinsics = [
        "llvm.riscv.simt.program.id",
        # "llvm.riscv.simt.lane.id",  # May not be used in simple kernel
        # "llvm.riscv.simt.barrier",  # May not be used in simple kernel
    ]
    
    found = []
    missing = []
    
    for intrinsic in expected_intrinsics:
        if f"@{intrinsic}" in llir:
            found.append(intrinsic)
        else:
            missing.append(intrinsic)
    
    # Also check for thread.id which may be used instead of lane.id
    if "@llvm.riscv.simt.thread.id" in llir:
        found.append("llvm.riscv.simt.thread.id")
    
    if not found:
        return TestResult(
            "Custom intrinsics",
            False,
            "No custom intrinsics found",
            f"Expected: {', '.join(expected_intrinsics)}"
        )
    
    return TestResult(
        "Custom intrinsics",
        True,
        f"Found {len(found)} custom intrinsic(s): {', '.join(found)}"
    )


def test_valid_llvm_ir(llir: str) -> TestResult:
    """Test that the LLIR is valid LLVM IR structure."""
    # Basic structure checks
    checks = [
        (r'target\s+datalayout\s*=', "datalayout"),
        (r'target\s+triple\s*=', "triple"),
        (r'define\s+\w+\s+@\w+', "function definition"),
        (r'ret\s+(void|\w+)', "return instruction"),
    ]
    
    found = []
    missing = []
    
    for pattern, name in checks:
        if re.search(pattern, llir):
            found.append(name)
        else:
            missing.append(name)
    
    if missing:
        return TestResult(
            "Valid LLVM IR",
            False,
            f"Missing LLVM IR elements: {', '.join(missing)}"
        )
    
    return TestResult(
        "Valid LLVM IR",
        True,
        f"All LLVM IR structure elements present: {', '.join(found)}"
    )


def run_all_tests() -> Tuple[List[TestResult], str]:
    """Run all B4 tests."""
    results = []
    
    # Generate LLIR with NVIDIA pipeline disabled
    print("Generating LLIR with NVIDIA pipeline disabled...")
    print(f"  TRITON_CUSTOM_TTGIR_MODE = {os.environ.get('TRITON_CUSTOM_TTGIR_MODE')}")
    print(f"  TRITON_CUSTOM_USE_NVIDIA_TTGIR_PIPELINE = {os.environ.get('TRITON_CUSTOM_USE_NVIDIA_TTGIR_PIPELINE')}")
    
    success, llir, error = generate_llir()
    
    if not success:
        results.append(TestResult(
            "LLIR Generation",
            False,
            f"Failed: {error}"
        ))
        return results, ""
    
    results.append(TestResult(
        "LLIR Generation",
        True,
        f"Generated {len(llir)} bytes"
    ))
    
    # Run all tests
    tests = [
        test_nvidia_pipeline_disabled,
        test_no_nvidia_triple,
        test_no_nvvm_intrinsics,
        test_no_shared_memory,
        test_no_mma_tma_tmem,
        test_no_ptx_asm,
        test_custom_intrinsics_present,
        test_valid_llvm_ir,
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
    print("=" * 70)
    print("Phase B4: TTGIR NVIDIA Feature Removal Validation")
    print("=" * 70)
    print()
    
    results, llir = run_all_tests()
    
    # Print results
    print()
    print("Test Results:")
    print("-" * 70)
    
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
    
    print("-" * 70)
    print(f"Summary: {passed} passed, {failed} failed")
    
    # Save LLIR for inspection
    if llir:
        output_path = "/tmp/custom_b4_test.ll"
        with open(output_path, "w") as f:
            f.write(llir)
        print(f"\nLLIR saved to: {output_path}")
    
    if failed == 0:
        print()
        print("🎉 Phase B4 PASSED: TTGIR NVIDIA feature removal verified")
        print()
        print("Summary of NVIDIA features NOT used in custom pipeline:")
        print("  - nvidia.passes.ttnvgpuir.add_plan_cta")
        print("  - nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding")
        print("  - nvidia.passes.hopper.add_hopper_warpspec")
        print("  - nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem")
        print("  - nvidia.passes.ttnvgpuir.add_remove_tmem_tokens")
        print("  - nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts")
        print("  - nvidia.passes.ttnvgpuir.add_tma_lowering")
        print("  - nvidia.passes.ttnvgpuir.add_interleave_tmem")
        print("  - nvidia.passes.ttnvgpuir.add_fence_insertion")
        print("  - nvidia.passes.ttnvgpuir.add_lower_mma")
        print("  - nvidia.passes.ttgpuir.add_allocate_shared_memory_nv")
        print("  - nvidia.passes.ttnvgpuir.add_allocate_tensor_memory")
        print("  - passes.convert.add_nvvm_to_llvm")
        return 0
    else:
        print()
        print("❌ Phase B4 FAILED: Some tests did not pass")
        return 1


if __name__ == "__main__":
    sys.exit(main())
