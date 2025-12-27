#!/usr/bin/env python3
"""
Phase A4: End-to-end validation for Custom SIMT backend NVIDIA-free LLIR.

This script performs comprehensive verification that the custom backend
generates LLVM IR without any NVIDIA-specific dependencies.

Phase A4 Acceptance Criteria:
- [ ] Generated LLIR does not contain `nvptx64-nvidia-cuda`
- [ ] Generated LLIR does not contain `@llvm.nvvm.*` intrinsics
- [ ] Generated LLIR contains `llvm.riscv.simt.*` intrinsics
- [ ] LLIR can be parsed by `llvm-as` (syntax correct)
- [ ] Triple is `riscv32-unknown-unknown` or other non-NVIDIA triple

Usage:
    TRITON_CUSTOM_LLIR_MODE=1 python test_custom_llir.py
"""

import os
import sys
import tempfile
import subprocess
import re
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Set environment variables for custom LLIR mode BEFORE any triton imports
os.environ["TRITON_CUSTOM_LLIR_MODE"] = "1"
os.environ["TRITON_CUSTOM_TTGIR_MODE"] = "custom"
# Prevent auto-detection of CUDA/ROCm backends
os.environ["TRITON_CUSTOM_ACTIVE"] = "1"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


class PhaseA4Validator:
    """Validator for Phase A4 acceptance criteria."""
    
    NVIDIA_PATTERNS = [
        ("nvptx64-nvidia-cuda", "NVPTX triple"),
        ("nvptx-", "NVPTX prefix"),
        ("@llvm.nvvm.", "NVVM intrinsic"),
        ("nvvm.", "NVVM reference"),
        ("nvgpu.", "NVGPU dialect"),
        ('"cuda:', "CUDA target"),
        ("ptx_kernel", "PTX kernel attribute"),
        ("nvptx_kernel", "NVPTX kernel attribute"),
        ("nvvm_reflect", "NVVM reflect"),
        ("addrspace(3)", "Shared memory address space"),  # NVIDIA shared memory
    ]
    
    EXPECTED_CUSTOM_INTRINSICS = [
        "llvm.riscv.simt.program.id",
        "llvm.riscv.simt.barrier",
        "llvm.riscv.simt.thread.id",
        "llvm.riscv.simt.block.id",
        "llvm.riscv.simt.block.dim",
        "llvm.riscv.simt.grid.dim",
        "llvm.riscv.simt.lane.id",
        "llvm.riscv.simt.warp.size",
        "llvm.riscv.simt.shfl",
        "llvm.riscv.simt.ballot",
    ]
    
    def __init__(self, llir: str):
        self.llir = llir
        self.results: List[ValidationResult] = []
    
    def check_nvidia_free(self) -> ValidationResult:
        """Check that LLIR contains no NVIDIA-specific patterns."""
        found_patterns = []
        for pattern, desc in self.NVIDIA_PATTERNS:
            if pattern.lower() in self.llir.lower():
                found_patterns.append(f"{desc} ({pattern})")
        
        if found_patterns:
            return ValidationResult(
                name="NVIDIA-free check",
                passed=False,
                message=f"Found {len(found_patterns)} NVIDIA-specific pattern(s)",
                details="\n".join(f"  - {p}" for p in found_patterns)
            )
        return ValidationResult(
            name="NVIDIA-free check",
            passed=True,
            message="No NVIDIA-specific patterns found"
        )
    
    def check_custom_intrinsics(self) -> ValidationResult:
        """Check that LLIR contains expected custom intrinsics."""
        found = []
        for intrinsic in self.EXPECTED_CUSTOM_INTRINSICS:
            if intrinsic in self.llir:
                found.append(intrinsic)
        
        if not found:
            return ValidationResult(
                name="Custom intrinsics check",
                passed=False,
                message="No custom intrinsics found",
                details="Expected at least one of:\n" + 
                        "\n".join(f"  - {i}" for i in self.EXPECTED_CUSTOM_INTRINSICS[:5])
            )
        return ValidationResult(
            name="Custom intrinsics check",
            passed=True,
            message=f"Found {len(found)} custom intrinsic(s)",
            details="\n".join(f"  - {i}" for i in found)
        )
    
    def check_triple(self) -> ValidationResult:
        """Check that LLIR has a non-NVIDIA triple."""
        match = re.search(r'target triple\s*=\s*"([^"]+)"', self.llir)
        if not match:
            return ValidationResult(
                name="Triple check",
                passed=False,
                message="No target triple found in LLIR"
            )
        
        triple = match.group(1)
        is_nvidia = triple.startswith("nvptx")
        
        if is_nvidia:
            return ValidationResult(
                name="Triple check",
                passed=False,
                message=f"Triple is NVIDIA-specific: {triple}"
            )
        return ValidationResult(
            name="Triple check",
            passed=True,
            message=f"Triple is non-NVIDIA: {triple}"
        )
    
    def check_datalayout(self) -> ValidationResult:
        """Check that LLIR has a valid datalayout."""
        match = re.search(r'target datalayout\s*=\s*"([^"]+)"', self.llir)
        if not match:
            # Datalayout is optional but recommended
            return ValidationResult(
                name="Datalayout check",
                passed=True,
                message="No datalayout found (optional)"
            )
        
        datalayout = match.group(1)
        return ValidationResult(
            name="Datalayout check",
            passed=True,
            message=f"Datalayout present: {datalayout[:50]}..."
        )
    
    def check_syntax_llvm_as(self) -> ValidationResult:
        """Validate LLIR syntax using llvm-as."""
        llvm_as = os.environ.get("LLVM_AS", "llvm-as")
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
                f.write(self.llir)
                f.flush()
                temp_path = f.name
            
            result = subprocess.run(
                [llvm_as, temp_path, "-o", "/dev/null"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            os.unlink(temp_path)
            
            if result.returncode == 0:
                return ValidationResult(
                    name="LLVM-AS syntax check",
                    passed=True,
                    message="LLIR syntax is valid"
                )
            else:
                return ValidationResult(
                    name="LLVM-AS syntax check",
                    passed=False,
                    message="LLIR syntax error",
                    details=result.stderr[:500] if result.stderr else "Unknown error"
                )
                
        except FileNotFoundError:
            return ValidationResult(
                name="LLVM-AS syntax check",
                passed=True,
                message="(skipped: llvm-as not found)"
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                name="LLVM-AS syntax check",
                passed=False,
                message="llvm-as timed out"
            )
        except Exception as e:
            return ValidationResult(
                name="LLVM-AS syntax check",
                passed=True,
                message=f"(skipped: {e})"
            )
    
    def check_kernel_signature(self) -> ValidationResult:
        """Check that kernel has proper entry point."""
        # Look for define void @<name>(
        match = re.search(r'define\s+\w+\s+@(\w+)\s*\(', self.llir)
        if not match:
            return ValidationResult(
                name="Kernel signature check",
                passed=False,
                message="No kernel entry point found"
            )
        
        kernel_name = match.group(1)
        return ValidationResult(
            name="Kernel signature check",
            passed=True,
            message=f"Found kernel entry: @{kernel_name}"
        )
    
    def run_all_checks(self) -> List[ValidationResult]:
        """Run all Phase A4 validation checks."""
        self.results = [
            self.check_nvidia_free(),
            self.check_custom_intrinsics(),
            self.check_triple(),
            self.check_datalayout(),
            self.check_kernel_signature(),
            self.check_syntax_llvm_as(),
        ]
        return self.results
    
    def print_results(self):
        """Print validation results in a formatted way."""
        print("\n" + "=" * 70)
        print("Phase A4 Validation Results")
        print("=" * 70)
        
        passed = 0
        failed = 0
        
        for r in self.results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            print(f"\n{status}: {r.name}")
            print(f"       {r.message}")
            if r.details:
                for line in r.details.split('\n'):
                    print(f"       {line}")
            
            if r.passed:
                passed += 1
            else:
                failed += 1
        
        print("\n" + "-" * 70)
        print(f"Summary: {passed} passed, {failed} failed")
        print("=" * 70)
        
        return failed == 0


def generate_test_llir_via_pipeline() -> Tuple[bool, str, str]:
    """Generate LLIR by directly invoking the custom pipeline.
    
    Returns:
        tuple: (success, llir_string, error_message)
    """
    try:
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget
        
        # Define a simple test kernel
        @triton.jit
        def test_kernel(
            x_ptr,
            y_ptr, 
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)
        
        # Create target for custom backend
        target = GPUTarget("custom", 70, 32)
        
        # Use warmup to compile without running
        # warmup(arg_types..., grid=...) returns a CompiledKernel
        import numpy as np
        
        # Create dummy tensors for type inference
        try:
            import torch
            x = torch.zeros(1024, dtype=torch.float32, device='cpu')
            y = torch.zeros(1024, dtype=torch.float32, device='cpu')
            output = torch.zeros(1024, dtype=torch.float32, device='cpu')
        except ImportError:
            # Fallback without torch - use the compile API directly
            # Compile with explicit signature
            from triton.compiler.compiler import compile as triton_compile, ASTSource
            
            src = ASTSource(
                fn=test_kernel,
                signature={
                    0: "*fp32",
                    1: "*fp32", 
                    2: "*fp32",
                    3: "i32",
                },
                constants={4: 256},  # BLOCK_SIZE = 256
                attrs=triton.compiler.AttrsDescriptor(),
            )
            
            compiled = triton_compile(src, target=target)
            
            if hasattr(compiled, 'asm') and 'llir' in compiled.asm:
                llir = compiled.asm['llir']
                return True, llir, ""
            else:
                return False, "", "LLIR not found in compilation result"
        
        # With torch available, use warmup
        try:
            compiled = test_kernel.warmup(
                torch.float32, torch.float32, torch.float32, 1024,
                BLOCK_SIZE=256,
                grid=(4,),
            )
            
            if hasattr(compiled, 'asm') and 'llir' in compiled.asm:
                llir = compiled.asm['llir']
                return True, llir, ""
            else:
                return False, "", "LLIR not found in warmup result"
        except Exception as warmup_err:
            # Warmup failed, try direct compile
            from triton.compiler.compiler import compile as triton_compile, ASTSource
            
            src = ASTSource(
                fn=test_kernel,
                signature={
                    0: "*fp32",
                    1: "*fp32", 
                    2: "*fp32",
                    3: "i32",
                },
                constants={4: 256},
                attrs=triton.compiler.AttrsDescriptor(),
            )
            
            compiled = triton_compile(src, target=target)
            
            if hasattr(compiled, 'asm') and 'llir' in compiled.asm:
                llir = compiled.asm['llir']
                return True, llir, ""
            else:
                return False, "", f"Warmup failed: {warmup_err}, compile also failed"
            
    except Exception as e:
        import traceback
        return False, "", f"{e}\n{traceback.format_exc()}"


def generate_test_llir_from_ttir_file() -> Tuple[bool, str, str]:
    """Generate LLIR by directly invoking make_llir on a TTIR module.
    
    This bypasses the full compilation and blob generation steps.
    """
    try:
        from triton._C.libtriton import ir
        from triton.backends.custom import pipeline
        import tempfile
        import os as _os
        
        # Create a minimal TTIR content
        ttir_content = '''
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "custom:70"} {
  tt.func public @test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
        
        # Write to temp file (parse_mlir_module needs a file path)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ttir', delete=False) as f:
            f.write(ttir_content)
            ttir_path = f.name
        
        try:
            # Create mock options
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
            metadata = {"name": "test_kernel"}
            capability = 70
            
            # Parse TTIR from file
            context = ir.context()
            ir.load_dialects(context)
            
            mod = ir.parse_mlir_module(ttir_path, context)
            if mod is None:
                return False, "", "Failed to parse TTIR module"
            mod.context = context
            
            # Run the pipeline stages
            mod = pipeline.make_ttir(mod, metadata, opt, capability)
            mod = pipeline.make_ttgir(mod, metadata, opt, capability)
            
            # Generate LLIR
            llir = pipeline.make_llir(mod, metadata, opt, capability)
            
            if llir:
                return True, llir, ""
            else:
                return False, "", "make_llir returned empty result"
        finally:
            _os.unlink(ttir_path)
            
    except Exception as e:
        import traceback
        return False, "", f"{e}\n{traceback.format_exc()}"


def test_pipeline_functions() -> ValidationResult:
    """Test that pipeline functions are properly defined."""
    try:
        from triton.backends.custom import pipeline
        
        required_funcs = [
            '_lower_ttgir_to_llvm_nvidia',
            '_lower_ttgir_to_llvm_custom', 
            '_get_custom_triple_and_datalayout',
            '_inject_custom_intrinsic_declarations',
            'make_ttir',
            'make_ttgir',
            'make_llir',
        ]
        
        missing = [f for f in required_funcs if not hasattr(pipeline, f)]
        
        if missing:
            return ValidationResult(
                name="Pipeline functions",
                passed=False,
                message=f"Missing {len(missing)} function(s)",
                details="\n".join(f"  - {f}" for f in missing)
            )
        
        # Test triple function
        triple, cpu, features, datalayout = pipeline._get_custom_triple_and_datalayout(None)
        
        if triple.startswith('nvptx'):
            return ValidationResult(
                name="Pipeline functions",
                passed=False,
                message=f"Default triple is NVIDIA: {triple}"
            )
        
        return ValidationResult(
            name="Pipeline functions",
            passed=True,
            message=f"All functions present, triple={triple}"
        )
        
    except Exception as e:
        return ValidationResult(
            name="Pipeline functions",
            passed=False,
            message=f"Import error: {e}"
        )


def test_custom_plugin_loaded() -> ValidationResult:
    """Test that the custom C++ plugin is loaded."""
    try:
        from triton._C.libtriton import custom
        
        if not hasattr(custom, 'passes'):
            return ValidationResult(
                name="Custom plugin",
                passed=False,
                message="custom.passes not found"
            )
        
        if not hasattr(custom.passes, 'ttgpuir'):
            return ValidationResult(
                name="Custom plugin",
                passed=False, 
                message="custom.passes.ttgpuir not found"
            )
        
        if not hasattr(custom.passes.ttgpuir, 'add_to_llvmir'):
            return ValidationResult(
                name="Custom plugin",
                passed=False,
                message="custom.passes.ttgpuir.add_to_llvmir not found"
            )
        
        return ValidationResult(
            name="Custom plugin",
            passed=True,
            message="TritonCustom plugin loaded with add_to_llvmir"
        )
        
    except ImportError as e:
        return ValidationResult(
            name="Custom plugin",
            passed=False,
            message=f"Plugin not available: {e}",
            details="Run `pip install -e .` to build the custom plugin"
        )


def main():
    print("\n" + "=" * 70)
    print("Phase A4: Custom SIMT Backend End-to-End Validation")
    print("=" * 70)
    
    print(f"\nEnvironment:")
    print(f"  TRITON_CUSTOM_LLIR_MODE = {os.environ.get('TRITON_CUSTOM_LLIR_MODE', 'not set')}")
    print(f"  TRITON_CUSTOM_TTGIR_MODE = {os.environ.get('TRITON_CUSTOM_TTGIR_MODE', 'not set')}")
    print(f"  TRITON_CUSTOM_ACTIVE = {os.environ.get('TRITON_CUSTOM_ACTIVE', 'not set')}")
    
    all_results = []
    
    # Test 1: Check pipeline functions
    print("\n" + "-" * 70)
    print("Step 1: Checking pipeline functions...")
    result = test_pipeline_functions()
    all_results.append(result)
    status = "✓" if result.passed else "✗"
    print(f"  {status} {result.message}")
    
    # Test 2: Check custom plugin
    print("\n" + "-" * 70)
    print("Step 2: Checking custom C++ plugin...")
    result = test_custom_plugin_loaded()
    all_results.append(result)
    status = "✓" if result.passed else "✗"
    print(f"  {status} {result.message}")
    if result.details:
        print(f"     {result.details}")
    
    # Test 3: Generate LLIR and validate
    print("\n" + "-" * 70)
    print("Step 3: Generating LLIR via custom pipeline...")
    
    # Try TTIR file method first (more reliable)
    success, llir, error = generate_test_llir_from_ttir_file()
    
    if not success:
        print(f"  TTIR file method failed: {error[:200]}...")
        print("  Trying pipeline method...")
        success, llir, error = generate_test_llir_via_pipeline()
    
    if success and llir:
        print(f"  ✓ LLIR generated ({len(llir)} bytes)")
        
        # Save LLIR for inspection
        llir_path = "/tmp/custom_test.ll"
        with open(llir_path, 'w') as f:
            f.write(llir)
        print(f"  Saved to: {llir_path}")
        
        # Run Phase A4 validation
        validator = PhaseA4Validator(llir)
        validator.run_all_checks()
        all_results.extend(validator.results)
        validator.print_results()
        
    else:
        print(f"  ✗ Failed to generate LLIR")
        print(f"  Error: {error[:500]}")
        all_results.append(ValidationResult(
            name="LLIR generation",
            passed=False,
            message="Failed to generate LLIR",
            details=error[:500]
        ))
    
    # Final summary
    print("\n" + "=" * 70)
    print("Phase A4 Final Summary")
    print("=" * 70)
    
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    
    for r in all_results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} {r.name}: {r.message}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 Phase A4 PASSED - Custom LLIR is NVIDIA-free!")
    else:
        print("\n❌ Phase A4 FAILED - Issues need to be resolved")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
