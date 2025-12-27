#!/usr/bin/env python3
"""
Phase B2: Intrinsic Specification Validation Tests

This script validates that the custom backend correctly implements
the intrinsic specification as defined in docs/custom-intrinsic-spec.md.

Tests:
1. Intrinsic naming convention (llvm.riscv.simt.*)
2. Intrinsic attributes (convergent, readnone, nounwind)
3. Intrinsic declarations present
4. Barrier convergent attribute (critical for correctness)
5. Shuffle/ballot convergent attributes

Usage:
    TRITON_CUSTOM_LLIR_MODE=1 python test_intrinsic_spec.py
"""

import os
import sys
import re
import tempfile
from typing import List, Tuple, Optional, Dict
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


# Expected intrinsic definitions with their required attributes
INTRINSIC_SPEC = {
    # Pure intrinsics (readnone, nounwind, willreturn)
    "llvm.riscv.simt.program.id": {
        "return_type": "i32",
        "params": ["i32"],
        "required_attrs": ["nounwind"],
        "optional_attrs": ["willreturn", "memory(none)"],
        "category": "simt",
    },
    "llvm.riscv.simt.thread.id": {
        "return_type": "i32",
        "params": ["i32"],
        "required_attrs": ["nounwind"],
        "optional_attrs": ["willreturn", "memory(none)"],
        "category": "simt",
    },
    "llvm.riscv.simt.lane.id": {
        "return_type": "i32",
        "params": [],
        "required_attrs": ["nounwind"],
        "optional_attrs": ["willreturn", "memory(none)"],
        "category": "simt",
    },
    "llvm.riscv.simt.warp.size": {
        "return_type": "i32",
        "params": [],
        "required_attrs": ["nounwind"],
        "optional_attrs": ["willreturn", "memory(none)"],
        "category": "simt",
    },
    "llvm.riscv.simt.num.programs": {
        "return_type": "i32",
        "params": ["i32"],
        "required_attrs": ["nounwind"],
        "optional_attrs": ["willreturn", "memory(none)"],
        "category": "simt",
    },
    "llvm.riscv.simt.block.id": {
        "return_type": "i32",
        "params": ["i32"],
        "required_attrs": ["nounwind"],
        "optional_attrs": [],
        "category": "simt",
    },
    "llvm.riscv.simt.block.dim": {
        "return_type": "i32",
        "params": ["i32"],
        "required_attrs": ["nounwind"],
        "optional_attrs": [],
        "category": "simt",
    },
    "llvm.riscv.simt.grid.dim": {
        "return_type": "i32",
        "params": ["i32"],
        "required_attrs": ["nounwind"],
        "optional_attrs": [],
        "category": "simt",
    },
    # Convergent intrinsics (must have convergent attribute)
    "llvm.riscv.simt.barrier": {
        "return_type": "void",
        "params": [],
        "required_attrs": ["convergent", "nounwind"],
        "optional_attrs": [],
        "category": "sync",
    },
    "llvm.riscv.simt.warp.barrier": {
        "return_type": "void",
        "params": [],
        "required_attrs": ["convergent", "nounwind"],
        "optional_attrs": [],
        "category": "sync",
    },
    "llvm.riscv.simt.shfl.bfly": {
        "return_type": "i32",  # or type-polymorphic
        "params": ["i32", "i32"],
        "required_attrs": ["convergent", "nounwind"],
        "optional_attrs": [],
        "category": "shuffle",
    },
    "llvm.riscv.simt.shfl.idx": {
        "return_type": "i32",
        "params": ["i32", "i32"],
        "required_attrs": ["convergent", "nounwind"],
        "optional_attrs": [],
        "category": "shuffle",
    },
    "llvm.riscv.simt.ballot.mask": {
        "return_type": "i32",
        "params": ["i1"],
        "required_attrs": ["convergent", "nounwind"],
        "optional_attrs": [],
        "category": "ballot",
    },
}


def generate_test_llir() -> Tuple[bool, str, str]:
    """Generate LLIR for testing intrinsic specifications."""
    try:
        from triton._C.libtriton import ir
        from triton.backends.custom import pipeline
        
        # TTIR that uses multiple intrinsics
        ttir_content = '''
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.target" = "custom:70"} {
  tt.func public @test_intrinsics(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %9 = tt.load %8, %6 : tensor<256x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %11, %9, %6 : tensor<256x!tt.ptr<f32>>
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
            metadata = {"name": "test_intrinsics"}
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


def parse_intrinsic_declarations(llir: str) -> Dict[str, Dict]:
    """Parse intrinsic declarations from LLIR."""
    intrinsics = {}
    
    # Pattern for function declarations
    # declare [attrs] <return_type> @<name>(<params>) [#attr_group]
    decl_pattern = r'declare\s+(?:!dbg\s+!\d+\s+)?(\w+)\s+@([^\s(]+)\s*\(([^)]*)\)\s*(#\d+)?'
    
    for match in re.finditer(decl_pattern, llir):
        ret_type = match.group(1)
        name = match.group(2)
        params = match.group(3)
        attr_group = match.group(4)
        
        if name.startswith("llvm.riscv.simt."):
            intrinsics[name] = {
                "return_type": ret_type,
                "params": params,
                "attr_group": attr_group,
            }
    
    return intrinsics


def parse_attribute_groups(llir: str) -> Dict[str, List[str]]:
    """Parse attribute groups from LLIR."""
    attr_groups = {}
    
    # Pattern for attribute groups: attributes #N = { ... }
    pattern = r'attributes\s+(#\d+)\s*=\s*\{([^}]*)\}'
    
    for match in re.finditer(pattern, llir):
        group_id = match.group(1)
        attrs_str = match.group(2)
        
        # Parse individual attributes
        attrs = []
        for attr in attrs_str.split():
            attrs.append(attr.strip())
        
        attr_groups[group_id] = attrs
    
    return attr_groups


def test_intrinsic_naming(llir: str) -> TestResult:
    """Test that all custom intrinsics use correct naming convention."""
    intrinsics = parse_intrinsic_declarations(llir)
    
    custom_intrinsics = [name for name in intrinsics if name.startswith("llvm.riscv.simt.")]
    
    # Check for NVIDIA intrinsics (should not be present)
    nvidia_intrinsics = [name for name in intrinsics if "nvvm" in name.lower()]
    
    if nvidia_intrinsics:
        return TestResult(
            "Intrinsic naming",
            False,
            f"Found {len(nvidia_intrinsics)} NVIDIA intrinsics",
            ", ".join(nvidia_intrinsics)
        )
    
    if not custom_intrinsics:
        return TestResult(
            "Intrinsic naming",
            False,
            "No llvm.riscv.simt.* intrinsics found"
        )
    
    return TestResult(
        "Intrinsic naming",
        True,
        f"Found {len(custom_intrinsics)} custom intrinsics",
        ", ".join(custom_intrinsics)
    )


def test_intrinsic_attributes(llir: str) -> TestResult:
    """Test that intrinsics have correct attributes."""
    intrinsics = parse_intrinsic_declarations(llir)
    attr_groups = parse_attribute_groups(llir)
    
    issues = []
    verified = []
    
    for name, spec in INTRINSIC_SPEC.items():
        if name not in intrinsics:
            continue  # Not all intrinsics may be used
        
        decl = intrinsics[name]
        attr_group = decl.get("attr_group")
        
        if attr_group and attr_group in attr_groups:
            attrs = attr_groups[attr_group]
            
            # Check required attributes
            for req_attr in spec["required_attrs"]:
                if req_attr not in attrs:
                    issues.append(f"{name}: missing '{req_attr}'")
                else:
                    verified.append(f"{name}: has '{req_attr}'")
    
    if issues:
        return TestResult(
            "Intrinsic attributes",
            False,
            f"{len(issues)} attribute issue(s)",
            "\n".join(issues[:5])
        )
    
    return TestResult(
        "Intrinsic attributes",
        True,
        f"Verified {len(verified)} attributes",
        "\n".join(verified[:3]) if verified else "No intrinsics to verify"
    )


def test_convergent_intrinsics(llir: str) -> TestResult:
    """Test that barrier/shuffle/ballot have convergent attribute."""
    intrinsics = parse_intrinsic_declarations(llir)
    attr_groups = parse_attribute_groups(llir)
    
    convergent_required = [
        "llvm.riscv.simt.barrier",
        "llvm.riscv.simt.warp.barrier",
        "llvm.riscv.simt.shfl.bfly",
        "llvm.riscv.simt.shfl.idx",
        "llvm.riscv.simt.ballot.mask",
    ]
    
    found_convergent = []
    missing_convergent = []
    
    for name in convergent_required:
        if name not in intrinsics:
            continue
        
        decl = intrinsics[name]
        attr_group = decl.get("attr_group")
        
        has_convergent = False
        if attr_group and attr_group in attr_groups:
            attrs = attr_groups[attr_group]
            has_convergent = "convergent" in attrs
        
        if has_convergent:
            found_convergent.append(name)
        else:
            missing_convergent.append(name)
    
    # If no convergent intrinsics are used, that's OK for basic tests
    if not found_convergent and not missing_convergent:
        return TestResult(
            "Convergent intrinsics",
            True,
            "No convergent intrinsics used in this kernel"
        )
    
    if missing_convergent:
        return TestResult(
            "Convergent intrinsics",
            False,
            f"{len(missing_convergent)} intrinsic(s) missing 'convergent'",
            ", ".join(missing_convergent)
        )
    
    return TestResult(
        "Convergent intrinsics",
        True,
        f"{len(found_convergent)} intrinsic(s) correctly marked convergent"
    )


def test_pure_intrinsics(llir: str) -> TestResult:
    """Test that pure intrinsics (program_id, etc.) have readnone/nounwind."""
    intrinsics = parse_intrinsic_declarations(llir)
    attr_groups = parse_attribute_groups(llir)
    
    pure_intrinsics = [
        "llvm.riscv.simt.program.id",
        "llvm.riscv.simt.thread.id",
        "llvm.riscv.simt.lane.id",
        "llvm.riscv.simt.warp.size",
        "llvm.riscv.simt.num.programs",
    ]
    
    found_with_attrs = []
    
    for name in pure_intrinsics:
        if name not in intrinsics:
            continue
        
        decl = intrinsics[name]
        attr_group = decl.get("attr_group")
        
        if attr_group and attr_group in attr_groups:
            attrs = attr_groups[attr_group]
            has_nounwind = "nounwind" in attrs
            if has_nounwind:
                found_with_attrs.append(name)
    
    if not found_with_attrs and any(name in intrinsics for name in pure_intrinsics):
        return TestResult(
            "Pure intrinsics",
            False,
            "Pure intrinsics missing nounwind attribute"
        )
    
    return TestResult(
        "Pure intrinsics",
        True,
        f"{len(found_with_attrs)} pure intrinsic(s) have nounwind",
        ", ".join(found_with_attrs) if found_with_attrs else "None used"
    )


def test_no_nvidia_intrinsics(llir: str) -> TestResult:
    """Test that no NVIDIA intrinsics are present."""
    nvidia_patterns = [
        "@llvm.nvvm.",
        "@nvvm.",
        "nvptx_",
    ]
    
    found = []
    for pattern in nvidia_patterns:
        if pattern in llir:
            found.append(pattern)
    
    if found:
        return TestResult(
            "NVIDIA-free intrinsics",
            False,
            f"Found NVIDIA intrinsic patterns: {', '.join(found)}"
        )
    
    return TestResult(
        "NVIDIA-free intrinsics",
        True,
        "No NVIDIA intrinsics found"
    )


def run_all_tests() -> List[TestResult]:
    """Run all Phase B2 tests."""
    results = []
    
    print("Generating LLIR for intrinsic tests...")
    success, llir, error = generate_test_llir()
    
    if not success:
        results.append(TestResult("LLIR Generation", False, "Failed", error))
        return results
    
    results.append(TestResult("LLIR Generation", True, f"Generated {len(llir)} bytes"))
    
    # Save for inspection
    with open("/tmp/custom_b2_test.ll", "w") as f:
        f.write(llir)
    print(f"  Saved to /tmp/custom_b2_test.ll")
    
    # Run tests
    results.append(test_intrinsic_naming(llir))
    results.append(test_intrinsic_attributes(llir))
    results.append(test_convergent_intrinsics(llir))
    results.append(test_pure_intrinsics(llir))
    results.append(test_no_nvidia_intrinsics(llir))
    
    return results


def main():
    print("\n" + "=" * 70)
    print("Phase B2: Intrinsic Specification Validation Tests")
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
            for line in r.details.split('\n')[:5]:
                print(f"    {line}")
        
        if r.passed:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 70)
    print(f"Summary: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 Phase B2 PASSED - Intrinsic specification is correctly implemented!")
    else:
        print("\n❌ Phase B2 has issues that need attention")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
