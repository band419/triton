from __future__ import annotations

import os

from triton import knobs
from triton._C.libtriton import ir, llvm, passes


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().upper() in {"1", "ON", "YES", "TRUE"}


def _get_custom_warp_size(opt) -> int:
    # Custom backend should not hardcode warp size to 32.
    # Prefer an explicit option if present, else environment, else default.
    warp = getattr(opt, "warp_size", None)
    if warp is None:
        warp = os.environ.get("TRITON_CUSTOM_WARP_SIZE", "32")
    try:
        warp_i = int(warp)
    except Exception:
        warp_i = 32
    return warp_i


def make_ttir(mod, metadata: dict, opt, capability: int):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod, "custom.make_ttir")
    return mod


def make_ttgir(mod, metadata: dict, opt, capability: int):
    # TTGIR pipeline mode:
    # - "custom" (default): minimal, avoids NVIDIA-specific TTGIR passes
    # - "nvidia": matches upstream CUDA TTGIR pipeline (for regression comparison)
    mode = os.environ.get("TRITON_CUSTOM_TTGIR_MODE", "custom").strip().lower()

    # The `target` string is recorded onto the module as `ttg.target` and is
    # consulted by some downstream passes/utilities.
    #
    # In "custom" mode we default to a custom-prefixed target so that any
    # accidental NVIDIA-specific consumers (expecting `cuda:`) fail loudly
    # rather than silently taking CUDA-specific branches.
    default_target = f"custom:{capability}" if mode not in ("nvidia", "cuda") else f"cuda:{capability}"
    target = os.environ.get("TRITON_CUSTOM_TTGIR_TARGET", default_target)

    # When mode is "nvidia", we can still selectively drop NVIDIA-only passes.
    use_nvidia_pipeline = _env_flag("TRITON_CUSTOM_USE_NVIDIA_TTGIR_PIPELINE", "1")

    if getattr(opt, "maxnreg", None) is not None:
        mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

    warp_size = _get_custom_warp_size(opt)
    num_ctas = getattr(opt, "num_ctas", 1)

    pm = ir.pass_manager(mod.context)
    dump_enabled = pm.enable_debug()
    emuTF32 = (capability // 10 >= 8)
    passes.ttir.add_convert_to_ttgpuir(pm, target, opt.num_warps, warp_size, num_ctas)

    # ---------------------------------------------------------------------
    # Custom-hardware-friendly TTGIR path (default)
    # Phase B4: This path is NVIDIA-free, avoiding all NVIDIA-specific passes
    # ---------------------------------------------------------------------
    if mode not in ("nvidia", "cuda"):
        # Goal: keep the TTGIR legal and reasonably optimized without
        # introducing NVIDIA-only concepts (MMA/TMA/TMEM/cp.async/ptxas).
        #
        # NVIDIA passes explicitly NOT used in custom mode:
        # - nvidia.passes.ttnvgpuir.add_plan_cta (NVIDIA CTA planning)
        # - nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding (TMA)
        # - nvidia.passes.hopper.add_hopper_warpspec (Hopper warp specialization)
        # - nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem (TMEM)
        # - nvidia.passes.ttnvgpuir.add_remove_tmem_tokens (TMEM)
        # - nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts (TMEM)
        # - nvidia.passes.ttnvgpuir.add_tma_lowering (TMA)
        # - nvidia.passes.ttnvgpuir.add_interleave_tmem (TMEM)
        # - nvidia.passes.ttnvgpuir.add_fence_insertion (CUDA fences)
        # - nvidia.passes.ttnvgpuir.add_lower_mma (MMA operations)
        # - nvidia.passes.ttgpuir.add_allocate_shared_memory_nv (shared mem)
        # - nvidia.passes.ttnvgpuir.add_allocate_tensor_memory (TMEM)
        #
        # Generic passes that ARE safe for custom hardware:
        # - passes.ttgpuir.add_coalesce (memory coalescing - generic)
        # - passes.ttgpuir.add_remove_layout_conversions (layout - generic)
        # - passes.ttgpuir.add_optimize_thread_locality (thread - generic)
        # - passes.ttir.add_triton_licm (LICM - generic)
        # - passes.ttir.add_loop_aware_cse (CSE - generic)
        # - passes.common.add_canonicalizer (canonicalize - generic)
        # - passes.common.add_cse (CSE - generic)
        # - passes.common.add_symbol_dce (DCE - generic)
        
        if _env_flag("TRITON_CUSTOM_ENABLE_TTGIR_COALESCE", "1"):
            passes.ttgpuir.add_coalesce(pm)

        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)

        if _env_flag("TRITON_CUSTOM_ENABLE_TTGIR_LICM", "1"):
            passes.ttir.add_triton_licm(pm)

        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        pm.run(mod, "custom.make_ttgir")

        # Some Triton flows expect this key to exist. Keep it stable.
        try:
            metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        except Exception:
            metadata["tensordesc_meta"] = None

        return mod

    # ---------------------------------------------------------------------
    # NVIDIA-compatible TTGIR path (debug/regression)
    # ---------------------------------------------------------------------
    # Optimize TTGIR
    passes.ttgpuir.add_coalesce(pm)
    passes.ttgpuir.add_f32_dot_tc(pm, emuTF32)

    if use_nvidia_pipeline:
        from triton._C.libtriton import nvidia  # type: ignore

        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        nvidia.passes.ttnvgpuir.add_plan_cta(pm)

    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    passes.ttgpuir.add_accelerate_matmul(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)

    if use_nvidia_pipeline:
        from triton._C.libtriton import nvidia  # type: ignore

        nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)

    passes.ttir.add_loop_aware_cse(pm)

    if capability // 10 in [8, 9]:
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        if use_nvidia_pipeline:
            from triton._C.libtriton import nvidia  # type: ignore

            nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)

        passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
        passes.ttgpuir.add_schedule_loops(pm)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
    elif capability // 10 >= 10:
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.ttgpuir.add_optimize_accumulator_init(pm)
        passes.ttgpuir.add_hoist_tmem_alloc(pm, False)

        if use_nvidia_pipeline:
            from triton._C.libtriton import nvidia  # type: ignore

            nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)

        passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
        passes.ttgpuir.add_schedule_loops(pm)
        passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        passes.ttgpuir.add_optimize_partition_warps(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_hoist_tmem_alloc(pm, True)

        if use_nvidia_pipeline:
            from triton._C.libtriton import nvidia  # type: ignore

            nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
    else:
        passes.ttir.add_triton_licm(pm)

    passes.common.add_canonicalizer(pm)
    passes.ttir.add_loop_aware_cse(pm)
    passes.ttgpuir.add_prefetch(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
    passes.ttgpuir.add_coalesce_async_copy(pm)

    if use_nvidia_pipeline:
        from triton._C.libtriton import nvidia  # type: ignore

        nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
        if capability // 10 >= 9:
            nvidia.passes.ttnvgpuir.add_tma_lowering(pm)

    passes.ttgpuir.add_remove_layout_conversions(pm)

    if use_nvidia_pipeline:
        from triton._C.libtriton import nvidia  # type: ignore

        nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)

    passes.ttgpuir.add_reduce_data_duplication(pm)
    passes.ttgpuir.add_reorder_instructions(pm)
    passes.ttir.add_loop_aware_cse(pm)
    passes.common.add_symbol_dce(pm)

    if use_nvidia_pipeline:
        from triton._C.libtriton import nvidia  # type: ignore

        nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
        nvidia.passes.ttnvgpuir.add_lower_mma(pm)

    passes.common.add_sccp(pm)
    passes.common.add_cse(pm)
    passes.common.add_canonicalizer(pm)

    pm.run(mod, "custom.make_ttgir")

    # Some Triton flows expect this key to exist. Keep it stable.
    try:
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
    except Exception:
        metadata["tensordesc_meta"] = None

    return mod


def _lower_ttgir_to_llvm_nvidia(pm, mod, opt, capability: int):
    """NVIDIA-specific TTGIR → LLVM lowering (legacy path)."""
    from triton._C.libtriton import nvidia  # type: ignore

    ptx_version = int(os.environ.get("TRITON_CUSTOM_PTX_VERSION", "0"))
    if ptx_version == 0:
        from triton.backends.nvidia.compiler import get_ptx_version_from_options  # type: ignore
        ptx_version = get_ptx_version_from_options(opt, capability)

    nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
    nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
    nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm)

    if knobs.compilation.instrumentation_mode == "consan":
        passes.ttgpuir.add_concurrency_sanitizer(pm)

    passes.ttgpuir.add_allocate_global_scratch_memory(pm)
    nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)

    nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
    nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)

    if _env_flag("TRITON_CUSTOM_ENABLE_NVVM_TO_LLVM", "1"):
        passes.convert.add_nvvm_to_llvm(pm)


def _lower_ttgir_to_llvm_custom(pm, mod, opt):
    """Custom SIMT TTGIR → LLVM lowering (NVIDIA-free path).
    
    This path uses custom passes and intrinsics, avoiding all NVIDIA-specific
    dialects and lowering passes.
    
    Phase B4: NVIDIA passes explicitly NOT used:
    - nvidia.passes.ttgpuir.add_allocate_shared_memory_nv (shared memory)
    - nvidia.passes.ttnvgpuir.add_allocate_tensor_memory (TMEM)
    - nvidia.passes.ttnvgpuir.add_check_matmul_two_cta (multi-CTA matmul)
    - nvidia.passes.ttnvgpuir.add_proxy_fence_insertion (CUDA fences)
    - nvidia.passes.ttgpuir.add_to_llvmir (NVIDIA LLVM lowering)
    - nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm (NVGPU dialect)
    - nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm (warp spec)
    - passes.convert.add_nvvm_to_llvm (NVVM dialect)
    
    Note: Custom backend does NOT support shared memory. All memory operations
    use global memory only, as per the hardware spec.
    """
    try:
        from triton._C.libtriton import custom  # type: ignore
        warp_size = _get_custom_warp_size(opt)
        
        # Custom backend: emulate all CTA-level scratch/shared exchanges using
        # global scratch (global memory), not real shared memory.
        #
        # However, many generic TTGIR→LLVM lowerings still rely on the presence
        # of `allocation.offset` and `ttg.shared` metadata (normally produced by
        # AllocateSharedMemory) to compute a base pointer for scratch buffers.
        # We therefore run AllocateSharedMemory to attach offsets/sizes, and
        # then redirect the shared base to global scratch during lowering.
        mod.set_attr(
            "ttg.shared_memory_model",
            ir.builder(mod.context).get_string_attr("global_scratch"),
        )
        
        if knobs.compilation.instrumentation_mode == "consan":
            passes.ttgpuir.add_concurrency_sanitizer(pm)
        
        # Attach `allocation.offset` and `ttg.shared` (sizes) used by lowerings
        # like ConvertLayout/Reduce. This does NOT allocate real shared memory
        # for the custom backend; it only annotates the IR.
        passes.ttgpuir.add_allocate_shared_memory(pm)
        
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        
        # Convert to LLVM using custom pass (generates custom intrinsics)
        # This replaces nvidia.passes.ttgpuir.add_to_llvmir
        custom.passes.ttgpuir.add_to_llvmir(pm, warp_size)
        
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        
        # Convert standard dialects to LLVM
        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        
    except ImportError:
        # Fallback: if custom plugin is not built, use a minimal generic lowering
        # This allows the pipeline to still work for testing/development
        if knobs.compilation.instrumentation_mode == "consan":
            passes.ttgpuir.add_concurrency_sanitizer(pm)
        
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        passes.convert.add_index_to_llvmir(pm)
        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)


def _get_custom_triple_and_datalayout(opt):
    """Get triple and datalayout for Custom SIMT backend.
    
    Returns:
        tuple: (triple, cpu, features, datalayout)
    """
    triple = os.environ.get("TRITON_CUSTOM_LLVM_TRIPLE", "riscv32-unknown-unknown")
    cpu = os.environ.get("TRITON_CUSTOM_LLVM_CPU", "")
    features = os.environ.get("TRITON_CUSTOM_LLVM_FEATURES", "+f")
    
    # RV32IMF datalayout: 32-bit pointers, little endian
    # e = little endian
    # m:e = ELF mangling
    # p:32:32 = 32-bit pointers with 32-bit alignment
    # i64:64 = 64-bit integers with 64-bit alignment
    # n32 = native integer width is 32 bits
    # S128 = stack is 128-bit aligned
    datalayout = os.environ.get(
        "TRITON_CUSTOM_LLVM_DATALAYOUT",
        "e-m:e-p:32:32-i64:64-n32-S128"
    )
    
    return triple, cpu, features, datalayout


def _inject_custom_intrinsic_declarations(llvm_ir: str) -> str:
    """Inject custom intrinsic declarations into the LLVM IR if not present.
    
    This ensures all custom intrinsics are declared with proper attributes.
    """
    intrinsics = [
        # (name, return_type, param_types, attributes)
        ("llvm.riscv.simt.barrier", "void", "", "convergent nounwind"),
        ("llvm.riscv.simt.warp.barrier", "void", "", "convergent nounwind"),
        ("llvm.riscv.simt.lane.id", "i32", "", "readnone nounwind"),
        ("llvm.riscv.simt.warp.size", "i32", "", "readnone nounwind"),
        ("llvm.riscv.simt.program.id", "i32", "i32", "readnone nounwind"),
        ("llvm.riscv.simt.num.programs", "i32", "i32", "readnone nounwind"),
        ("llvm.riscv.simt.ballot.mask", "i32", "i1", "convergent nounwind"),
        ("llvm.riscv.simt.shfl.bfly", "i32", "i32, i32", "convergent nounwind"),
        ("llvm.riscv.simt.shfl.idx", "i32", "i32, i32", "convergent nounwind"),
    ]
    
    # Build declarations block
    decls = []
    for name, ret_ty, params, attrs in intrinsics:
        if f"@{name}" not in llvm_ir or f"declare {ret_ty} @{name}" in llvm_ir:
            # Already declared or not used
            continue
        param_str = f"({params})" if params else "()"
        attr_str = f" #{len(decls)}" if attrs else ""
        decls.append(f"declare {ret_ty} @{name}{param_str}{attr_str}")
    
    if not decls:
        return llvm_ir
    
    # Insert declarations after the module-level attributes
    # Find a good insertion point (after target triple/datalayout)
    lines = llvm_ir.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('target ') or line.startswith('source_filename'):
            insert_idx = i + 1
        elif line.startswith('define ') or line.startswith('@'):
            break
    
    # Insert declarations
    for decl in decls:
        lines.insert(insert_idx, decl)
        insert_idx += 1
    
    return '\n'.join(lines)


def make_llir(mod, metadata: dict, opt, capability: int) -> str:
    """Convert TritonGPU IR to LLVM IR.
    
    This function supports two modes:
    - NVIDIA mode (default for backward compatibility): Uses NVIDIA-specific
      passes and generates NVPTX-compatible LLVM IR.
    - Custom mode (TRITON_CUSTOM_LLIR_MODE=1): Uses custom passes and generates
      NVIDIA-free LLVM IR with custom intrinsics for SIMT operations.
    """
    # Determine which lowering path to use
    use_custom_llir = _env_flag("TRITON_CUSTOM_LLIR_MODE", "0")
    
    # Stage A: TritonGPU -> LLVM-IR (MLIR)
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    passes.ttgpuir.add_combine_tensor_select_and_if(pm)
    passes.ttgpuir.add_allocate_warp_groups(pm)
    passes.convert.add_scf_to_cf(pm)
    passes.gluon.add_inliner(pm)

    if use_custom_llir:
        # NVIDIA-free lowering path
        _lower_ttgir_to_llvm_custom(pm, mod, opt)
    else:
        # Legacy NVIDIA-compatible lowering path
        _lower_ttgir_to_llvm_nvidia(pm, mod, opt, capability)

    if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
        passes.llvmir.add_di_scope(pm)

    pm.run(mod, "custom.make_llir")

    if knobs.compilation.dump_ir_extract_di_local_variables:
        if not knobs.compilation.disable_line_info:
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_scope(pm)
            pm.run(mod, "custom.make_llir.disable_line_info")

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.llvmir.add_di_local_variable(pm)
        pm.run(mod, "custom.make_llir.dump_ir_extract_di_local_variables")

    # Stage B: LLVM-IR (MLIR) -> LLVM-IR (LLVM)
    llvm.init_targets()
    context = llvm.context()

    llvm_mod = llvm.to_module(mod, context)

    if use_custom_llir:
        # Custom SIMT triple and datalayout
        triple, cpu, features, datalayout = _get_custom_triple_and_datalayout(opt)
        # Note: We don't call llvm.attach_datalayout() because the target triple
        # may not be known to the host LLVM. Instead, we inject the triple and
        # datalayout directly into the LLIR string before returning.
        custom_triple = triple
        custom_datalayout = datalayout
    else:
        # NVIDIA-compatible triple and datalayout
        custom_triple = None
        custom_datalayout = None
        triple = os.environ.get("TRITON_CUSTOM_LLVM_TRIPLE", "nvptx64-nvidia-cuda")
        cpu = os.environ.get("TRITON_CUSTOM_LLVM_CPU", "")
        features = os.environ.get("TRITON_CUSTOM_LLVM_FEATURES", "")

        if cpu == "":
            if triple.startswith("nvptx"):
                from triton.backends.nvidia.compiler import sm_arch_from_capability, get_features  # type: ignore
                cpu = sm_arch_from_capability(capability)
                features = features or get_features(opt, capability)

        if triple.startswith("nvptx"):
            try:
                from triton._C.libtriton import nvidia  # type: ignore
                nvidia.set_short_ptr()
            except Exception:
                pass

        llvm.attach_datalayout(llvm_mod, triple, cpu, features)

    # Keep LLVM optimization level configurable.
    opt_level = os.environ.get("TRITON_CUSTOM_LLVM_OPT", "O3")
    if opt_level.upper() == "O0":
        pass
    else:
        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

    total_num_warps = mod.get_int_attr("ttg.total-num-warps")
    if total_num_warps is not None:
        metadata["num_warps"] = total_num_warps
    
    # Custom backend: no shared memory support (global memory only)
    if use_custom_llir:
        metadata["shared"] = 0
    else:
        metadata["shared"] = mod.get_int_attr("ttg.shared")
    
    metadata["tmem_size"] = mod.get_int_attr("ttg.tensor_memory_size")
    metadata["global_scratch_size"] = mod.get_int_attr("ttg.global_scratch_memory_size")
    metadata["global_scratch_align"] = mod.get_int_attr("ttg.global_scratch_memory_alignment")
    metadata["profile_scratch_size"] = mod.get_int_attr("ttg.profile_scratch_memory_size") or 0
    metadata["profile_scratch_align"] = mod.get_int_attr("ttg.profile_scratch_memory_alignment") or 1

    ret = str(llvm_mod)
    
    # Extract kernel name from LLVM IR (look for define void @<name>)
    import re
    kernel_names = re.findall(r'define void @([a-zA-Z_][a-zA-Z0-9_]*)\(', ret)
    if kernel_names:
        metadata["name"] = kernel_names[0]
    else:
        metadata["name"] = "triton_kernel"
    
    # For custom mode, inject triple/datalayout and intrinsic declarations
    if use_custom_llir:
        # Inject target triple and datalayout at the start of the module
        # We do this as string manipulation because the target may not be
        # known to the host LLVM's attach_datalayout function.
        lines = ret.split('\n')
        inject_lines = []
        
        # Find the right place to inject (after any comments/source_filename)
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(';') or line.strip().startswith('source_filename'):
                insert_pos = i + 1
            elif line.strip().startswith('target'):
                # Already has target, skip injection for this line type
                pass
            elif line.strip() and not line.strip().startswith(';'):
                break
        
        # Check if triple/datalayout already exist
        has_triple = any('target triple' in line for line in lines)
        has_datalayout = any('target datalayout' in line for line in lines)
        
        if not has_datalayout and custom_datalayout:
            inject_lines.append(f'target datalayout = "{custom_datalayout}"')
        if not has_triple and custom_triple:
            inject_lines.append(f'target triple = "{custom_triple}"')
        
        if inject_lines:
            for i, inject_line in enumerate(inject_lines):
                lines.insert(insert_pos + i, inject_line)
            ret = '\n'.join(lines)
        
        # Ensure all intrinsic declarations are present
        ret = _inject_custom_intrinsic_declarations(ret)
    
    del llvm_mod
    del context
    return ret


# =============================================================================
# Stage: LLVM IR -> RISCV Assembly
# =============================================================================

def _get_riscv_simt_config():
    """Get RISCV SIMT backend configuration.
    
    Returns:
        dict: Configuration for RISCV SIMT codegen
    """
    return {
        # Target triple: riscv32-unknown-unknown for custom SIMT
        "triple": os.environ.get("TRITON_CUSTOM_LLVM_TRIPLE", "riscv32-unknown-unknown"),
        # CPU: rv32imf or specific model
        "cpu": os.environ.get("TRITON_CUSTOM_LLVM_CPU", "generic-rv32"),
        # Features: +f for float, +m for multiply
        "features": os.environ.get("TRITON_CUSTOM_LLVM_FEATURES", "+f,+m"),
        # LLVM flags
        "flags": os.environ.get("TRITON_CUSTOM_LLC_FLAGS", "").split() if os.environ.get("TRITON_CUSTOM_LLC_FLAGS") else [],
    }


def make_asm(src: str, metadata: dict, opt, capability: int) -> str:
    """Convert LLVM IR to RISCV assembly.
    
    This is analogous to NVIDIA's make_ptx, but for RISCV SIMT.
    
    Args:
        src: LLVM IR string
        metadata: Compilation metadata dict
        opt: Compiler options
        capability: Target capability (e.g., 70)
        
    Returns:
        str: RISCV assembly code
    """
    import re
    
    config = _get_riscv_simt_config()
    triple = config["triple"]
    cpu = config["cpu"]
    features = config["features"]
    flags = config["flags"]
    
    # Enable FP fusion if requested
    enable_fp_fusion = getattr(opt, "enable_fp_fusion", True)
    
    # Find kernel names from LLVM IR
    # Look for: define void @kernel_name( or define dso_local void @kernel_name(
    names = re.findall(r'define\s+(?:dso_local\s+)?void\s+@([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', src)
    if names:
        metadata["name"] = names[0]
    else:
        metadata["name"] = "triton_kernel"
    
    # Translate LLVM IR to assembly using Triton's llvm module
    # This calls into the LLVM backend (which must have RISCV target enabled)
    try:
        asm = llvm.translate_to_asm(
            src,           # LLVM IR string
            triple,        # Target triple
            cpu,           # CPU model
            features,      # CPU features
            flags,         # Additional flags
            enable_fp_fusion,  # FP fusion
            False          # isObject=False for assembly
        )
    except Exception as e:
        # If LLVM backend translation fails, provide helpful error
        raise RuntimeError(
            f"Failed to translate LLVM IR to RISCV assembly.\n"
            f"Triple: {triple}, CPU: {cpu}, Features: {features}\n"
            f"Error: {e}\n"
            f"Make sure LLVM was built with RISCV target enabled:\n"
            f"  LLVM_TARGETS=Native;NVPTX;AMDGPU;RISCV ./scripts/build-llvm-project.sh"
        ) from e
    
    # Dump assembly if requested
    if _env_flag("TRITON_CUSTOM_DUMP_ASM"):
        print("// -----// RISCV SIMT Assembly Dump //----- //")
        print(asm)
    
    return asm


def make_obj(src: str, metadata: dict, opt, capability: int) -> bytes:
    """Convert LLVM IR to object file (ELF).
    
    This is analogous to NVIDIA's make_cubin, but for RISCV SIMT.
    
    Args:
        src: LLVM IR string (we go directly from LLIR to object)
        metadata: Compilation metadata dict
        opt: Compiler options
        capability: Target capability
        
    Returns:
        bytes: ELF object file bytes
    """
    import re
    
    config = _get_riscv_simt_config()
    triple = config["triple"]
    cpu = config["cpu"]
    features = config["features"]
    flags = config["flags"]
    
    enable_fp_fusion = getattr(opt, "enable_fp_fusion", True)
    
    # Find kernel names
    names = re.findall(r'define\s+(?:dso_local\s+)?void\s+@([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', src)
    if names:
        metadata["name"] = names[0]
    else:
        metadata["name"] = "triton_kernel"
    
    # Translate LLVM IR to object file
    try:
        obj = llvm.translate_to_asm(
            src,           # LLVM IR string
            triple,        # Target triple
            cpu,           # CPU model
            features,      # CPU features
            flags,         # Additional flags
            enable_fp_fusion,  # FP fusion
            True           # isObject=True for object file
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to translate LLVM IR to RISCV object file.\n"
            f"Triple: {triple}, CPU: {cpu}, Features: {features}\n"
            f"Error: {e}\n"
            f"Make sure LLVM was built with RISCV target enabled."
        ) from e
    
    # obj is returned as bytes when isObject=True
    if isinstance(obj, str):
        obj = obj.encode('latin-1')  # Should not happen, but handle gracefully
    
    if _env_flag("TRITON_CUSTOM_DUMP_OBJ_SIZE"):
        print(f"// RISCV object file size: {len(obj)} bytes")
    
    return obj
