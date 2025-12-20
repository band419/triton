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
    # ---------------------------------------------------------------------
    if mode not in ("nvidia", "cuda"):
        # Goal: keep the TTGIR legal and reasonably optimized without
        # introducing NVIDIA-only concepts (MMA/TMA/TMEM/cp.async/ptxas).
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
    """
    try:
        from triton._C.libtriton import custom  # type: ignore
        warp_size = _get_custom_warp_size(opt)
        
        # Allocate shared memory using custom pass
        custom.passes.ttgpuir.add_allocate_shared_memory(pm)
        
        if knobs.compilation.instrumentation_mode == "consan":
            passes.ttgpuir.add_concurrency_sanitizer(pm)
        
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        
        # Convert to LLVM using custom pass (generates custom intrinsics)
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
        ("llvm.custom.barrier", "void", "", "convergent nounwind"),
        ("llvm.custom.warp.barrier", "void", "", "convergent nounwind"),
        ("llvm.custom.lane.id", "i32", "", "readnone nounwind"),
        ("llvm.custom.warp.size", "i32", "", "readnone nounwind"),
        ("llvm.custom.program.id", "i32", "i32", "readnone nounwind"),
        ("llvm.custom.num.programs", "i32", "i32", "readnone nounwind"),
        ("llvm.custom.ballot", "i32", "i1", "convergent nounwind"),
        ("llvm.custom.shuffle.xor", "i32", "i32, i32", "convergent nounwind"),
        ("llvm.custom.shuffle.up", "i32", "i32, i32", "convergent nounwind"),
        ("llvm.custom.shuffle.idx", "i32", "i32, i32", "convergent nounwind"),
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
    else:
        # NVIDIA-compatible triple and datalayout
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
    metadata["shared"] = mod.get_int_attr("ttg.shared")
    metadata["tmem_size"] = mod.get_int_attr("ttg.tensor_memory_size")
    metadata["global_scratch_size"] = mod.get_int_attr("ttg.global_scratch_memory_size")
    metadata["global_scratch_align"] = mod.get_int_attr("ttg.global_scratch_memory_alignment")
    metadata["profile_scratch_size"] = mod.get_int_attr("ttg.profile_scratch_memory_size") or 0
    metadata["profile_scratch_align"] = mod.get_int_attr("ttg.profile_scratch_memory_alignment") or 1

    ret = str(llvm_mod)
    
    # For custom mode, ensure all intrinsic declarations are present
    if use_custom_llir:
        ret = _inject_custom_intrinsic_declarations(ret)
    
    del llvm_mod
    del context
    return ret
