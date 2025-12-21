from __future__ import annotations

"""Runtime driver scaffold for the 'custom' backend.

This is the runtime counterpart to the custom compiler backend.

Key points:
- The compiler backend emits a Scheme-A `.blob` artifact (flat bytes, no relocations).
- The Triton runtime expects the active driver to provide:
  - `utils.load_binary(name, binary, shared, device)`
  - `launcher_cls(src, metadata)` producing a callable launcher
  - `get_current_device()` / `get_current_stream(device)` (used for defaults)

We keep the implementation minimal and delegate actual device/simulator
interaction to a pluggable Python module.

Configure via environment:
- `TRITON_CUSTOM_RUNTIME_MODULE`: Python module path that provides:
    - `load_binary(name: str, blob: bytes, shared: int, device: int) -> (module, function, n_regs, n_spills, n_max_threads)`
    - `launch(gridX, gridY, gridZ, stream, function, packed_metadata, launch_metadata, enter_hook, exit_hook, *args) -> None`

If not configured, loading/launching will fail with a clear error.

Kernel ABI Contract (Phase B1):
- Kernel parameters are passed as expanded arguments (not a single param block pointer)
- Pointer arguments: 32-bit addresses pointing to global memory (addrspace 1)
- Scalar arguments: i32, i64, f32, f16 passed directly
- SIMT context (program_id, thread_id, etc.) accessed via llvm.custom.* intrinsics
- See docs/custom-kernel-abi.md for full specification
"""

import importlib
import os
import struct
from typing import Any, Tuple

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


# Type mapping for parameter packing (Triton type string -> struct format)
_TYPE_TO_STRUCT_FMT = {
    "*fp32": "I",   # 32-bit pointer
    "*fp16": "I",   # 32-bit pointer
    "*bf16": "I",   # 32-bit pointer
    "*i8": "I",     # 32-bit pointer
    "*i16": "I",    # 32-bit pointer
    "*i32": "I",    # 32-bit pointer
    "*i64": "I",    # 32-bit pointer
    "i1": "B",      # 8-bit (padded)
    "i8": "b",      # 8-bit signed
    "u8": "B",      # 8-bit unsigned
    "i16": "h",     # 16-bit signed
    "u16": "H",     # 16-bit unsigned
    "i32": "i",     # 32-bit signed
    "u32": "I",     # 32-bit unsigned
    "i64": "q",     # 64-bit signed
    "u64": "Q",     # 64-bit unsigned
    "fp16": "e",    # 16-bit float (half)
    "bf16": "e",    # 16-bit bfloat (treat as half for packing)
    "fp32": "f",    # 32-bit float
    "fp64": "d",    # 64-bit float
}


def _pack_kernel_args(signature: dict, args: tuple) -> bytes:
    """Pack kernel arguments according to the expanded args ABI.
    
    Args:
        signature: Dict mapping arg index to type string (e.g., {0: "*fp32", 1: "i32"})
        args: Tuple of argument values
    
    Returns:
        Packed bytes suitable for parameter block
    """
    packed_parts = []
    for i, arg in enumerate(args):
        ty = signature.get(i, "i32")  # Default to i32 if type unknown
        fmt = _TYPE_TO_STRUCT_FMT.get(ty, "I")
        
        # Handle pointer types - extract address
        if ty.startswith("*"):
            if hasattr(arg, "data_ptr"):
                # PyTorch tensor
                addr = arg.data_ptr() & 0xFFFFFFFF  # Truncate to 32-bit
            elif hasattr(arg, "__cuda_array_interface__"):
                addr = arg.__cuda_array_interface__["data"][0] & 0xFFFFFFFF
            elif isinstance(arg, int):
                addr = arg & 0xFFFFFFFF
            else:
                addr = 0
            packed_parts.append(struct.pack("<I", addr))
        else:
            packed_parts.append(struct.pack("<" + fmt, arg))
    
    return b"".join(packed_parts)


class _CustomRuntimeProxy:
    """Lazy import wrapper for TRITON_CUSTOM_RUNTIME_MODULE."""

    def __init__(self) -> None:
        self._mod = None

    def _get(self):
        if self._mod is not None:
            return self._mod
        mod_name = os.environ.get("TRITON_CUSTOM_RUNTIME_MODULE")
        if not mod_name:
            return None
        self._mod = importlib.import_module(mod_name)
        return self._mod

    def load_binary(self, *, name: str, blob: bytes, shared: int, device: int):
        mod = self._get()
        if mod is None or not hasattr(mod, "load_binary"):
            raise RuntimeError(
                "Custom runtime not configured. Set TRITON_CUSTOM_RUNTIME_MODULE to a module "
                "that implements load_binary(name, blob, shared, device)."
            )
        return mod.load_binary(name=name, blob=blob, shared=shared, device=device)

    def launch(self, *, gridX: int, gridY: int, gridZ: int, stream: int, function: Any, packed_metadata: bytes,
               launch_metadata: Any, enter_hook: Any, exit_hook: Any, args: tuple[Any, ...],
               signature: dict = None):
        """Launch a kernel with expanded arguments ABI.
        
        Args:
            gridX/Y/Z: Grid dimensions
            stream: Execution stream handle
            function: Kernel function handle from load_binary
            packed_metadata: Triton-packed metadata bytes
            launch_metadata: Additional launch metadata
            enter_hook/exit_hook: Optional profiling hooks
            args: Expanded kernel arguments
            signature: Type signature dict (arg_index -> type_string)
        """
        mod = self._get()
        if mod is None or not hasattr(mod, "launch"):
            raise RuntimeError(
                "Custom runtime not configured. Set TRITON_CUSTOM_RUNTIME_MODULE to a module "
                "that implements launch(...)."
            )
        return mod.launch(gridX, gridY, gridZ, stream, function, packed_metadata, launch_metadata, enter_hook,
                          exit_hook, args, signature or {})


class CustomUtils:
    def __init__(self) -> None:
        self._rt = _CustomRuntimeProxy()

    def load_binary(self, name: str, blob: bytes, shared: int, device: int):
        # Required by CompiledKernel._init_handles.
        return self._rt.load_binary(name=name, blob=blob, shared=shared, device=device)


class CustomLauncher:
    """Launcher for custom backend kernels.
    
    Implements the expanded arguments ABI as specified in docs/custom-kernel-abi.md.
    """
    
    def __init__(self, utils: CustomUtils, src, metadata):
        self._utils = utils
        self._src = src
        self._metadata = metadata
        # Extract signature for parameter packing
        self._signature = metadata.get("signature", {})

    def __call__(self, gridX: int, gridY: int, gridZ: int, stream: int, function: Any, packed_metadata: bytes,
                 launch_metadata: Any, enter_hook: Any, exit_hook: Any, *args: Any):
        """Launch the kernel with expanded arguments.
        
        The args are passed directly to the runtime module's launch function.
        The runtime is responsible for:
        1. Building the parameter block from args
        2. Setting up the kernel descriptor (start_pc, grid_size, block_size, param_block_addr)
        3. Triggering execution via tsync.launch or equivalent
        
        See docs/custom-kernel-abi.md for the complete ABI specification.
        """
        # Delegate launch to the custom runtime module.
        return self._utils._rt.launch(
            gridX=gridX,
            gridY=gridY,
            gridZ=gridZ,
            stream=stream,
            function=function,
            packed_metadata=packed_metadata,
            launch_metadata=launch_metadata,
            enter_hook=enter_hook,
            exit_hook=exit_hook,
            args=args,
            signature=self._signature,  # Pass signature for ABI-compliant packing
        )


class CustomDriver(DriverBase):
    @staticmethod
    def is_active():
        # Opt-in only. If you enable this on a machine that also has CUDA/HIP
        # available, you must ensure only one driver reports active.
        return os.environ.get("TRITON_CUSTOM_ACTIVE", "") in ("1", "ON", "YES", "TRUE")

    def __init__(self) -> None:
        self.utils = CustomUtils()
        # launcher_cls is called as launcher_cls(src, metadata)
        self.launcher_cls = lambda src, metadata: CustomLauncher(self.utils, src, metadata)

    # --- Required DriverBase API ---

    def map_python_to_cpp_type(self, ty: str) -> str:
        # For the custom runtime we don't generate a C launcher by default.
        # Keep this conservative.
        return ty

    def get_current_target(self):
        # Default target shape for compilation/runtime metadata.
        # (We currently reuse NVIDIA lowering for IR generation.)
        arch = int(os.environ.get("TRITON_CUSTOM_ARCH", "70"))
        warp = int(os.environ.get("TRITON_CUSTOM_WARP_SIZE", "32"))
        return GPUTarget("custom", arch, warp)

    def get_active_torch_device(self):
        # Optional: return a torch.device-like object if torch is available.
        try:
            import torch

            return torch.device("cpu")
        except Exception:
            return None

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    # --- Runtime conveniences expected by Triton ---

    def get_current_device(self) -> int:
        return int(os.environ.get("TRITON_CUSTOM_DEVICE", "0"))

    def set_current_device(self, idx: int) -> None:
        os.environ["TRITON_CUSTOM_DEVICE"] = str(idx)

    def get_current_stream(self, idx: int) -> int:
        # Stream is an opaque handle to the custom runtime.
        return 0
