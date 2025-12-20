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
"""

import importlib
import os
from typing import Any

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


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
               launch_metadata: Any, enter_hook: Any, exit_hook: Any, args: tuple[Any, ...]):
        mod = self._get()
        if mod is None or not hasattr(mod, "launch"):
            raise RuntimeError(
                "Custom runtime not configured. Set TRITON_CUSTOM_RUNTIME_MODULE to a module "
                "that implements launch(...)."
            )
        return mod.launch(gridX, gridY, gridZ, stream, function, packed_metadata, launch_metadata, enter_hook,
                          exit_hook, *args)


class CustomUtils:
    def __init__(self) -> None:
        self._rt = _CustomRuntimeProxy()

    def load_binary(self, name: str, blob: bytes, shared: int, device: int):
        # Required by CompiledKernel._init_handles.
        return self._rt.load_binary(name=name, blob=blob, shared=shared, device=device)


class CustomLauncher:
    def __init__(self, utils: CustomUtils, src, metadata):
        self._utils = utils
        self._src = src
        self._metadata = metadata

    def __call__(self, gridX: int, gridY: int, gridZ: int, stream: int, function: Any, packed_metadata: bytes,
                 launch_metadata: Any, enter_hook: Any, exit_hook: Any, *args: Any):
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
