from __future__ import annotations

"""Custom SIMT backend scaffold.

This backend reuses the NVIDIA lowering pipeline up to LLVM IR generation, then
emits a Scheme-A "blob" artifact (flat bytes, no relocations) per the contract
in docs/custom-simt-target-spec.md.

Actual ISA codegen is intentionally delegated to an external tool (or a future
LLVM backend). Triton only needs the semantic contract and an artifact format.
"""

import functools
import os
import subprocess
from typing import Any

from triton.backends.compiler import GPUTarget, Language

# We intentionally reuse the NVIDIA backend implementation up to (and including)
# LLVM IR generation.
import triton.backends.nvidia.compiler as nvidia_compiler  # type: ignore

from .blob import build_meta_tlv, pack_blob_v1
from . import pipeline


class CustomBackend(nvidia_compiler.CUDABackend):
    """Backend name: 'custom'.

    Targets look like: GPUTarget('custom', 70, 32)

    Pipeline stages:
            TRITON:  ttir -> ttgir -> llir -> blob
            GLUON:         ttgir -> llir -> blob

    The final artifact is a Scheme-A blob (bytes). Runtime loading/launching is
    expected to be provided by a custom driver (or simulator).
    """

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "custom"

    def get_target_name(self, options) -> str:
        # Keep target name distinct for caching.
        capability = self._parse_arch(options.arch)
        return f"custom:{capability}"

    def add_stages(self, stages, options, language):
        capability = self._parse_arch(options.arch)

        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options, capability)

        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
        stages["blob"] = lambda src, metadata: self.make_blob(src, metadata, options, capability)

        # Preserve the stages inspection hook behavior.
        from triton import knobs

        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, capability)

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        # Ensure the final stage name matches the binary extension Triton uses.
        self.binary_ext = "blob"

    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        return pipeline.make_ttir(mod, metadata, opt, capability)

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        return pipeline.make_ttgir(mod, metadata, opt, capability)

    def make_llir(self, src, metadata, options: Any, capability: int) -> str:
        return pipeline.make_llir(src, metadata, options, capability)

    def make_blob(self, src: str, metadata: dict, options: Any, capability: int) -> bytes:
        """LLVM IR (string) -> Scheme-A blob (bytes).

        This method expects an external codegen tool to turn LLVM IR into target
        ISA bytes for the `.text` segment.

        Configure via environment:
        - TRITON_CUSTOM_CODEGEN: path to an executable
          * stdin: LLVM IR (text)
          * stdout: raw `.text` bytes
        """

        codegen = os.environ.get("TRITON_CUSTOM_CODEGEN")
        if not codegen:
            raise RuntimeError(
                "Custom backend requires TRITON_CUSTOM_CODEGEN to be set. "
                "It must point to a tool that reads LLVM IR from stdin and writes raw .text bytes to stdout."
            )

        try:
            proc = subprocess.run(
                [codegen],
                input=src.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"TRITON_CUSTOM_CODEGEN failed (exit={e.returncode}):\n{stderr}") from e

        text = proc.stdout
        if not text:
            raise RuntimeError("TRITON_CUSTOM_CODEGEN produced empty .text")

        warp_size = int(os.environ.get("TRITON_CUSTOM_WARP_SIZE", "32"))
        stack_per_warp = int(os.environ.get("TRITON_CUSTOM_STACK_PER_WARP_BYTES", "1024"))
        num_warps_hint = int(getattr(options, "num_warps", 0) or 0)

        # Minimal `.meta` records (TLV), per contract.
        # Types:
        # 1 ABI_TAG, 2 KERNEL_NAME, 3 STACK_PER_WARP_BYTES, 4 USES_CTA_BARRIER, 5 NUM_WARPS_HINT, 6 WARP_SIZE
        meta = build_meta_tlv(
            [
                (1, b"ilp32f"),
                (2, str(metadata.get("name", "")).encode("utf-8")),
                (3, (stack_per_warp).to_bytes(4, "little")),
                (4, (1).to_bytes(1, "little")),
                (5, (num_warps_hint).to_bytes(2, "little")),
                (6, (warp_size).to_bytes(2, "little")),
            ]
        )

        return pack_blob_v1(text=text, rodata=b"", meta=meta)

    @functools.lru_cache()
    def hash(self):
        # Custom backend doesn't need ptxas, use a simpler hash
        import hashlib
        key = f"custom-{self.target.arch}-{self.target.warp_size}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
