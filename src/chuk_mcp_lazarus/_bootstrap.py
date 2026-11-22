"""
Bootstrap for optional chuk-lazarus dependencies.

chuk-lazarus v0.4 has a hard import on chuk-virtual-expert which
may not be installed. This module installs a lightweight stub so
that the import chain succeeds. Only the virtual-expert features
are unavailable; everything else (inference, introspection, hooks)
works normally.

This module must be imported before any chuk_lazarus imports.
"""

from __future__ import annotations

import sys
import types


def ensure_optional_stubs() -> None:
    """Install stubs for optional dependencies that may be missing."""
    if "chuk_virtual_expert" not in sys.modules:
        try:
            import chuk_virtual_expert  # noqa: F401
        except ImportError:
            stub = types.ModuleType("chuk_virtual_expert")
            stub.__version__ = "0.0.0-stub"  # type: ignore[attr-defined]
            # Create minimal stand-in classes
            for name in (
                "VirtualExpert",
                "VirtualExpertResult",
                "VirtualExpertAction",
                "VirtualExpertPlugin",
                "VirtualExpertRegistry",
            ):
                setattr(stub, name, type(name, (), {}))
            sys.modules["chuk_virtual_expert"] = stub


def patch_fp16_rmsnorm() -> None:
    """Make chuk-lazarus RMSNorm safe for float16 inputs.

    Its RMSNorm computes ``mean(x * x)`` in the input dtype. In float16
    any |x| > ~256 squares past 65504 -> rms = inf -> the norm outputs
    zeros (layers become no-ops, logits all zero). ``mx.fast.rms_norm``
    accumulates in float32. bfloat16/float32 keep the original path so
    their numerics are unchanged.
    """
    try:
        import mlx.core as mx
        from chuk_lazarus.models_v2.components.normalization.rmsnorm import RMSNorm
    except ImportError:
        return
    if getattr(RMSNorm, "_fp16_safe", False):
        return
    original = RMSNorm.__call__

    def __call__(self, x):
        if x.dtype == mx.float16:
            return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)
        return original(self, x)

    RMSNorm.__call__ = __call__
    RMSNorm._fp16_safe = True


# Run on import
ensure_optional_stubs()
patch_fp16_rmsnorm()
