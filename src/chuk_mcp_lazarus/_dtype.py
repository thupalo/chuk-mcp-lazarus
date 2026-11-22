"""
Weight-dtype resolution and enforcement.

Why this exists:

* ``auto`` picks the fastest *safe* dtype for the current device.
  bfloat16 is native on Apple Silicon (Metal) and NVIDIA SM >= 8.0
  (Ampere+). Older NVIDIA GPUs (e.g. Turing, SM 7.5) only emulate
  bfloat16 -- slower than float32 -- while float16 runs on tensor cores.
* Gemma-family models always stay in bfloat16: their residual stream
  exceeds the float16 range (|h| ~ 7e4 > 65504 on TranslateGemma 4B),
  so float16 changes predictions, and float32 inference is currently
  broken for Gemma in chuk-lazarus.
* chuk-lazarus ignores the requested dtype for models loaded through
  ``sanitize()`` (weights stay as stored); :func:`cast_model` enforces it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .model_state import WeightDType

logger = logging.getLogger(__name__)

# HF model_type prefixes whose activations overflow float16.
FP16_UNSAFE_MODEL_TYPES = ("gemma",)

_MLX_DTYPES = {
    WeightDType.BFLOAT16: "bfloat16",
    WeightDType.FLOAT16: "float16",
    WeightDType.FLOAT32: "float32",
}


def device_has_native_bf16() -> bool:
    """True on Metal and on CUDA devices with compute capability >= 8.0."""
    import mlx.core as mx

    try:
        if mx.metal.is_available():
            return True
    except Exception:  # pragma: no cover - mx.metal missing on some builds
        pass
    try:
        if mx.cuda.is_available():
            return int(mx.device_info().get("compute_capability_major", 0)) >= 8
    except Exception:  # pragma: no cover - defensive
        return True
    # CPU backend: no bf16 penalty worth avoiding; keep the historical default.
    return True


def read_model_type(model_id: str) -> str | None:
    """Return ``model_type`` from the model's config.json, if obtainable."""
    try:
        path = Path(model_id)
        if path.is_dir():
            cfg_file = path / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            cfg_file = Path(hf_hub_download(model_id, "config.json"))
        model_type = json.loads(cfg_file.read_text()).get("model_type")
        return str(model_type) if model_type else None
    except Exception as e:
        logger.debug("Could not read model_type for %s: %s", model_id, e)
        return None


def is_fp16_unsafe(model_type: str | None) -> bool:
    return bool(model_type) and str(model_type).lower().startswith(FP16_UNSAFE_MODEL_TYPES)


def resolve_dtype(
    requested: WeightDType, model_type: str | None, native_bf16: bool | None = None
) -> tuple[WeightDType, str | None]:
    """Pick the dtype to load with. Returns ``(dtype, note)``; note explains overrides."""
    if is_fp16_unsafe(model_type):
        if requested in (WeightDType.AUTO, WeightDType.BFLOAT16):
            return WeightDType.BFLOAT16, None
        return WeightDType.BFLOAT16, (
            f"{requested.value} is not supported for {model_type} models "
            "(float16 overflows the residual stream; float32 is broken in "
            "chuk-lazarus); loaded as bfloat16."
        )
    if requested is not WeightDType.AUTO:
        return requested, None
    if native_bf16 is None:
        native_bf16 = device_has_native_bf16()
    return (WeightDType.BFLOAT16 if native_bf16 else WeightDType.FLOAT16), None


def cast_model(model: Any, dtype: WeightDType) -> bool:
    """Cast floating-point parameters to ``dtype`` in place, module by module.

    Module-wise casting keeps peak memory near one copy of the model.
    Returns True if anything was cast.
    """
    import mlx.core as mx

    target = getattr(mx, _MLX_DTYPES[dtype])
    changed = False
    for _, module in model.named_modules():
        own = {
            k: v
            for k, v in module.items()
            if isinstance(v, mx.array) and mx.issubdtype(v.dtype, mx.floating) and v.dtype != target
        }
        if own:
            module.update({k: v.astype(target) for k, v in own.items()})
            mx.eval(module.parameters())
            changed = True
    if changed:
        mx.clear_cache()
    return changed
