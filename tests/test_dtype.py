"""Tests for weight-dtype resolution/enforcement and the fp16 RMSNorm patch."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest

from chuk_mcp_lazarus import _dtype
from chuk_mcp_lazarus._bootstrap import patch_fp16_rmsnorm
from chuk_mcp_lazarus._dtype import cast_model, is_fp16_unsafe, read_model_type, resolve_dtype
from chuk_mcp_lazarus.model_state import ModelState, WeightDType
from chuk_mcp_lazarus.tools.model_tools import load_model

D = WeightDType


class TestResolveDtype:
    @pytest.mark.parametrize("mt", ["gemma", "gemma3", "gemma3_text", "Gemma2"])
    def test_gemma_auto_is_bf16_without_note(self, mt: str) -> None:
        assert resolve_dtype(D.AUTO, mt, native_bf16=False) == (D.BFLOAT16, None)
        assert resolve_dtype(D.BFLOAT16, mt, native_bf16=False) == (D.BFLOAT16, None)

    @pytest.mark.parametrize("req", [D.FLOAT16, D.FLOAT32])
    def test_gemma_override_has_note(self, req: WeightDType) -> None:
        dtype, note = resolve_dtype(req, "gemma3", native_bf16=False)
        assert dtype is D.BFLOAT16
        assert note and req.value in note and "bfloat16" in note

    def test_auto_native_bf16(self) -> None:
        assert resolve_dtype(D.AUTO, "llama", native_bf16=True) == (D.BFLOAT16, None)

    def test_auto_pre_ampere_is_fp16(self) -> None:
        assert resolve_dtype(D.AUTO, "llama", native_bf16=False) == (D.FLOAT16, None)
        assert resolve_dtype(D.AUTO, None, native_bf16=False) == (D.FLOAT16, None)

    @pytest.mark.parametrize("req", [D.BFLOAT16, D.FLOAT16, D.FLOAT32])
    def test_explicit_non_gemma_is_kept(self, req: WeightDType) -> None:
        assert resolve_dtype(req, "llama", native_bf16=False) == (req, None)

    def test_auto_queries_device_when_unspecified(self) -> None:
        with patch.object(_dtype, "device_has_native_bf16", return_value=False):
            assert resolve_dtype(D.AUTO, "qwen2")[0] is D.FLOAT16

    def test_is_fp16_unsafe(self) -> None:
        assert is_fp16_unsafe("gemma3_text")
        assert not is_fp16_unsafe("llama")
        assert not is_fp16_unsafe(None)


class TestDeviceHasNativeBf16:
    def _cuda(self, available: bool) -> types.SimpleNamespace:
        return types.SimpleNamespace(is_available=lambda: available)

    @pytest.mark.parametrize("major,expected", [(7, False), (8, True), (9, True)])
    def test_cuda_compute_capability(self, major: int, expected: bool) -> None:
        with (
            patch.object(mx.metal, "is_available", lambda: False, create=True),
            patch.object(mx, "cuda", self._cuda(True), create=True),
            patch.object(
                mx, "device_info", lambda: {"compute_capability_major": major}, create=True
            ),
        ):
            assert _dtype.device_has_native_bf16() is expected

    def test_metal(self) -> None:
        with patch.object(mx.metal, "is_available", lambda: True, create=True):
            assert _dtype.device_has_native_bf16() is True

    def test_cpu_only(self) -> None:
        with (
            patch.object(mx.metal, "is_available", lambda: False, create=True),
            patch.object(mx, "cuda", self._cuda(False), create=True),
        ):
            assert _dtype.device_has_native_bf16() is True


class TestReadModelType:
    def test_local_dir(self, tmp_path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma3"}))
        assert read_model_type(str(tmp_path)) == "gemma3"

    def test_local_dir_without_config(self, tmp_path) -> None:
        assert read_model_type(str(tmp_path)) is None

    def test_hub_unreachable(self) -> None:
        # conftest blocks hf_hub_download
        assert read_model_type("org/model") is None

    def test_hub(self, tmp_path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"model_type": "llama"}))
        with patch("huggingface_hub.hf_hub_download", return_value=str(cfg)):
            assert read_model_type("org/model") == "llama"


class TestCastModel:
    def _model(self, *dtypes: str) -> MagicMock:
        arrays = {f"w{i}": mx.array(np.zeros(2), dtype=dt) for i, dt in enumerate(dtypes)}
        module = MagicMock()
        module.items.return_value = list(arrays.items())
        model = MagicMock()
        model.named_modules.return_value = [("layer", module)]
        return model, module

    def test_casts_mismatched_float_params(self) -> None:
        model, module = self._model("bfloat16", "float16")
        assert cast_model(model, D.FLOAT16) is True
        update = module.update.call_args[0][0]
        assert list(update) == ["w0"]
        assert update["w0"].dtype == "float16"

    def test_noop_when_already_target(self) -> None:
        model, module = self._model("float16")
        assert cast_model(model, D.FLOAT16) is False
        module.update.assert_not_called()

    def test_skips_non_float(self) -> None:
        model, module = self._model("uint32")
        assert cast_model(model, D.FLOAT32) is False
        module.update.assert_not_called()


class TestLoaderIntegration:
    def _pipeline(self) -> MagicMock:
        pipeline = MagicMock()
        cfg = pipeline.config
        cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads = 2, 8, 2
        cfg.num_key_value_heads, cfg.vocab_size, cfg.intermediate_size = 2, 10, 16
        cfg.max_position_embeddings, cfg.head_dim, cfg.model_type = 32, 4, "llama"
        cfg.num_local_experts = None
        pipeline.family.family_type.value = "llama"
        pipeline.model.parameters.return_value = {}
        pipeline.model.named_modules.return_value = []
        return pipeline

    def test_auto_on_pre_ampere_loads_fp16(self) -> None:
        with (
            patch("chuk_lazarus.inference.UnifiedPipeline.from_pretrained") as fp,
            patch.object(_dtype, "read_model_type", return_value="llama"),
            patch.object(_dtype, "device_has_native_bf16", return_value=False),
        ):
            fp.return_value = self._pipeline()
            md = ModelState.get().load("test/dtype-auto", D.AUTO)
        assert md.dtype == "float16" and md.dtype_note is None
        assert fp.call_args.kwargs["pipeline_config"].dtype.value == "float16"

    def test_gemma_fp16_request_reports_override(self) -> None:
        with (
            patch("chuk_lazarus.inference.UnifiedPipeline.from_pretrained") as fp,
            patch.object(_dtype, "read_model_type", return_value="gemma3"),
        ):
            fp.return_value = self._pipeline()
            md = ModelState.get().load("test/dtype-gemma", D.FLOAT16)
        assert md.dtype == "bfloat16"
        assert md.dtype_note and "float16" in md.dtype_note


class TestLoadModelTool:
    async def test_invalid_dtype_lists_auto(self) -> None:
        result = await load_model(model_id="test", dtype="half")
        assert result["error"] is True
        assert "auto" in result["message"]


class TestFp16RmsNormPatch:
    @pytest.fixture
    def rmsnorm_cls(self):
        calls: list[str] = []

        class RMSNorm:
            def __init__(self) -> None:
                self.weight = mx.array(np.ones(4), dtype="float32")
                self.eps = 1e-6

            def __call__(self, x):
                calls.append("original")
                return x

        names = [
            "chuk_lazarus.models_v2",
            "chuk_lazarus.models_v2.components",
            "chuk_lazarus.models_v2.components.normalization",
            "chuk_lazarus.models_v2.components.normalization.rmsnorm",
        ]
        mods = {n: types.ModuleType(n) for n in names}
        mods[names[-1]].RMSNorm = RMSNorm
        with patch.dict(sys.modules, mods):
            yield RMSNorm, calls

    def test_fp16_uses_fused_and_others_keep_original(self, rmsnorm_cls) -> None:
        RMSNorm, calls = rmsnorm_cls
        fused = MagicMock(side_effect=lambda x, w, eps: ("fused", w.dtype, eps))
        with patch.object(mx.fast, "rms_norm", fused, create=True):
            patch_fp16_rmsnorm()
            norm = RMSNorm()
            out = norm(mx.array(np.ones(4), dtype=mx.float16))
            assert out == ("fused", mx.float16, 1e-6)
            norm(mx.array(np.ones(4), dtype=mx.bfloat16))
            norm(mx.array(np.ones(4), dtype=mx.float32))
        assert calls == ["original", "original"]

    def test_idempotent(self, rmsnorm_cls) -> None:
        RMSNorm, _ = rmsnorm_cls
        patch_fp16_rmsnorm()
        first = RMSNorm.__call__
        patch_fp16_rmsnorm()
        assert RMSNorm.__call__ is first

    def test_missing_chuk_lazarus_is_noop(self) -> None:
        with patch.dict(sys.modules, {"chuk_lazarus.models_v2": None}):
            patch_fp16_rmsnorm()  # must not raise
