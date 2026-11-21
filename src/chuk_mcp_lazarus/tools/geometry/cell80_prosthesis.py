"""Cell80 circuit-prosthesis experiment.

Executes a Cell80 cell, ablates selected FFN outputs, and injects a
representation of the exact result into the same forward pass. The returned
causal ladder includes baseline, ablated/sham, correct-result, and
corrupt-result arms. No weights are updated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...errors import ToolError, make_error
from ...model_state import ModelState
from ...server import mcp

logger = logging.getLogger(__name__)
TOOL_NAME = "cell80_circuit_prosthesis"


def _single_token_id(tokenizer: Any, text: str) -> int:
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        direct = tokenizer.convert_tokens_to_ids(text)
        unk = getattr(tokenizer, "unk_token_id", None)
        if (
            isinstance(direct, int)
            and direct >= 0
            and direct != unk
            and tokenizer.decode([direct]).strip() == text.strip()
        ):
            return direct
    ids = tokenizer.encode(text, add_special_tokens=False)
    semantic_ids = [
        int(token_id)
        for token_id in ids
        if tokenizer.decode([int(token_id)]).strip()
    ]
    if len(semantic_ids) != 1:
        raise ValueError(
            f"Result representation requires one token; {text!r} encoded as {ids}."
        )
    return semantic_ids[0]


def _find_cell_source(cells_dir: Path, cell_name: str) -> Path:
    if not cell_name or any(ch in cell_name for ch in ("/", "\\", "..")):
        raise ValueError(f"Invalid cell name {cell_name!r}.")
    matches = sorted(cells_dir.rglob(f"{cell_name}.rs"))
    if not matches:
        raise FileNotFoundError(f"No {cell_name}.rs beneath {cells_dir}.")
    if len(matches) > 1:
        raise ValueError(
            f"Cell name {cell_name!r} is ambiguous beneath {cells_dir}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _execute_cell80(
    cell80_binary: Path,
    cells_dir: Path,
    cell_name: str,
    cell_args: list[int],
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run one cell through the production Cell80 CLI JSON contract."""
    if not cell80_binary.is_file():
        raise FileNotFoundError(cell80_binary)
    if not cells_dir.is_dir():
        raise FileNotFoundError(cells_dir)
    source = _find_cell_source(cells_dir, cell_name)
    command = [
        str(cell80_binary),
        "run",
        str(source),
        "--args",
        ",".join(str(int(arg)) for arg in cell_args),
        "--json",
    ]
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    elapsed_ns = time.perf_counter_ns() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cell80 exited {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Cell80 returned invalid JSON: {completed.stdout!r}"
        ) from exc
    payload["cell"] = cell_name
    payload["args"] = [int(arg) for arg in cell_args]
    payload["source"] = str(source)
    payload["elapsed_ns"] = elapsed_ns
    return payload


def _normal_forward(layer: Any, h: Any, mask: Any) -> Any:
    try:
        output = layer(h, mask=mask, cache=None)
    except TypeError:
        try:
            output = layer(h, cache=None)
        except TypeError:
            output = layer(h)
    if hasattr(output, "hidden_states"):
        return output.hidden_states
    if isinstance(output, tuple):
        return output[0]
    return output


def _ablate_ffn_forward(layer: Any, h: Any, mask: Any, neurons: set[int] | None) -> Any:
    """Run a block with all or selected FFN intermediate neurons zeroed."""
    import mlx.core as mx
    from ..._residual_helpers import _has_four_norms, _has_sublayers

    if not _has_sublayers(layer):
        raise ValueError("FFN ablation requires a transformer block with named sublayers.")

    normed = layer.input_layernorm(h)
    try:
        attn_out, _ = layer.self_attn(normed, mask=mask, cache=None)
    except TypeError:
        try:
            attn_out, _ = layer.self_attn(normed, cache=None)
        except (TypeError, ValueError):
            result = layer.self_attn(normed)
            attn_out = result[0] if isinstance(result, tuple) else result

    if hasattr(layer, "dropout") and layer.dropout is not None:
        attn_out = layer.dropout(attn_out)

    if _has_four_norms(layer):
        h_post_attn = h + layer.post_attention_layernorm(attn_out)
        ffn_input = layer.pre_feedforward_layernorm(h_post_attn)
    else:
        h_post_attn = h + attn_out
        ffn_input = layer.post_attention_layernorm(h_post_attn)

    if neurons is None:
        return h_post_attn

    mlp = layer.mlp
    required = ("gate_proj", "up_proj", "down_proj", "activation")
    if not all(hasattr(mlp, name) for name in required):
        raise ValueError(
            "Selected-neuron ablation requires a GLU/SwiGLU MLP with "
            "gate_proj, up_proj, down_proj, and activation."
        )
    intermediate = mlp.activation(mlp.gate_proj(ffn_input)) * mlp.up_proj(ffn_input)
    width = int(intermediate.shape[-1])
    invalid = sorted(index for index in neurons if index < 0 or index >= width)
    if invalid:
        raise ValueError(f"FFN neuron indices out of range 0..{width - 1}: {invalid}")
    neuron_mask = mx.ones((width,), dtype=intermediate.dtype)
    indices = mx.arange(width)
    for index in sorted(neurons):
        neuron_mask = mx.where(indices == index, 0, neuron_mask)
    ffn_out = mlp.down_proj(intermediate * neuron_mask)

    if _has_four_norms(layer):
        ffn_out = layer.post_feedforward_layernorm(ffn_out)
    elif hasattr(layer, "dropout") and layer.dropout is not None:
        ffn_out = layer.dropout(ffn_out)
    return h_post_attn + ffn_out


def _splice_direction(
    h: Any,
    position: int,
    direction: Any,
    coefficient: float,
    injection_mode: str,
) -> tuple[Any, float, float]:
    import mlx.core as mx

    seq_len = h.shape[1]
    pos = position if position >= 0 else seq_len + position
    pos = max(0, min(pos, seq_len - 1))
    unit = direction / mx.maximum(mx.linalg.norm(direction), 1e-12)
    current = h[0, pos, :]
    before_component = float(mx.sum(current * unit).item())
    if injection_mode == "set":
        delta = (coefficient - mx.sum(current * unit)) * unit
    else:
        delta = coefficient * unit
    injected = current + delta.astype(current.dtype)
    output = mx.concatenate(
        [
            h[:, :pos, :],
            injected.reshape(1, 1, -1),
            h[:, pos + 1 :, :],
        ],
        axis=1,
    )
    return output, before_component, float(mx.linalg.norm(delta).item())


def _run_composed_forward(
    model: Any,
    config: Any,
    input_ids: Any,
    ablate_layers: set[int],
    ablate_neurons: dict[int, set[int]],
    injection_layer: int,
    direction: Any | None,
    coefficient: float,
    injection_mode: str,
    token_position: int,
) -> tuple[Any, dict[str, Any]]:
    """One forward pass containing both ablation and optional injection."""
    import mlx.core as mx
    import mlx.nn as nn
    from chuk_lazarus.introspection.hooks import ModelHooks

    helper = ModelHooks(model, model_config=config)
    layers = helper._get_layers()
    embed = helper._get_embed_tokens()
    scale = helper._get_embedding_scale()
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    h = embed(input_ids)
    if scale is not None:
        h = h * mx.array(scale, dtype=h.dtype)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1])
    mask = mask.astype(h.dtype)

    trace: dict[str, Any] = {
        "ablate_layers": sorted(ablate_layers),
        "ablate_neurons": {
            str(layer): sorted(neurons)
            for layer, neurons in sorted(ablate_neurons.items())
        },
        "ablated_neuron_count": sum(len(neurons) for neurons in ablate_neurons.values()),
        "injection_layer": injection_layer,
        "injected": direction is not None,
        "injection_mode": injection_mode,
        "coefficient": coefficient if direction is not None else 0.0,
    }
    for layer_index, layer in enumerate(layers):
        if layer_index in ablate_layers:
            h = _ablate_ffn_forward(layer, h, mask, None)
        elif layer_index in ablate_neurons:
            h = _ablate_ffn_forward(layer, h, mask, ablate_neurons[layer_index])
        else:
            h = _normal_forward(layer, h, mask)
        if layer_index == injection_layer and direction is not None:
            h, before, delta_norm = _splice_direction(
                h,
                token_position,
                direction.astype(h.dtype),
                coefficient,
                injection_mode,
            )
            trace["component_before"] = before
            trace["delta_norm"] = delta_norm
    mx.eval(h)
    return h, trace


def _project_logits(model: Any, config: Any, hidden: Any, position: int) -> np.ndarray:
    import mlx.core as mx
    from chuk_lazarus.introspection.hooks import ModelHooks
    from ..._residual_helpers import _extract_position, _get_lm_projection, _norm_project

    helper = ModelHooks(model, model_config=config)
    final_norm = helper._get_final_norm()
    lm_head = _get_lm_projection(model)
    vector = _extract_position(hidden, position)
    logits = _norm_project(final_norm, lm_head, vector)
    mx.eval(logits)
    return np.asarray(logits, dtype=np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def _prediction_summary(
    logits: np.ndarray,
    tokenizer: Any,
    top_k: int,
    candidate_ids: list[int],
) -> dict[str, Any]:
    probs = _softmax(logits)
    order = np.argsort(-probs)[:top_k]
    raw = [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "probability": round(float(probs[token_id]), 8),
            "logit": round(float(logits[token_id]), 6),
        }
        for token_id in order
    ]
    candidate_logits = logits[candidate_ids]
    candidate_probs = _softmax(candidate_logits)
    candidate_order = np.argsort(-candidate_probs)
    candidates = [
        {
            "token_id": candidate_ids[int(index)],
            "token": tokenizer.decode([candidate_ids[int(index)]]),
            "probability": round(float(candidate_probs[index]), 8),
            "logit": round(float(candidate_logits[index]), 6),
        }
        for index in candidate_order
    ]
    return {
        "raw_top_k": raw,
        "raw_top1_id": raw[0]["token_id"],
        "raw_top1": raw[0]["token"],
        "candidate_ranking": candidates,
        "candidate_top1_id": candidates[0]["token_id"],
        "candidate_top1": candidates[0]["token"],
    }


def _direction_for_token(model: Any, token_id: int) -> Any:
    import mlx.core as mx
    from ..._residual_helpers import _get_unembed_vector

    direction = _get_unembed_vector(model, token_id)
    if direction is None:
        raise ValueError(f"Cannot access unembedding direction for token ID {token_id}.")
    mx.eval(direction)
    return direction


def _direction_for_candidate(
    model: Any,
    token_id: int,
    candidate_ids: list[int],
    direction_mode: str,
) -> Any:
    """Return a token row or its dual vector within the candidate subspace."""
    import mlx.core as mx

    if direction_mode == "token":
        return _direction_for_token(model, token_id)

    rows = np.stack(
        [
            np.asarray(_direction_for_token(model, candidate_id), dtype=np.float64)
            for candidate_id in candidate_ids
        ]
    )
    target_index = candidate_ids.index(token_id)
    desired = np.zeros((len(candidate_ids),), dtype=np.float64)
    desired[target_index] = 1.0
    gram = rows @ rows.T
    coefficients = np.linalg.solve(
        gram + np.eye(len(candidate_ids), dtype=np.float64) * 1e-8,
        desired,
    )
    dual = rows.T @ coefficients
    return mx.array(dual.astype(np.float32))


def _cell80_circuit_prosthesis_impl(
    model: Any,
    config: Any,
    tokenizer: Any,
    metadata: Any,
    prompt: str,
    cell_name: str,
    cell_args: list[int],
    cell80_binary: str,
    cells_dir: str,
    ablate_layers: list[int],
    ablate_neurons: dict[int, set[int]],
    injection_layer: int,
    coefficient: float,
    injection_mode: str,
    direction_mode: str,
    corrupt_result: int | None,
    result_token_text: str | None,
    corrupt_token_text: str | None,
    candidate_tokens: list[str],
    top_k: int,
    token_position: int,
    pre_result_token_ids: list[int],
) -> dict[str, Any]:
    import mlx.core as mx

    call = _execute_cell80(
        Path(cell80_binary),
        Path(cells_dir),
        cell_name,
        cell_args,
    )
    if str(call.get("halt", "")).lower() != "returned":
        return {
            "prompt": prompt,
            "cell80": call,
            "injection_performed": False,
            "reason": f"Cell80 did not return normally: {call.get('halt')}",
        }
    result = int(call["result"])
    if corrupt_result is None:
        corrupt_result = (result + 1) % 10
    if corrupt_result == result:
        raise ValueError("corrupt_result must differ from the Cell80 result.")

    result_text = result_token_text or str(result)
    corrupt_text = corrupt_token_text or str(corrupt_result)
    result_id = _single_token_id(tokenizer, result_text)
    corrupt_id = _single_token_id(tokenizer, corrupt_text)
    candidate_ids = [_single_token_id(tokenizer, token) for token in candidate_tokens]
    candidate_ids = list(dict.fromkeys(candidate_ids))
    if result_id not in candidate_ids or corrupt_id not in candidate_ids:
        raise ValueError("candidate_tokens must contain both correct and corrupt results.")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    invalid_prefix = [
        token_id
        for token_id in pre_result_token_ids
        if token_id < 0 or token_id >= metadata.vocab_size
    ]
    if invalid_prefix:
        raise ValueError(
            f"pre_result_token_ids outside vocabulary 0..{metadata.vocab_size - 1}: "
            f"{invalid_prefix}"
        )
    input_ids = mx.array(prompt_ids + pre_result_token_ids)
    layer_set = set(ablate_layers)
    arms: dict[str, Any] = {}
    arm_logits: dict[str, np.ndarray] = {}

    def run_arm(
        name: str,
        ablations: set[int],
        neurons: dict[int, set[int]],
        direction: Any | None,
    ) -> None:
        started = time.perf_counter_ns()
        hidden, trace = _run_composed_forward(
            model,
            config,
            input_ids,
            ablations,
            neurons,
            injection_layer,
            direction,
            coefficient,
            injection_mode,
            token_position,
        )
        logits = _project_logits(model, config, hidden, token_position)
        arm_logits[name] = logits
        elapsed_ns = time.perf_counter_ns() - started
        arms[name] = {
            "trace": trace,
            "elapsed_ns": elapsed_ns,
            **_prediction_summary(logits, tokenizer, top_k, candidate_ids),
        }

    run_arm("baseline", set(), {}, None)
    run_arm("ablated", layer_set, ablate_neurons, None)
    # Sham deliberately repeats the exact ablated computation after a
    # successful Cell80 call but performs no residual write.
    run_arm("sham", layer_set, ablate_neurons, None)
    run_arm(
        "correct",
        layer_set,
        ablate_neurons,
        _direction_for_candidate(model, result_id, candidate_ids, direction_mode),
    )
    run_arm(
        "corrupt",
        layer_set,
        ablate_neurons,
        _direction_for_candidate(model, corrupt_id, candidate_ids, direction_mode),
    )

    result_payload = {
        "prompt": prompt,
        "num_tokens": int(input_ids.shape[-1]),
        "pre_result_token_ids": pre_result_token_ids,
        "training_performed": False,
        "cell80": call,
        "ablate_component": "ffn",
        "ablate_layers": sorted(layer_set),
        "ablate_neurons": {
            str(layer): sorted(neurons)
            for layer, neurons in sorted(ablate_neurons.items())
        },
        "injection_layer": injection_layer,
        "injection_mode": injection_mode,
        "direction_mode": direction_mode,
        "coefficient": coefficient,
        "token_position": token_position,
        "result": result,
        "result_token_text": result_text,
        "result_token_id": result_id,
        "corrupt_result": corrupt_result,
        "corrupt_token_text": corrupt_text,
        "corrupt_token_id": corrupt_id,
        "candidate_tokens": candidate_tokens,
        "candidate_token_ids": candidate_ids,
        "arms": arms,
    }
    result_payload["summary"] = {
        "baseline_correct": arms["baseline"]["candidate_top1_id"] == result_id,
        "ablation_damaged": arms["ablated"]["candidate_top1_id"] != result_id,
        "correct_recovered": arms["correct"]["candidate_top1_id"] == result_id,
        "corrupt_control_followed": arms["corrupt"]["candidate_top1_id"] == corrupt_id,
        "sham_logits_exact": bool(
            np.array_equal(arm_logits["sham"], arm_logits["ablated"])
        ),
        "sham_max_abs_logit_delta": float(
            np.max(np.abs(arm_logits["sham"] - arm_logits["ablated"]))
        ),
    }
    return result_payload


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def cell80_circuit_prosthesis(
    prompt: str,
    cell_name: str,
    cell_args: list[int],
    cell80_binary: str,
    cells_dir: str,
    ablate_layers: list[int],
    injection_layer: int,
    coefficient: float,
    injection_mode: str = "add",
    direction_mode: str = "candidate_dual",
    corrupt_result: int | None = None,
    result_token_text: str | None = None,
    corrupt_token_text: str | None = None,
    candidate_tokens: list[str] | None = None,
    ablate_neurons: list[dict[str, Any]] | None = None,
    pre_result_token_ids: list[int] | None = None,
    top_k: int = 10,
    token_position: int = -1,
) -> dict[str, Any]:
    """Ablate FFNs and inject a Cell80-computed result in one forward pass.

    The causal ladder is baseline -> ablated/sham -> correct-result injection
    -> corrupt-result injection. Cell80 is executed by its production CLI,
    and model parameters are never updated.

    Current result injection requires a single tokenizer token, making this
    tool suitable for controlled one-token arithmetic experiments.
    """
    state = ModelState.get()
    if not state.is_loaded:
        return make_error(
            ToolError.MODEL_NOT_LOADED,
            "Call load_model() first.",
            TOOL_NAME,
        )
    if not prompt:
        return make_error(ToolError.INVALID_INPUT, "prompt must not be empty.", TOOL_NAME)
    if not ablate_layers and not ablate_neurons:
        return make_error(
            ToolError.INVALID_INPUT,
            "Provide at least one full FFN layer or selected FFN neuron to ablate.",
            TOOL_NAME,
        )
    num_layers = state.metadata.num_layers
    parsed_neurons: dict[int, set[int]] = {}
    try:
        for spec in ablate_neurons or []:
            layer = int(spec["layer"])
            neurons = {int(index) for index in spec["neurons"]}
            if not neurons:
                raise ValueError(f"Neuron list for layer {layer} is empty.")
            parsed_neurons.setdefault(layer, set()).update(neurons)
    except (KeyError, TypeError, ValueError) as exc:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Invalid ablate_neurons specification: {exc}",
            TOOL_NAME,
        )
    all_ablation_layers = list(ablate_layers) + list(parsed_neurons)
    invalid = [layer for layer in all_ablation_layers if layer < 0 or layer >= num_layers]
    if invalid:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"Invalid ablation layers {invalid}; model has {num_layers} layers.",
            TOOL_NAME,
        )
    intermediate_size = state.metadata.intermediate_size
    invalid_neurons = sorted(
        {
            neuron
            for neurons in parsed_neurons.values()
            for neuron in neurons
            if neuron < 0 or neuron >= intermediate_size
        }
    )
    if invalid_neurons:
        return make_error(
            ToolError.INVALID_INPUT,
            f"Neuron indices out of range 0..{intermediate_size - 1}: {invalid_neurons}",
            TOOL_NAME,
        )
    if injection_layer < 0 or injection_layer >= num_layers:
        return make_error(
            ToolError.LAYER_OUT_OF_RANGE,
            f"injection_layer must be 0..{num_layers - 1}.",
            TOOL_NAME,
        )
    if injection_mode not in ("add", "set"):
        return make_error(
            ToolError.INVALID_INPUT,
            "injection_mode must be 'add' or 'set'.",
            TOOL_NAME,
        )
    if direction_mode not in ("token", "candidate_dual"):
        return make_error(
            ToolError.INVALID_INPUT,
            "direction_mode must be 'token' or 'candidate_dual'.",
            TOOL_NAME,
        )
    if coefficient < 0 or not np.isfinite(coefficient):
        return make_error(
            ToolError.INVALID_INPUT,
            "coefficient must be finite and non-negative.",
            TOOL_NAME,
        )
    if top_k < 1 or top_k > 50:
        return make_error(
            ToolError.INVALID_INPUT,
            "top_k must be between 1 and 50.",
            TOOL_NAME,
        )
    candidates = candidate_tokens or [str(value) for value in range(10)]
    try:
        return await asyncio.to_thread(
            _cell80_circuit_prosthesis_impl,
            state.model,
            state.config,
            state.tokenizer,
            state.metadata,
            prompt,
            cell_name,
            [int(arg) for arg in cell_args],
            cell80_binary,
            cells_dir,
            sorted(set(int(layer) for layer in ablate_layers)),
            parsed_neurons,
            injection_layer,
            float(coefficient),
            injection_mode,
            direction_mode,
            corrupt_result,
            result_token_text,
            corrupt_token_text,
            candidates,
            top_k,
            token_position,
            [int(token_id) for token_id in (pre_result_token_ids or [])],
        )
    except (ValueError, FileNotFoundError) as exc:
        return make_error(ToolError.INVALID_INPUT, str(exc), TOOL_NAME)
    except Exception as exc:
        logger.exception("%s failed", TOOL_NAME)
        return make_error(ToolError.INTERVENTION_FAILED, str(exc), TOOL_NAME)
