"""Tests for the Cell80 circuit-prosthesis MCP tool."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from chuk_mcp_lazarus.model_state import ModelState
from chuk_mcp_lazarus.tools.geometry.cell80_prosthesis import (
    _find_cell_source,
    _single_token_id,
    cell80_circuit_prosthesis,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {"0": [4], "1": [5], "12": [5, 6]}
        return mapping.get(text, [99])

    def decode(self, ids):
        mapping = {4: "0", 5: "1", 6: "2", 99: "x"}
        return "".join(mapping[token_id] for token_id in ids)


def reset_state():
    ModelState._instance = None


def test_single_token_id_accepts_one_and_rejects_many():
    tokenizer = Tokenizer()
    assert _single_token_id(tokenizer, "1") == 5
    with pytest.raises(ValueError, match="requires one token"):
        _single_token_id(tokenizer, "12")


def test_find_cell_source(tmp_path):
    source = tmp_path / "safe-arith" / "add_sat.rs"
    source.parent.mkdir()
    source.write_text("fn run(a:u16,b:u16)->u16{a+b}")
    assert _find_cell_source(tmp_path, "add_sat") == source


def test_find_cell_source_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="Invalid cell name"):
        _find_cell_source(tmp_path, "../secret")


@pytest.mark.asyncio
async def test_requires_loaded_model():
    reset_state()
    result = await cell80_circuit_prosthesis(
        prompt="1 + 1 = ",
        cell_name="add_sat",
        cell_args=[1, 1],
        cell80_binary="/tmp/cell80",
        cells_dir="/tmp/cells",
        ablate_layers=[0],
        injection_layer=1,
        coefficient=10,
    )
    assert result["error_type"].value == "ModelNotLoaded"


def loaded_state():
    reset_state()
    state = ModelState.get()
    state._state.loaded = True
    state._state.model = object()
    state._state.config = object()
    state._state.tokenizer = Tokenizer()
    state._state.metadata = SimpleNamespace(num_layers=4, intermediate_size=16)
    return state


@pytest.mark.asyncio
async def test_validates_layers():
    loaded_state()
    result = await cell80_circuit_prosthesis(
        prompt="1 + 1 = ",
        cell_name="add_sat",
        cell_args=[1, 1],
        cell80_binary="/tmp/cell80",
        cells_dir="/tmp/cells",
        ablate_layers=[4],
        injection_layer=1,
        coefficient=10,
    )
    assert result["error_type"].value == "LayerOutOfRange"


@pytest.mark.asyncio
async def test_validates_mode():
    loaded_state()
    result = await cell80_circuit_prosthesis(
        prompt="1 + 1 = ",
        cell_name="add_sat",
        cell_args=[1, 1],
        cell80_binary="/tmp/cell80",
        cells_dir="/tmp/cells",
        ablate_layers=[1],
        injection_layer=2,
        coefficient=10,
        injection_mode="replace",
    )
    assert result["error_type"].value == "InvalidInput"


@pytest.mark.asyncio
async def test_validates_selected_neuron_range():
    loaded_state()
    result = await cell80_circuit_prosthesis(
        prompt="1 + 1 = ",
        cell_name="add_sat",
        cell_args=[1, 1],
        cell80_binary="/tmp/cell80",
        cells_dir="/tmp/cells",
        ablate_layers=[],
        ablate_neurons=[{"layer": 1, "neurons": [16]}],
        injection_layer=2,
        coefficient=10,
    )
    assert result["error_type"].value == "InvalidInput"


@pytest.mark.asyncio
async def test_dispatches_valid_request():
    state = loaded_state()
    expected = {"summary": {"correct_recovered": True}}
    with patch(
        "chuk_mcp_lazarus.tools.geometry.cell80_prosthesis._cell80_circuit_prosthesis_impl",
        return_value=expected,
    ) as implementation:
        result = await cell80_circuit_prosthesis(
            prompt="1 + 1 = ",
            cell_name="add_sat",
            cell_args=[1, 1],
            cell80_binary="/tmp/cell80",
            cells_dir="/tmp/cells",
            ablate_layers=[1],
            injection_layer=2,
            coefficient=10,
            result_token_text="two",
            corrupt_token_text="zero",
        )
    assert result == expected
    assert implementation.call_args.args[0] is state.model
