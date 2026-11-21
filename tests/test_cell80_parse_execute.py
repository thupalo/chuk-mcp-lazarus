"""Tests for the fail-closed Cell80 parse/execute tool."""

from unittest.mock import patch

import pytest

from chuk_mcp_lazarus.errors import ToolError
from chuk_mcp_lazarus.tools.geometry.cell80_parse_execute import (
    DEFAULT_CELL_MAP,
    _parse_execute_impl,
    cell80_parse_execute,
    parse_prompt,
)


@pytest.mark.parametrize(
    ("prompt", "operation", "args"),
    [
        ("compute six plus five as a sum:", "add", [6, 5]),
        ("give what remains after subtracting four from ten:", "sub", [10, 4]),
        ("give the total across one groups containing six:", "mul", [1, 6]),
        ("give how many groups of one fit within four:", "div", [4, 1]),
        ("give the largest divisor common to ten and two:", "gcd", [10, 2]),
        ("give the result of applying factorial to three:", "factorial", [3]),
        ("which shared multiple of one and three is least?", "lcm", [1, 3]),
        ("What is 17 + 24?", "add", [17, 24]),
    ],
)
def test_supported_frames(prompt, operation, args):
    parsed = parse_prompt(prompt)
    assert parsed["accepted"]
    assert parsed["selected"]["operation"] == operation
    assert parsed["selected"]["args"] == args


@pytest.mark.parametrize(
    "prompt",
    [
        "product model four replaced product model two",
        "the difference between trail five and trail one was shade",
        "divide the booklet between pages four and seven",
        "What is six plus five and then subtract two?",
        "What is ten divided by zero?",
        "Compute nine factorial.",
    ],
)
def test_fail_closed(prompt):
    assert not parse_prompt(prompt)["accepted"]


def test_ambiguous_distinct_calls_abstain():
    parsed = parse_prompt("What is 2 plus 3? What is 4 plus 5?")
    assert not parsed["accepted"]
    assert parsed["reason"] == "ambiguous_typed_parse"


def test_execute_and_direct_commit():
    call = {"halt": "returned", "result": 11, "cycles": 12}
    target = "chuk_mcp_lazarus.tools.geometry.cell80_parse_execute._execute_cell80"
    with patch(target, return_value=call) as execute:
        result = _parse_execute_impl(
            "compute six plus five as a sum:",
            "/tmp/cell80",
            "/tmp/cells",
            DEFAULT_CELL_MAP,
            "number_word",
            1.0,
        )
    execute.assert_called_once()
    assert result["commitment"] == {
        "method": "direct_cell80_result",
        "text": "eleven",
        "value": 11,
        "verified": True,
    }


def test_abstention_never_executes():
    target = "chuk_mcp_lazarus.tools.geometry.cell80_parse_execute._execute_cell80"
    with patch(target) as execute:
        result = _parse_execute_impl(
            "product model four replaced product model two",
            "/tmp/cell80",
            "/tmp/cells",
            DEFAULT_CELL_MAP,
            "auto",
            1.0,
        )
    execute.assert_not_called()
    assert result["abstained"]


@pytest.mark.asyncio
async def test_tool_validates_complete_mapping():
    result = await cell80_parse_execute(
        prompt="What is one plus one?",
        cell80_binary="/tmp/cell80",
        cells_dir="/tmp/cells",
        cell_map={"add": "add"},
    )
    assert result["error_type"] == ToolError.INVALID_INPUT
