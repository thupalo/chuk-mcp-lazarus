"""Fail-closed arithmetic parsing and exact Cell80 execution."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chuk_mcp_lazarus.errors import ToolError, make_error
from chuk_mcp_lazarus.server import mcp
from chuk_mcp_lazarus.tools.geometry.cell80_prosthesis import _execute_cell80

TOOL_NAME = "cell80_parse_execute"
MAX_U16 = 65535
COMMUTATIVE = {"add", "mul", "gcd", "lcm"}
DEFAULT_CELL_MAP = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "gcd": "gcd",
    "factorial": "factorial",
    "lcm": "lcm",
}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
VALUE_WORDS = {value: word for word, value in NUMBER_WORDS.items()}
NUMBER_ATOM = (
    r"(?<![\w.])(?:"
    + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    + r"|[0-9]{1,5})(?!\w)(?!\.[0-9])"
)


@dataclass(frozen=True)
class Grammar:
    name: str
    operation: str
    pattern: re.Pattern[str]
    arg_groups: tuple[str, ...]


def _grammar(
    name: str,
    operation: str,
    pattern: str,
    arg_groups: tuple[str, ...] = ("a", "b"),
) -> Grammar:
    return Grammar(name, operation, re.compile(pattern, re.IGNORECASE), arg_groups)


N = NUMBER_ATOM
PREFIX = r"(?:compute|calculate|evaluate|find|give|what(?:'s| is))\s+"
TAIL = r"(?=\s*(?:[?.!:,;]|$|\bas\b|\band give\b|\bthe result\b))"

GRAMMARS = (
    _grammar(
        "explicit_plus",
        "add",
        rf"{PREFIX}(?P<a>{N})\s*(?:\+|plus)\s*(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "increased_by",
        "add",
        rf"\bif\s+(?P<a>{N})\s+is\s+increased\s+by\s+(?P<b>{N})"
        rf"(?=,\s*(?:the\s+)?result\b)",
    ),
    _grammar(
        "sum_of",
        "add",
        rf"{PREFIX}(?:the\s+)?(?:sum|total)\s+of\s+(?P<a>{N})\s+and\s+(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "total_after_adding",
        "add",
        rf"\bgive\s+(?:the\s+)?total\s+after\s+adding\s+(?P<a>{N})\s+to\s+(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "explicit_minus",
        "sub",
        rf"{PREFIX}(?P<a>{N})\s*(?:-|minus)\s*(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "decreased_by",
        "sub",
        rf"\bif\s+(?P<a>{N})\s+is\s+decreased\s+by\s+(?P<b>{N})"
        rf"(?=,\s*(?:the\s+)?result\b)",
    ),
    _grammar(
        "subtract_from",
        "sub",
        rf"\b(?:give|find|what(?:'s| is))\b[^?.!:;]{{0,48}}?\bsubtracting\s+"
        rf"(?P<b>{N})\s+from\s+(?P<a>{N}){TAIL}",
    ),
    _grammar(
        "difference_of",
        "sub",
        rf"{PREFIX}(?:the\s+)?difference\s+(?:of|between)\s+(?P<a>{N})"
        rf"\s+(?:and|minus)\s+(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "explicit_times",
        "mul",
        rf"{PREFIX}(?P<a>{N})\s*(?:\*|×|x|times|multiplied\s+by)\s*(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "repeated_times",
        "mul",
        rf"\bif\s+(?P<a>{N})\s+is\s+repeated\s+(?P<b>{N})\s+times"
        rf"(?=,\s*(?:the\s+)?result\b)",
    ),
    _grammar(
        "groups_containing",
        "mul",
        rf"\bgive\s+(?:the\s+)?total\s+across\s+(?P<a>{N})\s+groups\s+"
        rf"containing\s+(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "product_of",
        "mul",
        rf"{PREFIX}(?:the\s+)?product\s+of\s+(?P<a>{N})\s+and\s+(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "explicit_divide",
        "div",
        rf"{PREFIX}(?P<a>{N})\s*(?:/|÷|divided\s+by)\s*(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "shared_across",
        "div",
        rf"\bif\s+(?P<a>{N})\s+is\s+shared\s+across\s+(?P<b>{N})\s+equal\s+parts"
        rf"(?=,\s*each\b)",
    ),
    _grammar(
        "groups_fit",
        "div",
        rf"\b(?:give|find|what(?:'s| is))\b[^?.!:;]{{0,32}}?\bhow\s+many\s+groups\s+"
        rf"of\s+(?P<b>{N})\s+fit\s+within\s+(?P<a>{N}){TAIL}",
    ),
    _grammar(
        "quotient_of",
        "div",
        rf"{PREFIX}(?:the\s+)?quotient\s+of\s+(?P<a>{N})\s+and\s+(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "explicit_gcd",
        "gcd",
        rf"{PREFIX}(?:the\s+)?(?:gcd|greatest\s+common\s+(?:divisor|factor))"
        rf"(?:\s+for|\s+of)?\s+(?P<a>{N})\s+(?:paired\s+with|and)\s+(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "largest_common_divisor",
        "gcd",
        rf"\b(?:give|find|which)\s+(?:the\s+)?(?:largest|greatest)\s+"
        rf"(?:divisor|factor)(?:\s+common\s+to|\s+of)\s+(?P<a>{N})\s+and\s+"
        rf"(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "shared_factor_greatest",
        "gcd",
        rf"\bwhich\s+shared\s+factor\s+of\s+(?P<a>{N})\s+and\s+(?P<b>{N})"
        rf"\s+is\s+greatest{TAIL}",
    ),
    _grammar(
        "explicit_factorial",
        "factorial",
        rf"{PREFIX}(?:(?P<a>{N})\s*(?:!|factorial)|(?:the\s+)?factorial\s+of\s+"
        rf"(?P<a_alt>{N})){TAIL}",
        ("a_or_alt",),
    ),
    _grammar(
        "multiply_down",
        "factorial",
        rf"\bmultiply\s+from\s+(?P<a>{N})\s+down\s+to\s+one\s+and\s+give{TAIL}",
        ("a",),
    ),
    _grammar(
        "applying_factorial",
        "factorial",
        rf"\bgive\s+(?:the\s+)?result\s+of\s+applying\s+factorial\s+to\s+"
        rf"(?P<a>{N}){TAIL}",
        ("a",),
    ),
    _grammar(
        "explicit_lcm",
        "lcm",
        rf"{PREFIX}(?:the\s+)?(?:lcm|least\s+common\s+multiple)"
        rf"(?:\s+for|\s+of)?\s+(?P<a>{N})\s+(?:paired\s+with|and)\s+(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "smallest_common_multiple",
        "lcm",
        rf"\b(?:give|find|which)\s+(?:the\s+)?(?:smallest|least)\s+multiple"
        rf"(?:\s+common\s+to|\s+of)\s+(?P<a>{N})\s+and\s+(?P<b>{N}){TAIL}",
    ),
    _grammar(
        "shared_multiple_least",
        "lcm",
        rf"\bwhich\s+shared\s+multiple\s+of\s+(?P<a>{N})\s+and\s+(?P<b>{N})"
        rf"\s+is\s+least{TAIL}",
    ),
)


def _parse_number(text: str) -> int:
    normalized = text.lower()
    value = NUMBER_WORDS.get(normalized)
    if value is None:
        value = int(normalized)
    if not 0 <= value <= MAX_U16:
        raise ValueError(f"number_out_of_range:{text}")
    return value


def _match_args(grammar: Grammar, match: re.Match[str]) -> tuple[int, ...]:
    if grammar.arg_groups == ("a_or_alt",):
        raw = match.groupdict().get("a") or match.groupdict().get("a_alt")
        if raw is None:
            raise ValueError("missing_factorial_argument")
        return (_parse_number(raw),)
    return tuple(_parse_number(match.group(name)) for name in grammar.arg_groups)


def parse_prompt(prompt: str) -> dict[str, Any]:
    """Return one unique typed parse or a structured abstention."""
    normalized = " ".join(prompt.strip().split())
    evidence: list[dict[str, Any]] = []
    unique: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    errors: list[str] = []
    for grammar in GRAMMARS:
        for match in grammar.pattern.finditer(normalized):
            try:
                args = _match_args(grammar, match)
            except ValueError as exc:
                errors.append(f"{grammar.name}:{exc}")
                continue
            row = {
                "rule": grammar.name,
                "operation": grammar.operation,
                "args": list(args),
                "span": [match.start(), match.end()],
                "matched_text": match.group(0),
            }
            evidence.append(row)
            unique.setdefault((grammar.operation, args), row)

    if errors:
        return {
            "accepted": False,
            "reason": "invalid_numeric_argument",
            "normalized_prompt": normalized,
            "evidence": evidence,
            "errors": errors,
        }
    if not unique:
        return {
            "accepted": False,
            "reason": "no_supported_computational_frame",
            "normalized_prompt": normalized,
            "evidence": [],
            "errors": [],
        }
    if len(unique) != 1:
        return {
            "accepted": False,
            "reason": "ambiguous_typed_parse",
            "normalized_prompt": normalized,
            "evidence": evidence,
            "errors": [],
            "candidates": [
                {"operation": operation, "args": list(args)}
                for operation, args in sorted(unique)
            ],
        }

    selected = next(iter(unique.values()))
    operation = selected["operation"]
    args = selected["args"]
    if operation == "div" and args[1] == 0:
        return {
            "accepted": False,
            "reason": "division_by_zero",
            "normalized_prompt": normalized,
            "evidence": evidence,
            "errors": [],
        }
    if operation == "factorial" and args[0] > 8:
        return {
            "accepted": False,
            "reason": "factorial_out_of_u16_domain",
            "normalized_prompt": normalized,
            "evidence": evidence,
            "errors": [],
        }
    return {
        "accepted": True,
        "reason": "unique_typed_parse",
        "normalized_prompt": normalized,
        "evidence": evidence,
        "selected": selected,
        "errors": [],
    }


def _format_result(result: int, result_format: str) -> str:
    if result_format == "decimal":
        return str(result)
    if result_format == "number_word":
        if result not in VALUE_WORDS:
            raise ValueError(f"No native number word is declared for {result}.")
        return VALUE_WORDS[result]
    if result_format == "auto":
        return VALUE_WORDS.get(result, str(result))
    raise ValueError("result_format must be 'decimal', 'number_word', or 'auto'.")


def _parse_execute_impl(
    prompt: str,
    cell80_binary: str,
    cells_dir: str,
    cell_map: dict[str, str],
    result_format: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    parsed = parse_prompt(prompt)
    payload: dict[str, Any] = {
        "prompt": prompt,
        "training_performed": False,
        "parse": parsed,
        "executed": False,
        "committed": False,
    }
    if not parsed["accepted"]:
        payload["abstained"] = True
        payload["abstention_reason"] = parsed["reason"]
        return payload

    selected = parsed["selected"]
    operation = selected["operation"]
    cell_name = cell_map[operation]
    call = _execute_cell80(
        Path(cell80_binary),
        Path(cells_dir),
        cell_name,
        selected["args"],
        timeout_seconds=timeout_seconds,
    )
    payload["executed"] = True
    payload["cell80"] = call
    if str(call.get("halt", "")).lower() != "returned":
        payload["abstained"] = True
        payload["abstention_reason"] = f"cell80_halt:{call.get('halt')}"
        return payload

    result = int(call["result"])
    commitment_text = _format_result(result, result_format)
    payload.update(
        {
            "abstained": False,
            "operation": operation,
            "args": selected["args"],
            "cell_name": cell_name,
            "result": result,
            "commitment": {
                "method": "direct_cell80_result",
                "text": commitment_text,
                "value": result,
                "verified": True,
            },
            "committed": True,
        }
    )
    return payload


@mcp.tool(read_only_hint=True, idempotent_hint=True)
async def cell80_parse_execute(
    prompt: str,
    cell80_binary: str,
    cells_dir: str,
    cell_map: dict[str, str] | None = None,
    result_format: str = "auto",
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Parse one unambiguous arithmetic frame, execute Cell80, and commit it.

    The tool fails closed: it abstains without running Cell80 unless a single
    supported grammar yields a single well-typed call.
    """
    if not prompt.strip():
        return make_error(ToolError.INVALID_INPUT, "prompt must not be empty.", TOOL_NAME)
    if not 0 < timeout_seconds <= 60:
        return make_error(
            ToolError.INVALID_INPUT,
            "timeout_seconds must be greater than zero and at most 60.",
            TOOL_NAME,
        )
    mapping = dict(DEFAULT_CELL_MAP)
    if cell_map is not None:
        unknown = sorted(set(cell_map) - set(DEFAULT_CELL_MAP))
        missing = sorted(set(DEFAULT_CELL_MAP) - set(cell_map))
        if unknown or missing:
            return make_error(
                ToolError.INVALID_INPUT,
                f"cell_map must contain exactly {sorted(DEFAULT_CELL_MAP)}; "
                f"unknown={unknown}, missing={missing}.",
                TOOL_NAME,
            )
        mapping = {operation: str(name) for operation, name in cell_map.items()}
    if any(not name for name in mapping.values()):
        return make_error(
            ToolError.INVALID_INPUT,
            "cell names must not be empty.",
            TOOL_NAME,
        )
    try:
        return await asyncio.to_thread(
            _parse_execute_impl,
            prompt,
            cell80_binary,
            cells_dir,
            mapping,
            result_format,
            float(timeout_seconds),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return make_error(ToolError.INVALID_INPUT, str(exc), TOOL_NAME)
    except Exception as exc:
        return make_error(ToolError.INTERVENTION_FAILED, str(exc), TOOL_NAME)
