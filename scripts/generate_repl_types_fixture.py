"""Record what dspy's REPL types render, by rendering them.

`REPLVariable`, `REPLEntry` and `REPLHistory` are the bytes RLM puts in a prompt on every single
iteration: what the model can reach, what it ran, and what came back. Each carries a rule that a
reimplementation guesses wrong in a way no small example reveals — Python's thousands separator,
a middle-out cut taken in code points off a floor-divided budget, `str()` versus
`json.dumps(indent=2)` for the value, and `json.dumps`' *default* `ensure_ascii=True`, which
escapes every non-ASCII character and changes the reported length with it.

    .venv/bin/python scripts/generate_repl_types_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.primitives.repl_types import REPLEntry, REPLHistory, REPLVariable

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "primitives"
PINNED = require("dspy")


class Described(dspy.Signature):
    """A signature whose fields carry real metadata."""

    plain: str = dspy.InputField()
    # `constraints` is not a kwarg upstream accepts: it is *derived* from pydantic's own
    # constraint arguments and rendered into prose, so it is stated the way a caller can reach it.
    annotated: int = dspy.InputField(desc="how many items", ge=0, le=100)
    answer: str = dspy.OutputField()


#: A string long enough that the preview is cut, in a script where one character is one byte.
LONG_ASCII = "".join(chr(ord("a") + i % 26) for i in range(2_500))

#: The same, in a script where it is not: a preview taken in bytes would slice a character in
#: half, and a length counted in bytes would be three times too large.
LONG_CJK = "日本語" * 800

#: Astral-plane characters, which `ensure_ascii` writes as surrogate *pairs* and which Python
#: counts as one character each.
EMOJI = "🙂🎯" * 400

#: (label, value, field, preview_chars) — `field` names a field on `Described` whose metadata is
#: passed as `field_info`, or None to pass none.
CASES = [
    # The scalars, each of which `from_value` stringifies with `str()` rather than as JSON.
    ("str", "hello", None, 1_000),
    ("int", 42, None, 1_000),
    ("float", 3.5, None, 1_000),
    ("float_integral", 1.0, None, 1_000),
    ("bool_true", True, None, 1_000),
    ("bool_false", False, None, 1_000),
    ("none", None, None, 1_000),
    # Containers, which go through `json.dumps(indent=2)` — a different writer with a different
    # ASCII policy from the flat `json.dumps` the adapters use.
    ("list", ["a", "b"], None, 1_000),
    ("list_empty", [], None, 1_000),
    ("dict", {"alpha": 1, "beta": [1, 2]}, None, 1_000),
    ("dict_empty", {}, None, 1_000),
    ("dict_nested", {"outer": {"inner": {"deep": [1, {"x": True}]}}}, None, 1_000),
    ("list_mixed_scalars", [1, 1.5, True, False, None, "s"], None, 1_000),
    # Non-ASCII, where `ensure_ascii=True` decides both the text and the reported length.
    ("str_unicode", "Voici de l'eau — 日本語", None, 1_000),
    ("list_unicode", ["日本語", "café"], None, 1_000),
    ("dict_unicode", {"ville": "Genève", "note": "日本語"}, None, 1_000),
    ("str_emoji", "🙂 hello 🎯", None, 1_000),
    ("list_emoji", ["🙂"], None, 1_000),
    # Over the preview budget: the middle-out cut, and the length that stays true.
    ("long_ascii", LONG_ASCII, None, 1_000),
    ("long_cjk", LONG_CJK, None, 1_000),
    ("long_emoji", EMOJI, None, 1_000),
    ("long_list", [f"item-{i:04d}" for i in range(200)], None, 1_000),
    # Exactly at the budget, one under and one over — the boundary the comparison is written on.
    ("at_budget", "x" * 1_000, None, 1_000),
    ("under_budget", "x" * 999, None, 1_000),
    ("over_budget", "x" * 1_001, None, 1_000),
    # An odd budget, which Python's floor division spends a character short of.
    ("odd_budget", "abcdefghij", None, 5),
    ("even_budget", "abcdefghij", None, 4),
    ("zero_budget", "abcdefghij", None, 0),
    # A length that crosses each thousands separator.
    ("grouped_thousand", "x" * 1_000, None, 100_000),
    ("grouped_million", "x" * 1_000_000, None, 100),
    # Field metadata: the placeholder desc a bare field carries is skipped, a real one is kept,
    # and constraints ride alongside it.
    ("field_placeholder_desc", "hello", "plain", 1_000),
    ("field_desc_and_constraints", "hello", "annotated", 1_000),
]

#: (label, output, max_output_chars) — the header states the true length either way, and the body
#: is cut in the middle once it runs past the cap.
OUTPUT_CASES = [
    ("short", "hi", 10_000),
    ("empty", "", 10_000),
    ("at_cap", "y" * 10, 10),
    ("one_over_cap", "y" * 11, 10),
    ("well_over_cap", "y" * 20, 10),
    ("odd_cap", "abcdefghij", 5),
    ("zero_cap", "abcdefghij", 0),
    ("cjk_over_cap", "日本語" * 10, 10),
    ("emoji_over_cap", "🙂" * 10, 10),
    ("grouped", "z" * 12_345, 10_000),
    ("newlines", "line one\nline two\n", 10_000),
]

#: (label, reasoning, code, output, index, max_output_chars)
ENTRY_CASES = [
    ("no_reasoning", "", "print(1)", "1", 0, 10_000),
    ("with_reasoning", "look first", "print(1)", "1", 1, 10_000),
    ("multiline_code", "", "x = 1\nfor i in range(3):\n    print(i)", "0\n1\n2", 4, 10_000),
    ("empty_output", "", "pass", "", 0, 10_000),
    ("truncated_output", "check", "print(big)", "q" * 30, 2, 10),
    ("fenced_code", "", "print('```')", "```", 0, 10_000),
]

#: (label, entries, max_output_chars) — entries index into ENTRY_CASES by label.
HISTORY_CASES = [
    ("empty", [], 10_000),
    ("one", ["no_reasoning"], 10_000),
    ("several", ["no_reasoning", "with_reasoning", "multiline_code"], 10_000),
    ("truncating", ["truncated_output", "no_reasoning"], 10),
]


def field_info(name: str | None):
    return None if name is None else Described.input_fields[name]


def variable_case(label, value, field, preview_chars) -> dict:
    variable = REPLVariable.from_value(
        label, value, field_info=field_info(field), preview_chars=preview_chars
    )
    return {
        "label": label,
        # The value as a Rust caller holds it, so the Rust side starts from the same input rather
        # than from a string this script already resolved.
        "value": value,
        "field": field,
        "preview_chars": preview_chars,
        "name": variable.name,
        "type_name": variable.type_name,
        "desc": variable.desc,
        "constraints": variable.constraints,
        "total_length": variable.total_length,
        "preview": variable.preview,
        "formatted": variable.format(),
        "serialized": variable.model_dump(),
    }


def entry_of(label: str) -> REPLEntry:
    _, reasoning, code, output, _, _ = next(c for c in ENTRY_CASES if c[0] == label)
    return REPLEntry(reasoning=reasoning, code=code, output=output)


def main() -> None:
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_repl_types_fixture.py",
        "dspy_version": PINNED,
        "variables": [variable_case(*case) for case in CASES],
        "outputs": [
            {
                "label": label,
                "output": output,
                "max_output_chars": cap,
                "formatted": REPLEntry.format_output(output, max_output_chars=cap),
            }
            for label, output, cap in OUTPUT_CASES
        ],
        "entries": [
            {
                "label": label,
                "reasoning": reasoning,
                "code": code,
                "output": output,
                "index": index,
                "max_output_chars": cap,
                "formatted": REPLEntry(reasoning=reasoning, code=code, output=output).format(
                    index=index, max_output_chars=cap
                ),
            }
            for label, reasoning, code, output, index, cap in ENTRY_CASES
        ],
        "histories": [
            {
                "label": label,
                "entries": labels,
                "max_output_chars": cap,
                "len": len(history),
                "truthy": bool(history),
                "formatted": history.format(),
                "serialized": history.model_dump(),
            }
            for label, labels, cap in HISTORY_CASES
            for history in [
                REPLHistory(entries=[entry_of(name) for name in labels], max_output_chars=cap)
            ]
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "repl_types.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
