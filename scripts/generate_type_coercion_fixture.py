#!/usr/bin/env python
"""Record which input shapes each dspy custom type accepts, and what it says when it refuses.

Every one of these types carries a `@model_validator(mode="before")` called `validate_input`, and
pydantic runs it on *every* construction — including parsing a model's reply. So the shapes it
accepts are part of the wire contract, not a Python-side convenience: a reply that arrives as a bare
string where the type is a `Reasoning` is a reply dspy reads and a stricter port drops.

The ledger said all eight were "reproduced as the type's `Deserialize`". Running them found two that
were not — `Audio` refused the data URI upstream accepts, `ToolCallResults` refused a bare list —
and two whose refusal message differed. None of that is visible without asking upstream directly.

Only a `value_error` message is recorded. Those are dspy's own strings, written in `validate_input`
and portable. Pydantic's structural complaints ("Input should be a valid dictionary") describe a
Python type system and have no counterpart worth matching, so those cases record the refusal alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import dspy
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.document import Document
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import ToolCallResults, ToolCalls

PINNED = "3.3.0b1"
OUT = Path(__file__).resolve().parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "adapter"

CITE = {
    "cited_text": "x",
    "document_index": 0,
    "document_title": "t",
    "start_char_index": 0,
    "end_char_index": 1,
}

#: One entry per branch of each `validate_input`, plus the value that falls off the end of it.
CASES: dict[str, tuple[type, list]] = {
    "Audio": (
        dspy.Audio,
        [
            "data:audio/wav;base64,AA==",
            "data:audio/x-wav;base64,AA==",
            {"data": "AA==", "audio_format": "wav"},
            {"data": "AA=="},
            "not a data uri",
            "data:audio/",
            42,
        ],
    ),
    "File": (
        dspy.File,
        [
            {"file_id": "f1"},
            {"filename": "a.pdf"},
            {"file_data": "data:application/pdf;base64,AA=="},
            {"file_id": "f1", "filename": "a.pdf"},
            {},
            {"unrelated": 1},
        ],
    ),
    "Code": (
        dspy.Code,
        [
            "print(1)",
            "```python\nprint(1)\n```",
            "```\nprint(1)\n```",
            {"code": "print(1)"},
            {},
            {"code": 5},
            5,
        ],
    ),
    "Citations": (
        Citations,
        [[CITE], {"citations": [CITE]}, [], {"citations": []}, "not a citation", 5],
    ),
    "Document": (
        Document,
        ["some text", {"data": "some text"}, {"data": "x", "title": "t"}, 5],
    ),
    "Reasoning": (
        Reasoning,
        ["thinking out loud", {"content": "thinking"}, {"reasoning": "x"}, {"content": 5}, {}, 5],
    ),
    "ToolCalls": (
        ToolCalls,
        [
            {"tool_calls": [{"name": "f", "args": {}}]},
            [{"name": "f", "args": {}}],
            {"name": "f", "args": {}},
            [{"name": "f", "arguments": {"a": 1}}],
            [{"function": {"name": "f", "arguments": '{"a": 1}'}}],
            {"tool_calls": []},
            "not a tool call",
            5,
        ],
    ),
    "ToolCallResults": (
        ToolCallResults,
        [
            {"tool_call_results": [{"name": "f", "value": 1}]},
            [{"name": "f", "value": 1}],
            {"name": "f", "value": 1},
            [],
            {"unrelated": 1},
            5,
        ],
    ),
}


def refusal(error: Exception) -> str | None:
    """dspy's own message for a refusal, or nothing where pydantic wrote it instead."""
    errors = getattr(error, "errors", None)
    if not callable(errors):
        return str(error)
    first = errors()[0]
    if first.get("type") != "value_error":
        return None
    return first["msg"].removeprefix("Value error, ")


def recorded(cls: type, shape) -> dict:
    try:
        cls.model_validate(shape)
    except Exception as error:  # noqa: BLE001 — whatever it raises is the answer
        return {"input": shape, "accepted": False, "refusal": refusal(error)}
    return {"input": shape, "accepted": True}


def main() -> None:
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_type_coercion_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "What each custom type's `validate_input` accepts, and dspy's own words when it does "
            "not. `refusal` is null where pydantic refused structurally rather than dspy refusing "
            "by name — those messages describe Python's type system and are not portable."
        ),
        "types": {
            name: [recorded(cls, shape) for shape in shapes]
            for name, (cls, shapes) in CASES.items()
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "type_coercion.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
