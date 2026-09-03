#!/usr/bin/env python3
"""The messages dspy's `XMLAdapter.format` writes for signatures with structured outputs.

dspy 3.3.1 writes a `list`, `dict`, TypedDict or model output as nested elements, sketches those
elements in the system prompt and in the request, and escapes `&` and `<` in a `str` output. Each
case is a string signature, the demos and inputs to render, and the messages dspy produced; the
Rust test renders the same and compares message by message.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

import dspy

PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()
ROOT = pathlib.Path(__file__).parent.parent
OUT = ROOT / "crates" / "dsrust" / "tests" / "conformance" / "adapter" / "xml_format.json"

NESTED = "question -> answer: str, tags: list[str], counts: dict[str, int], maybe: Optional[list[str]], score: int"

CASES = [
    {
        "name": "nested_outputs",
        "signature": NESTED,
        "demos": [
            {
                "question": "first?",
                "answer": "a < b & c > d",
                "tags": ["x", "y & z"],
                "counts": {"k": 1, "two words": 2, "a-b": 3, "1x": 4, "": 5},
                "maybe": None,
                "score": 7,
            }
        ],
        "inputs": {"question": "second?"},
    },
    {
        "name": "empty_containers",
        "signature": NESTED,
        "demos": [{"question": "q", "answer": "plain", "tags": [], "counts": {}, "maybe": ["a"], "score": 1}],
        "inputs": {"question": "second?"},
    },
    {
        "name": "missing_outputs_in_a_demo",
        "signature": NESTED,
        "demos": [{"question": "q", "answer": "only"}],
        "inputs": {"question": "second?"},
    },
    {
        "name": "plain_outputs",
        "signature": "question -> answer: str, score: int",
        "demos": [{"question": "q", "answer": "a & b", "score": 2}],
        "inputs": {"question": "second?"},
    },
    {
        "name": "list_input_is_text",
        "signature": "items: list[str], question -> out: list[str]",
        "demos": [],
        "inputs": {"items": ["a", "b & c"], "question": "which?"},
    },
    {
        "name": "nested_values_of_every_scalar",
        "signature": "question -> ratios: dict[str, float], flags: list[bool], notes: dict[str, Optional[int]], grid: list[list[int]]",
        "demos": [
            {
                "question": "q",
                "ratios": {"k": 1.0, "j": 2.5e-7, "i": 1e21},
                "flags": [True, False],
                "notes": {"k": None, "j": 3},
                "grid": [[1, 2], []],
            }
        ],
        "inputs": {"question": "second?"},
    },
]


def main() -> None:
    cases = []
    for case in CASES:
        signature = dspy.Signature(case["signature"])
        inputs = list(signature.input_fields)
        demos = [dspy.Example(**demo).with_inputs(*inputs) for demo in case["demos"]]
        messages = dspy.XMLAdapter().format(signature, demos, case["inputs"])
        cases.append(
            {
                **case,
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            }
        )
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from dspy=={PINNED} via scripts/generate_xml_format_fixture.py",
                "dspy_version": dspy.__version__,
                "cases": cases,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"  wrote {OUT.relative_to(ROOT)} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
