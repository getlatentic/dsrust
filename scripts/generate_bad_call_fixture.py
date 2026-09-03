"""Record what ReAct observes when a tool is called badly, by running dspy.

`Tool.__call__` validates each argument against its schema and raises `ValueError`; Python raises
`TypeError` for a required argument that is absent; pydantic raises its own error for a value the
schema did not reject but the parameter's type does. ReAct catches whichever lands and records
`Execution error in {tool}: ` followed by a traceback whose last line is the exception's own.

The traceback's frames are CPython's and are not reproduced. Its last line is, and the port asserts
it exactly for every case here except the ones pydantic raises, where the parser is serde and its
reason is its own — those cases are marked `parser`.

    .venv/bin/python scripts/generate_bad_call_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

import dspy

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "react"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()


def wikipedia_search(query: str) -> list[str]:
    """Search Wikipedia."""
    return ["Page about " + query]


def count_to(limit: int) -> str:
    """Count."""
    return ",".join(str(i) for i in range(limit))


def pair(prompt: str, answer: str) -> str:
    """Pair."""
    return prompt + "|" + answer


def triple(a: str, b: str, c: str) -> str:
    """Triple."""
    return a + b + c


def with_optional(prompt: str, worked: Optional[str]) -> str:
    """Optional by type, not by default."""
    return f"{prompt}|{worked}"


CASES = [
    ("missing", wikipedia_search, {}),
    ("mistyped", count_to, {"limit": "three"}),
    ("unknown_arg", wikipedia_search, {"topic": "haiku"}),
    ("missing_two", pair, {}),
    ("missing_three", triple, {}),
    ("integral_float", count_to, {"limit": 3.0}),
    ("bool_for_integer", count_to, {"limit": True}),
    ("optional_by_type_omitted", with_optional, {"prompt": "x"}),
    ("optional_by_type_null", with_optional, {"prompt": "x", "worked": None}),
    ("optional_by_type_wrong", with_optional, {"prompt": "x", "worked": 3}),
    ("good_call", count_to, {"limit": 3}),
]


def observe(tool, args):
    lm = dspy.utils.DummyLM(
        [
            {"next_thought": "call it", "next_tool_name": tool.__name__, "next_tool_args": args},
            {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "r", "answer": "a"},
        ]
    )
    dspy.configure(lm=lm)
    agent = dspy.ReAct("question -> answer", tools=[tool], max_iters=2)
    return agent(question="q").trajectory["observation_0"]


def main() -> None:
    cases = {}
    for label, tool, args in CASES:
        observation = observe(tool, args)
        entry = {"tool": tool.__name__, "args": args, "observation_0": observation}
        if isinstance(observation, str) and observation.startswith("Execution error in "):
            entry["exception_line"] = observation.rsplit("\n", 1)[-1]
            if "pydantic_core" in observation:
                entry["parser"] = True
        cases[label] = entry
        shown = entry.get("exception_line", observation)
        print(f"    {label}: {shown!r}", file=sys.stderr)
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_bad_call_fixture.py",
        "dspy_version": PINNED,
        "cases": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bad_call_observation.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.name}: {len(cases)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
