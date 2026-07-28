"""Record what dspy's ProgramOfThought builds and parses, by running it.

Two things drift when this module is ported. The three signatures it derives — generate,
regenerate, answer — decide the prompts, so their fields, order and instructions are recorded
verbatim. And `_parse_code` is a pair of regexes over whatever the model wrote; the inputs below
are chosen where a hand-written matcher parts company with them: a fence with no language line, a
two-backtick close, a `---` tail that keeps its newline, a trailing assignment that is echoed only
when there is more than one line, and the two shapes upstream refuses.

    .venv/bin/python scripts/generate_pot_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.predict.program_of_thought import ProgramOfThought

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance" / "predict"
PINNED = require("dspy")

#: The signatures to derive the three asks from — one plain, one with several fields either side.
TASKS = ["question -> answer", "question, context -> answer, confidence"]

#: What the model might have written in `generated_code`, at the edges of upstream's two regexes.
WRITTEN = [
    "```python\nx = 1\nprint(x)\n```",
    "```python x = 1 ```",
    "```python\nprint(1)\n``",
    "print(1)\n---\nnotes",
    "print(1)\n\n\nnotes",
    "y = 2\nanswer = y + 1",
    "answer = 1",
    "",
    "a = 1 = 2",
    "```python\n```",
    "no fence here\nresult = 4",
    "```python\nresult  =  4\n```",
    "text before\n```python\nprint(2)\n```\ntext after",
]


def described(signature, instructions: str) -> dict:
    """A signature as the fields and instructions that reach a prompt.

    `_generate_signature` carries dspy's *default* instruction; the real one comes from
    `_generate_instruction` and is combined with the fields in the constructor
    (`dspy.Signature(fields, instruction)`), which is what the module actually asks with.
    """
    def described_fields(fields):
        # dspy stores `${name}` as the placeholder for "no description given" and renders it
        # blank; the crate stores the blank directly. Normalising here compares what reaches a
        # prompt rather than how each side spells the absence.
        return [
            {
                "name": name,
                "desc": "" if (desc := field.json_schema_extra.get("desc", "")) == f"${{{name}}}" else desc,
            }
            for name, field in fields.items()
        ]

    return {
        "instructions": instructions,
        "inputs": described_fields(signature.input_fields),
        "outputs": described_fields(signature.output_fields),
    }


class Unused:
    """ProgramOfThought builds an interpreter in its constructor; the fixture never runs code, so
    this stands in rather than requiring deno to be installed."""

    def execute(self, code, variables=None):
        raise AssertionError("the fixture does not execute code")

    def shutdown(self):
        pass


def main() -> None:
    cases = []
    for task in TASKS:
        pot = ProgramOfThought(task, interpreter=Unused())
        cases.append(
            {
                "task": task,
                "modes": {
                    mode: described(
                        pot._generate_signature(mode), pot._generate_instruction(mode)
                    )
                    for mode in ("generate", "regenerate", "answer")
                },
            }
        )
    pot = ProgramOfThought("question -> answer", interpreter=Unused())
    parsed = []
    for written in WRITTEN:
        code, error = pot._parse_code({"generated_code": written})
        parsed.append({"written": written, "code": code, "error": error})

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_pot_fixture.py",
        "dspy_version": PINNED,
        "signatures": cases,
        "parse_code": parsed,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "program_of_thought.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
