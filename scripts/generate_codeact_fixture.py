"""Record the signatures dspy's CodeAct builds, by constructing it.

CodeAct's two signatures decide two prompts. The turn signature carries the task's inputs, the
trajectory, and the `generated_code`/`finished` pair the model answers with; its instructions
embed the tool catalogue, which is where a port drifts — the numbering, the blank line after the
task's own instructions, and each tool's string form.

    .venv/bin/python scripts/generate_codeact_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.predict.code_act import CodeAct

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "predict"
PINNED = require("dspy")


def factorial(n: int) -> int:
    """Compute the factorial of n."""
    return 1 if n <= 1 else n * factorial(n - 1)


def lookup(name: str, year: int) -> str:
    """Look a thing up.

    Spans two lines, so the newline flattening in a tool's string form is exercised.
    """
    return name


class Unused:
    """CodeAct builds an interpreter in its constructor; the fixture never runs code."""

    def execute(self, code, variables=None):
        raise AssertionError("the fixture does not execute code")

    def shutdown(self):
        pass


class Described(dspy.Signature):
    """Answer the question carefully."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


#: (label, signature, tools) — a plain task with one tool, one with several fields and two tools,
#: and one whose signature carries its own instructions (which lead the prompt).
CASES = [
    ("plain", "question -> answer", [factorial]),
    ("multi", "question, context -> answer, confidence", [factorial, lookup]),
    ("described", Described, [factorial]),
    ("no_tools", "question -> answer", []),
]


def described(signature) -> dict:
    def fields(items):
        # dspy stores `${name}` for "no description given" and renders it blank; the crate stores
        # the blank. Normalising compares what reaches a prompt.
        return [
            {
                "name": name,
                "desc": "" if (d := f.json_schema_extra.get("desc", "")) == f"${{{name}}}" else d,
            }
            for name, f in items.items()
        ]

    return {
        "instructions": signature.instructions,
        "inputs": fields(signature.input_fields),
        "outputs": fields(signature.output_fields),
    }


def main() -> None:
    cases = []
    for label, signature, tools in CASES:
        act = CodeAct(signature, tools=tools, interpreter=Unused())
        cases.append(
            {
                "label": label,
                "task": signature if isinstance(signature, str) else "Described",
                "task_instructions": act.signature.instructions,
                "tools": [str(tool) for tool in act.tools.values()],
                "codeact": described(act.codeact.signature),
                "extract": described(act.extractor.predict.signature),
            }
        )
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_codeact_fixture.py",
        "dspy_version": PINNED,
        "cases": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "code_act.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
