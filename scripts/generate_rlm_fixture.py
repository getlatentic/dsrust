"""Record what dspy's RLM does with the code a model wrote, by running it.

`_strip_code_fences` is the gate every RLM iteration passes through, and it is a pile of edge
cases rather than one rule: decorative fence pairs are peeled off in a loop, a bare ``` fence is
accepted as Python, an explicit non-Python tag *raises*, an unterminated fence keeps its
remainder, and a fence opener with no newline after it is left alone entirely. The inputs below
are chosen at exactly those edges.

    .venv/bin/python scripts/generate_rlm_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.adapters.utils import get_annotation_name
from dspy.predict.rlm import RLM, _strip_code_fences

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "predict"
PINNED = require("dspy")

WRITTEN = [
    # No fence at all, and the surrounding whitespace that gets stripped.
    "print(1)",
    "  print(1)  ",
    "\n\nprint(1)\n\n",
    # The ordinary shapes.
    "```python\nprint(1)\n```",
    "```py\nprint(1)\n```",
    "```python3\nprint(1)\n```",
    "```py3\nprint(1)\n```",
    "```\nprint(1)\n```",
    # A tag that is not Python at all — upstream raises rather than guessing.
    "```javascript\nconsole.log(1)\n```",
    "```json\n{}\n```",
    # Case and extra words on the language line.
    "```PYTHON\nprint(1)\n```",
    "```python title=example\nprint(1)\n```",
    # Decorative outer pairs, peeled in a loop.
    "```\n```python\nprint(1)\n```\n```",
    "```\n```\n```python\nprint(1)\n```\n```\n```",
    # Prose before the fence, which is skipped to reach it.
    "Here is the code:\n```python\nprint(1)\n```",
    "Here is the code:\n```python\nprint(1)\n```\nAnd that is all.",
    # An unterminated fence keeps whatever followed it.
    "```python\nprint(1)",
    # A fence opener with no newline after it.
    "```python print(1)```",
    "```python",
    # Empty-ish inputs.
    "",
    "```\n```",
    # Two fenced blocks: the first one wins.
    "```python\nprint(1)\n```\n```python\nprint(2)\n```",
]


def factorial(n: int) -> int:
    """Compute the factorial of n."""
    return 1 if n <= 1 else n * factorial(n - 1)


class Unused:
    """RLM builds an interpreter per forward pass; the fixture never runs code."""

    def execute(self, code, variables=None):
        raise AssertionError("the fixture does not execute code")

    def shutdown(self):
        pass


class Described(dspy.Signature):
    """Answer from the context."""

    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Typed(dspy.Signature):
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()
    count: int = dspy.OutputField(desc="how many")


#: (label, signature, tools, max_llm_calls) — the template interpolates the input names, the
#: output-field list, the SUBMIT() names and the call cap, and tool docs are appended after it.
SIGNATURE_CASES = [
    ("plain", "context, query -> answer", [], 50),
    ("described", Described, [], 50),
    ("typed", Typed, [], 7),
    ("tools", "context -> answer", [factorial], 50),
]


def described_signature(signature) -> dict:
    def fields(items):
        return [
            {
                "name": name,
                "desc": "" if (d := f.json_schema_extra.get("desc", "")) == f"${{{name}}}" else d,
                "annotation": get_annotation_name(f.annotation),
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
    for written in WRITTEN:
        try:
            cases.append({"written": written, "code": _strip_code_fences(written), "error": None})
        except SyntaxError as error:
            cases.append({"written": written, "code": None, "error": str(error)})
    signatures = []
    for label, signature, tools, max_llm_calls in SIGNATURE_CASES:
        # dspy 3.3.0 takes a zero-argument *factory* rather than an interpreter: one is built per
        # forward pass and shut down after, where 3.3.0b1 held one for the module's lifetime.
        # The fixture never runs code, so `Unused` stands in either way — but it has to be passed
        # as the callable now, which is `Unused` itself rather than `Unused()`.
        rlm = RLM(signature, tools=tools or None, max_llm_calls=max_llm_calls, interpreter_factory=Unused)
        signatures.append(
            {
                "label": label,
                "task": signature if isinstance(signature, str) else signature.__name__,
                "task_instructions": rlm.signature.instructions,
                "max_llm_calls": max_llm_calls,
                "tools": [str(tool) for tool in rlm._user_tools.values()],
                "action": described_signature(rlm.generate_action.signature),
                "extract": described_signature(rlm.extract.signature),
            }
        )
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_rlm_fixture.py",
        "dspy_version": PINNED,
        "strip_code_fences": cases,
        "signatures": signatures,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rlm.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
