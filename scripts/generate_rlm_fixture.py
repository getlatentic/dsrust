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

from dspy.predict.rlm import _strip_code_fences

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance" / "predict"
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


def main() -> None:
    cases = []
    for written in WRITTEN:
        try:
            cases.append({"written": written, "code": _strip_code_fences(written), "error": None})
        except SyntaxError as error:
            cases.append({"written": written, "code": None, "error": str(error)})
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_rlm_fixture.py",
        "dspy_version": PINNED,
        "strip_code_fences": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rlm.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
