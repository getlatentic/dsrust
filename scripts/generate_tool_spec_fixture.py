"""Record what one tool looks like on a native request, by running dspy.

`Tool.format_as_litellm_function_call` decides which arguments the provider is told it must send,
and it is not "all of them": an argument whose schema carries a `default` is omitted from
`required`. The port asserted the opposite in a doc comment — `required` is `list(self.args.keys())`
— which was upstream's rule once and is not upstream's rule now.

The corpus below is chosen so that a wrong rule fails on it: an argument that is optional *by
default* and one that is optional *by type* sit side by side, and only the first is exempt.

    .dspy-venv/bin/python scripts/generate_tool_spec_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "react"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()


def every_argument_required(query: str, limit: int) -> str:
    """Look something up."""
    return f"{query}{limit}"


def a_default_of_empty_string(prompt: str, answer: str, worked_solution: str = "") -> str:
    """Append one practice question and the answer the learner checks it against."""
    return f"{prompt}{answer}{worked_solution}"


def defaults_of_every_scalar(a: int = 1, b: bool = False, c: float = 0.5, d: str = "x") -> str:
    """Four arguments, each optional by default."""
    return f"{a}{b}{c}{d}"


def optional_by_type_and_by_default(
    by_type: str | None,
    by_default: str | None = None,
) -> str:
    """An argument may be optional by type or by default, and only one of those exempts it.

    `by_type` has no default, so upstream requires it however nullable its schema is.
    """
    return f"{by_type}{by_default}"


def no_arguments() -> str:
    """Read the whole draft as written so far."""
    return "read"


def a_container_with_a_default(names: list[str], tags: dict = {}) -> str:  # noqa: B006
    """A mutable default is still a default."""
    return f"{names}{tags}"


def undocumented(x: int) -> str:
    return str(x)


def indented_docstring(term: str) -> str:
    """Look one term up.

        query is a phrase, not a sentence.
    Answers with a summary.
    """
    return term


TOOLS = [
    indented_docstring,
    undocumented,
    every_argument_required,
    a_default_of_empty_string,
    defaults_of_every_scalar,
    optional_by_type_and_by_default,
    no_arguments,
    a_container_with_a_default,
]


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")

    recorded = []
    for func in TOOLS:
        tool = dspy.Tool(func)
        recorded.append(
            {
                "name": tool.name,
                "desc": tool.desc,
                "args": tool.args,
                "native": tool.format_as_litellm_function_call(),
            }
        )

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_tool_spec_fixture.py",
        "dspy_version": PINNED,
        "tools": recorded,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tool_spec.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.name}: {len(recorded)} tools", file=sys.stderr)
    for entry in recorded:
        print(f"    {entry['name']}: required {entry['native']['function']['parameters']['required']}", file=sys.stderr)


if __name__ == "__main__":
    main()
