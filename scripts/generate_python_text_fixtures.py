"""Two things the port takes from CPython itself: `inspect.cleandoc`, and `str.isidentifier`.

Both were committed as goldens with **no generator and no version stamp**, which makes them the
only bytes here that a pin bump could not challenge — they could not be re-derived at all.

  - `inspect.cleandoc` is what dspy runs over a signature's class docstring, so it decides the
    instruction line breaks of every declared signature. It expands tabs, drops the common leading
    whitespace of every line *after* the first, and strips leading and trailing blank lines — none
    of which a `trim()` reproduces.
  - `str.isidentifier` decides whether a `Flex` tool's name can be written into generated code, and
    whether a signature's class name is usable. It is Unicode's XID_Start/XID_Continue, not ASCII:
    `café` and `αβ` are identifiers, `½` and an Arabic-Indic digit are not, and `Ⅶ` is.

    python3 scripts/generate_python_text_fixtures.py
"""

from __future__ import annotations

import inspect
import json
import pathlib
import platform
import sys

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "predict"

#: The docstring shapes a signature is written in, including the ones a `trim()` gets wrong.
DOCSTRINGS = {
    # The nine the derive was already held to, kept verbatim: three of them disagree with a
    # per-line one-space strip, which is what that code did before they were recorded.
    "conventional": " Answer the question.\n Be brief.",
    "uniformly indented": "     Answer the question.\n     Be brief.",
    "first line flush": "Answer.\n     Be brief.",
    "ragged": " Answer.\n   Deeper.\n Back.",
    "with a tab": " Answer.\tBriefly.",
    "leading blank line": "\n Answer.",
    "trailing newline": " Answer.\n",
    "blank line between": " Answer.\n\n Be brief.",
    "one line": " Answer.",
    # And the edges they did not reach.
    "trailing blank lines": "    Answer.\n    Be brief.\n\n\n",
    "tab on its own line": "Answer.\n\tTabbed.",
    "empty": "",
    "only whitespace": "   \n  \n",
}

#: Names that decide whether a tool can be referenced from generated code. Chosen for the
#: boundaries: a combining mark, a Roman numeral, a vulgar fraction, a non-ASCII digit, the micro
#: sign, and the ideographic zero.
NAMES = [
    "ok", "_ok", "ok1", "1no", "no-dash", "café", "αβ", "número", "", "_",
    "Ⅶ", "½", "٢", "a٢", "µ", "Ⅰ", "x́", "〇", "a b", "class", "None", "ok_", "__dunder__",
]


def main() -> None:
    stamp = {
        "source": f"CPython {platform.python_version()}, via scripts/generate_python_text_fixtures.py",
        "python_version": platform.python_version(),
    }
    (OUT / "cleandoc.json").write_text(
        json.dumps(
            {
                **stamp,
                "note": (
                    "`inspect.cleandoc` over the docstring shapes a signature is written in. dspy "
                    "runs it on a signature class's docstring, so this decides the instruction "
                    "line breaks of every declared signature."
                ),
                "cases": {
                    name: {"raw": raw, "cleandoc": inspect.cleandoc(raw)}
                    for name, raw in DOCSTRINGS.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    (OUT / "identifiers.json").write_text(
        json.dumps(
            {
                **stamp,
                "note": (
                    "`str.isidentifier`, which is Unicode's XID_Start/XID_Continue rather than "
                    "ASCII. It decides whether a `Flex` tool's name can be written into generated "
                    "code. Keyword-ness is *not* part of it: `class` is an identifier by this test "
                    "and refused later."
                ),
                "python": {name: name.isidentifier() for name in NAMES},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"  wrote cleandoc.json ({len(DOCSTRINGS)} cases) and identifiers.json ({len(NAMES)})", file=sys.stderr)


if __name__ == "__main__":
    main()
