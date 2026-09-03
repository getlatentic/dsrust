"""Record what dspy.Flex renders, by running it.

`Flex`'s state is a *string of Python source* — the module the optimizer rewrites — and the guest
runs it in a sandbox. Two strings decide whether a port's guest sees what upstream's does: the
signature rendered back to `"in: T -> out: T2"`, and the baseline module built around it. Both are
generated rather than transcribed, and both are byte comparisons: a class name, an attribute name,
the `dspy.Signature(...)` call when instructions are present and the bare string when they are not.

The signatures below are picked where a renderer drifts — a bare field, a typed one, several of
each, a non-scalar annotation, a literal set, and instructions present and absent — rather than for
prose any implementation agrees on.

    .venv/bin/python scripts/generate_flex_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "predict"
PINNED = require("dspy")

#: `(label, signature string, instructions)` — instructions change which branch builds `sig_arg`.
CASES = [
    ("bare", "question -> answer", None),
    ("bare with instructions", "question -> answer", "Answer well."),
    ("typed", "question: str -> answer: str", None),
    ("several", "question: str, context: str -> answer: str, confidence: float", None),
    ("non-scalar", "question: str -> tags: list[str]", None),
    ("int and bool", "n: int, flag: bool -> doubled: int", None),
    ("instructions with a quote", "question -> answer", "Say \"hello\" first."),
    # `repr` escapes control characters, and the generated source is Python that has to parse: a
    # literal newline inside a single-quoted string is an unterminated string, not a long one.
    ("multi-line instructions", "question -> answer", "Answer the question.\n\nBe brief."),
    ("instructions with a tab", "question -> answer", "Answer.\tBriefly."),
    ("instructions with a backslash", "question -> answer", "Escape \\n literally."),
    ("instructions with an apostrophe and a newline", "question -> answer", "Don't.\nEver."),
]


def named_with_descriptions():
    """A signature whose fields carry descriptions, and one that carries dspy's own placeholder.

    `render_signature_spec` prints a description unless it starts with `${`, which is what dspy fills
    in for a field nobody described — so a signature with no descriptions and one with placeholder
    descriptions render the same, and only a real one shows the branch.
    """

    class Described(dspy.Signature):
        """Answer carefully."""

        question: str = dspy.InputField(desc="The question asked.")
        context: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="A short answer.")

    return Described


def named_signature():
    """A declared subclass, whose own name is what `_class_name` builds from.

    Every string-built signature is called `StringSignature`, so the string cases cannot show that
    the name travels at all — they all render `StringSignatureModule`.
    """

    class QA(dspy.Signature):
        """Answer the question."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    return QA


def main() -> None:
    cases = []
    for label, signature, instructions in CASES:
        built = dspy.Signature(signature, instructions) if instructions else dspy.Signature(signature)
        flex = dspy.Flex(built)
        cases.append(
            {
                "label": label,
                "signature": signature,
                # What dspy *stored*, not what was passed: `Signature` runs a docstring through
                # `inspect.cleandoc`, which expands tabs — so a tab never reaches `repr` and a
                # comparison against the raw string would be testing two different inputs.
                "instructions": built.instructions,
                "rendered_signature": flex._flex_ctx.render_signature_string(),
                "class_name": flex._class_name(),
                "baseline_src": flex._baseline_src(),
            }
        )

    # The declared-subclass name, and the tools branch, which builds an `RLM` instead of a
    # `Predict` and names each tool in the constructor.
    def shout(text: str) -> str:
        """Shout it."""
        return text.upper()

    for label, built, tools in [
        ("declared subclass", named_signature(), None),
        ("with a tool", dspy.Signature("question -> answer"), [shout]),
    ]:
        flex = dspy.Flex(built, tools=tools) if tools else dspy.Flex(built)
        cases.append(
            {
                "label": label,
                "signature": None,
                "instructions": None,
                "rendered_signature": flex._flex_ctx.render_signature_string(),
                "class_name": flex._class_name(),
                "baseline_src": flex._baseline_src(),
            }
        )

    # A program holding one of each, so the state map carries both shapes at once. This is what a
    # map typed to predictor states cannot read back, and it is not a hypothetical: `dump_state`
    # writes them side by side.
    class Mixed(dspy.Module):
        def __init__(self):
            super().__init__()
            self.plain = dspy.Predict("question -> answer")
            self.flexed = dspy.Flex(dspy.Signature("question -> answer"))

        def forward(self, **kwargs):
            return self.plain(**kwargs)

    mixed_state = Mixed().dump_state()

    # The two string helpers the code proposer stands on. Both are small and both are exactly the
    # shape a transcription rounds off: `repr()` of a dict is Python's, not JSON's, and the fence
    # stripper drops the opening line whatever it says while only dropping a closing fence.
    from dspy.predict.flex.ctx import _strip_code_fences
    from dspy.teleprompt.gepa.gepa_flex_utils import _format_failures

    failure_cases = [
        ("none", []),
        ("one", [{"Inputs": {"question": "Where?"}, "Generated Outputs": {"answer": "x"}, "Feedback": "wrong"}]),
        ("two", [
            {"Inputs": {"a": 1}, "Generated Outputs": {"b": "it's"}, "Feedback": "no"},
            {"Inputs": {}, "Generated Outputs": None, "Feedback": None},
        ]),
        ("missing keys", [{"Feedback": "only feedback"}]),
    ]
    fence_cases = [
        "plain source",
        "```python\nclass M:\n    pass\n```",
        "```\nclass M:\n    pass\n```",
        "```python\nclass M:\n    pass",
        "  \n```py\n\tclass M:\n\t\tpass\n```\n  ",
        "```",
        "",
        "no fence but\ta tab",
        # A tab must advance to the next multiple of four *within its line*, not add four. Every
        # case above happens to sit at a column already divisible by four, so all of them agree
        # with a plain `replace("\\t", "    ")` — these do not, which is the point of them.
        "a\tb",
        "ab\tc",
        "abc\td",
        "abcd\te",
        "x\ty\tz",
    ]

    # What the code proposer is *shown*: the signature spelled out, and the tools in scope. Both are
    # rendering with edges — a description that is dspy's own `${placeholder}` is skipped, a tool's
    # blurb is the first line of its docstring, and each has an empty form the prompt branches on.
    def a_tool(name, doc):
        def fn(text: str) -> str:
            return text
        fn.__name__ = name
        fn.__doc__ = doc
        return fn

    context_cases = []
    for label, signature, tools in [
        ("bare", dspy.Signature("question -> answer"), None),
        ("described", named_with_descriptions(), None),
        ("one tool", dspy.Signature("question -> answer"), [a_tool("shout", "Shout it.\nSecond line ignored.")]),
        ("tool with no docstring", dspy.Signature("question -> answer"), [a_tool("quiet", None)]),
        ("two tools", dspy.Signature("question -> answer"),
         [a_tool("shout", "Shout it."), a_tool("whisper", "  Whisper it.  ")]),
    ]:
        flex = dspy.Flex(signature, tools=tools) if tools else dspy.Flex(signature)
        ctx = flex._flex_ctx
        context_cases.append({
            "label": label,
            "signature_spec": ctx.render_signature_spec(),
            "context_blurb": ctx.render_context_blurb(sandboxed=True),
            "context_blurb_unsandboxed": ctx.render_context_blurb(),
        })

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_flex_fixture.py",
        "dspy_version": PINNED,
        # Both vendored files, so a pin bump fails a test rather than going unnoticed. Written here
        # rather than by hand: this key was once added by a one-off script and the next run of *this*
        # generator dropped it, which left a doc comment naming a test that could not pass.
        "primitives_catalog": (
            pathlib.Path(dspy.__file__).parent / "predict" / "flex" / "primitives_doc.py"
        ).read_text().split('"""', 2)[1].removeprefix("\\\n"),
        "shim": (
            pathlib.Path(dspy.__file__).parent / "predict" / "flex" / "_sandbox_shim.py"
        ).read_text(),
        "cases": cases,
        "mixed_state": mixed_state,
        "format_failures": [
            {"label": label, "records": records, "rendered": _format_failures(records)}
            for label, records in failure_cases
        ],
        "task_context": context_cases,
        "strip_code_fences": [
            {"raw": raw, "stripped": _strip_code_fences(raw)} for raw in fence_cases
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "flex.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
