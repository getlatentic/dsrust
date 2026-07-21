"""Record what dspy's MultiChainComparison builds and sends, by running it.

Upstream ships one test for this module and it asserts on neither the signature it builds nor
the string it formats, so nothing but a recording holds the port to the bytes. Three of those
bytes are easy to get wrong in the other direction: the appended field names count from 1 while
the loop counts from 0, the corrected-reasoning field is *prepended* to the outputs rather than
appended, and the caller's own kwargs override the attempts rather than the other way round.

`prefix` is recorded beside every field even though dspy 3.2.1 no longer reads it — see the
`prefixes_are_recorded_but_never_rendered` note below. `system`/`user` are the messages a model
would actually receive, which is the only claim about faithfulness worth making.

    .dspy-venv/bin/python scripts/generate_mcc_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings

import dspy
from dspy.predict.multi_chain_comparison import MultiChainComparison

OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance" / "predict"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()

# Passing `prefix=` to a field is deprecated in 3.2.1 and MultiChainComparison still does it, so
# importing the module under test is itself a DeprecationWarning. Recording that fact is the
# point; being told about it three times a run is not.
warnings.filterwarnings("ignore", category=DeprecationWarning)


def base(instructions: str, inputs: list[tuple[str, str]], outputs: list[tuple[str, str]]):
    """A signature declared field by field, so the fixture can carry it to Rust as data."""
    fields = {
        **{name: (str, dspy.InputField(desc=desc)) for name, desc in inputs},
        **{name: (str, dspy.OutputField(desc=desc)) for name, desc in outputs},
    }
    return dspy.Signature(fields, instructions)


QA = base(
    "Answer questions with short factoid answers.",
    [("question", None)],
    [("answer", "often between 1 and 5 words")],
)

# Two outputs that differ, which is what makes "the *last* output field" a claim rather than a
# coincidence: reading the first would quote the citation where the answer belongs.
CITED = base(
    "Answer with a citation.",
    [("question", None), ("context", "what was retrieved")],
    [("citation", "where it came from"), ("answer", "the answer itself")],
)

# (name, signature, M, completions, caller kwargs)
CASES = [
    (
        "basic_qa",
        QA,
        3,
        [
            {"rationale": "I recall that during clear days, the sky often appears this color.",
             "answer": "blue"},
            {"rationale": "Based on common knowledge, I believe the sky is typically seen as "
                          "this color.", "answer": "green"},
            {"rationale": "From images and depictions in media, the sky is frequently "
                          "represented with this hue.", "answer": "blue"},
        ],
        {"question": "What is the color of the sky?"},
    ),
    # `c.get("rationale", c.get("reasoning"))`: a ChainOfThought completion names the field
    # `reasoning`, and only the fallback lets one be compared at all.
    (
        "reasoning_fallback",
        QA,
        2,
        [
            {"reasoning": "the sky scatters short wavelengths", "answer": "blue"},
            {"rationale": "rationale wins", "reasoning": "over reasoning", "answer": "azure"},
        ],
        {"question": "Why is the sky blue?"},
    ),
    # Only the first line survives, and both ends are stripped before and after that cut. The
    # \x1c is not idle: Python's str.strip() drops it and Rust's trim() does not.
    (
        "first_line_only",
        QA,
        3,
        [
            {"rationale": "  first line  \n  second line  ", "answer": " blue \n green "},
            {"rationale": "\x1c\x1d\x1e\x1f padded by separators \x1c", "answer": "\tgreen\t"},
            {"rationale": " non-breaking ", "answer": "　ideographic　"},
        ],
        {"question": "What is the color of the sky?"},
    ),
    # str(...) reaches the answer and nothing reaches the rationale, so a non-string answer is
    # spelled Python's way while a non-string rationale would raise.
    (
        "non_string_answers",
        QA,
        4,
        [
            {"rationale": "counted them", "answer": 5},
            {"rationale": "measured it", "answer": 1.5},
            {"rationale": "checked the box", "answer": True},
            {"rationale": "found nothing", "answer": None},
        ],
        {"question": "How many?"},
    ),
    (
        "last_output_field",
        CITED,
        2,
        [
            {"rationale": "the paper says so", "citation": "Smith 2019", "answer": "42"},
            {"rationale": "the table says so", "citation": "Jones 2020", "answer": "43"},
        ],
        {"question": "How many?", "context": "a pile of papers"},
    ),
    # M=1 still appends a field named _1, so the 1-indexing shows up where an off-by-one would
    # be invisible in a longer list.
    (
        "single_attempt",
        QA,
        1,
        [{"rationale": "only one student answered", "answer": "blue"}],
        {"question": "What is the color of the sky?"},
    ),
    # M=0 appends nothing and still prepends `rationale`, which separates the two edits.
    ("no_attempts", QA, 0, [], {"question": "What is the color of the sky?"}),
    # A caller kwarg of the same name overrides the attempt built for it, and keeps the
    # attempt's position — `{**attempts, **kwargs}`, not the reverse.
    (
        "caller_overrides_an_attempt",
        QA,
        3,
        [
            {"rationale": "one", "answer": "a"},
            {"rationale": "two", "answer": "b"},
            {"rationale": "three", "answer": "c"},
        ],
        {"reasoning_attempt_2": "the caller's own text", "question": "Which letter?"},
    ),
]

# Strings whose ends separate Python's str.strip() from Rust's trim(): the four separator
# controls are whitespace to Python and not to Unicode.
STRIPS = [
    "  padded  ",
    "\t\n\r\x0b\x0c ends \t\n\r\x0b\x0c",
    "\x1c\x1d\x1e\x1f separators \x1c\x1d\x1e\x1f",
    "     　 spaces 　",
    "\x85 next line \x85",
    "",
    "   ",
    "\x1c",
    "no padding",
]


def field_record(name: str, info) -> dict:
    extra = info.json_schema_extra
    return {
        "name": name,
        "desc": extra["desc"],
        "prefix": extra["prefix"],
        "annotation": info.annotation.__name__,
    }


def signature_record(signature) -> dict:
    return {
        "instructions": signature.instructions,
        "inputs": [field_record(n, f) for n, f in signature.input_fields.items()],
        "outputs": [field_record(n, f) for n, f in signature.output_fields.items()],
    }


class Recorder:
    """Stands in for the inner Predict, so `forward` runs and stops at the model call."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return None


def case(name, signature, m, completions, caller) -> dict:
    module = MultiChainComparison(signature, M=m)
    built = module.predict.signature
    recorder = Recorder()
    module.predict = recorder
    module([dspy.Prediction(**c) for c in completions], **caller)

    system, user = dspy.ChatAdapter().format(
        signature=built, demos=[], inputs=recorder.kwargs
    )
    return {
        "name": name,
        "m": m,
        "base": signature_record(signature),
        "last_key": module.last_key,
        "built": signature_record(built),
        "completions": completions,
        "caller_inputs": caller,
        # A list of pairs, not an object: the order dspy hands to Predict is part of what is
        # being recorded, and a JSON object would not carry it.
        "sent": list(recorder.kwargs.items()),
        "system": system["content"],
        "user": user["content"],
    }


def count_mismatch(given: int, m: int) -> dict:
    module = MultiChainComparison(QA, M=m)
    module.predict = Recorder()
    attempts = [dspy.Prediction(rationale="r", answer="a") for _ in range(given)]
    try:
        module(attempts, question="q")
    except AssertionError as error:
        return {"m": m, "given": given, "message": str(error)}
    raise SystemExit(f"expected {given} attempts against M={m} to fail")


def no_output_fields() -> str:
    try:
        MultiChainComparison(base("Nothing to say.", [("question", None)], []))
    except ValueError as error:
        return str(error)
    raise SystemExit("expected a signature with no output fields to fail")


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_mcc_fixture.py",
        "dspy_version": PINNED,
        "prefixes_are_recorded_but_never_rendered": (
            "dspy 3.2.1 deprecated the prefix= argument to InputField/OutputField: it is still "
            "stored on the field and still compared by Signature.equals, but no adapter reads "
            "it. Compare a field's prefix here against the system message and it is absent."
        ),
        "cases": [case(*spec) for spec in CASES],
        "count_mismatch": [count_mismatch(2, 3), count_mismatch(4, 3), count_mismatch(1, 0)],
        "no_output_fields": no_output_fields(),
        "python_strip": [{"text": text, "stripped": text.strip()} for text in STRIPS],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "mcc.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
