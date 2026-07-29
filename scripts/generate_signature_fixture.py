"""Record what dspy's signature layer decides, by running it.

`signatures/test_signature.py` asserts on behaviour that never reaches an adapter, so running it
through the bridge proves nothing until the crate owns the decisions it makes. This fixture is
what the Rust side is held to, and it is generated rather than transcribed for the usual reason:
a hand-copied expectation only tests the copying.

    .dspy-venv/bin/python scripts/generate_signature_fixture.py

`infer_prefix` turns a field name into the prefix an adapter prints. Upstream reaches it through
four regular expressions applied in order, and the cases below walk the boundaries between them:
where a capital run meets a lowercase one, where a letter meets a digit, and where a word is an
acronym that must survive title casing intact.
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.signatures.signature import infer_prefix

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "signature"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()

NAMES = [
    # The four cases upstream's own docstring names.
    "camelCaseText",
    "snake_case_text",
    "text2number",
    "HTMLParser",
    # Single words, and the empty edge.
    "",
    "a",
    "A",
    "question",
    "Question",
    # Capital runs against lowercase runs, which is where the first two expressions disagree.
    "HTML",
    "HTMLParserError",
    "parseHTML",
    "parseHTMLNow",
    "AParser",
    "ABCdef",
    "aB",
    "aBc",
    "aBcDe",
    # Digits on either side of a letter, which the third and fourth expressions handle.
    "2text",
    "text2",
    "text2number3more",
    "x2y",
    "abc123",
    "123abc",
    "123",
    "v2Model",
    "modelV2",
    # Underscores already present, including the degenerate runs.
    "_leading",
    "trailing_",
    "double__underscore",
    "_",
    "__",
    "snake_case_HTML",
    "MixedCase_with_snake",
    # Non-ASCII, because title casing is per-word and the crate cannot assume one byte per char.
    "café_au_lait",
    "naïveParser",
    "ΑΒΓdelta",
    "ПриветМир",
    # `[A-Z]` and `[a-z]` are ASCII in upstream's expressions while `\\d` is not, so a non-ASCII
    # digit splits where a non-ASCII capital does not. Pinned because a port reaching for a
    # Unicode-aware `is_uppercase` would silently disagree here and nowhere else.
    "text٢number",
    "٢text",
    # `\\d` is decimal digits only. Rust's `is_numeric` also takes letter-numerics like Ⅻ and
    # other-numerics like ½, so these pin the two apart.
    "Ⅻtext",
    "text½number",
    # An ASCII capital opening a non-ASCII lowercase word. The first expression's `[a-z]+` does
    # not reach it, and the second cannot either because the character before is not `[a-z0-9]`,
    # so this stays one word where a Unicode-aware port would split it.
    "ABécole",
    # A non-ASCII letter against a digit, on both sides. `[a-zA-Z]` does not reach either.
    "café2",
    "2école",
]


# Signature strings, spelled the way a caller writes them. The refusals are here too, because
# what a malformed signature is told is as much upstream's behaviour as what a good one becomes.
SIGNATURES = [
    "email -> sentiment",
    "question -> answer",
    "question, context -> answer",
    "question, context -> reasoning, answer",
    "a: int -> b: float",
    "a: int, b: float, c: bool -> d: str",
    "ctx: list[str] -> answer",
    "ctx: list[str], weights: dict[str, int] -> answer",
    "nested: dict[str, list[int]] -> out: tuple[int, str]",
    "  spaced   ,  out  ->   answer  ",
    "x: Optional[int] -> y: str",
    "x: int | None -> y: str",
    "x: int | str -> y: str",
    "x: int | str | None -> y: str",
    "x: list[int] | None -> y: str",
    "x: dict[str, int | None] -> y: str",
    "x: list[int | str] -> y: str",
    "x: tuple[int | None, str] -> y: str",
    "x: Union[int, str] -> y: str",
    "x: Optional[list[str]] -> y: str",
    # Refused, each for its own reason.
    "",
    "question",
    "a -> b -> c",
    "a, b -> b, a",
    "a -> a",
]


def parsed(spelling: str) -> dict:
    """What dspy makes of one signature string, or what it says when it will not."""
    try:
        signature = dspy.Signature(spelling)
    except Exception as error:  # noqa: BLE001 - the message is the recorded behaviour
        return {"signature": spelling, "error": str(error)}
    return {
        "signature": spelling,
        "instructions": signature.instructions,
        "inputs": [
            {"name": name, "annotation": _annotation(field)}
            for name, field in signature.input_fields.items()
        ],
        "outputs": [
            {"name": name, "annotation": _annotation(field)}
            for name, field in signature.output_fields.items()
        ],
    }


def _annotation(field) -> str:
    """The annotation as dspy holds it, which is a resolved type rather than the source text.

    That resolution is why `int | None` comes back as `Optional[int]`: upstream parses the
    annotation into a Python type and the spelling it prints afterwards is that type's own. A
    port owns the structure around the annotation, not the type system that canonicalises it.
    """
    annotation = field.annotation
    if annotation in (str, int, float, bool):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_signature_fixture.py",
        "dspy_version": PINNED,
        "infer_prefix": [{"name": name, "prefix": infer_prefix(name)} for name in NAMES],
        "parse": [parsed(spelling) for spelling in SIGNATURES],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "signature.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)
    print(f"  infer_prefix: {len(NAMES)} names", file=sys.stderr)


if __name__ == "__main__":
    main()
