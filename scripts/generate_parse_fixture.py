"""Record what dspy's adapters parse *out* of a reply, and what they refuse.

Nineteen fixtures in this repo record the prompt the crate **sends**. None recorded what it **reads**,
and mutation testing said so plainly: 35 survivors in `adapter/parse.rs`, 20 in `next_tag` alone,
which could be made to return `Some(("xyzzy", "xyzzy", "xyzzy"))` with the whole suite green. The
byte claim runs in both directions and only one of them had an oracle.

The corpus is chosen from `ChatAdapter.parse`'s actual branches rather than from what a well-behaved
model emits — every shape here is one the end-to-end tests never produce:

  - a marker at the start of a line *with content after it on the same line*, which dspy keeps
    (`line[match.end():].strip()`) rather than treating as an empty field;
  - the same field twice, where the **first** occurrence wins (`k not in fields`);
  - a field the signature never declared, which is dropped rather than rejected;
  - a missing field, which raises — the keys are compared as a set at the end;
  - text before the first marker, which lands in a `None` section and is discarded;
  - blank lines around markers, since every section is `.strip()`ed;
  - a marker-looking line that is *indented*, which still matches because the pattern is applied to
    `line.strip()`;
  - `[[ ## completed ## ]]` present, absent, and repeated — it is not an output field, so it is
    simply not one of the keys.

**The refusals are recorded too, and matter as much as the successes.** A reply the crate accepts
where dspy raises is a divergence that reaches the caller as a wrong value rather than an error,
which is worse than the reverse.

    .venv/bin/python scripts/generate_parse_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "parse"
PINNED = require("dspy")


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()


class Typed(dspy.Signature):
    """Answer with types."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    score: int = dspy.OutputField()
    tags: list[str] = dspy.OutputField()


#: The XML adapter scans with `<(?P<name>\w+)>(.*?)</\1>` under DOTALL — a *global* find, a
#: non-greedy body, a backreferenced closing name, and `\w+` for the name. Every case below is one
#: of those properties, and none of them is a shape a well-behaved model emits.
XML_CASES = [
    ("xml_plain", QA, "<reasoning>Because.</reasoning>\n<answer>Paris</answer>"),
    # Non-greedy: the body stops at the *first* close, and the trailing one is left as text.
    ("xml_non_greedy", QA, "<reasoning>Because.</reasoning><answer>Paris</answer>extra</answer>"),
    # Same name nested: the non-greedy body ends at the inner close, so the body keeps a tag.
    ("xml_same_name_nested", QA, "<reasoning><reasoning>inner</reasoning></reasoning>\n<answer>Paris</answer>"),
    # A name that is not `\w+` does not open a tag at all.
    ("xml_hyphenated_name", QA, "<my-reasoning>Because.</my-reasoning>\n<reasoning>R</reasoning>\n<answer>Paris</answer>"),
    # An attribute puts a space after the name, so `\w+>` never matches.
    ("xml_tag_with_attribute", QA, '<reasoning id="1">Because.</reasoning>\n<answer>Paris</answer>'),
    ("xml_first_wins", QA, "<reasoning>first</reasoning>\n<reasoning>second</reasoning>\n<answer>Paris</answer>"),
    ("xml_unknown_tag", QA, "<reasoning>Because.</reasoning>\n<confidence>high</confidence>\n<answer>Paris</answer>"),
    ("xml_field_missing", QA, "<answer>Paris</answer>"),
    ("xml_unclosed", QA, "<reasoning>Because.\n<answer>Paris</answer>"),
    ("xml_mismatched_close", QA, "<reasoning>Because.</thinking>\n<answer>Paris</answer>"),
    # DOTALL, so a body spans lines; and the body is stripped.
    ("xml_multiline_body", QA, "<reasoning>\n  line one\n  line two\n</reasoning>\n<answer>Paris</answer>"),
    # Tags buried in prose are still found — `finditer` scans the whole string.
    ("xml_buried_in_prose", QA, "Sure! <reasoning>Because.</reasoning> and so <answer>Paris</answer> there."),
    ("xml_empty_body", QA, "<reasoning></reasoning>\n<answer>Paris</answer>"),
    ("xml_nothing_at_all", QA, ""),
    (
        "xml_typed_fields",
        Typed,
        '<answer>Paris</answer>\n<score>7</score>\n<tags>["a", "b"]</tags>',
    ),
    (
        "xml_typed_that_will_not_parse",
        Typed,
        "<answer>Paris</answer>\n<score>very high</score>\n<tags>[\"a\"]</tags>",
    ),
]

#: (name, signature, completion). Each is a branch of `parse`, not a plausible reply.
CASES = [
    ("plain", QA, "[[ ## reasoning ## ]]\nBecause.\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"),
    ("no_completed_marker", QA, "[[ ## reasoning ## ]]\nBecause.\n\n[[ ## answer ## ]]\nParis"),
    # Content on the same line as the marker: dspy keeps it rather than reading an empty field.
    ("content_on_the_header_line", QA, "[[ ## reasoning ## ]] Because.\n[[ ## answer ## ]] Paris"),
    # First occurrence wins.
    (
        "field_repeated",
        QA,
        "[[ ## reasoning ## ]]\nfirst\n\n[[ ## answer ## ]]\nParis\n\n[[ ## reasoning ## ]]\nsecond",
    ),
    # Declared order is reasoning then answer; the reply gives them the other way round.
    ("fields_out_of_order", QA, "[[ ## answer ## ]]\nParis\n\n[[ ## reasoning ## ]]\nBecause."),
    # A field the signature never declared.
    (
        "unknown_field",
        QA,
        "[[ ## reasoning ## ]]\nBecause.\n\n[[ ## confidence ## ]]\nhigh\n\n[[ ## answer ## ]]\nParis",
    ),
    # Missing one: raises.
    ("field_missing", QA, "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"),
    ("nothing_at_all", QA, ""),
    ("prose_only", QA, "I think the answer is Paris."),
    # Prose before the first marker goes into a None section and is dropped.
    (
        "prose_before_the_first_marker",
        QA,
        "Sure! Here you go.\n\n[[ ## reasoning ## ]]\nBecause.\n\n[[ ## answer ## ]]\nParis",
    ),
    # Every section is stripped, so the blank lines vanish.
    (
        "blank_lines_around_markers",
        QA,
        "\n\n[[ ## reasoning ## ]]\n\n\nBecause.\n\n\n\n[[ ## answer ## ]]\n\nParis\n\n\n",
    ),
    # The pattern is matched against `line.strip()`, so an indented marker still counts.
    (
        "indented_marker",
        QA,
        "[[ ## reasoning ## ]]\nBecause.\n\n    [[ ## answer ## ]]\nParis",
    ),
    # A marker inside a value: this line does not *start* with one, so it stays content.
    (
        "marker_inside_a_value",
        QA,
        "[[ ## reasoning ## ]]\nthe model wrote [[ ## answer ## ]] in the middle\n\n[[ ## answer ## ]]\nParis",
    ),
    # A value that spans lines keeps its interior newlines.
    (
        "multiline_value",
        QA,
        "[[ ## reasoning ## ]]\nline one\nline two\n\nline four\n\n[[ ## answer ## ]]\nParis",
    ),
    ("completed_repeated", QA, "[[ ## reasoning ## ]]\nBecause.\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]\n\n[[ ## completed ## ]]"),
    # Typed fields: a good parse and one that fails the annotation.
    (
        "typed_fields",
        Typed,
        '[[ ## answer ## ]]\nParis\n\n[[ ## score ## ]]\n7\n\n[[ ## tags ## ]]\n["a", "b"]\n\n[[ ## completed ## ]]',
    ),
    (
        "typed_field_that_will_not_parse",
        Typed,
        '[[ ## answer ## ]]\nParis\n\n[[ ## score ## ]]\nvery high\n\n[[ ## tags ## ]]\n["a"]',
    ),
    (
        "typed_field_empty",
        Typed,
        '[[ ## answer ## ]]\nParis\n\n[[ ## score ## ]]\n\n\n[[ ## tags ## ]]\n["a"]',
    ),
]


def spelling(annotation) -> str:
    """The annotation as a signature string spells it: `list[str]`, not `list`."""
    text = str(annotation)
    return text.removeprefix("<class '").removesuffix("'>")


def parsed(adapter, signature, completion: str) -> dict:
    """What the adapter returns, or the kind of error it raises."""
    try:
        return {"ok": True, "fields": adapter.parse(signature, completion)}
    except Exception as error:
        return {"ok": False, "error": type(error).__name__}


def main() -> None:
    adapters = {
        "chat": dspy.ChatAdapter(),
        "json": dspy.JSONAdapter(),
        "xml": dspy.XMLAdapter(),
    }
    cases = []
    for name, signature, completion in CASES + XML_CASES:
        cases.append(
            {
                "name": name,
                # Spelled the way the crate parses a signature string, `list[str]` and all —
                # `annotation.__name__` reports `list` and would lose the element type, which is
                # what decides whether `["a"]` parses.
                "signature": " -> ".join(
                    [
                        ", ".join(signature.input_fields),
                        ", ".join(
                            f"{field}: {spelling(signature.output_fields[field].annotation)}"
                            for field in signature.output_fields
                        ),
                    ]
                ),
                "completion": completion,
                "adapter": "xml" if name.startswith("xml_") else "chat",
                "chat": parsed(
                    adapters["xml" if name.startswith("xml_") else "chat"], signature, completion
                ),
                # The crate casts a scalar during *validation* rather than during parse. So a good
                # `int` comes back as the text that spells it, and the two values that will not fit
                # are accepted here and fail later with a typed message instead of at parse.
                # Recorded as a divergence so the comparison stays honest and turns red when
                # `parse-time-casting` resolves it, rather than being quietly skipped.
                "diverges": name
                in {"typed_fields", "typed_field_that_will_not_parse", "typed_field_empty"},
            }
        )

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_parse_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "What ChatAdapter.parse returns for each reply, and which it refuses. The refusals "
            "matter as much as the successes: a reply the crate accepts where dspy raises reaches "
            "the caller as a wrong value rather than an error."
        ),
        "cases": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "chat_parse.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)

    refused = [case["name"] for case in cases if not case["chat"]["ok"]]
    accepted = [case["name"] for case in cases if case["chat"]["ok"]]
    # A corpus of only-valid replies pins nothing about the refusals, which is half of what parse
    # decides. Refuse to write one.
    if not refused or not accepted:
        raise SystemExit("the corpus must contain both accepted and refused replies")
    print(f"  {len(accepted)} accepted, {len(refused)} refused: {refused}", file=sys.stderr)


if __name__ == "__main__":
    main()
