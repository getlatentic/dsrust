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
from typing import Any
import pathlib
import sys

from typing import Literal

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


class Chosen(dspy.Signature):
    """Pick from a closed set.

    `parse_value`'s `Literal` branch, which is reached before every generic one and is the only
    place upstream refuses a value for *not being a member*. It also unwraps what a model tends to
    wrap a member in — surrounding quotes, and a `Literal[...]` or `str[...]` spelling of the
    annotation — so those arrive as the bare member rather than as a refusal.
    """

    question: str = dspy.InputField()
    colour: Literal["red", "blue"] = dspy.OutputField()


class Underscored(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    final_answer: str = dspy.OutputField()


class Unicode(dspy.Signature):
    """Answer in the caller's language.

    Both scans are spelled `\\w+`, and Python's `\\w` is `str.isalnum()` plus `_` rather than ASCII.
    A Python identifier may be any of it, so these are field names dspy renders markers for and
    reads back — and the crate refused every one of them until this signature was added.
    """

    question: str = dspy.InputField()
    réponse: str = dspy.OutputField()
    答え: str = dspy.OutputField()


class Freeform(dspy.Signature):
    """Answer, and note anything.

    `note: Any` is the one annotation that reaches `parse_value`'s literal fallback and hands the
    result back unvalidated — a scalar drags in `parse-time-casting`, and a `list[str]` turns the
    interesting cases into refusals whose messages may not separate the branch under test.
    """

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    note: Any = dspy.OutputField()


class Tagged(dspy.Signature):
    """Answer with a list, and no scalar to cast.

    `Typed` cannot isolate what json-repair does to a structured field, because its `score: int`
    drags in `parse-time-casting` and every case using it is recorded as a divergence for a reason
    that has nothing to do with the field under test.
    """

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
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
    # `\w` is `str.isalnum()` plus `_`, which is neither ASCII nor `char::is_alphanumeric`. A
    # combining mark is Alphabetic and *not* alnum, so `<xֺ>` opens a tag in Rust's predicate and
    # not in Python's — and a tag is consumed whole, so the `<reasoning>` it wraps goes with it.
    ("xml_name_with_a_combining_mark", QA, "<xְ><reasoning>Because.</reasoning></xְ>\n<answer>Paris</answer>"),
    # And the other end of the same predicate: a tag named in a script an ASCII test never reaches.
    ("xml_name_beyond_ascii", Unicode, "<réponse>Paris</réponse>\n<答え>はい</答え>"),
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
    # `\w` includes the underscore, so an underscored name opens a tag like any other. Nothing
    # exercised one, and a mutation reading `letter != '_'` instead of `== '_'` survived on that:
    # it rejects exactly the names most signatures use.
    ("xml_underscored_name", Underscored, "<final_answer>Paris</final_answer>"),
    # A declared field *inside* a non-word tag. The hyphenated case above only showed that such a
    # tag is not itself a field; this shows the scan resumes one character past the `<` and finds
    # what the tag wraps, rather than swallowing it whole.
    (
        "xml_field_inside_a_non_word_tag",
        QA,
        "<my-wrapper><reasoning>Because.</reasoning></my-wrapper>\n<answer>Paris</answer>",
    ),
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

#: `JSONAdapter.parse` is `json_repair.loads`, then — if that did not yield a dict — a *recursive*
#: brace regex to pull the outermost object out of surrounding text and repair that, then a filter to
#: the declared fields, then a cast. Each case below is one of those steps.
JSON_CASES = [
    ("json_plain", QA, '{"reasoning": "Because.", "answer": "Paris"}'),
    # Not a dict at the top: the brace regex has to find the object inside.
    ("json_in_prose", QA, 'Sure! {"reasoning": "Because.", "answer": "Paris"} there you go.'),
    ("json_in_a_fence", QA, '```json\n{"reasoning": "Because.", "answer": "Paris"}\n```'),
    # The regex is recursive, so a nested object does not stop the outer match early.
    (
        "json_nested_object_in_prose",
        QA,
        'text {"reasoning": {"why": "Because."}, "answer": "Paris"} more',
    ),
    ("json_array_holding_the_object", QA, '[{"reasoning": "Because.", "answer": "Paris"}]'),
    # Shapes json_repair fixes rather than refuses.
    ("json_trailing_comma", QA, '{"reasoning": "Because.", "answer": "Paris",}'),
    ("json_single_quotes", QA, "{'reasoning': 'Because.', 'answer': 'Paris'}"),
    ("json_unquoted_keys", QA, '{reasoning: "Because.", answer: "Paris"}'),
    ("json_missing_closing_brace", QA, '{"reasoning": "Because.", "answer": "Paris"'),
    ("json_python_literals", QA, "{'reasoning': None, 'answer': 'Paris'}"),
    # Filtered, not refused.
    ("json_extra_field", QA, '{"reasoning": "Because.", "answer": "Paris", "confidence": "high"}'),
    ("json_field_missing", QA, '{"answer": "Paris"}'),
    ("json_not_an_object_at_all", QA, "just some prose with no braces"),
    ("json_empty", QA, ""),
    # Typed values, native and as the strings a model often writes.
    (
        "json_typed_native",
        Typed,
        '{"answer": "Paris", "score": 7, "tags": ["a", "b"]}',
    ),
    (
        "json_typed_as_strings",
        Typed,
        # Built rather than written out: a `tags` whose value is a *string* containing JSON
        # needs escaped quotes, and spelling those in a literal is how this case silently
        # became malformed instead of testing what it meant to.
        json.dumps({"answer": "Paris", "score": "7", "tags": json.dumps(["a", "b"])}),
    ),
    (
        # The malformed shape that mistake produced, kept because it is a real one: an
        # unescaped quote inside a string, which only `json_repair`'s heuristics recover.
        # It was recorded as a divergence until `dsrust-json-repair` landed, and the test
        # asserting the divergence is what said so.
        "json_unescaped_quote_inside_a_string",
        Typed,
        '{"answer": "Paris", "score": "7", "tags": "["a", "b"]"}',
    ),
    (
        "json_typed_that_will_not_parse",
        Typed,
        '{"answer": "Paris", "score": "very high", "tags": ["a"]}',
    ),
    (
        # Two objects in one reply: dspy's `\\{(?:[^{}]|(?R))*\\}` takes the *first*, where a search
        # from the first brace to the last would take both and the prose between them.
        "json_two_objects_and_prose",
        QA,
        'first {"reasoning": "Because.", "answer": "Paris"} then {"answer": "Berlin"}',
    ),
]

#: (name, signature, completion). Each is a branch of `parse`, not a plausible reply.
CASES = [
    # `parse_value`'s Literal branch: a member as it stands, the three wrappings upstream unwraps,
    # and the two refusals. Case matters — a member is matched exactly, never folded.
    ("literal_member", Chosen, "[[ ## colour ## ]]\nred"),
    ("literal_quoted", Chosen, '[[ ## colour ## ]]\n"red"'),
    ("literal_single_quoted", Chosen, "[[ ## colour ## ]]\n'red'"),
    ("literal_annotation_spelled", Chosen, "[[ ## colour ## ]]\nLiteral[red]"),
    ("literal_str_spelled", Chosen, "[[ ## colour ## ]]\nstr[red]"),
    ("literal_annotation_and_quotes", Chosen, '[[ ## colour ## ]]\nLiteral["red"]'),
    ("literal_surrounded_by_space", Chosen, "[[ ## colour ## ]]\n  red  "),
    ("literal_not_a_member", Chosen, "[[ ## colour ## ]]\ngreen"),
    ("literal_wrong_case", Chosen, "[[ ## colour ## ]]\nRED"),
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
    # dspy's header pattern is `\[\[ ## (?P<name>\w+) ## \]\]`, so a name with punctuation in it
    # does not open a section and the line is content. A mutation reading the name check as an
    # *or* survived without this: it makes every such line a header.
    (
        "non_word_marker_name",
        QA,
        "[[ ## reasoning ## ]]\nBecause.\n[[ ## my-note ## ]]\nstill reasoning\n"
        "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]",
    ),
    # The same, with an empty name — the other half of that check.
    (
        "empty_marker_name",
        QA,
        "[[ ## reasoning ## ]]\nBecause.\n[[ ##  ## ]]\nstill reasoning\n"
        "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]",
    ),
    (
        "indented_marker",
        QA,
        "[[ ## reasoning ## ]]\nBecause.\n\n    [[ ## answer ## ]]\nParis",
    ),
    # `\[\[ ## (\w+) ## \]\]` names the field with `\w+`, which is `str.isalnum()` plus `_` and not
    # ASCII. A Python identifier may be non-ASCII, so dspy renders and reads these markers — and
    # `split_header` required `is_ascii_alphanumeric`, refusing the whole reply as having no
    # sections at all.
    (
        "marker_name_beyond_ascii",
        Unicode,
        "[[ ## réponse ## ]]\nParis\n\n[[ ## 答え ## ]]\nはい\n\n[[ ## completed ## ]]\n",
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
    # A structured field written the ways a model writes one. Each reaches `parse_value`, which
    # hands the section to json-repair before the annotation ever sees it — so an unclosed bracket
    # and Python's quoting both land as `list[str]` rather than as the text spelling one.
    (
        "tags_unclosed",
        Tagged,
        '[[ ## answer ## ]]\nParis\n\n[[ ## tags ## ]]\n["a", "b"',
    ),
    (
        "tags_single_quoted",
        Tagged,
        "[[ ## answer ## ]]\nParis\n\n[[ ## tags ## ]]\n['a', 'b']",
    ),
    (
        # The shapes only json-repair recovers. This crate's own literal reader closes a container
        # and rewrites Python's spelling; it does none of these, so each is a section that used to
        # arrive as the text spelling a list rather than as one.
        "tags_bare_words",
        Tagged,
        "[[ ## answer ## ]]\nParis\n\n[[ ## tags ## ]]\n[a, b]",
    ),
    (
        "tags_asymmetric_quotes",
        Tagged,
        '[[ ## answer ## ]]\nParis\n\n[[ ## tags ## ]]\n["a, "b"]',
    ),
    (
        "tags_smart_quotes",
        Tagged,
        "[[ ## answer ## ]]\nParis\n\n[[ ## tags ## ]]\n[\u201ca\u201d, \u201cb\u201d]",
    ),
    # The literal fallback itself, on the branch order the code comments name: json-repair first,
    # and Python's own literal syntax only where json-repair answered the empty string — its
    # "found nothing" report. Each case is one arm. A mutant deleting the `!` in the crate's
    # `!text.is_empty()` guard survived the whole adapter slice because nothing pinned any of
    # these: the fuzz corpus's signature has no field that reaches the fallback distinguishably.
    (
        "note_single_quoted_literal",
        Freeform,
        "[[ ## answer ## ]]\nParis\n\n[[ ## note ## ]]\n'a'",
    ),
    (
        "note_python_true",
        Freeform,
        "[[ ## answer ## ]]\nParis\n\n[[ ## note ## ]]\nTrue",
    ),
    (
        "note_bare_word",
        Freeform,
        "[[ ## answer ## ]]\nParis\n\n[[ ## note ## ]]\nhello",
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


def which(name: str) -> str:
    """Which adapter a case belongs to, by its name's prefix."""
    return name.split("_", 1)[0] if name.startswith(("xml_", "json_")) else "chat"


def spelling(annotation) -> str:
    """The annotation as a signature string spells it: `list[str]`, not `list`."""
    text = str(annotation)
    return text.removeprefix("<class '").removesuffix("'>")


def parsed(adapter, signature, completion: str) -> dict:
    """What the adapter returns, or how it refuses — the message, not only the class.

    Every refusal these three adapters raise is an `AdapterParseError`, so recording the class alone
    leaves the Rust side comparing a refusal against a refusal and nothing else. The message is
    where the adapter name, the reply it read and the fields it wanted actually appear.
    """
    try:
        return {"ok": True, "fields": adapter.parse(signature, completion)}
    except Exception as error:
        message = str(error)
        # A cast failure carries **pydantic's** rendering — the field's whole repr, the
        # `[type=int_parsing, input_value=...]` tail, and a versioned docs URL. Reproducing that
        # would mean reimplementing pydantic's error formatter and pinning its version in a Rust
        # string, so this one refuses with the crate's own text and says so.
        return {
            "ok": False,
            "error": type(error).__name__,
            "message": message,
            "message_diverges": "pydantic-error-text" if "errors.pydantic.dev" in message else None,
        }


def main() -> None:
    adapters = {
        "chat": dspy.ChatAdapter(),
        "json": dspy.JSONAdapter(),
        "xml": dspy.XMLAdapter(),
    }
    cases = []
    for name, signature, completion in CASES + XML_CASES + JSON_CASES:
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
                "adapter": which(name),
                "dspy": parsed(adapters[which(name)], signature, completion),
                # `parse-time-casting` closed 2026-08-24: the crate casts at parse as dspy does, so
                # the three typed cases that were flagged here are compared like the rest. Nothing
                # is flagged now, and the field stays so the next divergence has somewhere to go.
                "diverges": False,
            }
        )

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_parse_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "What each adapter's parse returns for a reply, and which it refuses. The refusals "
            "matter as much as the successes: a reply the crate accepts where dspy raises reaches "
            "the caller as a wrong value rather than an error."
        ),
        "cases": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "adapter_parse.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)

    refused = [case["name"] for case in cases if not case["dspy"]["ok"]]
    accepted = [case["name"] for case in cases if case["dspy"]["ok"]]
    # A corpus of only-valid replies pins nothing about the refusals, which is half of what parse
    # decides. Refuse to write one.
    if not refused or not accepted:
        raise SystemExit("the corpus must contain both accepted and refused replies")

    # Both scans name their field with `\w+`, and an all-ASCII corpus cannot tell that predicate
    # from `is_ascii_alphanumeric` or from `char::is_alphanumeric` — the crate shipped one of each,
    # in opposite directions, and 58 cases said nothing about either. A corpus that loses these is
    # blind to the whole class again, so it is refused rather than left to a reader.
    beyond_ascii = [
        case["name"]
        for case in cases
        if not (case["signature"] + case["completion"]).isascii()
    ]
    if not beyond_ascii:
        raise SystemExit(
            "no case has a field or tag name beyond ASCII — the corpus cannot see the `\\w+` "
            "predicate, which is what both scans decide a name with"
        )
    print(f"  {len(accepted)} accepted, {len(refused)} refused: {refused}", file=sys.stderr)
    print(f"  {len(beyond_ascii)} beyond ASCII: {beyond_ascii}", file=sys.stderr)


if __name__ == "__main__":
    main()
