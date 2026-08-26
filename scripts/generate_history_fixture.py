"""Record what a `dspy.History` input renders to, including the fields an exchange never recorded.

`format_conversation_history` replays each past exchange as a user turn and an assistant turn, and
it calls `format_assistant_message_content(signature, values)` with **no** `missing_field_message` —
so `outputs.get(name, None)` substitutes Python's `None`, which every field type formats as the four
letters `None`. The crate rendered the sentence "Not supplied for this conversation history
message. " there instead, on the strength of that string existing in `adapters/base.py`. It does
exist, at exactly one site: `format_demos`'s *complete* branch, where by construction no field is
missing. Nothing renders it.

Nothing caught that, because no fixture rendered a history at all. The cases here cover:

  - an exchange missing one output, and one missing several of different annotations, since the
    substituted value goes through `format_field_value` and a `list[str]` could plausibly have come
    out as `null` rather than `None`;
  - a complete exchange beside it, so the fixture also pins the ordinary path;
  - a demo missing an output, which is the *other* `missing_field_message` — the one that is real —
    so both strings are held by a golden rather than by a transcription.

    .venv/bin/python scripts/generate_history_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

from pins import require

OUT = (
    pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "history"
)
PINNED = require("dspy")


class Chat(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()
    tags: list[str] = dspy.OutputField()
    score: int = dspy.OutputField()


class Plain(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    confidence: str = dspy.OutputField()


def rendered(signature, demos: list, values: dict) -> dict:
    """The turns dspy's ChatAdapter produces, with no model involved."""
    messages = dspy.ChatAdapter().format(signature, demos, values)
    return {
        "system": messages[0]["content"],
        "turns": [
            {"role": message["role"], "content": message["content"]} for message in messages[1:]
        ],
    }


def main() -> None:
    exchanges = [
        # Every output missing but `answer` — the case the crate got wrong.
        {"question": "capital of Germany?", "answer": "Berlin"},
        # Complete, so the ordinary replay path is pinned beside it.
        {"question": "capital of Spain?", "answer": "Madrid", "tags": ["geo"], "score": 9},
        # Nothing but the input, so every output substitutes.
        {"question": "capital of Peru?"},
    ]
    history = rendered(
        Chat,
        [],
        {"question": "capital of France?", "history": dspy.History(messages=exchanges)},
    )

    # The other `missing_field_message`, which is real: an incomplete demo announces itself and
    # renders "Not supplied for this particular example. " for what it lacks.
    demos = rendered(
        Plain,
        [
            {"question": "capital of Italy?", "answer": "Rome"},  # confidence missing
            {"question": "capital of Japan?", "answer": "Tokyo", "confidence": "high"},
        ],
        {"question": "capital of France?"},
    )

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_history_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "A history exchange substitutes Python's None for a field it never recorded, whatever "
            "the annotation. An incomplete demo substitutes 'Not supplied for this particular "
            "example. '. They are different strings from different call sites and the crate had "
            "them confused."
        ),
        "history": {"exchanges": exchanges, **history},
        "demos": {
            "examples": [
                {"question": "capital of Italy?", "answer": "Rome"},
                {"question": "capital of Japan?", "answer": "Tokyo", "confidence": "high"},
            ],
            **demos,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "conversation_history.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)

    # A golden that renders no substituted field cannot fail the thing it was written for.
    if "None" not in json.dumps(fixture["history"]["turns"]):
        raise SystemExit("no history turn substitutes a missing field")
    if "Not supplied for this particular example." not in json.dumps(fixture["demos"]["turns"]):
        raise SystemExit("no demo turn substitutes a missing field")
    print("  both substitutions present", file=sys.stderr)


if __name__ == "__main__":
    main()
