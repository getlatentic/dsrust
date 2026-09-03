"""Record what restoring a saved signature does when its shape no longer fits, by running dspy.

A saved signature is a list of `{prefix, description}` with no names on it, zipped back onto the
live fields in order — inputs first, then outputs. Upstream passes `strict=False` to that `zip`
deliberately, so a program that has gained or lost a field since it was saved restores what lines
up and says nothing.

That is worth pinning rather than describing. A `ChainOfThought` grows a *leading* output, so every
saved field after the inputs lands one position late and the reasoning renders under the next
field's prefix — a silently wrong prompt from a file that loaded without complaint. The port has to
keep doing exactly this, and the only way to know it still does is to have asked dspy.

    .dspy-venv/bin/python scripts/generate_signature_state_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "state"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()


def saved(instructions: str, fields: list[tuple[str, str]]) -> dict:
    return {
        "instructions": instructions,
        "fields": [{"prefix": prefix, "description": desc} for prefix, desc in fields],
    }


CASES = [
    {
        "what": "the shape it was saved from",
        "live": "question -> answer",
        "state": saved("Answer it.", [("Q:", "the question"), ("A:", "the answer")]),
    },
    {
        # What a node becoming a `ChainOfThought` does: `reasoning` is an output, and outputs come
        # after every input, so it lands between them and the saved outputs slide one place.
        "what": "the program gained a leading output since it was saved",
        "live": "question -> reasoning, answer",
        "state": saved("Answer it.", [("Q:", "the question"), ("A:", "the answer")]),
    },
    {
        "what": "the program gained an input since it was saved",
        "live": "question, context -> answer",
        "state": saved("Answer it.", [("Q:", "the question"), ("A:", "the answer")]),
    },
    {
        "what": "the program lost a field since it was saved",
        "live": "question -> answer",
        "state": saved(
            "Answer it.",
            [("Q:", "the question"), ("A:", "the answer"), ("E:", "the evidence")],
        ),
    },
    {
        "what": "the saved state has no fields at all",
        "live": "question -> answer",
        "state": saved("Only the instructions survived.", []),
    },
    {
        "what": "every field is restored onto a signature of inputs alone",
        "live": "question, context -> answer",
        "state": saved(
            "Answer it.",
            [("Q:", "the question"), ("C:", "the context"), ("A:", "the answer")],
        ),
    },
]


def described(signature) -> list[dict]:
    """Each field in the order the loader zips onto: inputs, then outputs."""
    return [
        {
            "name": name,
            "prefix": field.json_schema_extra["prefix"],
            "description": field.json_schema_extra["desc"],
        }
        for name, field in signature.fields.items()
    ]


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")

    recorded = []
    for case in CASES:
        live = dspy.Signature(case["live"])
        restored = live.load_state(case["state"])
        recorded.append(
            {
                **case,
                "restored": {
                    "instructions": restored.instructions,
                    "fields": described(restored),
                },
            }
        )

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_signature_state_fixture.py",
        "dspy_version": PINNED,
        "cases": recorded,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "signature_state.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.name}: {len(recorded)} cases", file=sys.stderr)
    for entry in recorded:
        landed = ", ".join(
            f"{field['name']}={field['prefix']}" for field in entry["restored"]["fields"]
        )
        print(f"    {entry['what']}: {landed}", file=sys.stderr)


if __name__ == "__main__":
    main()
