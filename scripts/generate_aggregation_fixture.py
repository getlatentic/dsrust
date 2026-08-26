"""Record what dspy's majority vote decides, by running it.

The vote has three details a port gets wrong quietly: the field it reads is the *last* one rather
than the first, normalisation happens before counting rather than after, and a tie is broken
toward whichever value was seen first. Python's `max` over a dict returns the first key at the
maximum; Rust's `max_by_key` returns the last, which inverts exactly that.

    .dspy-venv/bin/python scripts/generate_aggregation_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.evaluate.metrics import normalize_text
from dspy.predict.aggregation import majority

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "predict"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()

# Strings that separate the five normalisation steps from one another.
TEXTS = [
    "The,  Eiffel  Tower!",
    "Paris",
    "  paris  ",
    "THE PARIS",
    "a paris",
    "an apple",
    "the",
    "a",
    "",
    "   ",
    "...",
    "don't",
    "Paris, France.",
    "hello   world",
    # Articles only bound at word boundaries: "another" keeps its "an".
    "another",
    "banana",
    # NFD decomposes an accent before lowercasing, and the combining mark is not punctuation.
    "café",
    "CAFÉ",
    "naïve",
    "Ω",
]

# (completions, field, mode). Three modes, because the parameter has three meanings and only
# one of them is "as written": omitting it uses `default_normalize`, which maps an empty result
# to None so the completion is skipped entirely; passing None means identity; passing
# normalize_text normalises without the None mapping. An article like "a" therefore vanishes
# under the default and survives under identity.
VOTES = [
    ([{"answer": "2"}, {"answer": "2"}, {"answer": "3"}], None, "default"),
    ([{"answer": "2"}, {"answer": "3"}, {"answer": "4"}], None, "default"),
    ([{"answer": "2"}, {"answer": " 2"}, {"answer": "3"}], None, "text"),
    # A tie under identity, where the earliest value wins.
    ([{"answer": "a"}, {"answer": "b"}, {"answer": "b"}, {"answer": "a"}], None, "identity"),
    # The same votes under the default, where "a" is an article and drops out entirely.
    ([{"answer": "a"}, {"answer": "b"}, {"answer": "b"}, {"answer": "a"}], None, "default"),
    ([{"answer": "x"}, {"answer": "y"}, {"answer": "y"}, {"answer": "x"}], None, "identity"),
    ([{"question": "q", "answer": "1"}, {"question": "q", "answer": "2"}], None, "default"),
    ([{"answer": "x", "other": "1"}, {"answer": "y", "other": "1"}], "other", "default"),
    # The two fields elect different completions, which is what makes "the last field" a claim
    # rather than a coincidence: voting on `first` returns index 0, voting on `last` index 1.
    (
        [
            {"first": "p", "last": "q"},
            {"first": "p", "last": "r"},
            {"first": "z", "last": "r"},
        ],
        None,
        "default",
    ),
    ([{"answer": "The Eiffel Tower"}, {"answer": "the  eiffel  tower"}], None, "text"),
    # Every value normalises to None, so the count falls back to the unfiltered list of Nones.
    ([{"answer": ""}, {"answer": "  "}], None, "default"),
    ([{"answer": "only"}], None, "default"),
]


def vote(completions: list[dict], field: str | None, mode: str) -> dict:
    prediction = dspy.Prediction.from_completions(
        {key: [c[key] for c in completions] for key in completions[0]}
    )
    kwargs = {}
    if field is not None:
        kwargs["field"] = field
    if mode == "identity":
        kwargs["normalize"] = None
    elif mode == "text":
        kwargs["normalize"] = normalize_text
    winner = majority(prediction, **kwargs)
    return {
        "completions": completions,
        "field": field,
        "normalize": mode,
        "winner": dict(winner.completions[0].items()),
    }


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_aggregation_fixture.py",
        "dspy_version": PINNED,
        "normalize_text": [{"text": text, "normalized": normalize_text(text)} for text in TEXTS],
        "majority": [vote(*case) for case in VOTES],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "aggregation.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
