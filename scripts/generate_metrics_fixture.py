"""Record what dspy's answer metrics score, by running them.

`normalize_text` is what every metric in `evaluate/metrics.py` agrees on, and it is where a port
drifts quietly: the NFD pass, the ASCII-only punctuation set, the `\\b(a|an|the)\\b` article regex
and the whitespace collapse each have an edge a reimplementation rounds off. So the inputs below
are chosen for those edges — accented text that NFD splits, Unicode punctuation the ASCII set does
*not* strip, articles against word boundaries (`theatre`, `the,cat`, `a_b`), the HotPotQA labels,
repeated tokens, and both empty sides — rather than for prose that any implementation agrees on.

    .venv/bin/python scripts/generate_metrics_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

from dspy.evaluate.metrics import (
    EM,
    F1,
    HotPotF1,
    em_score,
    f1_score,
    hotpot_f1_score,
    normalize_text,
    precision_score,
)

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance" / "evaluate"
PINNED = require("dspy")

#: Inputs picked for where a normalisation drifts, not where it agrees.
TEXTS = [
    "The,  Eiffel  Tower!",
    "A cat and an apple",
    "theatre",
    "the,cat",
    "a_b the_thing",
    "an",
    "a",
    "  ",
    "",
    "Paris",
    "café",
    "CAFÉ",
    "naïve résumé",
    "«quote» —dash—",
    "don't stop",
    "10 items, 20 boxes",
    "yes",
    "noanswer",
    "tower tower",
    "The The The",
    "a1 a-1 a_1",
    "ÅNGSTRÖM",
]

#: Pairs picked to separate the metrics from one another: exact vs partial overlap, label answers,
#: repeats, empty sides, and normalisation-only differences.
PAIRS = [
    ("Paris", "paris"),
    ("The Eiffel Tower", "Eiffel Tower"),
    ("paris", "Paris, France"),
    ("Eiffel Tower is in Paris", "Paris"),
    ("eiffel tower in paris", "eiffel tower"),
    ("tower tower", "tower"),
    ("yes", "no"),
    ("yes", "yes"),
    ("noanswer", "the answer is unknown"),
    ("no", "no way"),
    ("", ""),
    ("", "Paris"),
    ("Paris", ""),
    ("café", "cafe"),
    ("the a an", "an the a"),
    ("10 items", "10 items"),
]

#: (prediction, [answers]) for the max-over-references metrics.
SETS = [
    ("The Eiffel Tower", ["Eiffel Tower", "Louvre"]),
    ("Berlin", ["Eiffel Tower", "Louvre"]),
    ("Eiffel Tower is in Paris", ["Paris"]),
    ("yes", ["no", "yes"]),
    ("yes", ["no"]),
    ("café", ["cafe", "coffee"]),
]


def main() -> None:
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_metrics_fixture.py",
        "dspy_version": PINNED,
        "normalize_text": [{"text": text, "normalized": normalize_text(text)} for text in TEXTS],
        "pairs": [
            {
                "prediction": prediction,
                "truth": truth,
                "em_score": bool(em_score(prediction, truth)),
                "f1_score": float(f1_score(prediction, truth)),
                "hotpot_f1_score": float(hotpot_f1_score(prediction, truth)),
                "precision_score": float(precision_score(prediction, truth)),
            }
            for prediction, truth in PAIRS
        ],
        "sets": [
            {
                "prediction": prediction,
                "answers": answers,
                "em": bool(EM(prediction, answers)),
                "f1": float(F1(prediction, answers)),
                "hotpot_f1": float(HotPotF1(prediction, answers)),
            }
            for prediction, answers in SETS
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "metrics.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
