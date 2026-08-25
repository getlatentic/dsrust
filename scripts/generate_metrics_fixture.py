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

import dspy
from dspy.dsp.utils import DPR_normalize
from dspy.evaluate.metrics import (
    EM,
    F1,
    HotPotF1,
    answer_passage_match,
    em_score,
    f1_score,
    hotpot_f1_score,
    normalize_text,
    precision_score,
)

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "evaluate"
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


#: Inputs for the DPR tokenizer, picked where a hand-written scanner drifts from the regex.
#:
#: `([\\p{L}\\p{N}\\p{M}]+)|([^\\p{Z}\\p{C}])` is ordered, so the edges are: what counts as one run
#: (CJK has no spaces, so `北京市` is a single token and does not contain `北京`), what becomes a
#: token alone (punctuation and symbols, one character each), and what disappears (separators and
#: every "other" — controls, formats, private use, and codepoints Unicode has not assigned).
#:
#: Case is its own edge. Upstream lowercases the *token*, so Greek picks the final sigma at a word's
#: end and not otherwise, and `İ` becomes two codepoints.
DPR_TEXTS = [
    "The Eiffel Tower is in Paris.",
    "Paris, France",
    "a-b_c",
    "10 items, 20 boxes",
    "  spaced\tout\n",
    "",
    "café",
    "cafe\u0301",
    "\u00c5NGSTR\u00d6M",
    "\u039f\u0394\u03a5\u03a3\u03a3\u0395\u03a5\u03a3",
    "\u03a3",
    "\u0130stanbul",
    "\u01c5ungla",
    "\u5317\u4eac\u5e02",
    "\U0001f389!",
    "a\u200db",
    "x\u0378y",
    "\ue000private",
    # Assigned in Unicode 16, which is where two sets of tables part company: the Rust crate ships
    # 16.0 and upstream's `regex` its own. A token here proves they agree rather than assuming it.
    "\U00010d40 \U000105c0 \U0001e5d0 \U00016d40 \U0001cc00",
]

#: `(answer, context)` for the metric, picked for containment rather than for prose.
#:
#: Token containment is not substring containment, and every pair here is a case where the two
#: disagree or where a boundary decides: an answer at the end of a passage, an answer spanning two
#: tokens, a CJK answer inside a longer run, an answer longer than the passage, and both empty.
PASSAGE_CASES = [
    ("Eiffel Tower", ["The Eiffel Tower is in Paris.", "..."]),
    ("Eiffel Tower", ["The Louvre is in Paris."]),
    ("paris", ["The Eiffel Tower is in Paris."]),
    ("Paris", ["paris"]),
    (["Louvre", "Eiffel Tower"], ["The Eiffel Tower is in Paris."]),
    (["Louvre"], ["The Eiffel Tower is in Paris."]),
    ("the", ["theatre"]),
    ("the", ["the cat"]),
    ("\u5317\u4eac", ["\u5317\u4eac\u5e02\u4e2d\u5fc3"]),
    ("\u5317\u4eac\u5e02", ["\u5317\u4eac\u5e02\u4e2d\u5fc3"]),
    ("cafe", ["a caf\u00e9 in paris"]),
    ("a b c", ["a b"]),
    ("", ["anything"]),
    ("", [""]),
    ("Paris", []),
    ("Paris", ["", "Paris"]),
    # `context` as one string rather than a list: upstream iterates it, so every character is a
    # passage of its own and a one-letter answer lands.
    ("y", "xyz"),
    ("xyz", "xyz"),
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
        "dpr_normalize": [{"text": text, "tokens": DPR_normalize(text)} for text in DPR_TEXTS],
        "passage_match": [
            {
                "answer": answer,
                "context": context,
                "score": float(
                    answer_passage_match(
                        dspy.Example(answer=answer), dspy.Prediction(context=context)
                    )
                ),
            }
            for answer, context in PASSAGE_CASES
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
