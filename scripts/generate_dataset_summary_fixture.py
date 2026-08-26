"""Record what `data_aware_proposer` shows the proposer: the three signatures and the sample repr.

`create_dataset_summary` renders a slice of the trainset as **Python source** —
`order_input_keys_in_string(repr(trainset[a:b]))` — and puts it in an `examples` field. That is a
prompt-bytes claim like any other, and it has two edges worth pinning by running Python rather than
reading it:

  - CPython's `repr` of a string picks its quote from what the string holds. An apostrophe and no
    double quote switches to `"` and leaves the apostrophe unescaped; both quotes falls back to `'`
    and escapes. The crate escaped unconditionally before this, which disagreed on every value
    carrying an apostrophe.
  - `input_keys` is a Python `set`, so its order is randomised per process. `order_input_keys_in_string`
    sorts it with a regex, which is what makes the prompt reproducible at all — without it dspy would
    disagree with *itself* between runs.

The signatures are recorded as rendered system prompts, the same way every other proposer signature
here is.

    .venv/bin/python scripts/generate_dataset_summary_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.propose.dataset_summary_generator import (
    DatasetDescriptor,
    DatasetDescriptorWithPriorObservations,
    ObservationSummarizer,
    order_input_keys_in_string,
)

from pins import require

OUT = (
    pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "optimize"
)
PINNED = require("dspy")

#: Values chosen for what they do to `repr`, not for realism: an apostrophe alone, both quotes, a
#: backslash, a non-string scalar, None, True, and a list. Input keys that are *not* alphabetical in
#: field order, so the sort has something to do.
ROWS = [
    dspy.Example(question="capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(zebra="last", alpha="first", n=3, ok=True, nothing=None).with_inputs(
        "zebra", "alpha"
    ),
    dspy.Example(text="it's tricky", note='say "hi"', both="both ' and \"").with_inputs("text"),
    dspy.Example(path=r"a\b", tags=["x", "y"], score=1.5).with_inputs("path"),
]


def rendered(signature) -> str:
    """The system prompt this signature renders to, with no model involved."""
    inputs = {name: "" for name in signature.input_fields}
    return dspy.ChatAdapter().format(signature, [], inputs)[0]["content"]


def main() -> None:
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_dataset_summary_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "The examples field is Python source, not JSON. repr picks its quote from what the "
            "string holds, and order_input_keys_in_string sorts the input-key set — without which "
            "the prompt would differ between two runs of dspy itself."
        ),
        "signatures": {
            "dataset_descriptor": rendered(DatasetDescriptor),
            "dataset_descriptor_with_prior_observations": rendered(
                DatasetDescriptorWithPriorObservations
            ),
            "observation_summarizer": rendered(ObservationSummarizer),
        },
        "rows": [
            {"fields": dict(row.toDict()), "input_keys": sorted(row.inputs().keys())}
            for row in ROWS
        ],
        # Whole-slice renderings, since a batch is what a call actually shows.
        "slices": [
            {"start": start, "stop": stop, "repr": order_input_keys_in_string(repr(ROWS[start:stop]))}
            for start, stop in [(0, 1), (0, 2), (1, 3), (0, 4), (2, 4)]
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "dataset_summary.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)

    # A corpus that never exercises the quoting rule would pass against unconditional escaping.
    whole = json.dumps(fixture["slices"])
    if '\\"it' not in whole:
        raise SystemExit("no slice exercises repr's double-quote branch")
    print("  the double-quote branch is exercised", file=sys.stderr)


if __name__ == "__main__":
    main()
