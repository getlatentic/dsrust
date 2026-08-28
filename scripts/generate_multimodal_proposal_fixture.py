"""GEPA's multimodal instruction proposer, rendered by running it.

`teleprompt/gepa/instruction_proposal.py` is the proposer a caller hands GEPA when the reflective
examples carry images or audio. Instead of stringifying a custom type into the prompt, it replaces
each one with a numbered placeholder and sends the objects alongside the text, so the reflection
model actually sees them.

Everything it produces is prompt bytes, and three parts are easy to get subtly wrong:

* **Its own markdown renderer**, which is not the one gepa uses for a text-only dataset. Heading
  depth starts at 3 inside a section and is capped at 6; an empty dict or list still emits a blank
  line; every leaf is `str(value).strip()` followed by a blank line.
* **The placeholder numbering**, which restarts at 1 for each example, and the header line that is
  prepended only when at least one image was found — carrying the total across all examples.
* **The feedback pattern analysis**, a keyword scan over the lowercased `Feedback` of each example
  that prepends a summary block. Its three keyword lists overlap in ways worth pinning: one
  feedback string can count as an error, a success and a knowledge gap at once.

    .venv/bin/python scripts/generate_multimodal_proposal_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy
from dspy.teleprompt.gepa.instruction_proposal import (
    GenerateEnhancedMultimodalInstructionFromFeedback,
    SingleComponentMultiModalProposer,
)

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust"
    / "tests"
    / "conformance"
    / "optimize"
    / "multimodal_proposal.json"
)

#: A 1x1 PNG, so the recorded placeholder stands for a real custom type rather than a stub. What
#: the renderer checks is `isinstance(value, Type)` — the base every custom type shares — so an
#: `Audio` or a `Code` reaches the same branch.
PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def image() -> dspy.Image:
    return dspy.Image(url=PIXEL)


#: Each case is one reflective dataset, named for the part of the renderer it moves.
CASES: list[tuple[str, list[dict]]] = [
    (
        "text_only",
        [{"Inputs": {"question": "what?"}, "Generated Outputs": {"answer": "this"}, "Feedback": "fine"}],
    ),
    (
        "one_image_in_the_inputs",
        [
            {
                "Inputs": {"question": "what is this?", "photo": image()},
                "Generated Outputs": {"answer": "a pixel"},
                "Feedback": "incorrect, the colour is wrong",
            }
        ],
    ),
    (
        "images_across_two_examples_share_one_total",
        [
            {
                "Inputs": {"a": image(), "b": image()},
                "Generated Outputs": {"answer": "two"},
                "Feedback": "good",
            },
            {"Inputs": {"c": image()}, "Generated Outputs": {"answer": "one"}, "Feedback": "wrong"},
        ],
    ),
    (
        "nesting_is_capped_at_heading_six",
        [
            {
                "Inputs": {"a": {"b": {"c": {"d": {"e": {"f": {"g": "deep"}}}}}}},
                "Generated Outputs": {"answer": "x"},
                "Feedback": "fine",
            }
        ],
    ),
    (
        "lists_and_tuples_are_numbered_items",
        [
            {
                "Inputs": {"passages": ["one", "two"], "pair": ("left", "right")},
                "Generated Outputs": {"answer": "x"},
                "Feedback": "fine",
            }
        ],
    ),
    (
        "empty_containers_still_emit_a_line",
        [
            {
                "Inputs": {"nothing": {}, "none_either": []},
                "Generated Outputs": {},
                "Feedback": "fine",
            }
        ],
    ),
    (
        "a_leaf_is_stripped",
        [
            {
                "Inputs": {"padded": "   spaced out   \n"},
                "Generated Outputs": {"answer": "\n\ntrailing\n\n"},
                "Feedback": "fine",
            }
        ],
    ),
    (
        "outputs_can_be_a_bare_string",
        # What a failed parse leaves behind: `Generated Outputs` is a string, not a map.
        [
            {
                "Inputs": {"question": "what?"},
                "Generated Outputs": "Couldn't parse the output.\n",
                "Feedback": "failed to parse",
            }
        ],
    ),
    (
        "one_feedback_counts_in_every_bucket",
        # "incorrect" is an error word, "well" a success word, "context" a knowledge word — the
        # three lists are scanned independently, so a single string lands in all of them.
        [
            {
                "Inputs": {"q": "x"},
                "Generated Outputs": {"answer": "y"},
                "Feedback": "incorrect, though it read well given the context",
            }
        ],
    ),
    (
        "no_keyword_means_no_summary",
        [{"Inputs": {"q": "x"}, "Generated Outputs": {"answer": "y"}, "Feedback": "hm"}],
    ),
    (
        "keywords_are_matched_case_insensitively",
        [{"Inputs": {"q": "x"}, "Generated Outputs": {"answer": "y"}, "Feedback": "WRONG"}],
    ),
    (
        "a_substring_match_counts",
        # "wellington" contains "well", and the scan is a plain substring test.
        [{"Inputs": {"q": "x"}, "Generated Outputs": {"answer": "y"}, "Feedback": "wellington"}],
    ),
]


def main() -> None:
    proposer = SingleComponentMultiModalProposer()
    cases = []
    for name, dataset in CASES:
        text, image_map = proposer._format_examples_with_pattern_analysis(dataset)
        content = proposer._create_multimodal_examples(text, image_map)
        cases.append(
            {
                "name": name,
                "formatted": text,
                "images_per_example": {str(k): len(v) for k, v in image_map.items()},
                # The value handed to the predictor: the text alone, or a list whose head is the
                # text and whose tail is every image, flattened across examples in order.
                "content_is_a_list": isinstance(content, list),
                "content_length": len(content) if isinstance(content, list) else None,
            }
        )

    signature = GenerateEnhancedMultimodalInstructionFromFeedback
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"generated from dspy=={PINNED} via scripts/generate_multimodal_proposal_fixture.py",
                "signature": {
                    "instructions": signature.instructions,
                    "inputs": [
                        {"name": name, "desc": field.json_schema_extra.get("desc")}
                        for name, field in signature.input_fields.items()
                    ],
                    "outputs": [
                        {"name": name, "desc": field.json_schema_extra.get("desc")}
                        for name, field in signature.output_fields.items()
                    ],
                },
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    for case in cases:
        images = sum(case["images_per_example"].values())
        print(f"  {case['name']:44s} {len(case['formatted']):5d} chars  {images} image(s)")
    print(f"wrote {OUT.relative_to(pathlib.Path(__file__).parent.parent)}")


if __name__ == "__main__":
    main()
