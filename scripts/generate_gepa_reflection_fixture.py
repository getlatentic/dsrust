"""Record GEPA's reflection prompt and instruction extraction, by running its own signature.

`gepa.strategies.instruction_proposal.InstructionProposalSignature` is what both gepa's default
proposer and dspy's GEPA adapter use to rewrite a component's instruction: `prompt_renderer` turns the
current instruction and a reflective dataset (per-example inputs, outputs, and feedback) into one
prompt string, and `output_extractor` pulls the new instruction out of the reflection LM's reply. An
adapter feeds the LM this exact text, so both are byte-sensitive.

What is compared is the rendered prompt over nested / empty / deep reflective datasets, and the
extracted instruction over every branch of the fence parser. Reflective values are written in a tagged
form ({"text"|"map"|"list": ...}) so the Rust mirror reads them with map order preserved.

    .dspy-venv/bin/python scripts/generate_gepa_reflection_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

from gepa.strategies.instruction_proposal import InstructionProposalSignature

OUT = pathlib.Path(__file__).parent.parent / "gepa" / "tests" / "conformance"
PINNED = "0.0.27"


def encode(value):
    """Tag a reflective value so the Rust mirror can read it with map order preserved."""
    if isinstance(value, dict):
        return {"map": [[k, encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"list": [encode(v) for v in value]}
    return {"text": str(value)}


def encode_sample(sample: dict):
    """A reflective example is an ordered map of section -> value, kept as [key, node] pairs."""
    return [[key, encode(val)] for key, val in sample.items()]


# Reflective datasets that exercise the markdown renderer: the typical dspy shape, a failed-parse
# string output, nested and empty maps/lists, header-depth capping, and a multi-example dataset.
RENDER_CASES = [
    (
        "typical",
        "Answer the question.",
        [
            {
                "Inputs": {"question": "What is 2 + 2?"},
                "Generated Outputs": {"answer": "4"},
                "Feedback": "Correct.",
            }
        ],
    ),
    (
        "failed_parse_string_output",
        "Extract the entities.",
        [
            {
                "Inputs": {"text": "Ada Lovelace wrote the first program."},
                "Generated Outputs": "the model produced unparseable text",
                "Feedback": "Your output failed to parse.",
            }
        ],
    ),
    (
        "nested_and_empty",
        "  Solve it.  ",
        [
            {
                "Inputs": {
                    "context": {"passage": "  padded text  ", "meta": {}},
                    "hints": ["look closely", "check units"],
                    "nothing": [],
                },
                "Generated Outputs": {},
                "Feedback": "  Trim your reasoning.  ",
            }
        ],
    ),
    (
        "deep_nesting",
        "Reason step by step.",
        [
            {
                "Inputs": {"a": {"b": {"c": {"d": {"e": {"f": "very deep"}}}}}},
                "Generated Outputs": {"answer": "deep"},
                "Feedback": "OK",
            }
        ],
    ),
    (
        "multi_example",
        "Classify the sentiment.",
        [
            {"Inputs": {"review": "Loved it!"}, "Generated Outputs": {"label": "positive"}, "Feedback": "Right."},
            {"Inputs": {"review": "Terrible."}, "Generated Outputs": {"label": "negative"}, "Feedback": "Right."},
        ],
    ),
]

# Reflection-LM replies that exercise every branch of the fence extractor.
EXTRACT_CASES = [
    "```\nThe new instruction.\n```",
    "```markdown\nThe new instruction with a language tag.\n```",
    "Here is my answer:\n```\nWrapped instruction\n```\nHope that helps.",
    "```only an opening fence\nthen the body continues",
    "no fences at all, just the instruction text",
    "trailing fence only, body then```",
    "   ```lang\nleading whitespace before the fence```   ",
    "```",
    "```json\n{\"key\": \"value\"}\n```",
]


def render_case(label: str, current_instruction: str, dataset: list[dict]) -> dict:
    prompt = InstructionProposalSignature.prompt_renderer(
        {
            "current_instruction_doc": current_instruction,
            "dataset_with_feedback": dataset,
            "prompt_template": None,
        }
    )
    return {
        "label": label,
        "current_instruction": current_instruction,
        "dataset": [encode_sample(sample) for sample in dataset],
        "prompt": prompt,
    }


def extract_case(lm_out: str) -> dict:
    return {"lm_out": lm_out, "new_instruction": InstructionProposalSignature.output_extractor(lm_out)["new_instruction"]}


def main() -> None:
    fixture = {
        "source": f"generated from gepa=={PINNED} via scripts/generate_gepa_reflection_fixture.py",
        "gepa_version": PINNED,
        "default_template": InstructionProposalSignature.default_prompt_template,
        "render_cases": [render_case(*case) for case in RENDER_CASES],
        "extract_cases": [extract_case(lm_out) for lm_out in EXTRACT_CASES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "reflection.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
