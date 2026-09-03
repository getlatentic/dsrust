"""Record what a `Citations` output does to a prompt, on Anthropic and on everything else.

`Citations.adapt_to_native_lm_feature` is one line and it is not a formatting preference:

    if lm.model.startswith("anthropic/"):
        return signature.delete(field_name)
    return signature

So the field comes *out* of the rendered signature for an Anthropic model — the prompt never asks
for it — and `Citations.parse_lm_response` fills it afterwards from the citations the provider
attached to its own text blocks. Everywhere else the field renders and the model is asked in prose.

Both arms are recorded, because a port that deleted the field everywhere would pass a one-provider
fixture, and so would one that deleted it nowhere if only the OpenAI arm were kept. The two system
prompts are the evidence: one names the field, one does not.

The parse half is recorded from `Citations.parse_lm_response` directly — dspy reaches it with a dict
litellm assembled from Anthropic's `provider_specific_fields`, which this harness has no provider to
produce, and what crosses is the rule for reading that dict rather than litellm's plumbing.

    .venv/bin/python scripts/generate_citations_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.adapters.types.citation import Citations

from pins import require

OUT = (
    pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "adapter"
)
PINNED = require("dspy")


class Cited(dspy.Signature):
    """Answer the question with sources."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: Citations = dspy.OutputField()


def rendered_for(model: str) -> dict:
    """The system prompt a `Citations` signature renders to on `model`.

    `adapt_to_native_lm_feature` is what decides, and the adapter calls it while preparing the
    request — so this drives the real path rather than calling the classmethod directly.
    """
    lm = dspy.LM(model, api_key="not-used")
    adapter = dspy.ChatAdapter()
    prepared = Cited
    for name, field in Cited.output_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, dspy.Type):
            prepared = annotation.adapt_to_native_lm_feature(prepared, name, lm, {})
    messages = adapter.format(prepared, [], {"question": "Who wrote it?"})
    return {
        "model": model,
        "renders_the_field": "citations" in prepared.output_fields,
        "system": messages[0]["content"],
    }


def reflected_citations() -> dict:
    """What `Citations` says about itself on a field line, and the schema dspy notes beside it.

    A Rust-declared signature has no Python to reflect, so a byte comparison of the arm that *keeps*
    the field needs these carried across — the same crossing every other custom-type fixture makes.
    """
    # The schema dspy *renders*, taken from the prompt rather than from `model_json_schema()`:
    # upstream derives its own for the field line and the two differ in key order, which is a byte.
    rendered = rendered_for("openai/gpt-4o-mini")["system"]
    marker = "must adhere to the JSON schema: "
    start = rendered.index(marker) + len(marker)
    end = rendered.index("\n", start)
    return {
        "annotation": "Citations",
        "description": Citations.description(),
        "schema": json.loads(rendered[start:end]),
    }


#: What litellm hands dspy for an Anthropic reply carrying citations, flattened the way
#: `_extract_citations_from_response` flattens it: one list per text block, concatenated.
CITED_RESPONSE = {
    "text": "Bede wrote it.",
    "citations": [
        {
            "type": "char_location",
            "cited_text": "Bede completed it in 731.",
            "document_index": 0,
            "document_title": "Ecclesiastical History",
            "start_char_index": 10,
            "end_char_index": 35,
        },
        {
            "type": "char_location",
            "cited_text": "written at Jarrow",
            "document_index": 1,
            "document_title": None,
            "start_char_index": 0,
            "end_char_index": 17,
        },
    ],
}


def main() -> None:
    parsed = Citations.parse_lm_response(CITED_RESPONSE)
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_citations_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "Both arms of Citations.adapt_to_native_lm_feature: an anthropic/ model drops the "
            "field from the render, every other provider keeps it. `parsed` is what "
            "Citations.parse_lm_response makes of the citations litellm flattens out of "
            "provider_specific_fields."
        ),
        "renders": [
            rendered_for("anthropic/claude-sonnet-4-5"),
            rendered_for("openai/gpt-4o-mini"),
        ],
        "citations_type": reflected_citations(),
        "response": CITED_RESPONSE,
        "parsed": json.loads(parsed.model_dump_json()) if parsed else None,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "citations_native.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)
    for render in fixture["renders"]:
        print(f"    {render['model']}: renders the field = {render['renders_the_field']}", file=sys.stderr)


if __name__ == "__main__":
    main()
