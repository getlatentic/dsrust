"""Generate conformance fixtures by running Python DSPy itself.

The fixtures this writes are goldens: the exact messages a pinned DSPy renders for a given
signature and inputs. Transcribing them by hand would only test the transcription, so this
imports the real library and calls the real adapter. Fixtures are committed, so `cargo test`
never needs Python — only regeneration does.

    uv venv .dspy-venv --python 3.12
    uv pip install --python .dspy-venv/bin/python "dspy==$(cat scripts/DSPY_VERSION)"
    .dspy-venv/bin/python scripts/generate_fixtures.py

Adding a case: append to CASES. Keep each one inside the subset the Rust harness models
(scalar fields and demos today) until the matching Rust support lands, and widen together.
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()
OUT = pathlib.Path(__file__).parent.parent / "tests" / "conformance"

# Each case: the signature DSPy will build, and the input values to render with.
# `kind` names the Rust FieldKind the harness maps this annotation to.
CASES = [
    {
        "name": "simple_signature",
        "instructions": "Answer the question.",
        "inputs": [{"name": "question", "type": str, "kind": "str", "desc": None}],
        "outputs": [{"name": "answer", "type": str, "kind": "str", "desc": None}],
        "values": {"question": "What is the capital of France?"},
    },
    {
        "name": "described_fields",
        "instructions": "Pick a colour and justify it.",
        "inputs": [{"name": "request", "type": str, "kind": "str", "desc": "the request"}],
        "outputs": [
            {"name": "colour", "type": str, "kind": "str", "desc": "the chosen colour"},
            {"name": "why", "type": str, "kind": "str", "desc": "one short sentence"},
        ],
        "values": {"request": "something calm"},
    },
    {
        "name": "typed_scalar_outputs",
        "instructions": "Read the sentence and report the numbers.",
        "inputs": [{"name": "sentence", "type": str, "kind": "str", "desc": None}],
        "outputs": [
            {"name": "count", "type": int, "kind": "int", "desc": "how many items"},
            {"name": "score", "type": float, "kind": "float", "desc": "confidence 0-1"},
            {"name": "certain", "type": bool, "kind": "bool", "desc": "sure about it"},
        ],
        "values": {"sentence": "Three cats sat down."},
    },
    {
        # Demos are what an optimizer produces, so rendering them is the load-bearing case.
        "name": "one_demo",
        "instructions": "Answer the question.",
        "inputs": [{"name": "question", "type": str, "kind": "str", "desc": None}],
        "outputs": [{"name": "answer", "type": str, "kind": "str", "desc": None}],
        "demos": [{"question": "What is the capital of Germany?", "answer": "Berlin"}],
        "values": {"question": "What is the capital of France?"},
    },
    {
        "name": "two_demos_with_descriptions",
        "instructions": "Pick a colour and justify it.",
        "inputs": [{"name": "request", "type": str, "kind": "str", "desc": "the request"}],
        "outputs": [
            {"name": "colour", "type": str, "kind": "str", "desc": "the chosen colour"},
            {"name": "why", "type": str, "kind": "str", "desc": "one short sentence"},
        ],
        "demos": [
            {"request": "something calm", "colour": "blue", "why": "It reads as still."},
            {"request": "something warm", "colour": "amber", "why": "It holds light."},
        ],
        "values": {"request": "something bold"},
    },
    {
        "name": "multiline_instructions",
        "instructions": "Answer the question.\nBe brief.\nNever guess.",
        "inputs": [{"name": "question", "type": str, "kind": "str", "desc": None}],
        "outputs": [{"name": "answer", "type": str, "kind": "str", "desc": None}],
        "values": {"question": "Why is the sky blue?"},
    },
]


def build_signature(case: dict) -> type[dspy.Signature]:
    fields, annotations = {}, {}
    for spec in case["inputs"]:
        annotations[spec["name"]] = spec["type"]
        fields[spec["name"]] = (
            dspy.InputField(desc=spec["desc"]) if spec["desc"] else dspy.InputField()
        )
    for spec in case["outputs"]:
        annotations[spec["name"]] = spec["type"]
        fields[spec["name"]] = (
            dspy.OutputField(desc=spec["desc"]) if spec["desc"] else dspy.OutputField()
        )
    namespace = {"__doc__": case["instructions"], "__annotations__": annotations, **fields}
    return type(case["name"], (dspy.Signature,), namespace)


def render(signature: type[dspy.Signature], demos: list, values: dict) -> tuple[str, list]:
    """The messages DSPy's own ChatAdapter produces, with no LM involved."""
    messages = dspy.ChatAdapter().format(signature, demos, values)
    if messages[0]["role"] != "system":
        raise SystemExit(f"expected a leading system message, got {messages[0]['role']}")
    turns = [(message["role"], message["content"]) for message in messages[1:]]
    return messages[0]["content"], turns


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    OUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        demos = case.get("demos", [])
        system, turns = render(build_signature(case), demos, case["values"])
        fixture = {
            "source": f"generated from dspy=={PINNED} via scripts/generate_fixtures.py",
            "dspy_version": PINNED,
            "instructions": case["instructions"],
            "inputs": [{k: spec[k] for k in ("name", "kind", "desc")} for spec in case["inputs"]],
            "outputs": [{k: spec[k] for k in ("name", "kind", "desc")} for spec in case["outputs"]],
            "demos": demos,
            "values": case["values"],
            "expected_system": system,
            "expected_turns": [{"role": role, "content": content} for role, content in turns],
        }
        path = OUT / f"{case['name']}.json"
        path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
        print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
