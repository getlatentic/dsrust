"""Generate conformance fixtures by running Python DSPy itself.

The fixtures this writes are goldens: the exact messages a pinned DSPy renders for a given
signature and inputs. Transcribing them by hand would only test the transcription, so this
imports the real library and calls the real adapter. Fixtures are committed, so `cargo test`
never needs Python — only regeneration does.

    uv venv .dspy-venv --python 3.12
    uv pip install --python .dspy-venv/bin/python "dspy==$(cat scripts/DSPY_VERSION)"
    .dspy-venv/bin/python scripts/generate_fixtures.py

Adding a case: append to CASES. Keep each one inside the subset the Rust harness models
(scalar and JSON-valued fields, and demos, today) until the matching Rust support lands, and
widen together.
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.predict.refine import OfferFeedback
from dspy.teleprompt.copro_optimizer import (
    BasicGenerateInstruction,
    GenerateInstructionGivenAttempts,
)
from dspy.propose.grounded_proposer import (
    DescribeModule,
    DescribeProgram,
    generate_instruction_class,
)
from dspy.teleprompt.gepa.gepa_flex_utils import CodeProposalSignature
from dspy.propose.dataset_summary_generator import (
    DatasetDescriptor,
    DatasetDescriptorWithPriorObservations,
    ObservationSummarizer,
)

# GroundedProposer's instruction signature is built by a factory with every flag on — MIPROv2's
# default. Every reduced variant is a field-subset of this one, so verifying the full signature
# verifies each field's rendering.
GENERATE_MODULE_INSTRUCTION = generate_instruction_class(
    use_dataset_summary=True,
    program_aware=True,
    use_task_demos=True,
    use_instruct_history=True,
    use_tip=True,
).signature

PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()
OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance"

# Each case: the signature DSPy will build, and the input values to render with.
# `kind` names the Rust FieldKind the harness maps this annotation to.
CASES = [
    {
        # The one module that changes the signature before rendering it. Without this, nothing
        # compared the `reasoning` field dspy adds against the one this crate adds, and the two
        # disagreed on its description for as long as both existed.
        "name": "chain_of_thought_reasoning",
        "module": "chain_of_thought",
        "instructions": "Answer the question.",
        "inputs": [{"name": "question", "type": str, "kind": "str", "desc": None}],
        "outputs": [{"name": "answer", "type": str, "kind": "str", "desc": None}],
        "values": {"question": "What is the capital of France?"},
    },
    {
        # ReAct rewrites the signature far more than ChainOfThought does — a trajectory input,
        # three outputs, and instructions naming every tool. None of it was compared before.
        "name": "react_trajectory",
        "module": "react",
        "instructions": "Answer the question.",
        "inputs": [{"name": "question", "type": str, "kind": "str", "desc": None}],
        "outputs": [{"name": "answer", "type": str, "kind": "str", "desc": None}],
        "values": {"question": "What is the capital of France?"},
    },
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
        # A list- or dict-valued field goes through `json.dumps`, whose separators carry a space
        # that a compact JSON writer omits. Both a demo and the live turn render one.
        "name": "structured_field_values",
        "instructions": "Answer the question from the context.",
        "inputs": [
            {"name": "question", "type": str, "kind": "str", "desc": None},
            {"name": "context", "type": list[str], "kind": "json:list[str]", "desc": None},
            {
                "name": "weights",
                "type": dict[str, int],
                "kind": "json:dict[str, int]",
                "desc": None,
            },
        ],
        "outputs": [{"name": "answer", "type": str, "kind": "str", "desc": None}],
        "demos": [
            {
                "question": "Which capital?",
                "context": ["Berlin is in Germany.", "Paris is in France."],
                "weights": {"alpha": 1, "beta": 2},
                "answer": "Berlin",
            }
        ],
        "values": {
            "question": "Which river?",
            "context": ["The Seine runs through Paris.", "Voici de l'eau — 日本語."],
            "weights": {"alpha": 3, "beta": 4},
        },
    },
    {
        # The multimodal path, which had no golden at all: an image cannot reach a provider
        # inside a string, so the rendered message is content *blocks* rather than prose. The
        # value is a real dspy.Image; what lands in the fixture is the marker-wrapped string its
        # pydantic serialization produces, which is what a Rust caller has to reproduce.
        "name": "image_input_field",
        "instructions": "Describe the picture.",
        "inputs": [{"name": "photo", "type": dspy.Image, "kind": "json:Image", "desc": None}],
        "outputs": [{"name": "caption", "type": str, "kind": "str", "desc": None}],
        "values": {"photo": dspy.Image(url="https://example.com/a.jpg")},
    },
    {
        # Audio had no golden at all, where image had one: every audio render rested on
        # hand-written tests agreeing with the code they test. Unlike an image, audio travels as
        # an `input_audio` block carrying bare base64 beside a bare format name — the one media
        # type that does not ride in a data URI — so the block split is different enough to earn
        # its own measured render. The value is built with dspy's keyword form, which keeps the
        # payload exactly as given.
        "name": "audio_input_field",
        "instructions": "Transcribe the clip.",
        "inputs": [{"name": "clip", "type": dspy.Audio, "kind": "json:Audio", "desc": None}],
        "outputs": [{"name": "transcript", "type": str, "kind": "str", "desc": None}],
        "values": {"clip": dspy.Audio(data="YXVkaW8gYnl0ZXM=", audio_format="wav")},
    },
    {
        # Refine's feedback step, and the widest signature dspy ships: nine inputs, a float pair,
        # a list and a dict output. Its instructions and its `advice` description reach a prompt
        # verbatim — including upstream's "kind ofscenario", where two literals meet without a
        # space — so a transcription is exactly what must not be trusted here.
        "name": "offer_feedback",
        "module": "offer_feedback",
        "dspy_signature": OfferFeedback,
        "values": {
            "program_code": "class Program: ...",
            "modules_defn": "predict = Predict(question -> answer)",
            "program_inputs": '{\n  "question": "Why?"\n}',
            "program_trajectory": "[]",
            "program_outputs": '{\n  "answer": "Because."\n}',
            "reward_code": "def reward(inputs, outputs): ...",
            "target_threshold": 1.0,
            "reward_value": 0.5,
            "module_names": ["predict"],
        },
    },
    {
        "name": "multiline_instructions",
        "instructions": "Answer the question.\nBe brief.\nNever guess.",
        "inputs": [{"name": "question", "type": str, "kind": "str", "desc": None}],
        "outputs": [{"name": "answer", "type": str, "kind": "str", "desc": None}],
        "values": {"question": "Why is the sky blue?"},
    },
    {
        # COPRO's zero-shot proposal step. Its instructions and both output-field descriptions
        # reach the prompt verbatim, so a transcription is what must not be trusted — the crate
        # builds this signature from the same strings and this fixture proves they render alike.
        "name": "basic_generate_instruction",
        "dspy_signature": BasicGenerateInstruction,
        "values": {"basic_instruction": "Answer the question."},
    },
    {
        # COPRO's depth step. `attempted_instructions` is a `str` field handed a list, which
        # dspy lays out as a numbered `[N] «entry»` run rather than as JSON — the one rendering
        # path this signature exercises that the others do not.
        "name": "generate_instruction_given_attempts",
        "dspy_signature": GenerateInstructionGivenAttempts,
        "values": {
            "attempted_instructions": [
                "Instruction #1: Answer the question.",
                "Prefix #1: Answer:",
                "Resulting Score #1: 0.5",
            ]
        },
    },
    {
        # MIPROv2/GroundedProposer's program-aware summariser: describe the whole program.
        "name": "describe_program",
        "dspy_signature": DescribeProgram,
        "values": {
            "program_code": "class Program:\n    def forward(self, question):\n        return self.predict(question=question)",
            "program_example": "No task demos provided.",
        },
    },
    {
        # GroundedProposer's per-module summariser.
        "name": "describe_module",
        "dspy_signature": DescribeModule,
        "values": {
            "program_code": "class Program:\n    def forward(self, question):\n        return self.predict(question=question)",
            "program_example": "No task demos provided.",
            "program_description": "A question-answering pipeline.",
            "module": "Predict(question) -> answer",
        },
    },
    {
        # GroundedProposer's instruction generator, every flag on (MIPROv2's default). Nine inputs
        # whose descriptions all reach the prompt.
        "name": "generate_module_instruction",
        "dspy_signature": GENERATE_MODULE_INSTRUCTION,
        "values": {
            "dataset_description": "Short factual questions with one-word answers.",
            "program_code": "class Program:\n    def forward(self, question):\n        return self.predict(question=question)",
            "program_description": "A question-answering pipeline.",
            "module": "Predict(question) -> answer",
            "module_description": "Answers the question.",
            "task_demos": "No task demos provided.",
            "previous_instructions": "Instruction #1: Answer the question.",
            "basic_instruction": "Answer the question.",
            "tip": "Keep the instruction clear and concise.",
        },
    },
    {
        # GEPA's code proposer, which is what makes a `dspy.Flex` optimizable: five inputs, and an
        # instruction long enough that a transcription would drift on a line break alone. This is
        # the prompt that asks a model for a whole `dspy.Module` subclass instead of an instruction.
        "name": "code_proposal",
        "dspy_signature": CodeProposalSignature,
        "values": {
            "task_description": "StringSignature: question -> answer",
            "available_context": "(no extra context)",
            "primitives_catalog": "dspy.Predict(signature)",
            "current_source": "class M(dspy.Module):\n    def forward(self, **inputs):\n        return dspy.Prediction(answer='x')",
            "failures": "Example 1: answered 'x', expected 'Paris'.",
        },
    },
    {
        # The dataset-summary bootstrap: first batch of observations.
        "name": "dataset_descriptor",
        "dspy_signature": DatasetDescriptor,
        "values": {"examples": "Question: What is the capital of France? Answer: Paris"},
    },
    {
        # The dataset-summary continuation, folding in prior observations.
        "name": "dataset_descriptor_with_prior_observations",
        "dspy_signature": DatasetDescriptorWithPriorObservations,
        "values": {
            "examples": "Question: What is the capital of Spain? Answer: Madrid",
            "prior_observations": "The data is factual geography questions.",
        },
    },
    {
        # The final summariser that condenses observations into the dataset description.
        "name": "observation_summarizer",
        "dspy_signature": ObservationSummarizer,
        "values": {"observations": "The data is factual geography questions with one-word answers."},
    },
]


def build_signature(case: dict) -> type[dspy.Signature]:
    if "dspy_signature" in case:
        return case["dspy_signature"]
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


def specs_of(signature, case: dict, half: str) -> list[dict]:
    """The field list for the fixture, described by the case or read back off a real signature."""
    if "dspy_signature" not in case:
        return [{k: spec[k] for k in ("name", "kind", "desc")} for spec in case[half]]
    fields = signature.input_fields if half == "inputs" else signature.output_fields
    return [
        {"name": name, "kind": kind_of(field.annotation),
         "desc": field.json_schema_extra.get("desc") or None}
        for name, field in fields.items()
    ]


def kind_of(annotation) -> str:
    """The fixture's spelling of a declared type: a scalar by name, anything else as json:<python>."""
    scalars = {str: "str", int: "int", float: "float", bool: "bool"}
    if annotation in scalars:
        return scalars[annotation]
    return "json:" + python_spelling(annotation)


def python_spelling(annotation) -> str:
    origin, args = getattr(annotation, "__origin__", None), getattr(annotation, "__args__", ())
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    inner = ", ".join(python_spelling(a) for a in args)
    return f"{origin.__name__}[{inner}]"


def recorded(value):
    """What the Rust harness feeds its adapter.

    A custom type reaches dspy's adapter already serialized to a marker-wrapped string, so that
    string — not the object — is the input value a Rust caller supplies.
    """
    return value.serialize_model() if hasattr(value, "serialize_model") else value


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
        declared = build_signature(case)
        signature = declared
        # `chain_of_thought` renders the signature that module prepends `reasoning` to, which is
        # the only way a fixture sees the field it adds and the description it does *not* add.
        if case.get("module") == "chain_of_thought":
            signature = dspy.ChainOfThought(signature).predict.signature
        elif case.get("module") == "react":
            signature = dspy.ReAct(signature, tools=[]).react.signature
        system, turns = render(signature, demos, case["values"])
        fixture = {
            "source": f"generated from dspy=={PINNED} via scripts/generate_fixtures.py",
            "dspy_version": PINNED,
            "instructions": case.get("instructions", declared.instructions),
            "inputs": specs_of(declared, case, "inputs"),
            "outputs": specs_of(declared, case, "outputs"),
            "module": case.get("module", "predict"),
            "demos": demos,
            "values": {name: recorded(value) for name, value in case["values"].items()},
            "expected_system": system,
            "expected_turns": [{"role": role, "content": content} for role, content in turns],
        }
        path = OUT / f"{case['name']}.json"
        path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
        print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
