"""Record what dspy 3.3 actually serializes for every normalized LM type.

The Rust port of these types is checked against this rather than against a reading of
`dspy/core/types.py`: each entry is one instance dumped by pydantic itself, every field included, so a renamed field,
a changed discriminator literal, or a default that moved shows up as a parse failure in
`tests/lm_api_conformance.rs` instead of surviving as a plausible-looking struct.

    uv venv .dspy-venv-3.3 --python 3.12
    uv pip install --python .dspy-venv-3.3/bin/python "dspy==3.3.0"
    .dspy-venv-3.3/bin/python scripts/generate_lm_api_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy

from pins import require
import dspy.core.types as t

# Read from the pin rather than written here: a generator that names its own
# version cannot follow a bump, and six of them refused to run at 3.3.0 for
# exactly that reason while claiming the pin had drifted.
PINNED = require("dspy")
OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "lm_api" / "dspy_3_3.json"

CITATION = t.LMCitationPart(text="the quote", title="The Paper", url="https://example.com")
IMAGE = t.LMImagePart(url="https://example.com/a.jpg", detail="high")
AUDIO = t.LMAudioPart(data="YQ==", media_type="audio/wav")

# `rust` names the type the Rust side parses each entry into.
CASES = [
    ("LmPart", t.LMTextPart(text="plain")),
    ("LmPart", IMAGE),
    ("LmPart", AUDIO),
    ("LmPart", t.LMVideoPart(url="https://example.com/a.mp4")),
    ("LmPart", t.LMBinaryPart(file_id="f_1", filename="a.pdf")),
    ("LmPart", t.LMDocumentPart(source={"type": "text", "data": "the contract"}, title="Contract")),
    ("LmPart", t.LMToolCallPart(id="call_1", name="search", args={"q": "Paris"})),
    ("LmPart", t.LMToolResultPart(call_id="call_1", name="search", content=[t.LMTextPart(text="42")])),
    ("LmPart", t.LMThinkingPart(text="hmm", redacted=False)),
    ("LmPart", CITATION),
    ("LmPart", t.LMRefusalPart(text="no")),
    ("LmMessage", t.LMMessage(role="user", parts=[t.LMTextPart(text="Why?"), IMAGE])),
    ("LmToolSpec", t.LMToolSpec(name="search", description="look things up", parameters={"type": "object"})),
    ("LmReasoningConfig", t.LMReasoningConfig(effort="high", max_tokens=512, summary="concise")),
    ("LmToolChoice", t.LMToolChoice(mode="required", allowed=["search"], parallel=False)),
    ("LmCacheConfig", t.LMCacheConfig(enabled=True, rollout_id=7)),
    ("LmCacheConfig", t.LMCacheConfig(rollout_id="attempt-two")),
    ("LmPromptCacheConfig", t.LMPromptCacheConfig(enabled=True, key="k")),
    ("LmConfig", t.LMConfig.from_kwargs(temperature=0.7, max_tokens=100, top_p=0.9, stop=["\n\n"], n=2,
                                        logprobs=5, reasoning_effort="high", parallel_tool_calls=False,
                                        rollout_id=3, prompt_cache_key="k", anthropic_beta="tools-2024")),
    ("LmOutput", t.LMOutput(parts=[t.LMTextPart(text="Paris")], finish_reason="stop", truncated=False)),
    ("LmResponse", t.LMResponse(model="openai/gpt-4o", outputs=[t.LMOutput(parts=[t.LMTextPart(text="Paris")])],
                                usage=t.LMUsage(input_tokens=10, output_tokens=4), cost=0.002,
                                cache_hit=False, response_id="resp_1")),
    ("LmRequest", t.LMRequest(model="openai/gpt-4o",
                              messages=[t.LMMessage(role="user", parts=[t.LMTextPart(text="Why?")])],
                              tools=[t.LMToolSpec(name="search", parameters={})],
                              config=t.LMConfig.from_kwargs(temperature=0.7))),
    ("LmUsage", t.LMUsage(input_tokens=10, output_tokens=4, reasoning_tokens=2)),
    ("LmDelta", t.LMTextDelta(text="Par")),
    ("LmDelta", t.LMThinkingDelta(text="hmm")),
    ("LmDelta", t.LMToolCallDelta(id="call_1", name="search", args_delta='{"q"')),
    ("LmDelta", t.LMCitationDelta(citation=CITATION)),
    ("LmDelta", t.LMImageDelta(image=IMAGE)),
    ("LmDelta", t.LMAudioDelta(audio=AUDIO)),
    ("LmStreamEvent", t.LMStreamStartEvent(model="openai/gpt-4o")),
    ("LmStreamEvent", t.LMStreamDeltaEvent(output_index=0, part_index=0, delta=t.LMTextDelta(text="Par"))),
    ("LmStreamEvent", t.LMStreamOutputEndEvent(output_index=0, finish_reason="stop", truncated=False)),
    ("LmStreamEvent", t.LMStreamEndEvent(usage=t.LMUsage(input_tokens=1, output_tokens=1), cost=0.001)),
]


def main() -> None:
    if dspy.__version__ != PINNED:
        raise SystemExit(f"expected dspy {PINNED}, found {dspy.__version__}")
    entries = [
        {"rust": rust, "python": type(value).__name__, "json": value.model_dump(mode="json")}
        for rust, value in CASES
    ]
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_lm_api_fixture.py",
        "dspy_version": PINNED,
        "entries": entries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT.name}: {len(entries)} instances", file=sys.stderr)


if __name__ == "__main__":
    main()
