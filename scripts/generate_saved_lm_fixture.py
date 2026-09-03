"""The `lm` block dspy writes into a saved program, for models this crate can also build.

`generate_lm_state_fixture.py` records what *survives* a load — which keys `_sanitize_lm_state`
drops. This records the block itself: what `BaseLM.dump_state` emits, key by key and in order, for
an LM built the way a caller builds one. The two are separate corpora because they pin separate
things, and only this one can be compared against bytes Rust produced.

Key order is the point. `dspy.load` does not care, but the interop claim is that a file written here
is the file dspy would have written, and a diff is how anyone checks that. The order is not a rule
anybody wrote down — it falls out of the order `LM.__init__` puts things in `self.kwargs` — so it is
captured by running the constructor rather than by transcribing a list.

`finetuning_model`, `launch_kwargs` and `train_kwargs` are in every block. This crate has no
finetuning surface at all, so it emits the defaults dspy emits; they are recorded here to keep that
honest rather than assumed.

    .venv/bin/python scripts/generate_saved_lm_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy

from pins import require

OUT = (
    pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "state"
)
PINNED = require("dspy")

#: Each case is what a caller passed to `dspy.LM`. Every wire this crate supports appears, because
#: a block naming a provider the Rust side cannot build is a block it cannot honour — and the
#: reconstruction has to fail loudly there rather than quietly answer from somewhere else.
CASES = [
    ("openai default", "openai/gpt-4o-mini", {}),
    ("openai sampled", "openai/gpt-4o-mini", {"temperature": 0.7, "max_tokens": 100}),
    ("openai uncached", "openai/gpt-4o-mini", {"cache": False, "num_retries": 7}),
    ("openai responses", "openai/gpt-5", {"model_type": "responses"}),
    ("anthropic", "anthropic/claude-sonnet-4-20250514", {"temperature": 1.0, "max_tokens": 4096}),
    ("ollama", "ollama_chat/llama3.2", {}),
    ("openrouter", "openrouter/meta-llama/llama-3.1-8b-instruct", {}),
    # The key an api_key would have occupied: `dump_state` filters it, so a saved program never
    # carries a credential. Asserted rather than assumed, below.
    ("with credential", "openai/gpt-4o-mini", {"api_key": "sk-not-a-real-key"}),
    # A redirect, which is what the sanitising exists for. Kept here whole: this corpus is what a
    # *dump* produces, and dspy drops these on the load rather than on the save.
    ("redirected", "openai/gpt-4o-mini", {"api_base": "https://elsewhere.example/v1"}),
]


def main() -> None:
    from dspy.clients.base_lm import LM_CLASS_STATE_KEY, _BUILTIN_LM_CLASS_PATH

    blocks = []
    for name, model, kwargs in CASES:
        state = dspy.LM(model, **kwargs).dump_state()
        blocks.append({"name": name, "model": model, "kwargs": kwargs, "block": state})

    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_saved_lm_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "What BaseLM.dump_state writes for a model a caller built. Key order is dspy's and is "
            "part of what a byte comparison checks."
        ),
        "class_key": LM_CLASS_STATE_KEY,
        "builtin_class": _BUILTIN_LM_CLASS_PATH,
        "cases": blocks,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "saved_lm.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)

    # A block that carried a credential would make every saved program a leak, and the filter that
    # prevents it is one line of upstream that a port could silently not have.
    leaked = [b["name"] for b in blocks if any("api_key" in k for k in b["block"])]
    if leaked:
        raise SystemExit(f"an api_key reached the saved block in {leaked}")

    # Two arms of the corpus have to differ or the reconstruction is untested: a block with the
    # unsafe keys, and one without.
    unsafe = {"api_base", "base_url", "model_list"}
    if not any(unsafe & set(b["block"]) for b in blocks):
        raise SystemExit("no case carries an unsafe key — sanitising cannot be exercised")
    if not any(not (unsafe & set(b["block"])) for b in blocks):
        raise SystemExit("every case carries an unsafe key — nothing shows the untouched path")

    orders = {tuple(b["block"]) for b in blocks}
    print(f"    {len(blocks)} blocks, {len(orders)} distinct key orders", file=sys.stderr)


if __name__ == "__main__":
    main()
