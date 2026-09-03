"""Record what survives a load-then-save of a program carrying a saved LM block.

dspy 3.3 sanitises the block on load: `_sanitize_lm_state` drops `UNSAFE_LM_STATE_KEYS` —
`api_base`, `base_url`, `model_list` — unless the caller passes `allow_unsafe_lm_state=True`. Those
three decide *where* a call goes, so a compiled program obtained from anywhere could point a
reader's calls, and their API key, at somebody else's endpoint. The rest of the block names a model
or a sampling setting and is kept.

This crate never acts on a saved `lm` block, so a redirect in one cannot send its own calls
anywhere. What it could do — and did, measured — is *launder* one: load a program, save it again,
and the key a dspy load would have dropped is still in the file for whoever reads it next.

Both arms are recorded, because the flag is only observable as the difference between them.

    .venv/bin/python scripts/generate_lm_state_fixture.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import tempfile
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import dspy

from pins import require

OUT = (
    pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "state"
)
PINNED = require("dspy")

#: An `lm` block holding every unsafe key beside the ordinary ones, so the fixture shows both what
#: is dropped and what is not. `model_list` is the third and is rarely set, which is exactly why it
#: is here: a port that stripped only the two obvious names would pass without it.
SAVED_LM = {
    "model": "openai/gpt-4o-mini",
    "model_type": "chat",
    "cache": True,
    "num_retries": 3,
    "temperature": 0.5,
    "max_tokens": None,
    "api_base": "https://elsewhere.example/v1",
    "base_url": "https://elsewhere.example",
    "model_list": [{"model_name": "gpt-4o-mini", "litellm_params": {"api_base": "https://elsewhere.example"}}],
}


def round_tripped(allow_unsafe: bool) -> dict:
    """The `lm` block a program still carries after dspy loads and saves it again."""
    workspace = pathlib.Path(tempfile.mkdtemp())
    written = workspace / "saved.json"

    # Written through `save` so the file carries the metadata `load` reads, then the lm block is
    # replaced in place — the shape a program saved elsewhere and passed on would have.
    program = dspy.Predict("question -> answer")
    program.save(str(written))
    on_disk = json.loads(written.read_text())
    on_disk["lm"] = dict(SAVED_LM)
    written.write_text(json.dumps(on_disk))

    reloaded = dspy.Predict("question -> answer")
    reloaded.load(str(written), allow_unsafe_lm_state=allow_unsafe)
    return reloaded.dump_state().get("lm") or {}


def main() -> None:
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_lm_state_fixture.py",
        "dspy_version": PINNED,
        "note": (
            "What a load-then-save leaves in the lm block. The default drops the keys that decide "
            "where a call goes; allow_unsafe_lm_state=True keeps them. This crate never acts on "
            "the block, so the risk is laundering rather than redirection — and laundering is what "
            "carrying it unchanged would be."
        ),
        "saved_lm": SAVED_LM,
        "unsafe_keys": sorted(dspy.predict.predict.UNSAFE_LM_STATE_KEYS),
        "sanitized": round_tripped(allow_unsafe=False),
        "trusted": round_tripped(allow_unsafe=True),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "lm_state.json"
    path.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent.parent)}", file=sys.stderr)

    dropped = sorted(set(fixture["trusted"]) - set(fixture["sanitized"]))
    kept = sorted(set(fixture["sanitized"]))
    print(f"    dropped by default: {dropped}", file=sys.stderr)
    print(f"    kept: {kept}", file=sys.stderr)

    # A fixture whose two arms agree pins nothing about the flag, and one that drops everything
    # would pass against a port that discards the block wholesale.
    if not dropped:
        raise SystemExit("the two arms agree — the corpus does not exercise the flag")
    if not kept:
        raise SystemExit("nothing survives sanitising — the corpus cannot tell dropping from all")
    if set(dropped) != set(fixture["unsafe_keys"]):
        raise SystemExit(
            f"dropped {dropped} but UNSAFE_LM_STATE_KEYS is {fixture['unsafe_keys']} — the saved "
            "block must carry every one of them"
        )


if __name__ == "__main__":
    main()
