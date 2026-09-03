"""What dspy raises and what dsrust returns, for the same live failure.

`tests/exceptions_conformance.rs` compares constructed exception objects and
`openai_compatible.rs` compares stubbed statuses. Neither makes a call that actually fails, so
neither can catch a failure mode nothing on either side thought to construct. This does: both
libraries are pointed at the same endpoint and asked the same impossible thing.

It found one. A refused connection was `LMTransportError` and retryable in dspy, and an untyped
`anyhow` here — `LmErrorKind::Transport` existed and nothing produced it, so a caller could not see
the most ordinary transient failure there is as retryable.

Needs a live engine for the first case, and nothing for the second.

    .venv/bin/python scripts/compare_error_shapes.py
"""
import json, os, pathlib, subprocess, sys

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8080/v1")
CASES = [
    ("an unknown model", "no-such-model-anywhere", BASE),
    ("a port with nothing on it", "gemma", "http://127.0.0.1:9/v1"),
]

PY_SIDE = '''
import dspy, sys
from dspy.utils.exceptions import LMError, is_retryable_lm_error
dspy.configure(lm=dspy.LM("openai/{model}", api_base="{base}", api_key="x", cache=False,
                          num_retries=0))
try:
    dspy.Predict("question -> answer")(question="hi")
    print("NO ERROR")
except BaseException as e:
    kind = type(e).__name__
    code = getattr(e, "code", None)
    status = getattr(e, "status", None)
    retryable = is_retryable_lm_error(e) if isinstance(e, LMError) else None
    print(json.dumps({{"class": kind, "code": code, "status": status,
                      "retryable": retryable, "message": str(e)[:120]}}))
'''

for label, model, base in CASES:
    print(f"\n=== {label} ===")
    out = subprocess.run(
        [sys.executable, "-c", "import json\n" + PY_SIDE.format(model=model, base=base)],
        capture_output=True, text=True, timeout=180,
    )
    print("  dspy   :", (out.stdout.strip() or out.stderr.strip().splitlines()[-1])[:200])
    rust = subprocess.run(
        ["cargo", "run", "--quiet", "--example", "error_shape"],
        cwd=str(pathlib.Path(__file__).parent.parent / "crates" / "dsrust"),
        capture_output=True, text=True,
        env={**os.environ, "PROBE_MODEL": model, "PROBE_BASE": base}, timeout=600,
    )
    print("  dsrust :", (rust.stdout.strip() or rust.stderr.strip()[-200:])[:200])
