"""What litellm puts on the wire for `dspy.Embedder("openai/...")`, recorded at the HTTP layer.

`Embedder` hands litellm `embedding(model=..., input=[...], caching=..., **kwargs)`; litellm
builds an OpenAI `/embeddings` request from that. The body is captured where the OpenAI SDK sends
it — `httpx.Client.send` — which is the one place nothing above it can drift from. The Rust client
builds the same body from the same arguments and is held to these bytes.

    .venv/bin/python scripts/generate_embedding_wire_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from unittest import mock

import httpx

import dspy
from pins import require

PINNED = require("dspy")
OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "lm_api" / "embedding_wire.json"

CASES = [
    ("plain", "openai/text-embedding-3-small", ["hello", "world"], {}),
    ("with_dimensions", "openai/text-embedding-3-small", ["one input"], {"dimensions": 256}),
    ("bare_model", "text-embedding-ada-002", ["hello"], {}),
]


def main() -> None:
    captured = []

    def send(self, request, **kwargs):  # noqa: ANN001
        captured.append({
            "url": str(request.url),
            "method": request.method,
            "body": json.loads(request.content.decode()),
            "authorization": request.headers.get("authorization"),
        })
        n = len(captured[-1]["body"]["input"]) if isinstance(captured[-1]["body"].get("input"), list) else 1
        reply = {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": [0.1 * (i + 1), 0.2, 0.3]} for i in range(n)], "model": captured[-1]["body"]["model"], "usage": {"prompt_tokens": 0, "total_tokens": 0}}
        return httpx.Response(200, json=reply, request=request)

    recorded = []
    with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-recorded"}), mock.patch.object(httpx.Client, "send", send):
        for label, model, inputs, kwargs in CASES:
            embedder = dspy.Embedder(model, caching=False, **kwargs)
            vectors = embedder(inputs)
            wire = captured[-1]
            recorded.append({
                "label": label,
                "model": model,
                "inputs": inputs,
                "kwargs": kwargs,
                "url": wire["url"],
                "body": wire["body"],
                "bearer": wire["authorization"],
                "vectors": [[float(v) for v in row] for row in vectors],
            })
            print(f"    {label}: {wire['method']} {wire['url']} body {json.dumps(wire['body'])[:120]}", file=sys.stderr)
    fixture = {"source": f"generated from dspy=={PINNED} via scripts/generate_embedding_wire_fixture.py", "dspy_version": PINNED, "cases": recorded}
    OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT.name}: {len(recorded)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
