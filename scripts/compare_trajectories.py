"""What dspy and this crate actually PUT on the wire, for the same signature, to the same engine.

The fixtures under `tests/conformance/lm_api/` already compare request bodies, and they do it more
strongly than a live run could — `generate_litellm_wire_fixture.py` mocks litellm's HTTP handler and
captures the exact bytes it was about to send. What they cannot do is start from a *declared
signature*: they begin at a typed `LMRequest`, so everything between "a caller writes a struct" and
"a request exists" is checked by other means, in pieces.

This closes that path end to end, and closes it from outside both libraries. A proxy sits between
the clients and a local engine; dspy asks through it, this crate asks through it, and neither is
asked to report on itself. What comes out is a diff or the word IDENTICAL.

It needs a running OpenAI-compatible engine, so it is a tool rather than a gate — the same standing
as the `#[ignore]`d live tests.

    python3 scripts/compare_trajectories.py --engine http://127.0.0.1:8099/v1 --model gemma-4-e2b
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).parent.parent
PROXY_PORT = 8098

#: Both sides declare this, field for field. The Rust half is `examples/trajectory.rs`; keeping the
#: two literally side by side is what makes a rendered difference the libraries' and not ours.
PYTHON_SIDE = '''
import dspy

class PracticeForStep(dspy.Signature):
    """Write independent practice for this lesson step. The practice must set a
    different task from the worked example a learner has just been shown the
    answer to: keep the skill, change the values. Give the expected answer."""

    learning_goal: str = dspy.InputField()
    worked_example_problem: str = dspy.InputField()
    worked_example_answer: str = dspy.InputField()
    practice_question: str = dspy.OutputField()
    expected_answer: str = dspy.OutputField()

CASE = {{
    "learning_goal": "Order common fractions.",
    "worked_example_problem": "Order 1/6, 1/3 and 1/2 from least to greatest.",
    "worked_example_answer": "1/6 < 1/3 < 1/2",
}}

dspy.configure(lm=dspy.LM("openai/{model}", api_base="{proxy}", api_key="python-dspy", cache=False))

for module in (dspy.Predict(PracticeForStep), dspy.ChainOfThought(PracticeForStep)):
    try:
        print(f"{{type(module).__name__}}: {{module(**CASE).practice_question}}")
    except Exception as error:
        print(f"{{type(module).__name__}}: {{type(error).__name__}}: {{error}}")
'''


def recorder(upstream: str, recording: list[dict]) -> type[BaseHTTPRequestHandler]:
    """A proxy that writes down every request before forwarding it, tagged by the client's key."""

    class Recorder(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _relay(self, body: bytes | None) -> None:
            asked = urllib.request.Request(
                upstream + self.path,
                data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
                method="POST" if body is not None else "GET",
            )
            with urllib.request.urlopen(asked, timeout=3600) as answer:
                payload = answer.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802 — the base class's spelling
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            client = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
            recording.append({"client": client, "body": json.loads(body)})
            self._relay(body)

        def do_GET(self) -> None:  # noqa: N802
            self._relay(None)

        def log_message(self, *_: object) -> None:
            pass

    return Recorder


def diff(label: str, ours: dict, theirs: dict) -> bool:
    print(f"\n{label}\n{'-' * len(label)}")
    same = True
    for index, (mine, yours) in enumerate(zip(ours["messages"], theirs["messages"])):
        if mine == yours:
            print(f"  message {index} ({mine['role']}): IDENTICAL, {len(mine['content'])} bytes")
            continue
        same = False
        print(f"  message {index}: DIFFERS")
        for line in difflib.unified_diff(
            str(yours.get("content", yours)).splitlines(),
            str(mine.get("content", mine)).splitlines(),
            fromfile="python/dspy", tofile="rust/dsrust", lineterm="",
        ):
            print(f"    {line}")
    rest = ({k: v for k, v in ours.items() if k != "messages"},
            {k: v for k, v in theirs.items() if k != "messages"})
    if rest[0] != rest[1]:
        same = False
        print("  the rest of the request differs:")
        for key in sorted(set(rest[0]) | set(rest[1])):
            if rest[0].get(key) != rest[1].get(key):
                print(f"    {key}: python={rest[1].get(key)!r}  rust={rest[0].get(key)!r}")
    else:
        print(f"  every other request field: IDENTICAL {sorted(rest[0])}")
    return same


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="http://127.0.0.1:8099/v1")
    parser.add_argument("--model", default="gemma-4-e2b")
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    args = parser.parse_args()

    upstream = urllib.parse.urlparse(args.engine)
    recording: list[dict] = []
    proxy = ThreadingHTTPServer(
        ("127.0.0.1", PROXY_PORT), recorder(f"{upstream.scheme}://{upstream.netloc}", recording)
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    through = f"http://127.0.0.1:{PROXY_PORT}{upstream.path}"
    print(f"proxy on {PROXY_PORT} -> {args.engine}\n")

    print("==> dsrust")
    subprocess.run(
        ["cargo", "run", "--release", "--quiet", "--example", "trajectory"],
        cwd=ROOT, check=True,
        env={**__import__("os").environ, "TRAJECTORY_BASE_URL": through, "TRAJECTORY_MODEL": args.model},
    )
    print("\n==> dspy")
    subprocess.run([args.python, "-c", PYTHON_SIDE.format(model=args.model, proxy=through)], check=True)
    proxy.shutdown()

    ours = [ask["body"] for ask in recording if ask["client"] == "rust-dsrust"]
    theirs = [ask["body"] for ask in recording if ask["client"] == "python-dspy"]
    print(f"\nrecorded {len(ours)} asks from dsrust, {len(theirs)} from dspy")
    if len(ours) != len(theirs):
        print("the two made a different number of calls, which is itself the finding")
        raise SystemExit(1)

    verdicts = [diff(name, o, t) for name, o, t in zip(["Predict", "ChainOfThought"], ours, theirs)]
    print("\nVERDICT:", "identical on every ask" if all(verdicts) else "DIVERGENCE ABOVE")
    raise SystemExit(0 if all(verdicts) else 1)


if __name__ == "__main__":
    main()
