"""Reproduce the `TypeError` upstream's parallelizer raises when SIGINT is not callable.

`upstream_test_evaluate.py::test_multi_thread_evaluate_call_cancelled` fails intermittently in a
full-suite run with `TypeError: 'Handlers' object is not callable`. This is that mechanism,
isolated: dspy's parallelizer saves `signal.getsignal(SIGINT)` and calls it from inside the handler
it installs, without checking that what it saved is callable. `SIG_IGN` and `SIG_DFL` are members of
an enum, so the call raises instead of cancelling the run.

Run this and the failure is deterministic. What is *not* known is what leaves SIGINT as a sentinel
during a real suite run. Established so far:

  * No earlier test does it — a teardown probe over the whole suite reported only the failing test.
  * It is already `SIG_IGN` at that test's *setup*, so it arrives before the test body.
  * Forcing `trap "" INT` on the launching shell does not reproduce it.
  * Foreground runs pass; some backgrounded ones fail, and some pass.
  * Nothing in this repository installs a signal handler, and a `Predict` crossing into Rust leaves
    SIGINT untouched. Both checked rather than assumed.

Two fixes were written and both removed: each was a conftest fixture that never once fired, so the
green run after it was evidence of nothing. Start here, with the mechanism confirmed, and find the
trigger before writing a third.

    .venv/bin/python scripts/reproduce_parallelizer_sigint.py
"""

from __future__ import annotations

import os
import signal
import threading
import time

import dspy
from dspy.utils.dummies import DummyLM


class SlowLM(DummyLM):
    """Slow enough that the signal lands mid-evaluation, as upstream's own test arranges."""

    def __call__(self, *args, **kwargs):
        time.sleep(1)
        return super().__call__(*args, **kwargs)


def evaluate_with(sentinel, name: str) -> str:
    signal.signal(signal.SIGINT, sentinel)
    dspy.configure(lm=SlowLM([{"answer": "x"} for _ in range(40)]))
    program = dspy.Predict("question -> answer")
    devset = [dspy.Example(question=f"q{i}", answer="x").with_inputs("question") for i in range(10)]
    evaluate = dspy.Evaluate(devset=devset, num_threads=4, display_progress=False)

    threading.Timer(0.6, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    try:
        evaluate(program, metric=lambda example, prediction, trace=None: True)
        return f"SIGINT was {name}: completed, no TypeError"
    except TypeError as error:
        return f"SIGINT was {name}: TypeError -> {error}"
    except (KeyboardInterrupt, SystemExit):
        return f"SIGINT was {name}: cancelled cleanly, which is what the test asserts"


def main() -> None:
    for sentinel, name in ((signal.SIG_IGN, "SIG_IGN"), (signal.default_int_handler, "the default")):
        print("  " + evaluate_with(sentinel, name))
    signal.signal(signal.SIGINT, signal.default_int_handler)


if __name__ == "__main__":
    main()
