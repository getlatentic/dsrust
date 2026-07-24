"""Open a program this crate saved, using dspy's own reader.

The portability claim is that someone can take the JSON a Rust optimizer wrote and run it in
Python. Asserting that in Rust could only check the shape against a description of dspy's format;
this checks it against dspy, by handing the file to `Module.load` and printing what came back.

    .dspy-venv/bin/python scripts/check_saved_program.py <path-to-saved.json>
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import dspy


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <path-to-saved.json>")
    path = sys.argv[1]

    # The same program the Rust side compiled, fresh: `load` restores state onto a program that
    # already has the right shape, exactly as a user reloading their own program would.
    program = dspy.ChainOfThought("question -> answer")
    program.load(path)

    predictor = program.predict
    print(f"instructions={predictor.signature.instructions!r}")
    print(f"demos={len(predictor.demos)}")
    for name, field in predictor.signature.fields.items():
        extra = field.json_schema_extra
        print(f"field {name}: prefix={extra.get('prefix')!r} desc={extra.get('desc')!r}")

    # Loading is not running. Ask the restored program for a prediction through a scripted LM, so
    # the check covers the prompt dspy builds from the restored state rather than only the parse.
    with dspy.context(lm=dspy.utils.DummyLM([{"reasoning": "because", "answer": "4"}])):
        answered = program(question="2+2?")
    print(f"answered={answered.answer!r}")


if __name__ == "__main__":
    main()
