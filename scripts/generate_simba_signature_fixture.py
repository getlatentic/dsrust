"""SIMBA's own `OfferFeedback`, which is not `refine.py`'s despite the shared name.

Both are called `OfferFeedback` and both ask a model to advise a program's modules. They are
different signatures: refine's assigns blame against a *threshold* over one trajectory, and this one
contrasts a **worse trajectory against a better one** and asks for advice keyed by module name. Thirteen
input fields against seven, a `dict[str, str]` output rather than a `str`, and an instruction block
of its own.

`inspect_modules` is the opposite case — SIMBA's and refine's differ only by an unused `enumerate`
and render identically, which is asserted here rather than assumed so the shared Rust helper is
justified by a comparison instead of by a reading.

    .venv/bin/python scripts/generate_simba_signature_fixture.py
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
from dspy.predict.refine import OfferFeedback as RefineOfferFeedback
from dspy.predict.refine import inspect_modules as refine_inspect_modules
from dspy.teleprompt.simba_utils import OfferFeedback, inspect_modules

from pins import require

PINNED = require("dspy")
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates" / "dsrust" / "tests" / "conformance" / "optimize" / "simba_offer_feedback.json"
)

FILLED = {
    "program_code": "class Program(dspy.Module): ...",
    "modules_defn": "Module predict\n\tInput Fields:\n\t\t1. `question`",
    "program_inputs": '{\n  "question": "capital of Spain?"\n}',
    "oracle_metadata": '{\n  "answer": "Madrid"\n}',
    "worse_program_trajectory": '[\n  {"module_name": "predict"}\n]',
    "worse_program_outputs": '{\n  "answer": "Barcelona"\n}',
    "worse_reward_value": 0.0,
    "worse_reward_info": "{}",
    "better_program_trajectory": '[\n  {"module_name": "predict"}\n]',
    "better_program_outputs": '{\n  "answer": "Madrid"\n}',
    "better_reward_value": 1.0,
    "better_reward_info": "{}",
    "module_names": '[\n  "predict"\n]',
}


def fields(cls) -> dict:
    return {
        "instructions": cls.instructions,
        "inputs": [
            {"name": name, "desc": info.json_schema_extra.get("desc", ""),
             "annotation": getattr(info.annotation, "__name__", str(info.annotation))}
            for name, info in cls.input_fields.items()
        ],
        "outputs": [
            {"name": name, "desc": info.json_schema_extra.get("desc", ""),
             "annotation": getattr(info.annotation, "__name__", str(info.annotation))}
            for name, info in cls.output_fields.items()
        ],
    }


def main() -> None:
    adapter = dspy.ChatAdapter()
    rendered = adapter.format(signature=OfferFeedback, demos=[], inputs=FILLED)

    program = dspy.Predict("question -> answer")
    same = inspect_modules(program) == refine_inspect_modules(program)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"dspy=={PINNED} teleprompt/simba_utils.py::OfferFeedback",
                "dspy_version": PINNED,
                "note": (
                    "SIMBA's `OfferFeedback`, which shares a name with `refine.py`'s and is a "
                    "different signature. Recorded ahead of the port, with the rendered prompt so "
                    "the field order and every description are compared as bytes."
                ),
                "signature": fields(OfferFeedback),
                "refine_signature_field_count": len(RefineOfferFeedback.input_fields),
                "inspect_modules_matches_refines": same,
                "inspect_modules": inspect_modules(program),
                "filled_inputs": FILLED,
                "rendered": rendered,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(
        f"  wrote {OUT.name}: {len(OfferFeedback.input_fields)} inputs vs refine's "
        f"{len(RefineOfferFeedback.input_fields)}; inspect_modules matches refine's: {same}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
