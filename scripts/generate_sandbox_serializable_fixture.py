"""What dspy's `build_repl_variable` produces for a SandboxSerializable value.

Upstream's own `test_sandbox_serializable.py` runs green against this crate and proves nothing about
it: every test there asserts on the ABC in Python — that an incomplete subclass cannot be
instantiated, that a duck-typed class fails `isinstance`, that the pydantic hook is a pass-through.
A Rust trait has none of those, which is why the whole file is declared not-crossing.

What *is* portable is the behaviour: given a value, `build_repl_variable` decides the preview the
model reads, the length reported beside it, and how `sandbox_setup` is folded into the description.
This captures that from running dspy, so `interpreter/sandbox.rs` is checked against the oracle
rather than against what its author expected.

    .venv/bin/python scripts/generate_sandbox_serializable_fixture.py
"""

from __future__ import annotations

import base64
import json
import pathlib
import sys

from dspy.primitives.sandbox_serializable import SandboxSerializable, build_repl_variable
from pydantic.fields import FieldInfo

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "interpreter"
PINNED = require("dspy")


class TextValue(SandboxSerializable):
    """Text that crosses as itself, with imports the model is told about."""

    def __init__(self, body: str) -> None:
        self.body = body

    def sandbox_setup(self) -> str:
        return "import json\nimport io"

    def to_sandbox(self) -> bytes:
        return self.body.encode("utf-8")

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = json.loads({data_expr})"

    def rlm_preview(self, max_chars: int = 500) -> str:
        return f"TextValue: {len(self.body)} characters"


class BinaryValue(SandboxSerializable):
    """The docstring's DataFrame shape: binary in, base64 across, no setup line."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def sandbox_setup(self) -> str:
        return ""

    def to_sandbox(self) -> bytes:
        return base64.b64encode(self.payload)

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = pd.read_parquet(io.BytesIO(base64.b64decode({data_expr})))"

    def rlm_preview(self, max_chars: int = 500) -> str:
        return "DataFrame: 3 rows x 2 columns"


class LongPreview(SandboxSerializable):
    """A preview longer than the description, so `total_length` is visibly the preview's."""

    def sandbox_setup(self) -> str:
        return "import pandas as pd"

    def to_sandbox(self) -> bytes:
        return b""

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = {data_expr}"

    def rlm_preview(self, max_chars: int = 500) -> str:
        return "x" * 120


#: Each case is (label, value, name, the field's json_schema_extra or None).
#:
#: `desc` is read out of `json_schema_extra`, not `FieldInfo.description` — a detail worth capturing
#: because a generator that sets the wrong one silently records the no-description branch. The
#: `${...}` case is upstream's placeholder guard: dspy fills an unstated desc with `${name}`, and
#: that must not reach the model as if someone wrote it.
CASES = [
    ("setup and a desc, which concatenate", TextValue("col_a,col_b"), "sales",
     {"desc": "last quarter"}),
    ("setup and no desc", TextValue("abc"), "corpus", None),
    ("a desc and constraints, no setup", BinaryValue(b"\x00\x01\x02"), "frame",
     {"desc": "the frame", "constraints": "at most 3 rows"}),
    ("a preview that outruns the desc", LongPreview(), "wide", {"desc": "a wide thing"}),
    ("a placeholder desc, which is dropped", TextValue("x"), "unnamed", {"desc": "${unnamed}"}),
]


def captured(value: SandboxSerializable, name: str, extra: dict | None) -> dict:
    """Every field of the REPLVariable dspy builds, plus the four protocol answers."""
    field_info = FieldInfo(json_schema_extra=extra) if extra is not None else None
    variable = build_repl_variable(value, name, field_info=field_info)
    return {
        "name": variable.name,
        "type_name": variable.type_name,
        "desc": variable.desc,
        "constraints": variable.constraints,
        "total_length": variable.total_length,
        "preview": variable.preview,
        "sandbox_setup": value.sandbox_setup(),
        "to_sandbox_base64": base64.b64encode(value.to_sandbox()).decode("ascii"),
        "sandbox_assignment": value.sandbox_assignment(name, "_raw_" + name),
        "rlm_preview": value.rlm_preview(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = {
        "_source": f"dspy {PINNED} build_repl_variable, via {pathlib.Path(__file__).name}",
        "cases": [
            {"label": label, **captured(value, name, extra)}
            for label, value, name, extra in CASES
        ],
    }
    path = OUT / "sandbox_serializable.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
