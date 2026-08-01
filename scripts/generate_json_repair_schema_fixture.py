"""Record what `json_repair` does with a schema — and every question it asked `jsonschema`.

The schema modules are the half of the library dspy never reaches, and they are also the half that
delegates: `SchemaRepairer.is_valid` and `.validate` are `jsonschema`, a *separate* package again.
Reproducing a JSON Schema validator would be porting a fourth library and guessing at which draft,
so the crate exposes that as a seam — and this fixture is what lets the seam be tested rather than
merely declared.

Every call the repairer makes to the validator is recorded here with its answer. The Rust test
plugs in a validator that replays the table and **fails on a question that was never asked**, so a
port that validates a different value, validates one fewer time, or skips validation altogether is
caught. What is *not* being tested is `jsonschema` itself, which is the point.

    .venv/bin/python scripts/generate_json_repair_schema_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import json_repair  # noqa: E402
from json_repair.schema_repair import SchemaRepairer  # noqa: E402
from json_repair_schema_corpus import CASES  # noqa: E402
from pins import require  # noqa: E402

OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust-json-repair"
    / "tests"
    / "conformance"
    / "json_repair_schema.json"
)


class Recorder:
    """Wraps the two validator entry points, keeping what each was asked and what it answered."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._is_valid = SchemaRepairer.is_valid
        self._validate = SchemaRepairer.validate

    def __enter__(self) -> "Recorder":
        recorder = self

        def is_valid(repairer, value, schema):  # noqa: ANN001, ANN202
            try:
                answer = recorder._is_valid(repairer, value, schema)
            except Exception as error:  # noqa: BLE001
                # A validator that cannot read the schema at all. Not a `ValueError`, so nothing
                # in `json_repair` catches it — recording it is what lets the Rust seam raise the
                # same way instead of quietly answering False.
                recorder.record(repairer, "is_valid", value, schema, {"ok": False, "raised": str(error)})
                raise
            recorder.record(repairer, "is_valid", value, schema, {"ok": bool(answer)})
            return answer

        def validate(repairer, value, schema):  # noqa: ANN001, ANN202
            try:
                recorder._validate(repairer, value, schema)
            except ValueError as error:
                recorder.record(repairer, "validate", value, schema, {"ok": False, "message": str(error)})
                raise
            except Exception as error:  # noqa: BLE001
                # Not a `ValueError`, so nothing in `json_repair` catches it. Recording it is what
                # lets the Rust seam answer the same way rather than reporting a plain refusal.
                recorder.record(repairer, "validate", value, schema, {"ok": False, "raised": str(error)})
                raise
            recorder.record(repairer, "validate", value, schema, {"ok": True})

        SchemaRepairer.is_valid = is_valid
        SchemaRepairer.validate = validate
        return self

    def __exit__(self, *_exc: object) -> None:
        SchemaRepairer.is_valid = self._is_valid
        SchemaRepairer.validate = self._validate

    def record(self, repairer, method: str, value: object, schema: object, answer: dict) -> None:  # noqa: ANN001
        """Records a question that actually reached `jsonschema`.

        A boolean schema never does: `is_valid` and `validate` both answer it themselves, before
        the import. The Rust seam sits on the far side of that same short-circuit, so recording
        those would put a question in the table that the crate is right never to ask.
        """
        if isinstance(repairer.resolve_schema(schema), bool):
            return
        self.calls.append(
            {"method": method, "value": dumps(value), "schema": dumps(schema)} | answer
        )


def dumps(value: object) -> str:
    """A key for the replay table. `default=str` because a schema may hold a pydantic default."""
    return json.dumps(value, default=str)


def run(case: dict) -> dict[str, object]:
    with Recorder() as recorder:
        try:
            value, log = json_repair.loads(
                case["input"],
                schema=case["schema"],
                schema_repair_mode=case.get("mode", "standard"),
                logging=True,
            )
            # The log is the more discriminating half here too: `schema_repair.py` narrates every
            # coercion, drop and fill it makes, so it says *which* rule produced the value.
            answer: dict[str, object] = {"ok": True, "dumps": json.dumps(value), "log": log}
        except Exception as error:  # noqa: BLE001 — a refusal is half of what a schema does
            answer = {"ok": False, "error": type(error).__name__, "message": str(error)}
    return answer | {"calls": recorder.calls}


def check_the_corpus_discriminates(cases: list[dict]) -> None:
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate case names: {sorted({n for n in names if names.count(n) > 1})}")

    accepted = [case for case in cases if case["ok"]]
    if not accepted or len(accepted) == len(cases):
        raise SystemExit(f"{len(accepted)} of {len(cases)} accepted — the corpus exercises one arm")

    # A corpus that never makes the repairer ask the validator anything would pass against a seam
    # that was never wired up at all, which is exactly the failure this fixture exists to rule out.
    asked = sum(len(case["calls"]) for case in cases)
    if asked < len(cases):
        raise SystemExit(f"only {asked} validator calls over {len(cases)} cases — the seam is idle")

    salvaged = [case for case in cases if case.get("mode") == "salvage"]
    if not salvaged:
        raise SystemExit("no salvage-mode case, so half of schema_repair.py is unreached")

    for keyword in ("oneOf", "anyOf", "$ref", "patternProperties", "additionalProperties", "enum"):
        if not any(keyword in json.dumps(case["schema"]) for case in cases):
            raise SystemExit(f"no case uses {keyword}")


def main() -> None:
    require("json_repair")
    cases = [dict(case) | run(case) for case in CASES]
    check_the_corpus_discriminates(cases)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"json_repair {version()} with jsonschema {version('jsonschema')}",
                "what": (
                    "What a schema-guided repair returns, and every question the repairer asked the "
                    "validator to get there. The Rust seam replays the answers and fails on a "
                    "question that was never asked, so the repair decisions are tested without "
                    "reimplementing JSON Schema."
                ),
                "cases": cases,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    asked = sum(len(case["calls"]) for case in cases)
    refused = [case["name"] for case in cases if not case["ok"]]
    print(f"  wrote {OUT} — {len(cases)} cases, {asked} validator calls", file=sys.stderr)
    print(f"  {len(refused)} refused: {refused}", file=sys.stderr)


def version(name: str = "json_repair") -> str:
    from importlib.metadata import version as installed

    return installed(name)


if __name__ == "__main__":
    main()
