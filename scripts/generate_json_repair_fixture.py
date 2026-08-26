"""Record what `json_repair` makes of malformed JSON, by running it.

The Rust port is line-for-line, which is worth nothing on its own: `parse_string.py` is nine
hundred lines of heuristics with no specification other than itself, and a transcription error
there reads exactly like the original until an input reaches it. So the oracle is the library —
imported, run, and its answers written down.

**The expected value is recorded as `json.dumps` text, not as JSON structure.** Python distinguishes
`7` from `7.0`, keeps a key at the position it was first assigned, holds an integer wider than a
machine word exactly, and spells infinity `Infinity`. A structural copy loses the first and the
third; the bytes `json.dumps` produces lose none of them, and comparing them exercises the crate's
own `json.dumps` at the same time.

**The corpus is measured, not asserted.** A hand-picked case list proves whatever its author
thought of, so this traces execution through `json_repair` itself and refuses to write a fixture
that leaves the modules the port depends on unreached — the failure this guards against is a
hundred green cases that between them never enter `_handle_right_delimiter_candidate`.

    .venv/bin/python scripts/generate_json_repair_fixture.py
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import trace

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import json_repair  # noqa: E402
from json_repair_corpus import CASES, Case  # noqa: E402
from pins import require  # noqa: E402

OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates"
    / "dsrust-json-repair"
    / "tests"
    / "conformance"
    / "json_repair.json"
)

#: Every module the default `loads(text)` path reaches, and how many distinct lines of each the
#: corpus must reach. These are the measured numbers rather than a margin under them, so deleting a
#: case group fails here instead of quietly narrowing what the fixture tests. Raise one only by
#: adding cases; a *lower* number is a corpus that stopped covering something it used to.
COVERAGE_FLOOR = {
    "json_parser.py": 154,
    "json_repair.py": 65,
    "parse_array.py": 41,
    "parse_comment.py": 47,
    "parse_number.py": 22,
    "parse_object.py": 202,
    "parse_string.py": 425,
    "parser_parenthesized.py": 71,
    "object_value_context.py": 47,
    "parse_boolean_or_null.py": 16,
    "parse_json_llm_block.py": 6,
    "json_context.py": 21,
    "object_comparer.py": 13,
}


def run(case: Case) -> dict[str, object]:
    """What `json_repair` answers for one case, or how it refuses.

    The **repair log** is recorded alongside the value, and it is the more discriminating of the
    two: it names which branch the parser took to get there, so a port that reaches the right
    answer by the wrong route is still caught. A value alone cannot tell those apart.
    """
    read = (
        (lambda text, **kwargs: json_repair.load(io.StringIO(text), **kwargs))
        if case.from_file
        else json_repair.loads
    )
    try:
        value, log = read(case.text, logging=True, **case.options)
    except Exception as error:  # noqa: BLE001 — the refusal itself is the record
        return {"ok": False, "error": type(error).__name__, "message": str(error)}
    return {"ok": True, "dumps": json.dumps(value), "log": log}


def traced_line_counts() -> dict[str, int]:
    """Distinct lines of `json_repair` the whole corpus reaches."""
    tracer = trace.Trace(count=1, trace=0)
    tracer.runfunc(lambda: [run(case) for case in CASES])
    package = str(pathlib.Path(json_repair.__file__).parent)
    reached: dict[str, set[int]] = {}
    for (filename, lineno), _count in tracer.results().counts.items():
        if not filename.startswith(package):
            continue
        reached.setdefault(pathlib.Path(filename).name, set()).add(lineno)
    return {name: len(lines) for name, lines in reached.items()}


def check_the_corpus_discriminates(cases: list[dict[str, object]], coverage: dict[str, int]) -> None:
    """Refuse to write a fixture that cannot fail for the reasons it exists to catch."""
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise SystemExit(f"duplicate case names, so one would silently shadow another: {duplicates}")

    # A corpus every case of which is already valid JSON tests CPython's scanner and never the
    # repairs — which is the whole library.
    repaired = [case for case in CASES if not is_valid_json(case.text)]
    if len(repaired) < len(CASES) // 2:
        raise SystemExit(
            f"only {len(repaired)} of {len(CASES)} cases are malformed — most of this corpus never "
            "reaches the repair parser at all"
        )

    # Both answers have to appear, or the comparison only ever checks one arm.
    accepted = [case for case in cases if case["ok"]]
    if not accepted or len(accepted) == len(cases):
        raise SystemExit(f"{len(accepted)} of {len(cases)} accepted — the corpus exercises one arm")

    short = {
        name: (coverage.get(name, 0), floor)
        for name, floor in COVERAGE_FLOOR.items()
        if coverage.get(name, 0) < floor
    }
    if short:
        detail = ", ".join(f"{name} reached {got} of {floor}" for name, (got, floor) in short.items())
        raise SystemExit(f"the corpus stops short of what it claims to cover: {detail}")


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def main() -> None:
    require("json_repair")
    cases = []
    for case in CASES:
        record = {"name": case.name, "why": case.why, "input": case.text}
        if case.options:
            record["options"] = case.options
        if case.diverges:
            record["diverges"] = case.diverges
        if case.from_file:
            record["from_file"] = True
        cases.append(record | run(case))

    coverage = traced_line_counts()
    check_the_corpus_discriminates(cases, coverage)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": f"json_repair {json_repair_version()}, CPython {sys.version.split()[0]}",
                "what": (
                    "What json_repair.loads returns for each input, as the bytes json.dumps writes "
                    "for it. Recorded by running the library, never by hand: the value distinguishes "
                    "int from float, keeps a key at the position it was first assigned, and holds an "
                    "integer wider than a machine word exactly, and all three are observable here."
                ),
                "coverage": dict(sorted(coverage.items())),
                "cases": cases,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    refused = [case["name"] for case in cases if not case["ok"]]
    diverging = [case["name"] for case in cases if case.get("diverges")]
    print(f"  wrote {OUT} — {len(cases)} cases, {len(refused)} refused: {refused}", file=sys.stderr)
    print(f"  declared divergences: {diverging}", file=sys.stderr)
    print(f"  lines reached: {sum(coverage.values())} over {len(coverage)} modules", file=sys.stderr)


def json_repair_version() -> str:
    from importlib.metadata import version

    return version("json_repair")


if __name__ == "__main__":
    main()
