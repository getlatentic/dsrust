"""Replay every committed json-repair corpus through the pinned library, and record what it enters.

A ledger entry that names a Rust counterpart asserts two functions do the same thing. Nothing made
the *Python* one run. `parse_string.py::_post_fence_container_starts_next_member` was mapped, was
ported correctly as it turned out, and no corpus had ever entered it — so its entry rested on a
reading and nothing else, in the one file this port calls the hard third.

Entry rather than line coverage: the question is "did any case make this function run", and call
events cost a second over twelve hundred cases where line tracing costs a minute.

Each corpus is replayed through the entry point it was *recorded* from. `load(fd)` and `loads(text)`
take different paths — upstream turns the suffix fast path off for file input — so a replay that
collapsed them would report `string_file_wrapper.py` as dead code.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "crates" / "dsrust-json-repair" / "tests" / "conformance"


def quietly(call) -> None:
    """Run it for its side effects on the tracer; a refusal executes lines too."""
    try:
        call()
    except Exception:  # noqa: BLE001 — every corpus records refusals as answers
        pass


def _main(case: dict) -> None:
    import json_repair

    text, options = case["input"], case.get("options") or {}
    if case.get("from_file"):
        quietly(lambda: json_repair.load(io.StringIO(text), logging=True, **options))
    else:
        quietly(lambda: json_repair.loads(text, logging=True, **options))
    quietly(lambda: json_repair.repair_json(text, **options))


def _schema(case: dict) -> None:
    import json_repair

    quietly(
        lambda: json_repair.loads(
            case["input"],
            schema=case["schema"],
            schema_repair_mode=case.get("mode", "standard"),
            logging=True,
        )
    )


def _plain(case: dict) -> None:
    import json_repair

    quietly(lambda: json_repair.loads(case.get("input", case.get("text", "")), logging=True))


def _bytes(case: dict) -> None:
    import json_repair

    text = case.get("input", case.get("text", ""))
    for ensure_ascii in (True, False):
        quietly(lambda: json_repair.repair_json(text, ensure_ascii=ensure_ascii))


#: Which runner replays which committed fixture.
REPLAY = {
    "json_repair.json": _main,
    "json_repair_schema.json": _schema,
    "json_repair_sweep.json": _plain,
    "json_repair_upstream.json": _plain,
    "json_repair_bytes.json": _bytes,
}


def cases_of(path: pathlib.Path) -> list[dict]:
    return json.loads(path.read_text()).get("cases", [])


def entered() -> tuple[set[str], int]:
    """Every `module.py::name` any corpus runs, and how many cases were replayed."""
    import json_repair

    package = pathlib.Path(json_repair.__file__).parent
    found: set[str] = set()

    def profiler(frame, event, _arg):
        if event != "call":
            return
        code = frame.f_code
        if code.co_filename.startswith(str(package)):
            module = pathlib.Path(code.co_filename).relative_to(package)
            found.add(f"{module}::{code.co_name}")

    work = [(runner, case) for name, runner in REPLAY.items() for case in cases_of(FIXTURES / name)]
    if not work:
        raise SystemExit(f"no corpus under {FIXTURES} — nothing to replay")
    sys.setprofile(profiler)
    try:
        for runner, case in work:
            runner(case)
    finally:
        sys.setprofile(None)
    return found, len(work)
