"""Hold the plan to what actually runs.

`backlog.toml` says which upstream suites a sprint shipped. `run_upstream_tests.sh` says which
ones this crate is held to. Nothing connected the two, so a sprint could be marked done while
naming a file the suite never ran — the plan claiming coverage the gates do not check.

Run by the upstream runner before the suite, so a claim and its evidence cannot part company.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).parent.parent
BACKLOG = ROOT / "backlog.toml"
RUNNER = ROOT / "scripts" / "run_upstream_tests.sh"
MANIFEST = ROOT / "scripts" / "upstream_tests.txt"
VERSION = (ROOT / "scripts" / "DSPY_VERSION").read_text().strip()


def running() -> set[str]:
    """The suites the runner names, read from the array itself."""
    block = re.search(r"SUITES=\((.*?)\n\)", RUNNER.read_text(), re.S)
    if block is None:
        raise SystemExit(f"{RUNNER.name} has no SUITES array to read")
    return set(re.findall(r"[\w/]+/test_\w+\.py", block.group(1)))


def shipped() -> dict[str, list[str]]:
    """Suites each finished sprint claims, keyed by sprint."""
    backlog = tomllib.loads(BACKLOG.read_text())
    return {
        sprint["id"]: sprint["suites"]
        for sprint in backlog.get("sprint", [])
        if sprint.get("state") in {"done", "in-progress"} and "suites" in sprint
    }


def stale_manifest() -> str | None:
    """Whether the manifest still lists the version it was generated at.

    The manifest is the denominator of every coverage number the runner prints, and it is a
    snapshot of another repository — so a version bump silently invalidates it. Re-listing the
    tree here would put the network on every run; the header says which version was listed, and
    holding that to the pin is enough to catch the bump that stranded it.
    """
    header = MANIFEST.read_text().splitlines()[0]
    if header.startswith(f"# dspy {VERSION}:"):
        return None
    return (
        f"the manifest lists {header.removeprefix('# ').split(':')[0]} but the pin is dspy "
        f"{VERSION}; run scripts/refresh_upstream_manifest.py"
    )


def complaints() -> list[str]:
    suites, found = running(), []
    manifest = {
        line.removeprefix("tests/")
        for line in MANIFEST.read_text().splitlines()
        if not line.startswith("#")
    }
    if (stale := stale_manifest()) is not None:
        found.append(stale)

    for sprint, claimed in shipped().items():
        for suite in claimed:
            # A sprint may describe a group it has not enumerated — "signatures/* (4 files)" —
            # which is prose about intent rather than a claim about a file.
            if "*" in suite:
                continue
            if suite not in manifest:
                found.append(f"{sprint} names {suite}, which dspy does not ship at this version")
            elif suite not in suites:
                found.append(f"{sprint} claims {suite}, which the runner does not run")

    for suite in sorted(suites):
        if suite not in manifest:
            found.append(f"the runner runs {suite}, which is not in the manifest")
    return found


def main() -> None:
    found = complaints()
    for complaint in found:
        print(f"  {complaint}", file=sys.stderr)
    if found:
        raise SystemExit(1)
    print(f"  the plan and the suite agree on {len(running())} files", file=sys.stderr)


if __name__ == "__main__":
    main()
