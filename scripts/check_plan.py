"""Hold the plan to what actually runs, in both directions.

`backlog.toml` says which upstream suites a sprint shipped. `run_upstream_tests.sh` says which
ones this crate is held to. Nothing connected the two, so a sprint could be marked done while
naming a file the suite never ran — the plan claiming coverage the gates do not check.

The other direction is the one nobody notices, because nothing goes red: a sprint left `planned`
while the modules it describes are built and its suites green. Four sat that way here. A plan that
under-reports reads as a to-do list with work still ahead of it, which is a claim about the project
as much as an over-report is.

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

#: States that read as "this shipped". Everything else is a claim that work remains.
FINISHED = {"done", "in-progress"}


def running() -> set[str]:
    """The suites the runner names, read from the array itself."""
    block = re.search(r"SUITES=\((.*?)\n\)", RUNNER.read_text(), re.S)
    if block is None:
        raise SystemExit(f"{RUNNER.name} has no SUITES array to read")
    return set(re.findall(r"[\w/]+/test_\w+\.py", block.group(1)))


def sprints() -> list[dict]:
    return tomllib.loads(BACKLOG.read_text()).get("sprint", [])


def stories() -> list[dict]:
    """Stories carry `suites` too, and were unchecked in both directions until `callback-trait`
    shipped naming a file the runner does not run — the same claim-without-evidence a sprint is
    held to, one level down. A story's `state` is spelled the same way, so the check is the same."""
    return tomllib.loads(BACKLOG.read_text()).get("story", [])


def named(entry: dict) -> list[str]:
    """The suites an entry names as files, as `area/test_x.py`.

    An entry naming a group rather than a file — "signatures/* (4 files)" — is prose about intent
    and cannot be held to anything, so neither direction checks it. Stories write the path from the
    dspy root and sprints write it from `tests/`, so the prefix is dropped rather than made a
    difference the gate reports.
    """
    return [suite.removeprefix("tests/") for suite in entry.get("suites", []) if "*" not in suite]


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


def claims_without_evidence(suites: set[str], manifest: set[str]) -> list[str]:
    """Sprints and stories reporting coverage of a file that does not run, or that upstream does
    not ship.

    A file dspy does not ship is wrong whatever the state — an unfinished plan cannot name it
    either. A file that merely does not *run* is only wrong once the work is claimed as shipped,
    since naming what will prove a story is what a plan is for.
    """
    found = []
    for entry in sprints() + stories():
        shipped = entry.get("state") in FINISHED
        for suite in named(entry):
            if suite not in manifest:
                found.append(f"{entry['id']} names {suite}, which dspy does not ship at this version")
            elif shipped and suite not in suites:
                found.append(f"{entry['id']} claims {suite}, which the runner does not run")
    return found


def evidence_without_claims(suites: set[str]) -> list[str]:
    """Sprints still pending while every suite they name already runs.

    A suite running is not by itself proof a sprint shipped — it can pass through dspy's own
    module rather than this crate's, which is the whole reason the crossing count exists. So the
    escape hatch is a written one: `still_pending` says what remains despite the green. A sprint
    with neither the `done` nor the sentence is the case this catches, where the state is simply
    older than the work.
    """
    found = []
    for sprint in sprints():
        if sprint.get("state") in FINISHED or "still_pending" in sprint:
            continue
        claimed = named(sprint)
        if claimed and all(suite in suites for suite in claimed):
            found.append(
                f"{sprint['id']} is {sprint.get('state', 'unstated')!r} while all {len(claimed)} "
                f"suite(s) it names run — mark it done, or say what is left in `still_pending`"
            )
    return found


def runs_without_manifest(suites: set[str], manifest: set[str]) -> list[str]:
    return [f"the runner runs {suite}, which is not in the manifest" for suite in sorted(suites)
            if suite not in manifest]


def sprints_naming_nothing() -> list[str]:
    """Sprints that never say what would prove them either way.

    An empty `suites` is an answer — s11 and s12 both carry one, with the reason written above it,
    and running the suites they first named would have exercised dspy's module rather than this
    crate's. A *missing* `suites` is the question never having been asked, and a sprint in that
    state is invisible to both directions above.
    """
    return [
        f"{sprint['id']} names no `suites` at all — list the files that prove it, or `suites = []` "
        f"with the reason there are none"
        for sprint in sprints()
        if "suites" not in sprint
    ]


def complaints() -> list[str]:
    suites = running()
    manifest = {
        line.removeprefix("tests/")
        for line in MANIFEST.read_text().splitlines()
        if not line.startswith("#")
    }
    found = [] if (stale := stale_manifest()) is None else [stale]
    return (found
            + claims_without_evidence(suites, manifest)
            + evidence_without_claims(suites)
            + runs_without_manifest(suites, manifest)
            + sprints_naming_nothing())


def main() -> None:
    found = complaints()
    for complaint in found:
        print(f"  {complaint}", file=sys.stderr)
    if found:
        raise SystemExit(1)
    # Neither direction can say anything about a sprint whose evidence is deliberately not a suite,
    # so the count states how far this gate reaches rather than letting a clean run imply it holds
    # the whole plan.
    holds = sum(1 for sprint in sprints() if named(sprint))
    print(f"  the plan and the suite agree on {len(running())} files "
          f"({holds} of {len(sprints())} sprints are held to one)", file=sys.stderr)


if __name__ == "__main__":
    main()
