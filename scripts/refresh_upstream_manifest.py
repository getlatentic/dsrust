"""Record every test file upstream ships at the pinned version.

The suite this crate runs is an allowlist, so a green run means the files named in it pass. It
says nothing about the ones that are not. Without the whole list beside it, green reads as done.

The manifest is committed so the gap is visible in the repository rather than only to whoever
last looked at GitHub, and `run_upstream_tests.sh` reports coverage against it on every run.
Refresh it when `scripts/DSPY_VERSION` changes:

    python3 scripts/refresh_upstream_manifest.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).parent.parent
VERSION = (ROOT / "scripts" / "DSPY_VERSION").read_text().strip()
OUT = ROOT / "scripts" / "upstream_tests.txt"
TREE = f"https://api.github.com/repos/stanfordnlp/dspy/git/trees/{VERSION}?recursive=1"


def test_files() -> list[str]:
    with urllib.request.urlopen(TREE, timeout=60) as response:
        tree = json.load(response)["tree"]
    paths = [
        entry["path"]
        for entry in tree
        if entry["path"].startswith("tests/")
        and pathlib.PurePath(entry["path"]).name.startswith("test_")
        and entry["path"].endswith(".py")
    ]
    if not paths:
        raise SystemExit("the tree listing named no test files, which cannot be right")
    return sorted(paths)


def main() -> None:
    paths = test_files()
    OUT.write_text(f"# dspy {VERSION}: every test file upstream ships.\n" + "\n".join(paths) + "\n")
    print(f"  wrote {OUT.relative_to(ROOT)} ({len(paths)} files)", file=sys.stderr)


if __name__ == "__main__":
    main()
