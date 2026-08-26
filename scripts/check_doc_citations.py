#!/usr/bin/env python
"""Every file and function a doc comment names is one that exists.

`check_ledger_claims.py` does this for the ledger's reasons. Doc comments make the same kind of
claim — "held by `the_vendored_shim_is_upstreams_own`", "see `crates/dsrs-bridge/python/reflect.py`"
— and nothing read them. Six were wrong when this was written, every one written by me:

  - a test cited under a name it never had (`a_vendored_shim_matches_upstreams`)
  - a conformance test cited as a file that does not exist (`tests/openai_text_conformance.rs`)
  - three paths written `bridge/python/…` for files under `crates/dsrs-bridge/python/…`
  - an upstream test cited singular where it is plural (`..._input_...` for `..._inputs_...`)
  - a gepa function cited by a substring of its name

A citation is checked against *both* trees: this workspace, and the pinned Python this port follows.
Naming a dspy function or a gepa test is legitimate and common — the port is written in terms of
them — so the check is that the name exists *somewhere*, not that it exists here.

Run:

    python3 scripts/check_doc_citations.py
"""

from __future__ import annotations

import glob
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: A snake_case name of four or more words, in backticks. Shorter than that and it is prose — `to`,
#: `max_tokens`, `json_schema_extra` — which no rule could tell from a citation.
CITED_NAME = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+){3,})`")
#: A path ending in a source extension, backticked or bare.
#: `.json` as well: a doc comment naming a golden makes the same claim a reason does, and a
#: renamed fixture would leave the sentence pointing nowhere.
CITED_FILE = re.compile(
    r"`?((?:[A-Za-z_][A-Za-z0-9_-]*/)*[A-Za-z_][A-Za-z0-9_]*\.(?:rs|py|json))(?![A-Za-z0-9_])`?"
)
RUST_FN = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([a-z_][a-z0-9_]*)", re.M)
#: `async def` as well as `def`: dspy's streaming tests are coroutines, and leaving the keyword out
#: reported one of them missing while it sat in the file a `grep` found immediately.
PY_NAME = re.compile(
    r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)|^\s*([a-z_][a-z0-9_]*)\s*[:=]",
    re.M,
)

#: Trees this port is written in terms of, whose names a comment may cite. Read from the venv where
#: they are dependencies rather than vendored, so a missing venv weakens the check rather than
#: breaking it — reported below, not swallowed.
FOREIGN = ["third_party/dspy", ".venv/lib/*/site-packages/gepa", ".venv/lib/*/site-packages/litellm",
           ".venv/lib/*/site-packages/json_repair", ".venv/lib/*/site-packages/dspy"]

#: Named, not inferred: these belong to CPython, numpy and litellm's own vendored sources, which no
#: tree here holds. A checker that guessed at this would either miss real errors or cry wolf, and
#: the list is short enough to read.
EXTERNAL = {
    # CPython's json encoder, whose C accelerator this crate reproduces.
    "py_encode_basestring_ascii",
    # litellm's own helper, cited where the ollama wire differs from it.
    "convert_content_list_to_str",
    # A conformance *case* name, not a function — the golden's key.
    "chain_of_thought_n3",
}
EXTERNAL_FILES = {
    # Written by a fuzz campaign into `target/` and deliberately not committed: twenty thousand
    # random strings are evidence, not a golden.
    "target/parse_fuzz.json",
    "target/json_repair_fuzz.json",
    # CPython's and numpy's own suites, cited as the source of a vector table.
    "Lib/test/test_random.py",
    "numpy/random/tests/test_randomstate.py",
    "test_direct.py",
}


def ours() -> tuple[set[str], set[str]]:
    """Every file in this workspace, and every function it defines."""
    files: set[str] = set()
    names: set[str] = set()
    for pattern in (
        "crates/**/*.rs",
        "crates/**/*.py",
        "crates/**/*.json",
        "scripts/**/*.py",
        "examples/**/*.rs",
    ):
        for path in ROOT.glob(pattern):
            files.add(str(path.relative_to(ROOT)))
            if path.suffix == ".rs":
                names |= set(RUST_FN.findall(path.read_text(errors="ignore")))
    return files, names


def theirs() -> tuple[set[str], set[str], list[str]]:
    """The same for the pinned Python, and whichever roots could not be read."""
    files: set[str] = set()
    names: set[str] = set()
    unread: list[str] = []
    for pattern in FOREIGN:
        found = [pathlib.Path(hit) for hit in glob.glob(str(ROOT / pattern))]
        if not found:
            unread.append(pattern)
            continue
        for base in found:
            for path in base.rglob("*.py"):
                files.add(str(path))
                for first, second in PY_NAME.findall(path.read_text(errors="ignore")):
                    names.add(first or second)
    return files, names, unread


def main() -> int:
    our_files, our_names = ours()
    their_files, their_names, unread = theirs()
    known_files = our_files | their_files
    known_names = our_names | their_names | EXTERNAL

    missing: list[tuple[str, str, str]] = []
    checked = 0
    for path in sorted(ROOT.glob("crates/**/*.rs")):
        where = path.relative_to(ROOT)
        for at, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            comment = line.strip()
            if not (comment.startswith("///") or comment.startswith("//!")):
                continue
            for named in set(CITED_FILE.findall(comment)):
                if named in EXTERNAL_FILES:
                    continue
                checked += 1
                if not any(k.endswith("/" + named) or k == named for k in known_files):
                    missing.append((f"{where}:{at}", "file", named))
            for named in set(CITED_NAME.findall(comment)):
                checked += 1
                if named not in known_names:
                    missing.append((f"{where}:{at}", "name", named))

    for pattern in unread:
        print(f"  note: {pattern} was not read, so a name cited from it is not checked")
    print(f"doc citations: {checked} reference(s) resolved against both trees")
    if missing:
        print(f"\nDoc-citation gate FAILED: {len(missing)} that do not exist:")
        for where, kind, named in missing:
            print(f"    {where}\n      -> {kind} {named}")
        return 1
    print("\nDoc-citation gate: OK (every file and function a doc comment names exists)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
