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

Qualified paths — `JsonType::reflected`, `Watch::shown` — are checked the way the ledger's reasons
are: the member must belong to *that* type. rustdoc checks an intra-doc link, but only for items it
renders, so a `pub(crate)` type's links are unread — which is where `Parzen::pdf` sat, naming a
method called `log_pdf`. Two more were wrong: `FieldKind::reflected_json` (the derive reaches
`JsonType`) and a `with_config` builder whose name is `config`.

A path is skipped where no index can speak for it: `Self::x` needs the enclosing impl, a primitive's
methods are std's, `default` is derived, and a handful of names are cited precisely *because* they
do not exist — `FieldKind::Citations` is written to say this crate has no such variant.

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
#: Definitions only, for *our* Python. `PY_NAME` below also matches an assignment, which is right
#: for the pinned trees — a keyword argument is a name a reason may cite — and wrong here, where it
#: would put every local in the generators into the set a doc comment resolves against.
PY_DEF = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
#: Any identifier-shaped token, for the pinned trees only — see `theirs`.
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: `async def` as well as `def`: dspy's streaming tests are coroutines, and leaving the keyword out
#: reported one of them missing while it sat in the file a `grep` found immediately.
#: A qualified path in a doc comment — `JsonType::reflected`, `Watch::shown`. The ledger has had
#: this rule since two of its own reasons named a member of the wrong type; doc comments make the
#: same claim in the same spelling and nothing read them. 603 are cited, and it found
#: `FieldKind::reflected_json`, a method that does not exist (the derive reaches `JsonType`), and
#: `with_config`, a builder whose name is `config`.
QUALIFIED = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)\b")
#: A member name no index can speak for: `default` is derived, and asking whether a type "has" it
#: is a question about a `#[derive]` rather than about a block.
DERIVED = {"default", "clone", "fmt", "from", "into", "eq", "hash", "serialize", "deserialize"}
#: Paths that are correct *because* the name does not exist. `FieldKind::Citations` is cited to say
#: this crate has no such variant, which is the sentence's point.
ABSENT_ON_PURPOSE = {
    # Cited to say this crate has no such variant, which is the sentence's point.
    "FieldKind::Citations",
    # A stand-in name in the derive's own prose: `predict` is generated on whatever type the
    # macro is applied to, and `GiftTask` is what the example calls it.
    "GiftTask::predict",
}
#: Types from other crates named without their crate — `tracing::Span`, `tokio::runtime::Handle`.
FOREIGN_TYPES = {"Span", "Handle"}
#: Rust's own scalar types, whose inherent methods are std's.
PRIMITIVES = {"char", "str", "bool", "u8", "u32", "u64", "usize", "i32", "i64", "f32", "f64"}
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
    # deno's `npm:` resolution rule, cited where the sandbox runner is staged in temp: deno walks up
    # from the script for this file and switches to node_modules resolution on finding one, which is
    # what would lose pyodide. It only ever exists in somebody else's directory.
    "package.json",
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
            elif path.suffix == ".py":
                # The generators' own names. A doc comment citing `generate_fixtures` or a
                # bridge shim's function is naming this workspace, and only the `.rs` half of it
                # was indexed — so such a citation resolved against the pinned Python or not at
                # all.
                names |= set(PY_DEF.findall(path.read_text(errors="ignore")))
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
                text = path.read_text(errors="ignore")
                files.add(str(path))
                for first, second in PY_NAME.findall(text):
                    names.add(first or second)
                # And every identifier-shaped token, for the pinned trees only.
                # `list_of_named_predictors` is `self.list_of_named_predictors = ...` and
                # `ignored_args_for_cache_key` a keyword argument, so neither is a definition nor
                # at the start of a line — but both are names this port is written in terms of.
                # They had been resolving against a *generator's* local of the same name, which is
                # the right answer for the wrong reason.
                names |= set(TOKEN.findall(text))
    return files, names, unread


def qualified_paths(
    known: set[str], ours_only: set[str], owners: dict, everything: dict
) -> tuple[list[tuple[str, str, str]], int]:
    """Every `Type::member` a doc comment names, resolved as the ledger's reasons are.

    Answers the ones that do not resolve *and* how many were looked at — a summary line that
    counted only the failures would report the check shrinking as it found less.
    """
    import check_ledger_claims as ledger

    missing: list[tuple[str, str, str]] = []
    looked_at = 0
    for path in sorted(ROOT.glob("crates/**/*.rs")):
        where = path.relative_to(ROOT)
        for at, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            comment = line.strip()
            if not (comment.startswith("///") or comment.startswith("//!")):
                continue
            for ident in sorted(set(QUALIFIED.findall(comment))):
                if ident in ABSENT_ON_PURPOSE:
                    continue
                parts = ident.split("::")
                # `Self::x` names the enclosing impl, which needs the impl and not a name index;
                # a primitive's methods belong to std.
                if parts[0] == "Self" or parts[0] in PRIMITIVES:
                    continue
                if parts[0] in FOREIGN_TYPES:
                    continue
                if parts[0] in ledger.FOREIGN_ROOTS or parts[0] in ledger.EXTERNAL:
                    continue
                if parts[0] in ledger.OURS:
                    parts = parts[1:]
                elif parts[0][:1].islower() and parts[0] not in ours_only:
                    # Against *this workspace's* names, not the union: `py::TestRandomDist` roots
                    # at a namespace meaning "the Python side", and `py` is a token in the pinned
                    # trees, so the wider set makes every such path look like a claim about us.
                    continue
                if not parts or parts[-1] in DERIVED:
                    continue
                looked_at += 1
                owner = parts[-2] if len(parts) >= 2 and parts[-2][:1].isupper() else None
                if owner in ledger.EXTERNAL:
                    continue
                wanted = [parts[-1]] + ([owner] if owner else [])
                if any(part not in known for part in wanted):
                    missing.append((f"{where}:{at}", "qualified path", ident))
                elif owner in owners and parts[-1] not in owners[owner]:
                    missing.append((f"{where}:{at}", "member of", ident))
                elif owner in everything and parts[-1] not in everything[owner]:
                    missing.append((f"{where}:{at}", "member of", ident))
    return missing, looked_at


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

    import check_ledger_claims as ledger

    tree_names, _tree_files, _by_file = ledger.tree()
    known = tree_names | their_names | ledger.EXTERNAL
    paths, looked_at = qualified_paths(
        known, tree_names, ledger.members_by_type(), ledger.everything_by_type()
    )
    missing += paths
    checked += looked_at

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
