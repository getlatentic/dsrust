#!/usr/bin/env python3
"""Hold the ledger's *prose* to the tree, which nothing did until a claim was wrong twice.

`check_api_surface.py` verifies that a `mapped` entry's `rust` names something real. It says nothing
about the reasons, and the reasons are where the substitutions live: a `divergence` justifies an
absence by pointing at what stands in for it, and that pointer is a claim about this crate.

Two of them have been false. `callbacks` said dspy's six callback points "emit tracing spans here"
when there was not one span in the tree. `GEPA.log_dir` and `MIPROv2.log_dir` said "tracing spans are
the record here" when neither optimizer emits one — the only points are in `evaluate.rs`, which a
GEPA run does not go through. Both read as resolved for as long as nobody looked.

So this checks what can be checked mechanically:

  **Hard failure** — a reason that names a `.rs` file or a Rust identifier which does not exist.
  That is the cheap half and it should stay at zero.

  **A ratchet** — a reason that asserts a *capability* (emits, records, validates, retries…) while
  naming nothing a grep can find. Those cannot be verified here; they can only be read. The number
  is the reading list, and it may fall and never rise.

    ./scripts/check_ledger_claims.py
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "scripts" / "api_ledger.toml"

#: Capability claims that name nothing a checker can look at — "this crate caches", with no
#: identifier, file or path to go and read.
#:
#: **Zero, and it can stay zero.** The four that were here were all groundable: each asserted
#: something true about this crate and simply did not say where to look. Naming
#: `LmRequest::cache_key`, `LmHistoryEntry` and `DiskCache` turned three of them into claims the
#: qualified-path check verifies on every run, and the fourth into one that names the method a
#: response is returned from. A new entry that cannot name anything is either vague or wrong;
#: raise this only with a note saying which.
UNVERIFIED = 0

#: A reason that points at something: "reproduced as X", "handled by Y".
SUBSTITUTION = re.compile(
    # `reproduced in` was missing while `reproduced via|as|by` were there, so
    # "reproduced in lm/api" went unchecked — one of a run of very short reasons that name a
    # substitution and nothing else, which is exactly the shape most worth checking.
    r"\b(reproduced (via|as|by|in)|handled by|covered by|answered by|done by"
    r"|lives (in|on)|reached (through|by)|provided by|supplied by|folded into)\b",
    re.I,
)
#: A reason that claims the crate *does* something.
CAPABILITY = re.compile(
    r"\b(emits?|fires?|records?|reports?|tracks?|logs?|watches|observes?"
    r"|validates?|enforces?|retries|caches?|branches on)\b",
    re.I,
)
RS_FILE = re.compile(r"\b((?:[a-z_][a-z0-9_]*/)*[a-z_][a-z0-9_]*\.py)\b")
#: The mirror of `RS_FILE`, for the other tree. A reason naming `predict/predict.py` is making a
#: claim about where upstream keeps something, and that claim goes stale the day the pin moves and
#: a file is renamed — which is the one thing nothing here re-derives. Paths this repo owns are
#: excluded by prefix, since a reason may cite `scripts/generate_fixtures.py` as ours.
PY_FILE = re.compile(r"\b((?:[a-z_][a-z0-9_]*/)*[a-z_][a-z0-9_]*\.py)\b")
OURS_PY = ("scripts/", "crates/")
RS_FILE = re.compile(r"\b((?:[a-z_][a-z0-9_]*/)*[a-z_][a-z0-9_]*\.rs)\b")
#: `chat.rs::chat_user` — a file and an item *in* it. No other rule sees this spelling: `BARE_PATH`
#: matches from the `rs`, finds no such crate, and skips the whole path as foreign. So a reason
#: naming a renamed private helper this way went unchecked, which is how 21 of them were written.
#: The file must exist and must be where the item is defined, which is the claim being made.
RS_PATH = re.compile(r"\b((?:[a-z_][a-z0-9_]*/)*[a-z_][a-z0-9_]*\.rs)::([A-Za-z_][A-Za-z0-9_]*)\b")
IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)`")
#: A qualified path *anywhere* in a reason or a `rust` field, backticks or not, and on **every**
#: status rather than only on divergences. `LM::with_callbacks` was named eight times in plain text
#: and had been renamed; `LM::with_retry` twice more, in two `mapped` entries the check skipped
#: entirely — 258 entries name a qualified path and are not divergences.
BARE_PATH = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)\b")
#: A `with_`-prefixed name, which is a builder setter in this workspace and almost never a dspy one
#: — upstream has only `with_inputs`, `with_instructions` and `with_updated_fields`. Checked against
#: *both* trees, so citing a Python name stays legal while a renamed Rust one does not: `with_lm`
#: was named in three entries and `with_extract_lm` in a fourth, none of them qualified, so no
#: `::` rule could see them.
WITH_NAME = re.compile(r"\b(with_[a-z][a-z0-9_]*)\b")
DEFINITION = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r"(?:fn|struct|enum|const|static|type|trait|mod|union)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.M,
)
#: Enum variants, which a reason names as often as a type — `LmPart::Text`.
VARIANT = re.compile(r"^\s{4}([A-Z][A-Za-z0-9_]*)\s*[({,]", re.M)
#: Public struct fields. A reason names `Parallel::max_in_flight` as readily as a method, and a
#: walk that collected only definitions reported three such fields missing — the third time in one
#: pass that this checker's own gaps read as ledger errors.
FIELD = re.compile(r"^\s+pub(?:\([^)]*\))?\s+([a-z_][a-z0-9_]*)\s*:", re.M)
#: Names that legitimately come from outside these crates, so their absence proves nothing.
#: The crates this repo owns, so a path rooted in one is ours to have.
OURS = {"dsrust", "gepa", "pyrng", "tpe", "json_repair", "dsrust_gepa", "dsrust_tpe"}
EXTERNAL = {
    "Serialize", "Deserialize", "Clone", "Debug", "Default", "Display", "PartialEq", "Eq",
    "Hash", "Iterator", "From", "Into", "Option", "Result", "Vec", "String", "Arc", "Box",
    "Send", "Sync", "JsonSchema", "Value", "Map", "Duration", "Instant", "Path", "PathBuf",
}


def tree() -> tuple[set[str], set[str], dict[str, set[str]]]:
    """Every name defined anywhere in the crates, every `.rs` path, and each file's own names.

    Definitions are collected regardless of visibility: a reason may legitimately point at a private
    helper, and `numbered_block` — the first substitution in the file — is one. A `pub`-only walk
    reported it missing, which is how this function came to be written the way it is.
    """
    names, files, by_file = set(EXTERNAL), set(), {}
    for path in (ROOT / "crates").rglob("*.rs"):
        text = path.read_text(errors="ignore")
        own = set(DEFINITION.findall(text)) | set(VARIANT.findall(text)) | set(FIELD.findall(text))
        names |= own
        rel = str(path.relative_to(ROOT))
        files.add(rel)
        by_file[rel] = own
    return names, files, by_file


def impl_target(tail: str) -> str | None:
    """The type an `impl` line is *for*, past its generics.

    A regex with `<[^>]*>` cannot skip the parameter list of
    `impl<E: Iterator<Item = LmStreamEvent>> LmStream<E>` — it stops at the inner `>` and matches
    nothing, so `LmStream` looked like a type with no impl and its methods read as missing. Angle
    brackets nest; counting them is the only thing that reads them.
    """
    if " for " in tail:
        tail = tail.split(" for ", 1)[1]
    depth, name = 0, []
    for character in tail:
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
        elif depth == 0:
            if character.isalnum() or character == "_":
                name.append(character)
            elif name:
                break
    found = "".join(name)
    return found or None


def members_by_type() -> dict[str, set[str]]:
    """The members of every `struct` and `enum` that has no `impl` block — and only those.

    A flat name set cannot tell `Evaluation::results` from `Outcome::results`. Five reasons named
    an `Outcome` this crate does not have, two of them as `Outcome::results`, and the check passed
    because `refine.rs` defines an unrelated `Outcome` and something somewhere has a `results`
    field. The gate's own comment says a *bare* name resolves against any type that has one; a
    qualified path was doing the same, which is worse — writing `Type::member` is how you say
    *which* type.

    Restricted to impl-less data types because that is the whole of what a regex can be sure of: a
    `struct` or `enum` declaration lists its members and there is nowhere else for one to come
    from. Anything with an `impl` can also carry associated consts, associated types, and trait
    methods it never overrides, and asking about those produced fourteen false positives —
    `Predict::dump_state` is defaulted on `Module` and appears in no block belonging to `Predict`.
    A gate that cries wolf gets worked around, so this one only speaks where it knows.

    What it therefore does **not** catch: a wrong member on a type that has an `impl`.
    `Evaluation::rows` passes, because `Evaluation` has impls and `rows` is a field on something
    else. Closing that needs the trait methods each type inherits, which means resolving
    `impl Trait for Type` against the trait's own defaults — worth doing, not done here.
    """
    owners: dict[str, set[str]] = {}
    #: A name is only usable here if the workspace declares it exactly once, as a data type with no
    #: `impl`. `Module` is a trait in `dsrust` and an unrelated `enum` in `dsrust-derive`, and
    #: `Outcome` is one enum here and a different idea in five ledger reasons — a second declaration
    #: anywhere means the members of one of them are not the members of the other.
    implemented: set[str] = set()
    times_declared: dict[str, int] = {}
    declared = re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M
    )
    trait_or_type = re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M
    )
    implements = re.compile(r"^\s*impl\b(.*)$")
    member = re.compile(
        r"^\s+(?:pub(?:\([^)]*\))?\s+)?([a-z_][a-z0-9_]*)\s*:|^\s+([A-Z][A-Za-z0-9_]*)\s*[({,]"
    )
    for path in (ROOT / "crates").rglob("*.rs"):
        lines = path.read_text(errors="ignore").splitlines()
        for line in lines:
            found = implements.match(line)
            if found:
                target = impl_target(found.group(1))
                if target:
                    implemented.add(target)
            found = trait_or_type.match(line)
            if found:
                name = found.group(1)
                times_declared[name] = times_declared.get(name, 0) + 1
                if line.lstrip().startswith(("trait", "pub trait")):
                    implemented.add(name)
        for start, line in enumerate(lines):
            found = declared.match(line)
            if not found:
                continue
            name = found.group(1)
            depth, at, entered = 0, start, False
            # The block this opener owns, by brace balance. `entered` matters: an
            # `impl<M> Type<M>\nwhere\n    …\n{` opens its brace three lines below the name, and
            # breaking on depth alone cut every such block to nothing — which reported
            # `MIPROv2::score` missing while it sat in one.
            while at < len(lines):
                depth += lines[at].count("{") - lines[at].count("}")
                entered = entered or depth > 0
                if entered and depth <= 0:
                    break
                at += 1
                if at < len(lines):
                    hit = member.match(lines[at])
                    if hit:
                        owners.setdefault(name, set()).add(next(g for g in hit.groups() if g))
    return {
        name: members
        for name, members in owners.items()
        if name not in implemented and times_declared.get(name, 0) == 1
    }


def upstream_files() -> set[str]:
    """Every `.py` path under the pinned dspy, relative to its package root."""
    root = ROOT / "third_party" / "dspy" / "dspy"
    if not root.is_dir():
        return set()
    return {str(path.relative_to(root)) for path in root.rglob("*.py")}


def package_names() -> tuple[set[str], list[str]]:
    """Every `def`/`class` name in the pinned Python packages, and the ones that could not be read.

    dspy is vendored, so it is always there. `gepa` is a dependency in the venv, and a reason may
    cite it as readily — eight do. Missing it does not fail the run, because this gate is also run
    where the venv is not built; it is *reported*, since the alternative is a check that quietly
    knows less than its summary line claims.
    """
    found: set[str] = set()
    unread: list[str] = []
    for pinned in (ROOT / "third_party" / "dspy" / "dspy", *sorted((ROOT / ".venv" / "lib").glob("*/site-packages/gepa"))):
        found |= names_in(pinned) if pinned.is_dir() else set()
        if not pinned.is_dir():
            unread.append(str(pinned.relative_to(ROOT)))
    if not any((ROOT / ".venv" / "lib").glob("*/site-packages/gepa")):
        unread.append(".venv/…/site-packages/gepa")
    return found, unread


def names_in(pinned: pathlib.Path) -> set[str]:
    """Every `def`/`class` name under one package directory."""
    found: set[str] = set()
    if not pinned.is_dir():
        return found
    for path in pinned.rglob("*.py"):
        found |= set(re.findall(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", path.read_text(errors="ignore"), re.M))
    return found


def main() -> int:
    ledger = tomllib.loads(LEDGER.read_text())
    names, files, by_file = tree()
    theirs, unread = package_names()
    their_files = upstream_files()
    owners = members_by_type()

    missing: list[tuple[str, str, str]] = []
    unverified: list[tuple[str, str]] = []
    substitutions = 0
    checked = 0
    for table in ledger.values():
        if not isinstance(table, dict):
            continue
        for key, entry in table.items():
            if not isinstance(entry, dict):
                continue
            reason = str(entry.get("reason", ""))
            divergence = entry.get("status") == "divergence"
            if divergence and SUBSTITUTION.search(reason):
                substitutions += 1
                # A substitution points at Rust, so a bare name in one is checkable — and often is
                # a private helper: `numbered_block` is the first in the file.
                for named in set(RS_FILE.findall(reason)):
                    checked += 1
                    if not any(f.endswith("/" + named) or f == named for f in files):
                        missing.append((key, "file", named))
                for ident in set(IDENT.findall(reason)):
                    checked += 1
                    parts = ident.split("::")
                    if not (parts[0][0].isupper() or len(parts) > 1):
                        continue
                    if not any(part in names for part in parts):
                        missing.append((key, "identifier", ident))

            # A **qualified** path is unambiguously Rust wherever it appears, so it is checked on
            # every reason rather than only on substitutions. That is what a bare-name rule misses:
            # `LM::with_callbacks` was named eight times, had been renamed to
            # `LmBuilder::callbacks`, and every one of those reasons said "there is no" — which no
            # substitution-only check ever looks at. A reason may still name a Python symbol or an
            # environment variable in backticks, and those carry no `::`.
            for named in set(PY_FILE.findall(reason)):
                if named.startswith(OURS_PY) or not their_files:
                    continue
                checked += 1
                if named not in their_files and not any(
                    f.endswith("/" + named) for f in their_files
                ):
                    missing.append((key, "dspy file", named))
            for path_named, item in set(RS_PATH.findall(reason + " " + str(entry.get("rust") or ""))):
                checked += 1
                # Every file the suffix names, not the first — two files here are called `demos.rs`,
                # and picking one of them reported a reason wrong that named the other.
                owning_files = [f for f in files if f.endswith("/" + path_named) or f == path_named]
                if not owning_files:
                    missing.append((key, "file", path_named))
                elif not any(item in by_file[f] for f in owning_files):
                    missing.append((key, "item in file", f"{path_named}::{item}"))
            for named in set(WITH_NAME.findall(reason + " " + str(entry.get("rust") or ""))):
                checked += 1
                if named not in names and named not in theirs:
                    missing.append((key, "with_ name", named))
            for ident in set(BARE_PATH.findall(reason + " " + str(entry.get("rust") or ""))):
                checked += 1
                parts = ident.split("::")
                if parts[0] in OURS:
                    parts = parts[1:]          # a crate-rooted path; the crate itself is not an item
                elif parts[0][:1].islower() and parts[0] not in names:
                    continue                   # `anyhow::Error`, `reqwest::Client` — not ours to have
                # The item, and its owning type when the path names one. Intermediate modules are
                # not checked: `mod` may be private and re-exported, which says nothing about the
                # item at the end.
                wanted = [parts[-1]]
                if len(parts) >= 2 and parts[-2][:1].isupper():
                    wanted.append(parts[-2])
                for part in wanted:
                    if part not in names:
                        missing.append((key, "qualified path", ident))
                        break
                else:
                    # And the member must belong to *that* type, not merely exist somewhere.
                    owner = parts[-2] if len(parts) >= 2 and parts[-2][:1].isupper() else None
                    if owner in owners and parts[-1] not in owners[owner]:
                        missing.append((key, "member of", ident))
            claim = CAPABILITY.search(reason) if divergence else None
            if claim and not entry.get("rust"):
                named_something = bool(RS_FILE.search(reason)) or any(
                    i[0].isupper() or "::" in i or "_" in i for i in IDENT.findall(reason)
                )
                if not named_something:
                    unverified.append((key, claim.group(0)))

    for missed in unread:
        print(f"  note: {missed} was not read, so a name cited from it is not checked")
    print(
        f"ledger claims: {checked} reference(s) resolved against the tree, "
        f"{substitutions} of them in a row phrased as a substitution"
    )
    if missing:
        print(f"\nLedger-claims gate FAILED: {len(missing)} reference(s) that do not exist:")
        for key, kind, named in missing:
            print(f"    {key}\n      -> {kind} {named}")
        return 1

    print(f"  capability claims naming nothing checkable: {len(unverified)} (floor {UNVERIFIED})")
    for key, verb in unverified:
        print(f"      ? {key}  (\"{verb}\")")
    if len(unverified) > UNVERIFIED:
        print(f"\nLedger-claims gate FAILED: {len(unverified)} unverified, floor {UNVERIFIED}")
        return 1
    if len(unverified) < UNVERIFIED:
        print(f"\n{UNVERIFIED - len(unverified)} below the floor — lower UNVERIFIED to {len(unverified)}")
    print("\nLedger-claims gate: OK (every name a reason points at exists, and a `file.rs::item` is in that file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
