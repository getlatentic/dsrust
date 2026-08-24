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

#: Reasons asserting a capability while naming nothing checkable. Lower it as they gain a name.
#: The four left describe *dspy's* behaviour or an absence here — "nothing pickles" names nothing
#: because there is nothing to name — so this is a floor rather than a target of zero.
UNVERIFIED = 4

#: A reason that points at something: "reproduced as X", "handled by Y".
SUBSTITUTION = re.compile(
    r"\b(reproduced (via|as|by)|handled by|covered by|answered by|done by"
    r"|lives (in|on)|reached (through|by)|provided by|supplied by)\b",
    re.I,
)
#: A reason that claims the crate *does* something.
CAPABILITY = re.compile(
    r"\b(emits?|fires?|records?|reports?|tracks?|logs?|watches|observes?"
    r"|validates?|enforces?|retries|caches?|branches on)\b",
    re.I,
)
RS_FILE = re.compile(r"\b((?:[a-z_][a-z0-9_]*/)*[a-z_][a-z0-9_]*\.rs)\b")
IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)`")
#: A qualified path *anywhere* in a reason or a `rust` field, backticks or not, and on **every**
#: status rather than only on divergences. `LM::with_callbacks` was named eight times in plain text
#: and had been renamed; `LM::with_retry` twice more, in two `mapped` entries the check skipped
#: entirely — 258 entries name a qualified path and are not divergences.
BARE_PATH = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)+)\b")
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


def tree() -> tuple[set[str], set[str]]:
    """Every name defined anywhere in the crates, and every `.rs` path.

    Definitions are collected regardless of visibility: a reason may legitimately point at a private
    helper, and `numbered_block` — the first substitution in the file — is one. A `pub`-only walk
    reported it missing, which is how this function came to be written the way it is.
    """
    names, files = set(EXTERNAL), set()
    for path in (ROOT / "crates").rglob("*.rs"):
        text = path.read_text(errors="ignore")
        names |= set(DEFINITION.findall(text)) | set(VARIANT.findall(text)) | set(FIELD.findall(text))
        files.add(str(path.relative_to(ROOT)))
    return names, files


def main() -> int:
    ledger = tomllib.loads(LEDGER.read_text())
    names, files = tree()

    missing: list[tuple[str, str, str]] = []
    unverified: list[tuple[str, str]] = []
    substitutions = 0
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
                    if not any(f.endswith("/" + named) or f == named for f in files):
                        missing.append((key, "file", named))
                for ident in set(IDENT.findall(reason)):
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
            for ident in set(BARE_PATH.findall(reason + " " + str(entry.get("rust") or ""))):
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
            claim = CAPABILITY.search(reason) if divergence else None
            if claim and not entry.get("rust"):
                named_something = bool(RS_FILE.search(reason)) or any(
                    i[0].isupper() or "::" in i or "_" in i for i in IDENT.findall(reason)
                )
                if not named_something:
                    unverified.append((key, claim.group(0)))

    print(f"ledger claims: {substitutions} substitutions checked against the tree")
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
    print("\nLedger-claims gate: OK (every substitution points at something that exists)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
