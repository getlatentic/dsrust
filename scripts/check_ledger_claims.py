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
#: Both classification tables. `api_ledger.toml` justifies a symbol inside a ported module;
#: `unported_modules.toml` justifies a whole module being outside them. The reasons in the second
#: make the same kind of claim as the reasons in the first — `COPRO.__init__` dropped `verbose`,
#: `named_parameters` sets `a.b[0].c` — so they earn the same rules rather than a prose exemption.
LEDGERS = [
    ROOT / "scripts" / "api_ledger.toml",
    ROOT / "scripts" / "unported_modules.toml",
]

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

#: Rows every rule walks past: a `divergence` or `deferred` reason in which no `.rs` file, no
#: upstream `.py`, no golden, no name, no type and no qualified path appears — so nothing in it is
#: resolved against either tree, and the only thing holding it up is that somebody read it once.
#:
#: **Zero, and it can stay zero.** Every dspy-facing row names something now, and the walk down was
#: not a rewording exercise: naming what a sentence was about found `_get_max_tokens_key`, a
#: function dspy has never had, and the divergence behind it changed five model names' request
#: bodies. It also found `adapter/stream.rs` cited four times for a file that is
#: `adapter/stream/mod.rs`, a `JSONAdapter.format_finetune_data` described as formatting something
#: when upstream raises `NotImplementedError`, and a `requires_permission_to_run` described as
#: blocking on `input()` three releases after it stopped.
#:
#: This is the metric the whole ledger audit was run against, and for most of that run it lived
#: nowhere but in a command typed at a shell. It read 24 with a looser test than the checker
#: actually applies — a backticked name counted as reached whether or not any rule looked at it —
#: where the checker's own walk read 130. A number that is remembered rather than run is the second
#: one in this repo to have been wrong.
UNREACHED = 0

#: The same question for the other half of the ledger: a `[rust_only]` row justifying an invented
#: Rust item. Its *key* is gated — `check_rust_surface.py` fails if the item is not there — and so
#: is whether anything calls it. What nothing read is the reason, and these reasons are where the
#: port's claims about the other side live: that numpy normalises a cumulative sum by its last
#: entry, that CPython's `&` iterates the smaller operand, that dspy reads litellm's registry per
#: call. Each is a fact the Rust item was built from, and a wrong one is a wrong item.
#:
#: **Zero.** Naming what each row was about found two claims with nothing behind them: `system_of`
#: said dspy reads `messages[0]` inline for a system prompt, and nothing in the pinned tree
#: separates a system message at all; and a `DummyLM` mode described as taking an answers dict by
#: some other shape than the dict-of-dicts upstream matches against the final message.
INVENTED = 0

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
#: A golden named in a reason. Ten reasons cite one — "held by evaluate/max_errors.json" — and
#: neither this checker nor the doc one resolved a `.json`, so a renamed fixture would have left the
#: sentence pointing nowhere. All ten resolved when the rule was added; the rule is what keeps that
#: true.
GOLDEN = re.compile(r"\b((?:[a-z_][a-z0-9_]*/)*[a-z_][a-z0-9_]*\.json)\b")
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
#: A backticked snake_case name — `use_instruct_history`, `dump_state`, `__init__`. Half the ledger
#: is written about Python, and until this rule existed *none* of that half was checked: the Rust
#: side of every reason resolved against the tree while a sentence could say dspy's
#: `_get_max_tokens_key` decides the token key and nothing would notice that no such function has
#: ever existed. It was the first thing this rule caught, and the real one behind it was a
#: divergence in five model names' request bodies.
#:
#: Checked against both trees, as `WITH_NAME` is: a name is legitimate if this workspace defines it
#: *or* it appears in the pinned Python. The pinned side is every word token rather than every
#: `def` — `use_instruct_history` is a keyword argument and `reasoning_effort` a dict key, and a
#: definitions-only index would call both of them missing. That makes it a weak check for what a
#: name *is* and an exact one for whether it is there at all, which is the claim being made and the
#: one a pin bump falsifies.
PY_IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*)`")
#: A backticked type name, on **every** row rather than only on substitutions. 774 reasons name
#: one and the substitution-only rule looked at a fraction of them, against the Rust tree alone —
#: so `NotImplementedError`, `Protocol` and `FieldInfo` were unchecked for being Python, and a
#: renamed Rust type was unchecked for sitting in a sentence that did not say "reproduced as".
TYPE_NAME = re.compile(r"`([A-Z][A-Za-z0-9_]*)`")
#: A dotted Python path — `inspect.getfile`, `gepa.optimize`, `dspy.load`. The last segment is the
#: item, and it is what a pin bump renames.
DOTTED = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")
#: A backticked *expression* rather than a bare name, and the names inside it. `LM(cache=...)`,
#: `dependency_versions: {dspy: DSPY_VERSION}`, `lm.copy(**kwargs)` — 173 reasons name a dspy call
#: form this way and every rule above walked past all of them, because each anchors its backticks
#: to the whole name. A parameter named inside a call form is exactly as much of a claim as one
#: written alone.
#:
#: Spans holding a path are left to the file and qualified-path rules, which read them properly; a
#: long one is prose in backticks rather than an expression.
SPAN = re.compile(r"`([^`]+)`")
SPAN_TOKEN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*|[A-Z][A-Za-z0-9_]*)\b")
DEFINITION = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r"(?:fn|struct|enum|const|static|type|trait|mod|union)\s+([A-Za-z_][A-Za-z0-9_]*)"
    # `macro_rules! name` — a definition no other arm matches, and `asks_with_a_prediction` is
    # one of three the workspace exports.
    r"|^\s*macro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)",
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
#: Roots that are never ours however the workspace spells its own items. The lowercase-and-unknown
#: rule below is not enough on its own: a field named `std` — dspy's own name for a standard
#: deviation — put `std` in the tree's names and turned every `std::thread` in a reason into a
#: claim about this crate.
FOREIGN_ROOTS = {
    "std", "core", "alloc", "tokio", "serde", "serde_json", "anyhow", "reqwest", "futures_util",
    "tracing", "schemars", "regex", "chrono", "base64", "rand", "url", "uuid", "pyo3",
    "image", "sha2", "futures_channel",
}
#: Names in no tree here, which is not an error for any of them. Listed rather than inferred: a
#: rule that guessed would either miss a real absence or cry wolf, and this is short enough to read.
CITED_ELSEWHERE = {
    # Members of crates this workspace depends on but does not vendor.
    "buffer_unordered", "unwrap_or", "spawn_blocking", "from_slice", "deny_unknown_fields",
    "serde_json", "vorbis_rs",
    # CPython's and numpy's C sources, which `pyrng` reproduces and no `.py` file holds.
    "genrand_res53", "genrand_uint32", "init_by_array", "init_genrand", "rk_double", "rk_random",
    "pcg64_next64", "random_poisson", "random_poisson_mult", "next_double",
    # This repo's own names that are not code: an environment variable, and a case *inside* a
    # golden rather than a definition anywhere.
    "DSRS_CACHEDIR", "evaluate_abandoned",
    # Prose: the stem `infer_prefix` splits an attribute name on, written as a fragment.
    "attribute_",
    # An environment variable, a cache directory, and a doc example's own local.
    "RUST_LOG", "dsrs_cache", "a_set",
    # gepa 0.1.1's *old* placeholder names. Their absence from the pinned tree is the sentence's
    # whole point — the entry says the rename happened — so this is the one case where a name in
    # neither tree is the claim rather than a mistake.
    "curr_instructions", "inputs_outputs_feedback",
    # `String::push_str`, written on a receiver rather than as a path.
    "push_str",
}
EXTERNAL = {
    "Serialize", "Deserialize", "Clone", "Debug", "Default", "Display", "PartialEq", "Eq",
    "Hash", "Iterator", "From", "Into", "Option", "Result", "Vec", "String", "Arc", "Box",
    "Send", "Sync", "JsonSchema", "Value", "Map", "Duration", "Instant", "Path", "PathBuf",
    "BTreeMap", "AsRef", "Fn", "EnvFilter",
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
        own = {first or second for first, second in DEFINITION.findall(text)}
        own |= set(VARIANT.findall(text)) | set(FIELD.findall(text))
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


def everything_by_type() -> dict[str, set[str]]:
    """Every member reachable on a type, impls and inherited trait defaults included.

    Wider and less certain than [`members_by_type`], which speaks only for impl-less data types.
    This one has to guess at three things a regex cannot see cleanly — associated consts, associated
    types, and the methods a type gets from a trait it never overrides — and it is used only to
    *warn*, never to fail, because the first attempt at it produced fourteen false positives.

    It exists because the certain check does not cover the case that actually went wrong: three
    reasons in one afternoon named a member of a type *with* impls — `Refine::code` for
    `program_code`, and two more — and every one of them passed.
    """
    members: dict[str, set[str]] = {}
    trait_methods: dict[str, set[str]] = {}
    implements_trait: list[tuple[str, str]] = []
    declarations: dict[str, int] = {}
    opener = re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)"
    )
    impl_line = re.compile(r"^\s*impl\b(.*)$")
    impl_for = re.compile(r"^\s*impl(?:\s*<[^>]*>)?\s+([A-Za-z_][A-Za-z0-9_]*)[^ ]*\s+for\s+")
    any_member = re.compile(
        r"^\s+(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:async\s+)?(?:unsafe\s+)?"
        r"(?:const\s+|type\s+)?fn\s+([a-z_][A-Za-z0-9_]*)"
        r"|^\s+(?:pub(?:\([^)]*\))?\s+)?const\s+([A-Z_][A-Z0-9_]*)\s*:"
        r"|^\s+(?:pub(?:\([^)]*\))?\s+)?type\s+([A-Za-z_][A-Za-z0-9_]*)\s*[=;:]"
        r"|^\s+(?:pub(?:\([^)]*\))?\s+)?([a-z_][a-z0-9_]*)\s*:"
        r"|^\s+([A-Z][A-Za-z0-9_]*)\s*[({,]"
    )

    def block(lines: list[str], start: int) -> list[str]:
        depth, at, entered = 0, start, False
        out = []
        while at < len(lines):
            depth += lines[at].count("{") - lines[at].count("}")
            entered = entered or depth > 0
            if entered and depth <= 0:
                break
            at += 1
            if at < len(lines):
                out.append(lines[at])
        return out

    for path in (ROOT / "crates").rglob("*.rs"):
        lines = path.read_text(errors="ignore").splitlines()
        for start, line in enumerate(lines):
            found = opener.match(line)
            if found:
                name = found.group(1)
                declarations[name] = declarations.get(name, 0) + 1
                bucket = trait_methods if line.lstrip().startswith(("trait", "pub trait")) else members
                for own in block(lines, start):
                    hit = any_member.match(own)
                    if hit:
                        bucket.setdefault(name, set()).add(next(g for g in hit.groups() if g))
                continue
            found = impl_line.match(line)
            if not found:
                continue
            target = impl_target(found.group(1))
            if not target:
                continue
            declarations.setdefault(target, 1)
            for own in block(lines, start):
                hit = any_member.match(own)
                if hit:
                    members.setdefault(target, set()).add(next(g for g in hit.groups() if g))
            named = impl_for.match(line)
            if named:
                implements_trait.append((target, named.group(1)))

    # A type gets every method of every trait it implements, overridden or not.
    for target, trait in implements_trait:
        members.setdefault(target, set()).update(trait_methods.get(trait, set()))
    for name, own in trait_methods.items():
        members.setdefault(name, set()).update(own)
    # A name declared more than once cannot speak for either declaration — `Task` is a doc
    # example's struct in one crate and a real one elsewhere, and asking whether `Task::predict`
    # exists gets the wrong `Task`. The certain index guards this; so must this one.
    return {name: own for name, own in members.items() if declarations[name] == 1}


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
    """Every `.py` path under the pinned packages, relative to each package root.

    gepa as well as dspy: this crate reproduces that package too, and eight reasons cite one of its
    files — `proposer/merge.py`, `core/adapter.py` — which read as dspy paths that do not exist
    until it is here.
    """
    roots = [ROOT / "third_party" / "dspy" / "dspy"]
    roots += sorted((ROOT / ".venv" / "lib").glob("*/site-packages/gepa"))
    found: set[str] = set()
    for root in roots:
        if root.is_dir():
            found |= {str(path.relative_to(root)) for path in root.rglob("*.py")}
    return found


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


def pinned_words() -> tuple[set[str], list[str]]:
    """Every identifier-shaped token in the pinned Python, and the roots that could not be read.

    dspy's `tests/` tree as well as its package: a reason names an upstream test as readily as an
    upstream function, and reading only the package reported `IN_PROGRESS` — an enum member in
    `tests/predict/test_predict.py` — as a name that exists nowhere. litellm and json_repair for
    the same reason: this crate reproduces both, and its reasons say so in their words.
    """
    found: set[str] = set()
    unread: list[str] = []
    roots = [ROOT / "third_party" / "dspy"]
    # numpy as well: `pyrng` reproduces two of its generators, so a reason naming `SeedSequence`
    # or `default_rng` is a claim about a package this port follows exactly as dspy is. Its *C*
    # sources — `rk_random`, `rk_double` — stay in `CITED_ELSEWHERE`, since no tree here holds them.
    for package in ("gepa", "litellm", "json_repair", "numpy"):
        hits = sorted((ROOT / ".venv" / "lib").glob(f"*/site-packages/{package}"))
        roots += hits
        if not hits:
            unread.append(f".venv/…/site-packages/{package}")
    for root in roots:
        if not root.is_dir():
            unread.append(str(root))
            continue
        for path in root.rglob("*.py"):
            found |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", path.read_text(errors="ignore")))
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
    ledger = {}
    for path in LEDGERS:
        for section, table in tomllib.loads(path.read_text()).items():
            ledger.setdefault(section, {}).update(table)
    names, files, by_file = tree()
    everything = everything_by_type()
    goldens = {str(path.relative_to(ROOT)) for path in (ROOT / "crates").rglob("*.json")}
    theirs, unread = package_names()
    words, unread_words = pinned_words()
    unread += unread_words
    their_files = upstream_files()
    owners = members_by_type()

    missing: list[tuple[str, str, str]] = []
    unverified: list[tuple[str, str]] = []
    unreached: list[tuple[str, str]] = []
    invented: list[tuple[str, str]] = []
    substitutions = 0
    checked = 0
    for section, table in ledger.items():
        if not isinstance(table, dict):
            continue
        for key, entry in table.items():
            if not isinstance(entry, dict):
                continue
            reason = str(entry.get("reason", ""))
            divergence = entry.get("status") == "divergence"
            reached_at = checked
            if divergence and SUBSTITUTION.search(reason):
                substitutions += 1
                # A substitution points at Rust, so a bare name in one is checkable — and often is
                # a private helper: `numbered_block` is the first in the file.
                for ident in set(IDENT.findall(reason)):
                    checked += 1
                    parts = ident.split("::")
                    if not (parts[0][0].isupper() or len(parts) > 1):
                        continue
                    # In *either* tree. A substitution usually points at Rust, but not always:
                    # `RoundRobinReflectionComponentSelector` is the gepa class a round-robin
                    # cursor lives on, and rejecting it for being Python would be crying wolf at
                    # a correct name because the sentence happened to say "lives on".
                    if not any(part in names or part in words for part in parts):
                        missing.append((key, "identifier", ident))

            # A **qualified** path is unambiguously Rust wherever it appears, so it is checked on
            # every reason rather than only on substitutions. That is what a bare-name rule misses:
            # `LM::with_callbacks` was named eight times, had been renamed to
            # `LmBuilder::callbacks`, and every one of those reasons said "there is no" — which no
            # substitution-only check ever looks at. A reason may still name a Python symbol or an
            # environment variable in backticks, and those carry no `::`.
            # Outside the substitution gate: four reasons named `adapter/stream.rs` in a
            # parenthesis and three more named a provider module after "reproduced natively in",
            # which the phrase list does not match. A `.rs` path is a claim about this tree
            # wherever it sits in the sentence.
            for named in set(RS_FILE.findall(reason)):
                checked += 1
                if not any(f.endswith("/" + named) or f == named for f in files):
                    missing.append((key, "file", named))
            for named in set(PY_FILE.findall(reason)):
                if named.startswith(OURS_PY) or not their_files:
                    continue
                checked += 1
                if named not in their_files and not any(
                    f.endswith("/" + named) for f in their_files
                ):
                    missing.append((key, "dspy file", named))
            for named in set(GOLDEN.findall(reason)):
                checked += 1
                if not any(f.endswith("/" + named) for f in goldens):
                    missing.append((key, "golden", named))
            for path_named, item in set(RS_PATH.findall(reason + " " + str(entry.get("rust") or ""))):
                checked += 1
                # Every file the suffix names, not the first — two files here are called `demos.rs`,
                # and picking one of them reported a reason wrong that named the other.
                owning_files = [f for f in files if f.endswith("/" + path_named) or f == path_named]
                if not owning_files:
                    missing.append((key, "file", path_named))
                elif not any(item in by_file[f] for f in owning_files):
                    missing.append((key, "item in file", f"{path_named}::{item}"))
            cited: dict[str, str] = {}
            for pattern, what in (
                (PY_IDENT, "name"),
                (TYPE_NAME, "type"),
                (DOTTED, "path"),
            ):
                for named in pattern.findall(reason):
                    cited.setdefault(named, what)
            # And the names inside a backticked expression, which none of the three can see.
            for span in SPAN.findall(reason):
                if "/" in span or "::" in span or span.count(" ") > 6:
                    continue          # a path: the file and qualified-path rules read these
                if re.search(r"\.(rs|py|json|toml|sh|md)\b", span):
                    continue          # a bare filename: the file and golden rules read these
                for named in SPAN_TOKEN.findall(span):
                    cited.setdefault(named, "name")
            for named, what in cited.items():
                if named in CITED_ELSEWHERE or named in EXTERNAL or not words:
                    continue
                checked += 1
                wanted = named.rsplit(".", 1)[-1]
                if wanted not in words and wanted not in names:
                    missing.append((key, f"{what} in neither tree", named))
            for named in set(WITH_NAME.findall(reason + " " + str(entry.get("rust") or ""))):
                checked += 1
                if named not in names and named not in theirs:
                    missing.append((key, "with_ name", named))
            for ident in set(BARE_PATH.findall(reason + " " + str(entry.get("rust") or ""))):
                checked += 1
                parts = ident.split("::")
                if parts[0] in FOREIGN_ROOTS:
                    continue                   # `std::thread`, `serde_json::Value` — another crate's
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
                    # The wider index covers types *with* impls, which the certain one skips: it
                    # reads inherent methods, associated consts and types, and the methods a type
                    # inherits from every trait it implements. Three reasons named a member of such
                    # a type wrongly in one afternoon and every one of them passed until this ran.
                    elif owner in everything and parts[-1] not in everything[owner]:
                        missing.append((key, "member of", ident))
            # A row every rule above walked past. Not an error — a reason may legitimately be
            # about Python machinery — but it is the reading list, and it only shrinks by someone
            # naming what the sentence is about.
            if checked == reached_at and entry.get("status") != "mapped":
                (invented if section == "rust_only" else unreached).append(
                    (key, entry.get("status"))
                )
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

    print(
        f"  dspy-facing rows no rule reaches: {len(unreached)} (floor {UNREACHED})"
    )
    print(f"  [rust_only] rows no rule reaches: {len(invented)} (floor {INVENTED})")
    if len(invented) > INVENTED or "--invented" in sys.argv:
        for key, status in invented:
            print(f"      ? {key}  ({status})")
    if len(invented) > INVENTED:
        print(f"\nLedger-claims gate FAILED: {len(invented)} unreached, floor {INVENTED}")
        return 1
    if len(invented) < INVENTED:
        print(f"  {INVENTED - len(invented)} below the floor — lower INVENTED to {len(invented)}")
    if len(unreached) > UNREACHED or "--unreached" in sys.argv:
        for key, status in unreached:
            print(f"      ? {key}  ({status})")
    if len(unreached) > UNREACHED:
        print(f"\nLedger-claims gate FAILED: {len(unreached)} unreached, floor {UNREACHED}")
        return 1
    if len(unreached) < UNREACHED:
        print(f"  {UNREACHED - len(unreached)} below the floor — lower UNREACHED to {len(unreached)}")
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
