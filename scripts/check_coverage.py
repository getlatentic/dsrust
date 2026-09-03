"""Every upstream test file is either running, or excused by name.

#23 was "a ported module's tests are in the manifest and not in SUITES". Enumerating that gap by
hand used `tests/<area>/test_<name>.py` -> `<area>/<name>.py`, and the heuristic quietly missed six
modules whose test does not sit under a matching directory — `dsp/utils/settings.py` is tested by
`tests/utils/test_settings.py`, and the usage tracker was not in PORTED_MODULES at all despite the
crate implementing it. Two live bugs were sitting behind that miss.

So the question stops being "which ported module has an unrun test", which needs a mapping that can
be wrong, and becomes "which upstream file is not running", which needs none. Every file upstream
ships is either in SUITES or has a line here saying why not. A file added upstream arrives
unexcused and fails this check, which is the point: a gap has to be *decided*, not overlooked.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from api_surface import PORTED_MODULES  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
RUNNER = ROOT / "scripts" / "run_upstream_tests.sh"
MANIFEST = ROOT / "scripts" / "upstream_tests.txt"

#: Upstream files this crate does not run, and why. A reason is a claim about scope, so it names
#: what the file tests rather than saying "not ported" — the two are different, and only the first
#: survives someone asking.
#: **The whole remainder was probed on 2026-08-27, not just reasoned about.** Every excused file
#: that does not need a live provider or a corpus was copied into the harness and run. The result
#: is why the list below is the length it is: they pass, and *none of their tests cross into the
#: crate* — `test_magicattr` 18 of 18 non-crossing, `test_disk_serialization` 23 of 23,
#: `test_resource_loading` 31 of 31. Adding them would buy a hundred `DOES_NOT_EXERCISE_RUST`
#: lines and no conformance.
#:
#: Six excuses did not survive the same probe and are gone: the six `tests/flex/` files, and
#: `tests/streaming/test_streaming.py`, which was refused by one string — dspy keys its streaming
#: allowlist by `adapter.__class__.__name__`, so `RustChatAdapter` was "unsupported" while being a
#: `dspy.ChatAdapter`. Five more went the same way for having a reason that was a label rather than
#: a fact. Re-probe after a pin bump; the cost is one afternoon and the last one found 150 tests.
EXCUSED = {
    # New in dspy 3.3.0, and each decided rather than overlooked.
    "tests/clients/test_lm_direct_live.py": "talks to a live provider",
    # Thirty cases, and not one renders or parses: every one drives dspy's own `Image`/`Audio`/`File`
    # constructors, its `from_path`/`from_url` factories, its deprecation warnings, and `requests`.
    # Wiring it would add thirty passing tests all declared as not exercising Rust, which inflates a
    # count without adding conformance.
    #
    # What it is *about* does reach here, and is held where the rules live rather than by running
    # this file: construction never dereferences a locator (`Image::new`, `Audio::new`, and their
    # own tests), the explicit factories do the reading (`Image::from_path`, `Audio::from_path`,
    # `File::from_path`, each measured against dspy's bytes), and a non-audio suffix is refused
    # rather than guessed at. The excuse used to say "not yet wired", which read as a scheduling
    # problem; it is a shape problem, and the work it implied is done.
    "tests/adapters/test_resource_loading.py": (
        "dspy's own media-type constructors and factories; nothing in it renders or parses. The "
        "rules it states are held in adapter/types/{image,audio,file}.rs"
    ),

    # Providers and features outside the port's ceiling.
    "tests/clients/test_databricks.py": "a provider this crate does not speak",
    "tests/retrievers/test_colbertv2.py": "a retriever, out of scope",
    "tests/teleprompt/test_finetune.py": "finetuning, deferred past 1.0",
    "tests/teleprompt/test_bootstrap_finetune.py": "finetuning, deferred past 1.0",
    "tests/teleprompt/test_grpo.py": "RL, deferred past 1.0",
    # Python-runtime machinery with no Rust counterpart to reach.
    "tests/utils/test_asyncify.py": "Python's sync/async bridging",
    "tests/utils/test_syncify.py": "Python's sync/async bridging",
    "tests/utils/test_lazy_import.py": "Python's import machinery",
    "tests/clients/test_lazy_litellm_import.py": "litellm's import cost",
    "tests/utils/test_magicattr.py": "Python attribute-path access",
    "tests/utils/test_annotation.py": "Python decorators",
    "tests/utils/test_unbatchify.py": "a Python batching helper",
    "tests/utils/test_langchain_tool.py": "a LangChain adapter",
    "tests/callback/test_callback.py": (
        "the protocol is ported — `Callback` in src/callback.rs, twelve defaulted methods, "
        "registered by configure_callbacks or LM::callbacks — but five of these nine tests "
        "drive Python machinery with no Rust surface to reach: four put `@with_callbacks` on a "
        "plain method and assert on inspect.getcallargs reading a call's arguments back and an "
        "attribute list on a Python object, and the fifth resets a ContextVar token. The other "
        "four assert something portable and all four are held by tests/callback.rs against "
        "sequences recorded from this same dspy by scripts/generate_callback_fixture.py: the "
        "handler sequence one ChainOfThought(n=3) call fires (test_callback_complex_module and "
        "its async twin), the tool handlers (test_tool_calls), and the call-id parent chain "
        "(test_active_id). This said eight tests and named two as portable, both written when "
        "there were eight; `miscounts` checks the total now"
    ),
    # The taxonomy stayed small on purpose: upstream branches on error *identity* in one place,
    # and that one — ContextWindowExceeded, which ReAct catches to truncate — is built and tested.
    # What this file tests is the other ~17 classes existing, which is Python's class hierarchy.
    # Nine of its ten tests assert on a code, a retryability, a metadata field or an exact
    # rendered string, and `tests/exceptions_conformance.rs` holds this crate to all of them
    # against a golden generated by running dspy — all four ContextWindowExceeded renderings,
    # all three AdapterParseError ones, and the code and retryability of thirteen classes.
    # The tenth is `test_lm_errors_are_exported_from_dspy`, which asserts the classes are bound
    # on the `dspy` namespace: a Python name binding with nothing to compare against.
    #
    # Two of the nine also call `isinstance`, three assertions naming three classes between them
    # (`LMInvalidRequestError`, `LMError`, `DSPyError`). That is the part a Rust enum has no
    # answer for, and it is all of it — this said "the tenth" was an `isinstance` against a
    # "fourteen-class Python tree", which is neither the tenth nor fourteen nor most of the file.
    "tests/utils/test_exceptions.py": "isinstance against dspy's Python exception tree",
    # Eighteen of its nineteen are pickle mechanics — allowlists, restricted unpicklers, numpy,
    # a corrupt pickle — and this cache is JSON, so there is nothing to restrict. The nineteenth
    # idea does transfer, and did not: a cache that round-trips a string can still lose a
    # structured reply, and `DiskCache::get` treats an unparseable entry as a *miss*, so a lossy
    # write would silently re-ask the provider for ever rather than fail.
    # `a_reply_carrying_parts_survives_the_round_trip` holds that, and
    # `the_providers_raw_choice_is_not_kept_on_disk` holds the one thing deliberately dropped.
    "tests/clients/test_disk_serialization.py": "Python pickling policy; the round-trip idea is held by lm/cache/disk.rs",
    "tests/clients/test_lm_local.py": "launching a locally-served model",
    # The three modules behind this file are in PORTED_MODULES now, so their symbols are held to
    # the ledger and not only their behaviour to a golden. They were cited by name throughout
    # `adapter/stream/` while being ported and were on no list — which is the tell an audit of that
    # list looks for, and the same shape as the usage tracker this file's own docstring records.
    #
    # Excused for a reason that is *not* "unported": streamify, the three listeners and the status
    # provider all landed with `lm-streamify`. It is excused because running it would prove
    # nothing about them. The bridge carries rendering and parsing and no streaming crossing, so
    # this file would drive dspy's own Python `StreamListener` over a Rust-rendered prompt and
    # never reach this crate's. `tests/streaming_conformance.rs` and
    # `tests/partial_json_conformance.rs` are the direct oracle instead: dspy's own listener driven
    # over sixteen recorded streams, and `jiter`'s answer for 342 accumulated prefixes.
    # Not conformance at all.
    "tests/docs/test_mkdocs_links.py": "upstream's documentation links",
    "tests/metadata/test_metadata.py": "upstream's package metadata",
    "tests/examples/test_baleen.py": "an end-to-end example needing a live retriever",
    "tests/reliability/test_generated.py": "upstream's generated reliability corpus",
    "tests/reliability/test_pydantic_models.py": "upstream's reliability corpus",
    "tests/predict/test_retry.py": (
        "nothing to run: every test in the file is commented out upstream, since dspy.Retry and "
        "dspy.primitives.assertions were removed. Not a feature this crate skips — the retry it "
        "names is ported (src/lm/retry.rs) and held by tests/lm_retry.rs"
    ),
}


#: Number words an excuse might count in, so "eight tests" is as checkable as "8 tests".
NUMBERS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
#: "nine of its ten tests", "eight tests", "30 cases" — the total, wherever the excuse states one.
STATED = re.compile(
    rf"\b(?:{'|'.join(NUMBERS)}|\d+)\s+(?:of\s+(?:its\s+)?({'|'.join(NUMBERS)}|\d+)\s+)?"
    rf"(?:cases|tests)\b",
    re.I,
)


def counted(reason: str) -> int | None:
    """How many tests an excuse says the file has, if it says."""
    for match in STATED.finditer(reason):
        stated = (match.group(1) or match.group(0).split()[0]).lower()
        if stated in NUMBERS:
            return NUMBERS[stated]
        if stated.isdigit():
            return int(stated)
    return None


def tests_in(path: str) -> int | None:
    """How many the file actually has, read from the pinned tree."""
    source = ROOT / "third_party" / "dspy" / path
    if not source.exists():
        return None
    return sum(
        1
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    )


def miscounts() -> list[str]:
    """Excuses whose stated test count is not the file's.

    An excuse that says "six of these eight tests drive the decorator" is making a claim about a
    file upstream owns, and upstream adds tests. That one was written when there were eight; the
    ninth arrived with `_bind_active_call_id` — a fix for carrying the parent call id across a
    thread — and the excuse read as settled for as long as nobody counted.

    **Only the reason string is read**, not the comment above it. `tests/utils/test_exceptions.py`
    carries its detail in a comment, so its "nine of its ten tests" is unchecked here — and that
    comment was wrong about something else nobody could have counted: it called the tenth an
    `isinstance` against a "fourteen-class Python tree", where the file has three `isinstance`
    assertions naming three classes and the tenth is a namespace check. A count is gateable; what
    the tests are *about* is not, and that half stays a reading.
    """
    found = []
    for path, reason in EXCUSED.items():
        stated, actual = counted(reason), tests_in(path)
        if stated is not None and actual is not None and stated != actual:
            found.append(f"{path}: the excuse says {stated} test(s), the file has {actual}")
    return found


def experimental_is_not_a_line() -> list[str]:
    """No excuse may rest on `@experimental` while the crate ports things carrying it.

    `dspy.Flex`'s six files were excused because its docstring says the API "may change or be
    removed in a future release without warning". They run as of 2026-08-27 — 109 tests, 83 of them
    crossing into the crate — and the excuse that outlived its reason had by then become "not
    ported yet", of a module with 1246 lines and its own conformance test. The sentence quoted here
    is real — `@experimental` writes it.
    The standard is not this crate's: six modules in the pin carry that decorator and five are
    ported and classified, `GEPA` among them with forty ledger entries. An excuse naming a boundary
    the crate does not hold is the shape `SCOPE_EXCLUSIONS` gates for `deferred` ledger rows, and
    this is that shape one table over.
    """
    marked = {
        str(path.relative_to(ROOT / "third_party" / "dspy" / "dspy"))
        for path in (ROOT / "third_party" / "dspy" / "dspy").rglob("*.py")
        if "@experimental" in path.read_text()
    }
    ported = sorted(marked & set(PORTED_MODULES))
    resting = sorted(p for p, reason in EXCUSED.items() if "experimental" in reason.lower())
    if not ported or not resting:
        return []
    return [
        f"{len(resting)} excuse(s) resting on `@experimental` while {len(ported)} module(s) "
        f"carrying it are ported:",
        *(f"      ~ {path}" for path in resting),
        f"      ported anyway: {', '.join(ported)}",
        "  Experimental is not this crate's boundary. Give the real reason.",
    ]


#: The per-test declarations in the bridge's conftest, which make the same kind of claim a file-level
#: excuse does and are nine times as many.
CONFTEST = (
    pathlib.Path(__file__).parent.parent / "crates" / "dsrs-bridge" / "python" / "conftest.py"
)
DECLARATIONS = (
    "DOES_NOT_EXERCISE_RUST",
    "NOT_YET_IMPLEMENTED",
    "NOT_ADAPTER_CONFORMANCE",
    "SIGNATURE_CONFORMANCE",
)


def declared_reasons() -> dict[str, str]:
    """Every per-test declaration in the conftest, as `dict name -> its reason`.

    Read with `ast` rather than by importing: the conftest needs pytest, dspy and a built bridge,
    and this check has to run without any of them.
    """
    import ast

    found: dict[str, str] = {}
    tree = ast.parse(CONFTEST.read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id not in DECLARATIONS or not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if not isinstance(key, ast.Constant):
                continue
            try:
                reason = ast.literal_eval(value)
            except ValueError:
                continue
            if isinstance(reason, str):
                found[f"{node.targets[0].id}[{key.value}]"] = reason
    return found


def reasons_naming_nothing() -> list[str]:
    """A comment block inside a declaration table that introduces no entry.

    The tables are prose plus keys, and `_orphaned_declarations` in the conftest already refuses a
    key naming no test. Nothing read the other half. A paragraph explaining why some tests are
    deferred, left behind after its entries were deleted, still reads as a live gap — and it is the
    one claim no gate could see, because it names nothing to check it against.

    Found by reading `NOT_YET_IMPLEMENTED`, whose last paragraph argued that converting sets in the
    shim "would green all three" of tests that had all been passing since the 3.3.0 pin.

    A comment sharing a line with code is that entry's own note, not an introduction, so only runs
    of whole comment lines count.
    """
    import ast
    import io
    import tokenize

    source = CONFTEST.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    problems = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name not in DECLARATIONS or not isinstance(node.value, ast.Dict):
            continue
        span = range(node.value.lineno, (node.value.end_lineno or node.value.lineno) + 1)
        keys = [key.lineno for key in node.value.keys if key is not None]
        standalone = sorted(
            token.start[0]
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
            and token.start[0] in span
            and not lines[token.start[0] - 1][: token.start[1]].strip()
        )
        blocks: list[list[int]] = []
        for row in standalone:
            if blocks and blocks[-1][-1] == row - 1:
                blocks[-1].append(row)
            else:
                blocks.append([row])
        for block in blocks:
            if any(lineno > block[-1] for lineno in keys):
                continue
            problems.append(
                f"{name} carries a reason at {CONFTEST.name}:{block[0]}-{block[-1]} that "
                "introduces no entry — its tests are no longer declared, so the paragraph is "
                "about nothing"
            )
    return problems


def names_that_do_not_exist() -> list[str]:
    """Every `.rs` file, golden and `Type::member` an excuse names, resolved against the tree.

    An excuse is prose making the same kind of claim a ledger reason does, and nothing read it.
    `LM::with_callbacks` sat in the callback excuse after that builder had been renamed to
    `LM::callbacks` — one wrong name out of the seven these thirty-one cite, which is a low density
    and exactly the density that keeps a surface unchecked.

    The rules and the indexes are `check_ledger_claims`'s, imported rather than restated: a second
    copy of "is this a member of that type" is a second thing to keep true.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import check_ledger_claims as ledger

    names, files, _by_file = ledger.tree()
    everything = ledger.everything_by_type()
    owners = ledger.members_by_type()
    goldens = {str(path.relative_to(ROOT)) for path in (ROOT / "crates").rglob("*.json")}

    missing: list[str] = []
    # The conftest's per-test declarations join the file-level excuses: same rules, same
    # indexes, and 266 more reasons that nothing read. `LM::with_callbacks` sat wrong in an
    # excuse for months at a density of one in seven; there is no reason the larger corpus
    # would be cleaner for having been unchecked longer.
    for path, why in {**EXCUSED, **declared_reasons()}.items():
        for named in sorted(set(ledger.RS_FILE.findall(why))):
            if not any(f.endswith("/" + named) or f == named for f in files):
                missing.append(f"{path} names {named}, which is not a file here")
        for named in sorted(set(ledger.GOLDEN.findall(why))):
            if not any(f.endswith("/" + named) for f in goldens):
                missing.append(f"{path} names the golden {named}, which does not exist")
        for ident in sorted(set(ledger.BARE_PATH.findall(why))):
            parts = ident.split("::")
            if parts[0] in ledger.FOREIGN_ROOTS:
                continue
            if parts[0] in ledger.OURS:
                parts = parts[1:]
            elif parts[0][:1].islower() and parts[0] not in names:
                continue
            if not parts:
                continue
            owner = parts[-2] if len(parts) >= 2 and parts[-2][:1].isupper() else None
            wanted = [parts[-1]] + ([owner] if owner else [])
            if any(part not in names for part in wanted):
                missing.append(f"{path} names {ident}, which does not exist")
            elif owner in owners and parts[-1] not in owners[owner]:
                missing.append(f"{path} names {ident}, and {owner} has no such member")
            elif owner in everything and parts[-1] not in everything[owner]:
                missing.append(f"{path} names {ident}, and {owner} has no such member")
    return missing


def running() -> set[str]:
    block = re.search(r"SUITES=\((.*?)\n\)", RUNNER.read_text(), re.S)
    if block is None:
        raise SystemExit(f"{RUNNER.name} has no SUITES array to read")
    return {f"tests/{name}" for name in re.findall(r"[\w/]+/test_\w+\.py", block.group(1))}


#: Phrases an excuse uses when it rests on the module simply not being here. Each is a claim about
#: this crate rather than about the test, and each stops being true the day the module lands.
ABSENCE = (
    "deferred",
    "out of scope",
    "does not cover",
    "not ported",
    "the port does not",
)


def stale_excuses() -> list[str]:
    """An excuse that rests on a module this crate has since ported.

    `tests/A/test_B.py` covers `A/B.py`, near enough to check: when that module is in
    PORTED_MODULES, an excuse saying the feature is deferred or out of scope is quoting a state
    that has changed. The excuse may still be right — a ported module can have Python-only tests —
    but then it has to say *that*, and this makes it say it.

    Found by the history suite going stale: its excuse said the crate does not accumulate a history
    and therefore six of nine tests could not cross, which was true when written and wrong once
    `utils/inspect_history.py` was ported. All nine cross now.
    """
    import api_surface

    ported = set(api_surface.PORTED_MODULES)
    problems = []
    for path, why in sorted(EXCUSED.items()):
        parts = pathlib.PurePosixPath(path).parts
        if len(parts) != 3 or not parts[2].startswith("test_"):
            continue
        module = f"{parts[1]}/{parts[2][len('test_'):]}"
        if module not in ported:
            continue
        flat = " ".join(why.split()).lower()
        for phrase in ABSENCE:
            if phrase in flat:
                problems.append(
                    f"{path} is excused as {phrase!r}, but {module} is in PORTED_MODULES — "
                    "run the suite, or say what about the tests is Python-only"
                )
                break
    return problems


def main() -> None:
    manifest = [
        line.strip()
        for line in MANIFEST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    run = running()
    unexcused = [f for f in manifest if f not in run and f not in EXCUSED]
    miscounted = miscounts()
    resting = experimental_is_not_a_line()
    stale = [f for f in EXCUSED if f not in manifest]
    both = [f for f in EXCUSED if f in run]
    unresolved = names_that_do_not_exist()
    outdated = stale_excuses()
    about_nothing = reasons_naming_nothing()

    print(f"  {len(run)} of {len(manifest)} upstream files run, {len(EXCUSED)} excused by name")

    failures = []
    if unexcused:
        failures.append(f"{len(unexcused)} upstream file(s) neither running nor excused:")
        failures += [f"      + {name}" for name in sorted(unexcused)]
    if stale:
        failures.append(f"{len(stale)} excuse(s) naming a file upstream no longer ships:")
        failures += [f"      - {name}" for name in sorted(stale)]
    if both:
        failures.append(f"{len(both)} file(s) both excused and running — drop the excuse:")
        failures += [f"      ! {name}" for name in sorted(both)]
    if miscounted:
        failures.append(f"{len(miscounted)} excuse(s) counting a file wrong:")
        failures += [f"      # {line}" for line in sorted(miscounted)]
    if unresolved:
        failures.append(f"{len(unresolved)} excuse(s) naming something that does not exist:")
        failures += [f"      -> {line}" for line in sorted(unresolved)]
    failures += about_nothing
    failures += outdated
    failures += resting

    if failures:
        print("\nUpstream-coverage check FAILED:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)
    print("  upstream coverage: OK (every file runs or says why not)")


if __name__ == "__main__":
    main()
