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
EXCUSED = {
    # New in dspy 3.3.0, and each decided rather than overlooked.
    #
    # `dspy.Flex` is a whole feature this port does not have — its own class, its own GEPA
    # extension, its own interpreter binding: 1092 lines of source and 2570 of tests.
    #
    # This used to read "its docstring calls itself experimental ... porting an API upstream
    # reserves the right to delete is not a 1.0 obligation". The quote is real — `@experimental`
    # writes that sentence — but the standard is not this crate's. Six things in the pin carry
    # that decorator and five are ported and fully classified: `Citation`, `Document`, `Rlm`,
    # `ReActV2`, and `GEPA`, which alone holds forty ledger entries. Experimental has never been
    # the line here, so it cannot be the line for this one.
    #
    # The real reason is size, which is a schedule and not a boundary. Tracked as `dspy-flex`.
    "tests/flex/test_flex_binding.py": "dspy.Flex: 1092 lines of source and a host/guest bridge, not ported yet (`dspy-flex`)",
    "tests/flex/test_flex_gepa.py": "dspy.Flex: 1092 lines of source and a host/guest bridge, not ported yet (`dspy-flex`)",
    "tests/flex/test_flex_gepa_seed.py": "dspy.Flex: 1092 lines of source and a host/guest bridge, not ported yet (`dspy-flex`)",
    "tests/flex/test_flex_interpreter.py": "dspy.Flex: 1092 lines of source and a host/guest bridge, not ported yet (`dspy-flex`)",
    "tests/flex/test_flex_output_types.py": "dspy.Flex: 1092 lines of source and a host/guest bridge, not ported yet (`dspy-flex`)",
    "tests/flex/test_tools.py": "dspy.Flex: 1092 lines of source and a host/guest bridge, not ported yet (`dspy-flex`)",
    # Live, by its own docstring: "provider behavior that cannot be verified with the mocked unit
    # tests". The crate's own live tests carry `#[ignore]` for the same reason.
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
    "tests/clients/test_embedding.py": "embeddings, which the port does not cover",
    "tests/retrievers/test_colbertv2.py": "a retriever, out of scope",
    "tests/retrievers/test_embeddings.py": "a retriever, out of scope",
    "tests/teleprompt/test_finetune.py": "finetuning, deferred past 1.0",
    "tests/teleprompt/test_bootstrap_finetune.py": "finetuning, deferred past 1.0",
    "tests/teleprompt/test_grpo.py": "RL, deferred past 1.0",
    # The optimizers deliberately deferred past 1.0 (#9). Both need the KNN retriever rather than
    # anything the optimizer work builds, which is why they sit with retrieval in s10.
    "tests/teleprompt/test_knn_fewshot.py": "KNNFewShot is deferred (#9)",
    "tests/predict/test_knn.py": "KNN is deferred (#9)",
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
        "registered by configure_callbacks or LM::with_callbacks — but five of these nine tests "
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
    "tests/clients/test_inspect_global_history.py": "dspy's history printing",
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
    "tests/streaming/test_streaming.py": "dspy's async streaming plumbing around a Python program; it would exercise upstream's own listener rather than this crate's, which the streaming goldens compare directly",
    "tests/teleprompt/test_bootstrap_trace.py": "dspy's trace-collection helpers",
    "tests/teleprompt/test_utils.py": "dspy's optimizer helper functions",
    "tests/teleprompt/test_gepa_instruction_proposer.py": "dspy's proposer over multimodal inputs",
    "tests/evaluate/test_auto_evaluation.py": "dspy's built-in evaluation signatures",
    # Not conformance at all.
    "tests/docs/test_mkdocs_links.py": "upstream's documentation links",
    "tests/metadata/test_metadata.py": "upstream's package metadata",
    "tests/datasets/test_dataset.py": "dspy's dataset loaders",
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
    removed in a future release without warning". The sentence is real — `@experimental` writes it.
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


def running() -> set[str]:
    block = re.search(r"SUITES=\((.*?)\n\)", RUNNER.read_text(), re.S)
    if block is None:
        raise SystemExit(f"{RUNNER.name} has no SUITES array to read")
    return {f"tests/{name}" for name in re.findall(r"[\w/]+/test_\w+\.py", block.group(1))}


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
    failures += resting

    if failures:
        print("\nUpstream-coverage check FAILED:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)
    print("  upstream coverage: OK (every file runs or says why not)")


if __name__ == "__main__":
    main()
