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

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent.parent
RUNNER = ROOT / "scripts" / "run_upstream_tests.sh"
MANIFEST = ROOT / "scripts" / "upstream_tests.txt"

#: Upstream files this crate does not run, and why. A reason is a claim about scope, so it names
#: what the file tests rather than saying "not ported" — the two are different, and only the first
#: survives someone asking.
EXCUSED = {
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
    "tests/callback/test_callback.py": "dspy's Python callback protocol",
    # The taxonomy stayed small on purpose: upstream branches on error *identity* in one place,
    # and that one — ContextWindowExceeded, which ReAct catches to truncate — is built and tested.
    # What this file tests is the other ~17 classes existing, which is Python's class hierarchy.
    # The portable half is ported and tested in `lm/error/`: the HTTP-status map, the four
    # retryable kinds, and the provider/configuration split. What stays here is the part that is a
    # Python class tree — `isinstance` against fourteen types, and `raise ... from`.
    "tests/utils/test_exceptions.py": "dspy's exception classes, as a Python class hierarchy",
    "tests/clients/test_disk_serialization.py": "Python pickling policy",
    "tests/clients/test_inspect_global_history.py": "dspy's history printing",
    "tests/clients/test_lm_local.py": "launching a locally-served model",
    "tests/streaming/test_streaming.py": "dspy's streaming listeners over its own async plumbing",
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
    "tests/predict/test_retry.py": "dspy's retry wrapper over litellm",
}


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

    if failures:
        print("\nUpstream-coverage check FAILED:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)
    print("  upstream coverage: OK (every file runs or says why not)")


if __name__ == "__main__":
    main()
