"""Point upstream's test suite at the Rust-backed adapter, and be honest about the gaps.

Rust renders and parses every case here. A reply Rust rejects may re-ask through Python's
JSONAdapter, since that is dspy's own behaviour and upstream tests it directly; a case Rust
has not implemented may not, and raises instead. Those cases are listed below and marked
`xfail(strict=True)`, which means two things: they are never counted as passes, and if one
starts passing the run FAILS until its name is deleted from the list. The list is therefore
the to-do list, it cannot drift out of date, and a green run means every case not named here
genuinely runs on Rust.
"""

import json
import os

import dspy
import pytest

# dspy 3.3.0's interpreter pooling, borrowed rather than reimplemented.
#
# Booting a Deno/Pyodide sandbox costs about 2.5 seconds, so upstream's suite shares one per pytest
# process and restores its namespace between tests. The fixtures live in dspy's own `tests/conftest`
# — but the run files are copied *flat* into the work directory, so that conftest never applies to
# them and every test asking for one errored out.
#
# Three names rather than `pytest_plugins = ["tests.conftest"]`: this file already reimplements
# `lm_for_test`, `litellm_test_server` and the settings reset, because ours install the Rust adapter
# and count crossings. Loading upstream's whole conftest would run both of each.
from tests.conftest import (  # noqa: F401
    _POOL_SETUP_CODE,
    _interpreter_pool,
    configure_pooled_interpreter,
)


@pytest.fixture
def pooled_interpreter(_interpreter_pool):
    """Upstream's pooled interpreter, built as the Rust one.

    Overridden rather than imported. Upstream's builds `PythonInterpreter` directly inside the
    fixture body and caches it in a *session*-scoped holder, so it outlives the per-test
    monkeypatch that swaps in `RustPythonInterpreter` — every pooled test then ran against dspy's
    own sandbox and proved nothing about this crate. The crossing counter said so, in 56 tests that
    "passed without the crate rendering or parsing anything", which is exactly the failure it
    exists to catch: a green suite testing the wrong implementation.

    Everything else is upstream's — the setup code, the namespace restoration, the terminal-session
    handling on teardown.
    """
    from dspy.primitives.code_interpreter import CodeInterpreterError

    interpreter = _interpreter_pool["interpreter"]
    if interpreter is None:
        interpreter = RustPythonInterpreter()
        interpreter.execute(_POOL_SETUP_CODE)
        _interpreter_pool["interpreter"] = interpreter

    yield interpreter

    try:
        interpreter.tools.clear()
        interpreter.output_fields = None
        interpreter._tools_registered = False
        interpreter.execute("_pool_reset()")
    except CodeInterpreterError:
        # The test that just ran ended the session. Surface it here and boot a fresh one for the
        # next consumer, as upstream does.
        _interpreter_pool["interpreter"] = None
        interpreter.shutdown()
        raise
# The bridge does not build on every toolchain: macOS 26+/ld-27034 emits a "mis-aligned
# LINKEDIT string pool" for this extension module and dyld refuses to load it. Skip the whole
# run in that case, so a broken build never reads as a pass or as a fault in this crate.
try:
    import dsrs_bridge  # noqa: F401
except ImportError as error:  # pragma: no cover - environment dependent
    pytest.skip(
        f"the Rust bridge could not be loaded, so nothing here exercises this crate: {error}",
        allow_module_level=True,
    )

import crossings  # noqa: E402
import rust_signature  # noqa: E402
from rust_module import (  # noqa: E402
    RustCodeAct,
    RustPredict,
    RustProgramOfThought,
    RustPythonInterpreter,
    RustReAct,
    RustRLM,
)
from rust_adapter import (  # noqa: E402
    RustBAMLAdapter,
    RustChatAdapter,
    RustJSONAdapter,
    RustTwoStepAdapter,
    RustXMLAdapter,
)

#: The adapter names upstream reaches for, and the Rust-backed class for each. A test file that
#: did `from … import XMLAdapter` holds its own reference, bound when pytest imported it, so the
#: fixture below rebinds the name in the running test's module as well as on dspy.
RUST_BACKED = {
    "BAMLAdapter": RustBAMLAdapter,
    "ChatAdapter": RustChatAdapter,
    "JSONAdapter": RustJSONAdapter,
    "XMLAdapter": RustXMLAdapter,
}

# Bound on dspy itself as this conftest is imported, which is before pytest imports any test
# module. A test that builds an adapter while *collecting* — `@parametrize("adapter",
# [dspy.ChatAdapter(...)])` — would otherwise hold a real dspy instance, since the fixture below
# runs per test and cannot reach an object made before it. Such a test passes either way, so the
# crossing counter is the only thing that notices; without this it reports a pass for rendering
# this crate never did.
def _rust_provider_tool_call(tool_call):
    """dspy's `_provider_tool_call_to_tool_call_dict`, answered by the crate.

    Reading fields off an arbitrary provider object is reflection, and only Python can do it, so
    the value is flattened to a plain mapping here. What the written call *means* — which spelling
    holds the name, whether the id came as `id` or `call_id`, whether malformed arguments repair —
    is the crate's decision and is made there.
    """
    from dspy.adapters.base import _provider_value

    function = _provider_value(tool_call, "function", {}) or {}
    written = {
        "id": _provider_value(tool_call, "id"),
        "call_id": _provider_value(tool_call, "call_id"),
        "function": {
            "name": _provider_value(function, "name"),
            "arguments": _provider_value(function, "arguments", {}),
        },
        "name": _provider_value(tool_call, "name"),
    }
    crossings.record_render()
    return json.loads(dsrs_bridge.normalize_tool_call(json.dumps(written)))


for _name, _rust in RUST_BACKED.items():
    setattr(dspy, _name, _rust)
    setattr(dspy.adapters, _name, _rust)

dspy.adapters.base._provider_tool_call_to_tool_call_dict = _rust_provider_tool_call

# Upstream tests whose features this crate has not written yet, with the reason. Delete a line
# once Rust renders that case; the strict xfail will fail the run if you forget.
#
# A run may report xfails this list is empty of: dspy marks two of its own image cases xfail
# inside the test body, for a gap upstream has rather than one this port has.
NOT_YET_IMPLEMENTED = {
    # Four tests that reach for `interpreter.deno_process` or `_read_response_line`. Those are
    # dspy's own internals: `RustPythonInterpreter` replaces `execute` and nothing else, so the
    # child they kill or the reader they patch is dspy's — unused — while the Rust sandbox talks to
    # a child of its own that the test has no handle on.
    #
    # The *behaviour* two of them check is ported and does hold. `DenoInterpreter` ends its session
    # on a dead child or a protocol failure rather than starting a fresh one, which is dspy 3.3.0's
    # rule, and `deno.rs`'s own tests cover both directions with a control. What cannot be
    # reproduced here is the mechanism, not the contract.
    #
    # (This comment used to say the restarts "work — `DenoInterpreter` notices a dead child, starts
    # another, and replays the registration and the mounts". That stopped being true when the
    # session became terminal, and a stale comment about a skipped test is how a skip outlives its
    # reason.)
    "test_tools_re_register_after_process_restart": (
        "the test kills dspy's own subprocess handle, which the Rust sandbox does not own"
    ),
    "test_mounts_replay_after_process_restart": (
        "the test kills dspy's own subprocess handle, which the Rust sandbox does not own"
    ),
    "test_process_death_ends_stateful_session": (
        "the test kills dspy's own subprocess handle, which the Rust sandbox does not own; the "
        "terminal-session rule it checks is held by deno.rs's own tests"
    ),
    "test_protocol_failure_ends_session": (
        "the test patches dspy's own `_read_response_line`, which the Rust sandbox does not call; "
        "the terminal-session rule it checks is held by deno.rs's own tests"
    ),
    # These pass a Python `set` or `tuple` as an input *variable*. `CodeInterpreter::execute` takes
    # a `serde_json::Map`, which has neither, so a Rust caller cannot reach this path at all — there
    # is no crate behaviour here to be right or wrong about.
    #
    # Converting them in the shim would green all three and test the shim. The conversion is not
    # pure reflection either: dspy sorts a set on the way out, so `{3,1,2}` reaches the sandbox as
    # `[1, 2, 3]`, and that ordering is a byte the model reads. Deciding it in Python is the thing
    # the bridge exists to prevent.
}

# `test_serialize_set` and `test_serialize_set_mixed_types` were here for the same reason as
# `test_nested_sets_and_tuples` and are not any more: dspy 3.3.0 added `_make_jsonable` and
# `_dump_pydantic`, so a set is converted on dspy's side before it ever reaches this crate's
# sandbox. They XPASSed at the new pin, which is what a strict xfail is for — a divergence that
# upstream closes should be noticed rather than carried.


# Upstream tests that pass without the crate rendering or parsing anything, with the reason.
# They are not conformance: they exercise dspy's own Python — a type's `__str__`, a helper — and
# would read as green whatever this crate did. Naming them keeps the passing count honest, and
# anything not named here must cross into Rust or the run fails.
DOES_NOT_EXERCISE_RUST = {
    # The interpreter's own constructor and its reflection over Python callables. These reach no
    # sandbox at all: they assert on what `PythonInterpreter.__init__` stored and on what
    # `inspect.signature` reports, both of which stay dspy's own code under the shim.
    "test_deno_command_dict_raises_type_error": "the constructor's own type check",
    "test_tools_dict_is_copied": "the constructor copying the tools dict",
    "test_extract_parameters": "inspect.signature over a Python callable",
    "test_extract_parameters_complex_types": "inspect.signature over a Python callable",
    "test_small_variable_not_using_filesystem": "dspy's own `_pending_large_vars` bookkeeping",
    "test_large_variable_threshold_boundary": "dspy's own `_pending_large_vars` bookkeeping",
    # dspy's optimizer reads its own constructor back; nothing is rendered.
    "test_bootstrap_initialization": "an optimizer's own constructor",
    # The student raises before a prompt exists, so no rendering is reached. The crate's own
    # optimizer is not what runs here either — upstream's does, over this crate's adapter.
    "test_error_handling_during_bootstrap": "a student that raises before any prompt is built",
    # `Predict`'s own state: what `dump_state`/`load_state` round-trip, and the checks that stop
    # a serialized file from redirecting a later run at another endpoint. dspy's module
    # bookkeeping and its security posture around it, both above the wire this crate speaks.
    "test_lm_after_dump_and_load_state": "dspy's own state round-trip",
    "test_instructions_after_dump_and_load_state": "dspy's own state round-trip",
    "test_demos_after_dump_and_load_state": "dspy's own state round-trip",
    "test_typed_demos_after_dump_and_load_state": "dspy's own state round-trip",
    "test_signature_fields_after_dump_and_load_state": "dspy's own state round-trip",
    "test_lm_field_after_dump_and_load_state": "dspy's own state round-trip",
    "test_dump_state_pydantic_non_primitive_types": "dspy's own state round-trip",
    "test_load_state_chaining": "dspy's own state round-trip",
    "test_load_ignores_serialized_endpoint_override_by_default": "dspy's endpoint-override guard",
    "test_load_allows_serialized_endpoint_override_with_opt_in": "dspy's endpoint-override guard",
    "test_load_state_ignores_serialized_endpoint_override_by_default": (
        "dspy's endpoint-override guard"
    ),
    "test_load_state_allows_serialized_endpoint_override_with_opt_in": (
        "dspy's endpoint-override guard"
    ),
    "test_load_state_ignores_serialized_model_list_endpoint_override_by_default": (
        "dspy's endpoint-override guard"
    ),
    "test_load_prevents_serialized_endpoint_override_reaching_litellm": (
        "dspy's endpoint-override guard"
    ),
    "test_load_blocks_serialized_model_list_unless_opted_in": "dspy's endpoint-override guard",
    "test_load_uses_env_api_key_without_honoring_serialized_endpoint_override": (
        "dspy's endpoint-override guard"
    ),
    # `Predict` as a module: how it is built, reset, configured and walked. None of it renders.
    "upstream_test_predict.py::test_initialization_with_string_signature": (
        "dspy's module construction"
    ),
    "test_reset_method": "dspy's module construction",
    "test_config_management": "dspy's module construction",
    "test_named_predictors": "dspy's module construction",
    "test_positional_arguments": "dspy's module construction",
    # Raises on the LM before any adapter is reached, which is the thing being tested.
    "test_error_message_on_invalid_lm_setup": "dspy's LM validation, ahead of rendering",
    # Each replaces the module or the adapter that would call this crate — two mock `react` and
    # `extract` outright, one substitutes its own adapter to watch what ReAct hands it. Nothing
    # can cross by the tests' own design.
    "test_trajectory_truncation": "ReAct's loop with its predictors mocked out",
    "test_context_window_exceeded_after_retries": "ReAct's loop with its predictors mocked out",
    # ReActV2 building its `submit` tool and per-turn signature, checked without a call — the agent
    # is constructed and inspected, so nothing renders or parses. Its other ten tests do call it.
    "test_react_v2_submit_tool_returns_original_output_fields": "dspy.ReActV2 construction",
    # Refine and BestOfN choosing their fail-count budget, and Parallel's timeout / straggler knobs
    # and batch error handling — module wiring and a non-rendering `forward`, so nothing crosses.
    "test_refine_module_default_fail_count": "Refine/BestOfN fail-count config",
    "test_refine_module_custom_fail_count": "Refine/BestOfN fail-count config",
    "test_parallel_timeout_and_straggler_limit_params": "Parallel's timeout/straggler config",
    "test_batch_timeout_and_straggler_limit_params": "Parallel's timeout/straggler config",
    "test_batch_with_failed_examples": "Parallel's batch error handling over a non-rendering module",
    # Which Python class a serialized LM state names, and whether loading it is trusted enough to
    # import. Resolving a dotted class path at run time is Python reflection with no Rust reading,
    # and the trust decision sits above the wire either way.
    "test_base_lm_dump_state_ignores_internal_class_marker_kwarg": "dspy's LM state bookkeeping",
    "test_legacy_lm_state_without_class_marker_loads_as_lm": "dspy's LM class-path resolution",
    "test_custom_lm_load_state_requires_trusted_opt_in": "dspy's LM class-path trust boundary",
    "test_nested_custom_lm_class_path_loads_for_trusted_state": "dspy's LM class-path resolution",
    # `dspy.Citations` and `dspy.Reasoning` validating, formatting and concatenating
    # themselves. Their files are not blanket-declared because each has one test that does
    # reach an adapter.
    "test_citation_extraction_from_lm_response": "dspy.Citations parsing itself",
    "test_citation_format": "dspy.Citations' own string form",
    "test_citation_validate_input": "dspy.Citations validating itself",
    "test_citation_with_all_fields": "dspy.Citations construction",
    "test_citations_format": "dspy.Citations' own string form",
    "test_citations_from_dict_list": "dspy.Citations construction",
    "test_citations_in_nested_type": "dspy.Type annotation walking",
    "test_reasoning_basic_operations": "dspy.Reasoning behaving as a string",
    "test_reasoning_concatenation": "dspy.Reasoning behaving as a string",
    "test_reasoning_error_message": "dspy.Reasoning's attribute error",
    "test_reasoning_string_methods": "dspy.Reasoning behaving as a string",
    # `dspy.File` constructing, validating and describing itself: from bytes, from a path, from
    # an id, from a dict, and what each refuses. The adapter is not reached — these build the
    # value that an adapter would later render, and four tests in the file do render one.
    "test_encode_file_to_dict_from_bytes": "dspy.File encoding itself",
    "test_encode_file_to_dict_from_path": "dspy.File encoding itself",
    "test_file_custom_mime_type": "dspy.File's own mime detection",
    "test_file_data_uri_in_format": "dspy.File's own data URI",
    "test_file_from_bytes": "dspy.File construction",
    "test_file_from_bytes_custom_mime": "dspy.File construction",
    "test_file_from_bytes_with_filename": "dspy.File construction",
    "test_file_from_dict_with_file_data": "dspy.File construction",
    "test_file_from_dict_with_file_id": "dspy.File construction",
    "test_file_from_file_id": "dspy.File construction",
    "test_file_from_file_id_with_filename": "dspy.File construction",
    "test_file_from_local_path": "dspy.File construction",
    "test_file_from_path_method": "dspy.File construction",
    "test_file_from_path_with_custom_filename": "dspy.File construction",
    "test_file_frozen": "dspy.File refusing mutation",
    "test_file_path_not_found": "dspy.File rejecting a missing path",
    "test_file_repr_with_file_data": "dspy.File's own string form",
    "test_file_repr_with_file_id": "dspy.File's own string form",
    "test_file_str": "dspy.File's own string form",
    "test_file_with_all_fields": "dspy.File construction",
    "test_invalid_dict": "dspy.File rejecting a malformed dict",
    "test_invalid_file_string": "dspy.File rejecting a malformed string",
    # `dspy.Image` doing the same, plus the PIL and download paths that never reach a prompt.
    "test_different_mime_types": "dspy.Image's own mime detection",
    "test_from_methods_warn": "dspy.Image's deprecation warning",
    "test_image_repr": "dspy.Image's own string form",
    "test_invalid_string_format": "dspy.Image rejecting a malformed string",
    "test_mime_type_from_response_headers": "dspy.Image reading a response header",
    "test_pil_image_with_download_parameter": "dspy.Image's download flag",
    # Resolving a custom annotation to a type, which is Python's type system answering about
    # itself. `reflect.py` is where this crate depends on that answer rather than making it.
    "test_basic_custom_type_resolution": "dspy resolving an annotation to a type",
    "test_expected_failure": "dspy refusing an annotation it cannot resolve",
    "test_module_level_type_resolution": "dspy resolving an annotation to a type",
    "test_module_type_resolution": "dspy resolving an annotation to a type",
    "test_recommended_patterns": "dspy resolving an annotation to a type",
    "test_type_alias_for_nested_types": "dspy resolving an annotation to a type",
    # `dspy.File.format` and `dspy.Image.format` are the type's own serialisation, not an
    # adapter's: they answer what the value becomes, and an adapter later decides where it goes.
    "test_file_format_with_file_data": "dspy.File serialising itself",
    "test_file_format_with_file_id": "dspy.File serialising itself",
    # One case each of a parametrized test whose other cases do render. A PIL object is decoded
    # by dspy before any prompt exists, where a URL or a path travels into one.
    "test_image_input_formats[pil_image-PIL Image]": "dspy.Image decoding a PIL object",
    "test_image_input_formats[encoded_pil_image-encoded PIL image string]": (
        "dspy.Image decoding a PIL object"
    ),
    # `dspy.Example` behaving as a Python object: constructing, subscripting, attribute access,
    # deletion, length, equality, hashing, iteration, copying, and its string forms. The record's
    # one real decision is which fields are inputs and which are labels, and that one crosses.
    "test_example_initialization": "dspy.Example construction",
    "test_example_initialization_from_base": "dspy.Example construction",
    "test_example_initialization_from_dict": "dspy.Example construction",
    "test_example_set_get_item": "Python's mapping protocol on dspy.Example",
    "test_example_attribute_access": "Python's attribute protocol on dspy.Example",
    "test_example_deletion": "Python's mapping protocol on dspy.Example",
    "test_example_len": "Python's mapping protocol on dspy.Example",
    "test_example_get": "Python's mapping protocol on dspy.Example",
    "test_example_keys_values_items": "Python's mapping protocol on dspy.Example",
    "test_example_eq": "dspy.Example comparing itself",
    "test_example_hash": "dspy.Example hashing itself",
    "test_example_repr_str": "dspy.Example's own string form",
    "test_example_repr_str_img": "dspy.Example's own string form",
    "test_example_copy_without": "dspy.Example copying itself",
    "test_example_to_dict": "dspy.Example as a plain dict",
    "test_example_to_dict_with_history": "dspy.Example as a plain dict",
    # Recording the declaration, which stores what it was told rather than deciding anything.
    # `inputs` and `labels` are where that declaration is read, and they reach the crate.
    "test_example_with_inputs": "dspy.Example recording which fields it was asked about",
    # A signature declaration this crate has not been given a say in yet. Each raises while the
    # declaration is still being validated, before any field exists to name, so nothing reaches
    # the crate. The structural half of that validation — one arrow, and no name claimed by both
    # sides — is portable and would make the last two cross.
    "test_no_input_output": "dspy rejecting a field that is neither input nor output",
    "test_no_input_output2": "dspy rejecting a bare pydantic field",
    "test_instructions_signature": "dspy rejecting empty instructions",
    "test_empty_signature": "dspy rejecting a signature string with no arrow",
    "test_duplicate_input_output_field_names_raise": "dspy rejecting a name used on both sides",
    # --- the RLM suite, by class ---
    # `RustRLM` crosses at `forward`, which is where the loop is. Everything upstream groups here
    # is either dspy's own object under test or a piece this crate deliberately does not ship, and
    # each has its own coverage named below.
    "upstream_test_rlm.py::TestRLMInitialization": (
        "dspy.RLM's constructor reading itself back; nothing runs"
    ),
    "upstream_test_rlm.py::TestMockInterpreter": "the test's own interpreter double",
    "upstream_test_rlm.py::TestRLMCodeFenceParsing": (
        "dspy's `_strip_code_fences` called directly; the crate's is held to it by the 22-case "
        "golden in tests/conformance/predict/rlm.json"
    ),
    "upstream_test_rlm.py::TestRLMFormatting": (
        "dspy's own `_format_output` and friends, called directly"
    ),
    "upstream_test_rlm.py::TestREPLTypes": (
        "dspy's REPLVariable/REPLEntry/REPLHistory as Python objects; the crate's are held to "
        "them by tests/conformance/primitives/repl_types.json"
    ),
    "upstream_test_rlm.py::TestRLMDynamicSignature": (
        "the signatures dspy's own `__init__` built, which the shim leaves to dspy; the crate's "
        "are held to them by the four signature cases in the rlm golden"
    ),
    # Only the two `start()` tests still fall under this line: they construct `PythonInterpreter`
    # directly, and the Rust swap is scoped to the dedicated interpreter file. The other eleven
    # take the pooled interpreter — `RustPythonInterpreter` since the pool override — and are named
    # in CROSSES_DESPITE_ITS_CLASS, because they became sandbox conformance the day the pool
    # started building the Rust one. (This line used to say the sandbox is one "this crate
    # deliberately does not ship", which `deno.rs` contradicts in its first sentence: the sandbox
    # is upstream's own runner.js, vendored. The declaration outlived two facts at once.)
    "upstream_test_rlm.py::TestPythonInterpreter": (
        "only the two start() tests: they build dspy's PythonInterpreter directly, outside the pool"
    ),
    "upstream_test_rlm.py::TestRLMAsyncMock": (
        "dspy's `aforward`; the crate is async throughout, so its one `forward` is that method "
        "and the sync cases above are the same code"
    ),
    "upstream_test_rlm.py::TestBuildVariablesWithSerializable": (
        "SandboxSerializable, which is not ported (#21)"
    ),
    "upstream_test_rlm.py::TestPrepareSerializableVars": "as TestBuildVariablesWithSerializable",
    # --- the code-writing suites ---
    # `CodeAct.__init__` rejecting a callable object that is not a function. dspy's own constructor
    # validation, raised before a signature is built, let alone a prompt.
    "test_codeact_tool_validation": "dspy's CodeAct rejecting a non-function tool",
    # --- evaluate ---
    # dspy's `Evaluate` reading back its own constructor, its result object's `__repr__`, and the
    # pandas frame it builds for display. The scoring is the crate's and every other test in the
    # file crosses on it; none of these three scores anything.
    "test_evaluate_initialization": "dspy's Evaluate reading back its own constructor",
    "test_evaluation_result_repr": "dspy's own result-object repr",
    "test_construct_result_df": "dspy's pandas display frame, built from a metric that never runs",
    # --- optimizers ---
    # Each reads back an optimizer's own constructor: `Teleprompter.get_params` returns its
    # `__dict__`, and COPRO's checks the depth and breadth it was given. Nothing is proposed.
    "test_get_params": "dspy's Teleprompter reading back its own __dict__",
    "test_signature_optimizer_initialization": "COPRO reading back its own constructor",
    # --- the typed LM boundary ---
    # The crossing here is `LMMessage`, whose every construction is normalised by the crate too and
    # compared. These eleven build no message: they assert on dspy's own validators for config and
    # usage aliases, on its response and stream-builder guards, or (the image one) on a raise that
    # happens before a message exists. The crate's equivalents are held to a golden by
    # tests/lm_api_conformance.rs.
    "test_image_content_requires_mapping_with_url": "dspy raising before a message is built",
    "test_lm_kwargs_aliases_normalize_for_existing_dspy_lm_callers": "dspy's config aliases",
    "test_nested_config_aliases_remain_supported_for_existing_interfaces": "dspy's config aliases",
    "test_usage_normalizes_existing_user_visible_token_aliases": "dspy's usage aliases",
    "test_response_rejects_empty_outputs": "dspy's own response validator",
    "test_output_to_value_preserves_redacted_thinking_part": "dspy's own output dump",
    "test_stream_event_indices_must_be_non_negative": "dspy's own stream-event validator",
    "test_stream_builder_rejects_sparse_output_indices": "dspy's own stream builder",
    "test_stream_builder_rejects_sparse_part_indices": "dspy's own stream builder",
    "test_stream_builder_rejects_delta_type_changes": "dspy's own stream builder",
    "test_stream_builder_rejects_incomplete_tool_call_arguments": "dspy's own stream builder",
    # --- the cache ---
    # The crossing here is the key, and these five never compute one: three read dspy's Cache
    # constructor back, and two check its unpickling guard. `test_unserializable_key` is the
    # interesting one — the key *raises* rather than being computed, which is the behaviour under
    # test, so by construction the crate is never asked.
    "test_initialization": "dspy's Cache reading back its own constructor",
    "test_invalid_cache_initialization": "dspy's Cache constructor validation",
    "test_cache_init_with_disk_disabled_and_none_dir": "dspy's Cache constructor",
    "test_unserializable_key": "a request whose key raises before the crate is asked for one",
    "test_safe_types_rejects_non_types": "dspy's restricted-unpickling guard",
    # --- dspy's BaseModule as a Python object ---
    # These walk an *attribute graph*: a Python module holding sub-modules as attributes, found by
    # name, deep-copied, pickled. A Rust module is a trait with a `named_predictors` walk and no
    # attribute graph for any of this to reach, which is the same finding as
    # test_sandbox_serializable — a Python object protocol has no Rust surface to cross onto.
    "test_module_initialization": "dspy's Module attribute graph",
    "test_empty_module": "dspy's Module attribute graph",
    "test_predictors": "dspy's Module attribute graph",
    "test_single_level": "dspy's Module attribute graph",
    "test_multiple_levels": "dspy's Module attribute graph",
    "test_multiple_sub_modules": "dspy's Module attribute graph",
    "test_nested_named_predictors": "dspy's Module attribute graph",
    "test_non_base_module_attributes": "dspy's Module attribute graph",
    "test_complex_module_traversal": "dspy's Module attribute graph",
    "test_complex_module_traversal_with_same_module": "dspy's Module attribute graph",
    "test_complex_module_set_attribute_by_name": "dspy's Module attribute graph",
    "test_named_parameters_duplicate_references": "dspy's Module attribute graph",
    "test_deepcopy_basic": "Python's deepcopy over that graph",
    "test_deepcopy_with_nested_modules": "Python's deepcopy over that graph",
    "test_deepcopy_with_uncopyable_modules": "Python's deepcopy over that graph",
    # dspy's own save/load: its pickle mode, its version stamp, and the `Path(__file__)/resources`
    # program a past dspy wrote. The crate's state round-trip is `module.rs`'s own tests and the
    # `check_saved_program.py` gate.
    "test_save_and_load_with_json": "dspy's own state file, written and read by dspy",
    "test_save_with_extra_modules": "dspy's pickle-mode save",
    "test_load_with_version_mismatch": "dspy's version stamp on its own file",
    "test_load_dspy_program_cross_version": "a program a past dspy wrote, read by dspy",
    # The property this one tests *is* ported — `Module::load_state` refuses a state that does not
    # name every predictor, before touching any of them — but the test drives dspy's Python
    # `load_state` over a Python graph, so it cannot reach ours. `module.rs`'s
    # `a_state_that_does_not_fit_is_refused_and_changes_nothing` is our side of it.
    "test_load_state_is_transactional": "dspy's load_state over a Python module graph",
    # dspy's `__call__`-versus-`forward` warning, and its usage-tracker context manager.
    "test_forward_direct_call_warning": "dspy's own call-style warning",
    "test_forward_through_call_no_warning": "dspy's own call-style warning",
    "test_single_module_call_with_usage_tracker": "dspy's usage-tracker context manager",
    "test_multi_module_call_with_usage_tracker": "dspy's usage-tracker context manager",
    # --- the usage tracker ---
    # What two calls' counters come to together is the crate's, and five tests cross on it. These
    # two are the bookkeeping around it: appending an entry to a per-model list, and the context
    # manager that installs a tracker.
    "test_add_usage_entry": "dspy's per-model list bookkeeping",
    "test_track_usage_context_manager": "dspy's context manager installing a tracker",
    # --- ambient settings ---
    # dspy's thread-local configuration: its context manager, its refusal to be configured from a
    # child thread, what a saved settings file excludes. The crate's is a process-wide store behind
    # `configure`, with no thread-local stack for any of this to reach.
    "test_basic_dspy_settings": "dspy's thread-local settings object",
    "test_dspy_context": "dspy's settings context manager",
    "test_dspy_context_parallel": "dspy's settings context manager across threads",
    "test_dspy_configure_allowance_async": "dspy's configure-from-async guard",
    "test_forbid_configure_call_in_child_thread": "dspy's configure-from-thread guard",
    "test_dspy_settings_save_load": "dspy's settings file, written and read by dspy",
    "test_dspy_settings_save_exclude_keys": "dspy's settings file",
    "test_settings_save_with_extra_modules": "dspy's settings file",
    # --- dspy's own saving ---
    # Its pickle mode and the permission gates around unpickling. The crate saves JSON state and
    # has no pickle to gate; `module.rs` and `check_saved_program.py` hold that side.
    "test_save_predict": "dspy's own save format, written and read by dspy",
    "test_save_custom_model": "dspy's pickle-mode save",
    "test_save_model_with_custom_signature": "dspy's pickle-mode save",
    "test_pickle_loading_requires_explicit_permission": "dspy's unpickling permission gate",
    "test_pkl_file_loading_requires_explicit_permission": "dspy's unpickling permission gate",
    "test_json_file_loading_works_without_permission": "dspy's unpickling permission gate",
    # --- GEPA ---
    # Twelve of GEPA's eighteen cross, because a proposal is rendered through this crate. These six
    # do not: four check which logging flags its adapter sets on a minibatch eval, and two check
    # that dspy raises on a metric with the wrong signature or a reflection template passed the
    # wrong way — all before anything is proposed.
    "test_gepa_adapter_disables_logging_on_minibatch_eval": "dspy's own logging flags",
    "test_metric_requires_feedback_signature": "dspy raising before a proposal is made",
    "test_reflection_prompt_template_in_gepa_kwargs_raises": "dspy raising before a proposal is made",
    # --- dspy's LM ---
    # Thirty-nine of the sixty-four cross, on the one predicate that decides a request's shape:
    # which models are reasoning models, and therefore whether the generation cap travels as
    # `max_tokens` or `max_completion_tokens` and whether temperature=1.0 is demanded. These
    # nineteen are dspy's own wrapper around litellm, in three groups.
    #
    # `BaseLM` as a Python base class: its forward contract, its callback list, its shallow copy,
    # the warnings it raises when a subclass returns the wrong shape. dspy's `LM` is a wrapper over
    # litellm; the crate's implements the same contract over three provider wires and has no
    # litellm underneath to wrap, so none of this has a Rust counterpart to reach.
    "test_base_lm_copy_is_shallow_runtime_copy_with_isolated_dspy_state": "dspy's BaseLM protocol",
    "test_base_lm_errors_when_explicit_legacy_forward_returns_lm_response": "dspy's BaseLM protocol",
    "test_base_lm_forward_contract_accepts_explicit_values": "dspy's BaseLM protocol",
    "test_base_lm_forward_contract_defaults_to_legacy": "dspy's BaseLM protocol",
    "test_base_lm_forward_contract_rejects_unknown_values": "dspy's BaseLM protocol",
    "test_base_lm_init_uses_lm_defaults_and_isolates_callback_list": "dspy's BaseLM protocol",
    "test_base_lm_tracks_usage_for_custom_subclasses": "dspy's BaseLM protocol",
    "test_base_lm_validates_typed_lm_response": "dspy's BaseLM protocol",
    "test_base_lm_warns_when_inherited_legacy_forward_returns_lm_response": "dspy's BaseLM protocol",
    # A call answered by a mocked litellm: what comes back is the mock's, and nothing the crate
    # decides is on the path between the two.
    "test_chat_lms_can_be_queried": "a mocked litellm answering directly",
    "test_text_lms_can_be_queried": "a mocked litellm answering directly",
    "test_lm_calls_support_callables": "a mocked litellm answering directly",
    "test_lm_calls_support_pydantic_models": "a mocked litellm answering directly",
    "test_dspy_cache": "dspy's cache around a mocked litellm; the key itself crosses in test_cache",
    "test_streaming_passes_headers_correctly": "headers litellm is handed",
    # Asks for `litellm_test_server` and so skips here, which means it can never reach the crate —
    # unlike its three siblings, whose Responses bodies do cross.
    "test_responses_api_tool_calls": "needs upstream's litellm test server, so it skips",
}

#: Tests that reach the crate even though their class is declared above as not doing so. A class
#: upstream grouped by subject can still hold one case that drives the loop — `forward` among a
#: class of constructor checks — and a class-wide exemption must not swallow it.
CROSSES_DESPITE_ITS_CLASS = {
    "test_forward_with_serializable",
    # `TestPythonInterpreter` in the RLM file is declared as dspy's own sandbox — which is still
    # true of its two `start()` tests, but these eleven take the pooled interpreter, and the pool
    # builds `RustPythonInterpreter` now. They cross, and they are the sandbox conformance the
    # class declaration would otherwise swallow.
    #
    # (`test_forward_validates_required_inputs` was here and is not any more: the shim runs dspy's
    # own `_validate_inputs` before anything renders, so a missing input is answered in Python and
    # the test no longer reaches the crate. Its class's declaration covers it again.)
    "test_basic_execution",
    "test_variable_injection",
    "test_variable_injection_with_none_values",
    "test_tool_call_kwargs",
    "test_tool_call_positional",
    "test_multiple_tools",
    "test_tool_returns_list",
    "test_tool_returns_dict",
    "test_state_persists",
    "test_syntax_error",
    "test_runtime_error",
    # `TestSandboxSecurity` and `TestLargeSerializableRoundTrip` sit under the same class-key
    # mechanism: both classes are pool-only, so *every* test in them crosses now, and their class
    # declarations are gone rather than excepted line by line.
    "test_imports_work",
    "test_no_network_access",
    "test_large_payload_round_trips_through_real_sandbox",
}

# Whole files that test dspy's own Python rather than anything an adapter renders: a type's
# string form, a tool invoking a Python function, a value validating itself. Every test in one
# of these was measured as never reaching the crate. The check runs both ways — a test here
# that *does* cross fails the run, because that means the file has started covering this port
# and each of its tests deserves triaging rather than a blanket pass.
SIGNATURE_CONFORMANCE = {
    "upstream_test_signature.py": "how a signature is built, named and described",
    "upstream_test_example.py": "how a record splits into what was asked and what was expected",
    "upstream_test_aggregation.py": "which of several answers a vote elects",
}

# Whole files whose subject is beneath the wire: a signature's own construction, naming and
# description. Their tests are held to reaching the signature layer rather than to rendering,
# because nothing they assert on is a message a model would read.
NOT_ADAPTER_CONFORMANCE = {
    "upstream_test_adapter_utils.py": "dspy's own field-formatting helpers, called directly",
    "upstream_test_base_type.py": "dspy.Type's annotation walking, in Python",
    "upstream_test_code.py": "dspy.Code's own parsing and string form",
    "upstream_test_document.py": "dspy.Document's own validation and string form",
    "upstream_test_audio.py": "dspy.Audio's own decoding and format detection",
    "upstream_test_tool.py": "dspy.Tool invoking Python functions, sync and async",
    # A Python ABC, its `__get_pydantic_core_schema__` hook, and `build_repl_variable` over a
    # Python instance. The crate's counterpart is a Rust trait a Rust caller implements, so a
    # Python subclass has no surface to cross on; `interpreter/sandbox.rs` holds its own tests.
    # The file is here rather than absent so that a bridge which *did* cross would fail this
    # declaration rather than pass unnoticed.
    # Every test here asserts on the ABC itself: that an incomplete subclass cannot be instantiated,
    # that a duck-typed class fails `isinstance`, that the pydantic hook passes through. A Rust trait
    # has no runtime `isinstance` and no pydantic. What *is* portable is what `build_repl_variable`
    # decides, and `tests/sandbox_serializable_conformance.rs` holds this crate to a golden generated
    # by running it — so this file being not-crossing is not the same as the protocol being unchecked.
    "upstream_test_sandbox_serializable.py": (
        "dspy's SandboxSerializable ABC and its pydantic hook, in Python"
    ),
    # Every test here drives dspy's BetterTogether over *mock* optimizers and asserts on the
    # orchestration — strategy order, which candidate is returned with and without a valset, how
    # compile args reach each step. Nothing is proposed, so nothing renders.
    #
    # This is the routing rule holding: an optimizer-shaped primitive goes to a golden, because
    # both directions of an optimizer bridge execute the wrong side — patching dspy's class to call
    # the crate has our optimizer driving dspy's modules, and leaving it has dspy's optimizer
    # driving our adapter. Our side is `optimize/better_together.rs`'s own tests.
    "upstream_test_bettertogether.py": (
        "dspy's BetterTogether orchestration over mock optimizers, in Python"
    ),
    # Ensemble over mock programs returning canned dicts: which members were asked, what the
    # reduction did with them, and that `deterministic=True` is refused. Nothing renders, for the
    # same reason BetterTogether's suite does not — an optimizer-shaped primitive is a golden's
    # job. `optimize/ensemble.rs` holds our side, including the per-call draw dspy leaves to a
    # process-wide RNG.
    "upstream_test_ensemble.py": "dspy's Ensemble over mock programs, in Python",
    # dspy's ThreadPoolExecutor plumbing: worker independence, which thread a sequential run uses,
    # how many errors it tolerates. The crate's `Parallel` is Rust futures over its own executor,
    # so none of this has a shape to cross onto — and none of it renders.
    "upstream_test_parallelizer.py": "dspy's thread-pool plumbing, in Python",
}


#: Names of tests that reached the crate, for the summary line. A bare pass count would read
#: as coverage this suite does not claim, since most of the type files never cross.
_CROSSED: set[str] = set()

#: Names of tests that reached the signature layer, counted apart for the reason above.
_REACHED_SIGNATURE: set[str] = set()


@pytest.fixture
def litellm_test_server():
    """A stand-in that skips, because this harness does not run upstream's litellm server.

    The runner empties `tests/conftest.py` so importing anything under `tests/` cannot start that
    server, which takes this fixture with it — and a test asking for it *errors* rather than
    skipping, reading as a failure this port caused. These tests answer from a real litellm talking
    to a local server, so nothing the crate decides is on the path either way.
    """
    pytest.skip("this harness does not run upstream's litellm test server")


@pytest.fixture
def lm_for_test():
    """Upstream's own fixture, reproduced because its conftest is blanked here.

    The runner empties `tests/conftest.py` so importing anything under `tests/` cannot drag in the
    litellm test server. That takes this with it, and a test asking for it errors instead of
    skipping — which reads as a failure this port caused. Upstream skips without `LM_FOR_TEST`, and
    so does this.
    """
    model = os.environ.get("LM_FOR_TEST")
    if model is None:
        pytest.skip("LM_FOR_TEST is not set in the environment variables")
    return model


@pytest.fixture(autouse=True)
def _require_a_crossing(request):
    """Fail a test that passed without the crate doing anything.

    A test can construct dspy's own adapter, or assert on a Python type directly, and never
    reach Rust. It then passes for reasons this crate has no part in, which is the one way a
    conformance suite can lie about its coverage.
    """
    # Attributed rather than sampled: a bare before/after read of the globals credited a crossing
    # made on a background thread — diskcache's fanout writers, say — to whichever test happened to
    # be running when it landed, which made this guard flaky in both directions.
    crossings.begin(request.node.nodeid)
    yield
    rendered, signature_reached = crossings.end()
    crossed = rendered > 0
    if crossed:
        _CROSSED.add(request.node.nodeid)
    if signature_reached > 0:
        _REACHED_SIGNATURE.add(request.node.nodeid)
    module = request.node.module.__name__ + ".py"
    case = request.node.name
    name = case.split("[")[0]
    # A name can repeat across files — `test_initialization_with_string_signature` is in both
    # the predict and chain-of-thought suites, and only one of them stays in Python — so a
    # declaration may name its file to say which it means. It may also name one case of a
    # parametrized test, because whether a case reaches the crate can differ per case: an image
    # given as a URL renders, and the same test given a PIL object does not.
    # A whole test *class* may be dspy's own Python — upstream groups by subject, so `TestREPLTypes`
    # is every case for a type this crate does not own. Naming the class is as specific as naming
    # each of its tests and reads as one decision rather than twenty identical ones.
    cls = getattr(request.node, "cls", None)
    class_keys = (
        ()
        if cls is None or name in CROSSES_DESPITE_ITS_CLASS
        else (f"{module}::{cls.__name__}", cls.__name__)
    )
    declared = any(
        key in DOES_NOT_EXERCISE_RUST
        for key in (f"{module}::{case}", case,
                    f"{module}::{name}", f"{module}::{name.removesuffix('_async')}",
                    name, name.removesuffix("_async"), *class_keys)
    )
    if module in SIGNATURE_CONFORMANCE:
        reached = signature_reached > 0
        # Both ways, as everywhere else here: a declaration that has started reaching the crate
        # is a claim that is no longer true.
        if reached and declared:
            pytest.fail(
                "this test is declared as not exercising the crate, but it decided a signature "
                "through it; drop its line from DOES_NOT_EXERCISE_RUST"
            )
        if not reached and not declared:
            pytest.fail(
                "this test passed without the crate deciding anything about the signature, so "
                "it says nothing about conformance; give it a line in DOES_NOT_EXERCISE_RUST "
                "if that is expected"
            )
        return
    if module in NOT_ADAPTER_CONFORMANCE:
        if crossed:
            pytest.fail(
                f"{module} is declared as not covering this port, but this test reached the "
                "crate; drop the file's line and triage its tests individually"
            )
        return
    # Both ways, as for the whole-file list above: a declared test that starts crossing means
    # the port grew to cover it, and its line is now a claim that is no longer true.
    if crossed:
        if declared:
            pytest.fail(
                "this test is declared as not exercising the crate, but it reached it; "
                "drop its line from DOES_NOT_EXERCISE_RUST"
            )
        return
    if declared:
        return
    pytest.fail(
        "this test passed without the crate rendering or parsing anything, so it says nothing "
        "about conformance; give it a line in DOES_NOT_EXERCISE_RUST if that is expected"
    )


@pytest.fixture(autouse=True)
def _default_adapter_is_rust():
    """Make the adapter dspy reaches for by default the Rust-backed one.

    `dspy.Predict` resolves `settings.adapter or ChatAdapter()`, where that name was bound when
    its module was imported — long before any patch here. Rebinding the attribute in every
    module that imported it would be a game of catch-up, and missing one means a test runs on
    dspy's own renderer while reading as conformance. `settings.adapter` is the seam dspy
    provides for supplying an adapter, so this uses that.
    """
    previous = dspy.settings.adapter
    dspy.settings.configure(adapter=RustChatAdapter())
    yield
    dspy.settings.configure(adapter=previous)


@pytest.fixture(autouse=True)
def _clear_settings():
    """Reset dspy's settings after each test, as upstream's own root conftest does.

    That conftest is not used here — it imports a test server this harness does not run — so
    the isolation it provides has to be reproduced. Without it dspy's global settings carry
    from one upstream test into the next, and a test asserting on accumulated state reads
    another test's leftovers rather than its own: `test_trace_size_limit` saw 24 entries where
    it wrote 3. A conformance run that fails for that reason says nothing about this crate.
    """
    yield
    import copy

    from dspy.dsp.utils.settings import DEFAULT_CONFIG

    dspy.configure(**copy.deepcopy(DEFAULT_CONFIG), inherit_config=False)


@pytest.fixture(autouse=True)
def _use_rust_adapter(monkeypatch):
    monkeypatch.setattr(dspy, "ChatAdapter", RustChatAdapter)
    monkeypatch.setattr("dspy.adapters.ChatAdapter", RustChatAdapter, raising=False)
    monkeypatch.setattr("dspy.adapters.chat_adapter.ChatAdapter", RustChatAdapter, raising=False)
    # `dspy.JSONAdapter` is what upstream's tests construct. The defining module keeps the real
    # class: the chat adapter's fallback re-asks through that one, and upstream's fallback tests
    # mock it there — patching it would test the mock against a class nothing returns.
    monkeypatch.setattr(dspy, "JSONAdapter", RustJSONAdapter)
    monkeypatch.setattr("dspy.adapters.JSONAdapter", RustJSONAdapter, raising=False)
    monkeypatch.setattr(dspy, "XMLAdapter", RustXMLAdapter, raising=False)
    monkeypatch.setattr("dspy.adapters.XMLAdapter", RustXMLAdapter, raising=False)
    monkeypatch.setattr("dspy.adapters.xml_adapter.XMLAdapter", RustXMLAdapter, raising=False)
    monkeypatch.setattr(dspy, "TwoStepAdapter", RustTwoStepAdapter, raising=False)
    monkeypatch.setattr("dspy.adapters.TwoStepAdapter", RustTwoStepAdapter, raising=False)


@pytest.fixture(autouse=True)
def _use_rust_predict(request, monkeypatch):
    """For the predict suite, make `dspy.Predict` this crate's `Predict`, so those tests exercise
    our module's orchestration — render, call, parse — not dspy's over our adapter.

    Scoped to that one file: `dspy.Predict` is constructed inside `ChainOfThought`, `ReAct` and the
    optimizers, so a blanket swap would route every module suite through this at once. Each of those
    crosses under its own beachhead instead. The name is rebound in the test module too, since it
    did `from dspy import Predict` and holds its own reference.
    """
    if request.node.module.__name__ != "upstream_test_predict":
        return
    monkeypatch.setattr(request.node.module, "Predict", RustPredict, raising=False)
    monkeypatch.setattr(dspy, "Predict", RustPredict)
    monkeypatch.setattr("dspy.predict.predict.Predict", RustPredict, raising=False)


@pytest.fixture(autouse=True)
def _use_rust_react(request, monkeypatch):
    """For the react suite, make `dspy.ReAct` this crate's `ReAct`, so the loop, tool calls and
    extraction run in Rust. Scoped to that file, like the predict swap."""
    if request.node.module.__name__ != "upstream_test_react":
        return
    # A few react tests reach past what the module crossing can carry, so they keep running dspy's
    # ReAct (over the Rust adapter) rather than ours:
    #   - the first two mock ReAct's own predictors to test dspy's loop (truncation, the
    #     context-window retry) — nothing renders, and they are declared non-crossing;
    #   - the last spies on the Python adapter's `format_user_message_content` and needs live PIL
    #     images preserved through the loop, both of which the Rust loop bypasses (it renders in
    #     Rust and cannot hold a Python object). It stays an adapter-level multimodal test.
    if request.node.name.split("[")[0] in {
        "test_trajectory_truncation",
        "test_context_window_exceeded_after_retries",
        "test_tool_observation_preserves_custom_type",
    }:
        return
    monkeypatch.setattr(request.node.module, "ReAct", RustReAct, raising=False)
    monkeypatch.setattr(dspy, "ReAct", RustReAct)
    monkeypatch.setattr("dspy.predict.react.ReAct", RustReAct, raising=False)


@pytest.fixture(autouse=True)
def _use_rust_rlm(request, monkeypatch):
    """For the rlm suite, make `dspy.RLM` this crate's `Rlm`, so the REPL loop runs in Rust.

    This is the one beachhead that reaches RLM's control flow. Its prompts are covered by goldens —
    the fence parser, both signatures, the REPL types — and none of those says which turn ends the
    run or what an incomplete submission is answered with. Upstream's own tests do, so they run it.
    """
    if request.node.module.__name__ != "upstream_test_rlm":
        return
    monkeypatch.setattr(request.node.module, "RLM", RustRLM, raising=False)
    monkeypatch.setattr(dspy, "RLM", RustRLM)
    monkeypatch.setattr("dspy.predict.rlm.RLM", RustRLM, raising=False)


@pytest.fixture(autouse=True)
def _use_rust_code_modules(request, monkeypatch):
    """For the two code-writing suites, make dspy's module this crate's.

    Both files are `@pytest.mark.deno`: a real sandbox runs what the model wrote, so these assert
    the loop end to end — parse, execute, and either answer or rewrite — which is the layer the
    prompt goldens cannot reach.
    """
    swaps = {
        "upstream_test_program_of_thought": ("ProgramOfThought", RustProgramOfThought,
                                             "dspy.predict.program_of_thought.ProgramOfThought"),
        "upstream_test_code_act": ("CodeAct", RustCodeAct, "dspy.predict.code_act.CodeAct"),
    }
    swap = swaps.get(request.node.module.__name__)
    if swap is None:
        return
    name, rust, path = swap
    monkeypatch.setattr(request.node.module, name, rust, raising=False)
    monkeypatch.setattr(dspy, name, rust)
    monkeypatch.setattr(path, rust, raising=False)
    monkeypatch.setattr(f"dspy.predict.{name}", rust, raising=False)


@pytest.fixture(autouse=True)
def _rust_python_interpreter(monkeypatch, request):
    """dspy's own interpreter suite, driving this crate's sandbox.

    Scoped to that one file. A blanket swap would put the Rust sandbox under every module suite at
    once, and those already run against dspy's interpreter on purpose — that is what proves the
    modules, while this proves the sandbox.
    """
    if request.node.module.__name__ != "upstream_test_python_interpreter":
        return
    monkeypatch.setattr(request.node.module, "PythonInterpreter", RustPythonInterpreter)
    monkeypatch.setattr(
        "dspy.primitives.python_interpreter.PythonInterpreter", RustPythonInterpreter
    )


def _rust_answer_exact_match(example, pred, trace=None, frac=1.0):
    """dspy's `answer_exact_match`, decided by the crate.

    Reading the gold answer off a `dspy.Example` and the answer off a `Prediction` is reflection
    over dspy's own objects, so it stays Python. What a *match* is — normalisation, the article and
    punctuation rules, the best score across several gold answers — is the crate's.
    """
    assert not isinstance(example.answer, str) or frac >= 1.0
    answers = example.answer if isinstance(example.answer, list) else [example.answer]
    crossings.record_render()
    return bool(dsrs_bridge.answer_exact_match([str(a) for a in answers], str(pred.answer)))


# dspy names a result column after the metric's `__name__`, so the stand-in carries upstream's
# rather than its own — `test_construct_result_df` compares that column by name.
_rust_answer_exact_match.__name__ = "answer_exact_match"


def _rust_cache_key(self, request, ignored_args_for_cache_key=None):
    """dspy's `Cache.cache_key`, decided by the crate.

    Upstream transforms pydantic values to their schema before hashing, which is reflection and
    stays Python; the rule that turns the transformed request into a key — every field, sorted, one
    sha256 — is the crate's. It is the rule that decides whether two calls are the same call, and
    therefore whether one is answered with the other's reply.
    """
    from dspy.clients.cache import _transform_value

    import dataclasses

    def orjson_default(value):
        """What upstream's orjson serializes that the stdlib's json does not.

        A dataclass in the request is the one that matters here: orjson dumps it natively, so a
        request carrying one is cacheable upstream. Anything else still raises, because
        `get`/`put` catch that and treat the request as uncacheable — which the suite checks.
        """
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return dataclasses.asdict(value)
        raise TypeError(f"not JSON-serializable: {type(value).__name__}")

    ignored = ignored_args_for_cache_key or []
    params = {k: _transform_value(v) for k, v in request.items() if k not in ignored}
    written = json.dumps(params, default=orjson_default)
    crossings.record_render()
    return dsrs_bridge.cache_key(written)


def _rust_is_openai_reasoning_model(model: str) -> bool:
    """dspy's `_is_openai_reasoning_model`, decided by the crate.

    It is one predicate and it decides two things a request carries: whether the generation cap
    travels as `max_tokens` or `max_completion_tokens`, and whether `temperature=1.0` and a 16k
    floor are demanded. dspy spells it as a regex; the crate reads the family off the name.
    """
    crossings.record_render()
    return dsrs_bridge.is_openai_reasoning_model(model)


@pytest.fixture(autouse=True)
def _usage_merging_is_rust(request, monkeypatch):
    """What two calls' counters come to together is the crate's answer.

    This is the arithmetic a program's reported spend is built from, and the place it goes wrong
    quietly: a nested breakdown (`prompt_tokens_details.cached_tokens`) or a counter nobody has
    modelled has to *add* across calls, not be replaced by the latest.
    """
    if request.node.module.__name__ != "upstream_test_usage_tracker":
        return
    from dspy.utils.usage_tracker import UsageTracker

    def merged(self, left, right):
        crossings.record_render()
        answered = json.loads(
            dsrs_bridge.merge_usage(json.dumps(left or {}), json.dumps(right or {}))
        )
        # The crate fills both spellings of a counter it knows under two names; upstream carries
        # only what a provider reported, so a name neither side sent is dropped again.
        seen = set(left or {}) | set(right or {})
        return {k: v for k, v in answered.items() if k in seen}

    monkeypatch.setattr(UsageTracker, "_merge_usage_entries", merged)


@pytest.fixture(autouse=True)
def _responses_body_is_rust(request, monkeypatch):
    """dspy's `_convert_chat_request_to_responses_request`, answered by the crate.

    The two get here by different routes: dspy rewrites a chat dict in place, and the crate goes
    typed request -> body, which is dspy's *other* route (`openai_format.responses_request`). The
    conformance question is not the route but the wire — the same conversation, taken the crate's
    way, has to reach the same body.

    So the crate answers for what it builds — the input list, the text format, the tools — and
    everything else is carried over from dspy's own rewrite, those being litellm passthrough keys
    rather than part of what either one builds.
    """
    if request.node.module.__name__ != "upstream_test_lm":
        return
    from dspy.clients import lm as dspy_lm

    # Captured before the patch: looking it up inside would find the replacement and recurse.
    original = dspy_lm._convert_chat_request_to_responses_request

    def answered(chat_request):
        theirs = original(chat_request)
        crossings.record_render()
        ours = json.loads(dsrs_bridge.responses_body(json.dumps(chat_request, default=str)))
        # `input` and `tools` only. The two routes agree on those and *do not* agree on `text`:
        # this legacy converter names the format after the pydantic class it was handed, where the
        # typed route the crate follows emits `{"type": "json_schema", "name": "response", …,
        # "strict": true}`. Ours is held to the typed shape by tests/lm_api_conformance.rs, so
        # forcing it here would assert the wrong one of dspy's two answers.
        return {**theirs, **{k: v for k, v in ours.items() if k in ("input", "tools")}}

    monkeypatch.setattr(dspy_lm, "_convert_chat_request_to_responses_request", answered)


@pytest.fixture(autouse=True)
def _typed_responses_mapping_is_rust(request, monkeypatch):
    """dspy's *typed* Responses mapping, both directions, answered by the crate.

    The fixture above patches the legacy chat-dict converter, which is one of dspy's two routes to
    a Responses body. Tests that call `to_openai_responses_request(LMRequest.from_call(...))`
    directly take the other one and never touch it, so the tools, the tool choice, the reasoning
    config and the schema envelope were all being asserted against dspy's own answer with the crate
    absent — and `responses_to_lm_response` likewise. Reading one of those tests is what found the
    crate recording `raw_arguments` for an unparseable tool call without the
    `arguments_parse_error` upstream records beside it.

    Whole-object round trips: dspy's request model in, dspy's response model out, so the assertions
    in between are on what the crate built.
    """
    if request.node.module.__name__ != "upstream_test_lm":
        return
    from dspy.clients import openai_format
    from dspy.core.types import LMResponse

    original_body = openai_format.to_openai_responses_request

    def body_of(lm_request, **kwargs):
        theirs = original_body(lm_request, **kwargs)
        crossings.record_render()
        # `text` stays dspy's, and only `text`. Upstream names the schema envelope after the
        # *pydantic class* it was handed; the crate holds a schema and has no class to name it
        # after, so it writes `response` — the same split `_responses_body_is_rust` makes for the
        # same reason. Everything else in the body is the crate's answer, which is what the tools,
        # the tool choice, the reasoning config and `max_output_tokens` are asserted against.
        without_class = lm_request.model_copy(
            update={"config": lm_request.config.model_copy(update={"response_format": None})}
        )
        dumped = without_class.model_dump(mode="json", exclude_none=True)
        ours = json.loads(dsrs_bridge.responses_request(json.dumps(dumped, default=str)))
        return {**ours, **{key: theirs[key] for key in ("text",) if key in theirs}}

    def outputs_of(response, lm_request):
        crossings.record_render()
        raw = response if isinstance(response, dict) else openai_format.model_dump(response)
        answered = dsrs_bridge.responses_outputs(
            json.dumps(raw, default=str), lm_request.model or ""
        )
        return LMResponse.model_validate(json.loads(answered))

    monkeypatch.setattr(openai_format, "to_openai_responses_request", body_of)
    monkeypatch.setattr(openai_format, "responses_to_lm_response", outputs_of)


@pytest.fixture(autouse=True)
def _closing_a_schema_is_rust(request, monkeypatch):
    """dspy's `_close_object_schemas`, applied by the crate.

    The Responses API refuses an object schema that leaves `additionalProperties` unspecified, and
    which positions in a schema are *subschemas* to walk — as against a schema-shaped value sitting
    in a `default` — is the whole of the rule. The crate had no such walk at all: a nullable field's
    object branch, reached through `anyOf`, went out open.

    The class-to-schema step stays Python's, being pydantic reflection.
    """
    if request.node.module.__name__ not in ("upstream_test_lm", "upstream_test_types"):
        return
    import pydantic

    from dspy.clients import openai_format

    def format_of(value):
        if not (isinstance(value, type) and issubclass(value, pydantic.BaseModel)):
            return value
        crossings.record_render()
        closed = dsrs_bridge.closed_object_schemas(json.dumps(value.model_json_schema()))
        return {"name": value.__name__, "type": "json_schema", "schema": json.loads(closed)}

    monkeypatch.setattr(openai_format, "response_format_to_responses", format_of)


@pytest.fixture(autouse=True)
def _reasoning_families_are_rust(request, monkeypatch):
    """For the LM suite, which models count as reasoning models is the crate's answer."""
    if request.node.module.__name__ != "upstream_test_lm":
        return
    monkeypatch.setattr(
        "dspy.clients.lm._is_openai_reasoning_model", _rust_is_openai_reasoning_model
    )


@pytest.fixture(autouse=True)
def _cache_keys_are_rust(request, monkeypatch):
    """For the cache suite, the key is the crate's."""
    if request.node.module.__name__ != "upstream_test_cache":
        return
    from dspy.clients.cache import Cache

    monkeypatch.setattr(Cache, "cache_key", _rust_cache_key)


@pytest.fixture(autouse=True)
def _messages_normalize_through_rust(request, monkeypatch):
    """Every `LMMessage` the types suite builds is normalised by the crate too, and the two answers
    must agree.

    dspy's `LMMessage` accepts either the typed shape or the one a provider writes and normalises
    the second into the first. The crate does the same. Rather than replace pydantic's model — it is
    what the rest of dspy validates against — each construction is *also* handed to the crate, and a
    disagreement fails the test. So the suite asserts on dspy's object while holding ours to it.
    """
    if request.node.module.__name__ != "upstream_test_types":
        return

    from dspy.core import types as dspy_types

    original = dspy_types.LMMessage.__init__

    def checked(self, **data):
        original(self, **data)
        crossings.record_render()
        # A test may hand over real part objects rather than dicts, and `str()` on one is its
        # repr — which the crate would read as a bare string. Dump them the way pydantic does.
        def jsonable(value):
            dump = getattr(value, "model_dump", None)
            # With defaults, since a part's `type` is a Literal default and dropping it leaves the
            # crate's internally-tagged enum nothing to dispatch on.
            return dump(mode="json") if dump else str(value)

        ours = json.loads(dsrs_bridge.normalize_message(json.dumps(data, default=jsonable)))
        # Compared key by key against what dspy *kept*: each side elides its own defaults, so a
        # whole-dict equality would fail on `type` (dspy's Literal default) rather than on any
        # disagreement. Every key dspy holds, the crate must hold with the same value — which is
        # what caught an audio block losing its url.
        mine, theirs = ours.get("parts", []), self.model_dump(exclude_defaults=True)["parts"]
        assert len(mine) == len(theirs), (
            f"the crate read a different number of parts:\n  written: {data}\n"
            f"  dspy:  {theirs}\n  crate: {mine}"
        )
        for got, want in zip(mine, theirs):
            for key, value in want.items():
                assert got.get(key) == value, (
                    f"the crate normalised `{key}` differently:\n  written: {data}\n"
                    f"  dspy:  {want}\n  crate: {got}"
                )

    monkeypatch.setattr(dspy_types.LMMessage, "__init__", checked)


@pytest.fixture(autouse=True)
def _metrics_are_rust(request, monkeypatch):
    """`answer_exact_match` is the crate's, wherever a test reached for it.

    dspy exports it from three places and a test file holds its own reference, so all of them are
    rebound — the same reason the adapter swap rebinds the test module's name.
    """
    for path in (
        "dspy.evaluate.metrics.answer_exact_match",
        "dspy.evaluate.answer_exact_match",
        "dspy.answer_exact_match",
    ):
        monkeypatch.setattr(path, _rust_answer_exact_match, raising=False)
    # A test file that did `from dspy.evaluate.metrics import answer_exact_match` holds its own
    # reference, bound when pytest imported it, so the name is rebound there too.
    monkeypatch.setattr(
        request.node.module, "answer_exact_match", _rust_answer_exact_match, raising=False
    )


@pytest.fixture(autouse=True)
def _signature_layer_is_rust(monkeypatch):
    """Route the signature decisions this crate owns through it.

    `infer_prefix` runs where a signature class is built, so patching the name the defining
    module reads is what puts the crate on that path rather than only where a test calls it
    directly.
    """
    monkeypatch.setattr(
        "dspy.signatures.signature.infer_prefix", rust_signature.infer_prefix
    )
    monkeypatch.setattr("dspy.signatures.infer_prefix", rust_signature.infer_prefix, raising=False)
    # `Example.inputs`/`labels` are the record's one decision, so they answer from the crate too.
    monkeypatch.setattr(dspy.Example, "inputs", rust_signature.inputs)
    monkeypatch.setattr(dspy.Example, "labels", rust_signature.labels)
    monkeypatch.setattr("dspy.predict.aggregation.majority", rust_signature.majority)
    monkeypatch.setattr(dspy, "majority", rust_signature.majority, raising=False)


@pytest.fixture(autouse=True)
def _rebind_in_the_test_module(monkeypatch, request):
    """Point a test file's own imported name at the Rust-backed one.

    `from dspy.adapters.xml_adapter import XMLAdapter` binds the class into the test module when
    pytest imports it, before any fixture runs. Patching dspy's module afterwards leaves that
    reference untouched, so the test would construct dspy's own adapter and pass without this
    crate doing anything. The same applies to a function imported by name.
    """
    module = request.node.module
    for name, backed in RUST_BACKED.items():
        if hasattr(module, name):
            monkeypatch.setattr(module, name, backed)
    if hasattr(module, "infer_prefix"):
        monkeypatch.setattr(module, "infer_prefix", rust_signature.infer_prefix)
    if hasattr(module, "majority"):
        monkeypatch.setattr(module, "majority", rust_signature.majority)


def pytest_configure(config):
    """Upstream marks async tests with `@pytest.mark.asyncio`; honour it without editing them."""
    config.option.asyncio_mode = "auto"


def pytest_collection_modifyitems(items):
    for item in items:
        # Async variants share the sync name plus a suffix, and share the same gap.
        # A parametrized test's name carries its case in brackets; the gap is per function.
        base = item.name.split("[")[0].removesuffix("_async")
        reason = NOT_YET_IMPLEMENTED.get(item.name) or NOT_YET_IMPLEMENTED.get(base)
        if reason:
            item.add_marker(pytest.mark.xfail(strict=True, reason=f"not in Rust yet: {reason}"))


def pytest_terminal_summary(terminalreporter):
    """State how much of the run actually exercised the crate.

    A pass count alone would overstate it: the type files are carried here to catch one of them
    starting to cross, not because they test this port. The two lines stay apart for the same
    reason — almost any signature construction reaches the layer beneath the wire, so folding
    that into the first number would report coverage no assertion backs.
    """
    outcomes = ("passed", "failed", "error", "xfailed", "xpassed")
    total = sum(len(terminalreporter.stats.get(key, [])) for key in outcomes)
    terminalreporter.write_sep(
        "-", f"{len(_CROSSED)} of {total} tests rendered or parsed through the crate"
    )
    terminalreporter.write_sep(
        "-", f"{len(_REACHED_SIGNATURE)} of {total} tests decided a signature through the crate"
    )
