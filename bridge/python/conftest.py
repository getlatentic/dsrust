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

import dspy
import pytest

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
from rust_module import RustPredict, RustReAct, RustRLM  # noqa: E402
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
    # dspy.LM's predicted-outputs feature: a `prediction` kwarg passed straight through to
    # litellm's `completion`. The crate's typed `LmConfig` does not model it, so `RustPredict`
    # renders and calls but never forwards it. The other half of the test — `prediction` as an
    # ordinary input field, which must NOT reach the LM — does cross correctly.
    "test_predicted_outputs_piped_from_predict_to_lm_call": (
        "dspy.LM's predicted-outputs `prediction` kwarg passthrough to litellm"
    ),
    # `RLM._process_final_output` validates each submitted value against its output field's
    # annotation and feeds a `[Type Error] …` back so the model can submit again. The crate's
    # `Rlm::submitted` checks the shape and the field names and stops there, so a wrongly-typed
    # submission is accepted where upstream would retry. See #22 — this is the divergence the RLM
    # beachhead was built to find, and it is the whole reason these two are here.
    "test_type_error_retries": "RLM does not yet type-check a submission and retry (#22)",
    "test_with_input_variables_e2e": "as test_type_error_retries (#22)",
}


# Upstream tests that pass without the crate rendering or parsing anything, with the reason.
# They are not conformance: they exercise dspy's own Python — a type's `__str__`, a helper — and
# would read as green whatever this crate did. Naming them keeps the passing count honest, and
# anything not named here must cross into Rust or the run fails.
DOES_NOT_EXERCISE_RUST = {
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
    "upstream_test_rlm.py::TestPythonInterpreter": (
        "upstream's Deno/Pyodide sandbox, which this crate deliberately does not ship — the "
        "interpreter is the caller's"
    ),
    "upstream_test_rlm.py::TestSandboxSecurity": "as TestPythonInterpreter",
    "upstream_test_rlm.py::TestLargeSerializableRoundTrip": "as TestPythonInterpreter",
    "upstream_test_rlm.py::TestRLMAsyncMock": (
        "dspy's `aforward`; the crate is async throughout, so its one `forward` is that method "
        "and the sync cases above are the same code"
    ),
    "upstream_test_rlm.py::TestBuildVariablesWithSerializable": (
        "SandboxSerializable, which is not ported (#21)"
    ),
    "upstream_test_rlm.py::TestPrepareSerializableVars": "as TestBuildVariablesWithSerializable",
}

#: Tests that reach the crate even though their class is declared above as not doing so. A class
#: upstream grouped by subject can still hold one case that drives the loop — `forward` among a
#: class of constructor checks — and a class-wide exemption must not swallow it.
CROSSES_DESPITE_ITS_CLASS = {
    "test_forward_validates_required_inputs",
    "test_forward_with_serializable",
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
}


#: Names of tests that reached the crate, for the summary line. A bare pass count would read
#: as coverage this suite does not claim, since most of the type files never cross.
_CROSSED: set[str] = set()

#: Names of tests that reached the signature layer, counted apart for the reason above.
_REACHED_SIGNATURE: set[str] = set()


@pytest.fixture(autouse=True)
def _require_a_crossing(request):
    """Fail a test that passed without the crate doing anything.

    A test can construct dspy's own adapter, or assert on a Python type directly, and never
    reach Rust. It then passes for reasons this crate has no part in, which is the one way a
    conformance suite can lie about its coverage.
    """
    before = crossings.RENDERED
    before_signature = crossings.SIGNATURE
    yield
    crossed = crossings.RENDERED > before
    if crossed:
        _CROSSED.add(request.node.nodeid)
    if crossings.SIGNATURE > before_signature:
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
        reached = crossings.SIGNATURE > before_signature
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
