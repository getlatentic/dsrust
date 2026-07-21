"""Point upstream's test suite at the Rust-backed adapter, and be honest about the gaps.

Rust renders and parses every case here. A reply Rust rejects may re-ask through Python's
JSONAdapter, since that is dspy's own behaviour and upstream tests it directly; a case Rust
has not implemented may not, and raises instead. Those cases are listed below and marked
`xfail(strict=True)`, which means two things: they are never counted as passes, and if one
starts passing the run FAILS until its name is deleted from the list. The list is therefore
the to-do list, it cannot drift out of date, and a green run means every case not named here
genuinely runs on Rust.
"""

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

# Upstream tests whose features this crate has not written yet, with the reason. Delete a line
# once Rust renders that case; the strict xfail will fail the run if you forget.
#
# A run may report xfails this list is empty of: dspy marks two of its own image cases xfail
# inside the test body, for a gap upstream has rather than one this port has.
NOT_YET_IMPLEMENTED = {
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
    # Calls dspy's private `_call_postprocess` with outputs already parsed, so it exercises
    # dspy's own plumbing around an adapter rather than anything the adapter renders.
    "test_tool_call_with_null_content_does_not_raise": "dspy-internal postprocessing",
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
    # A signature declaration this crate has not been given a say in yet. Each raises while the
    # declaration is still being validated, before any field exists to name, so nothing reaches
    # the crate. The structural half of that validation — one arrow, and no name claimed by both
    # sides — is portable and would make the last two cross.
    "test_no_input_output": "dspy rejecting a field that is neither input nor output",
    "test_no_input_output2": "dspy rejecting a bare pydantic field",
    "test_instructions_signature": "dspy rejecting empty instructions",
    "test_empty_signature": "dspy rejecting a signature string with no arrow",
    "test_duplicate_input_output_field_names_raise": "dspy rejecting a name used on both sides",
}

# Whole files that test dspy's own Python rather than anything an adapter renders: a type's
# string form, a tool invoking a Python function, a value validating itself. Every test in one
# of these was measured as never reaching the crate. The check runs both ways — a test here
# that *does* cross fails the run, because that means the file has started covering this port
# and each of its tests deserves triaging rather than a blanket pass.
SIGNATURE_CONFORMANCE = {
    "upstream_test_signature.py": "how a signature is built, named and described",
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
    declared = any(
        key in DOES_NOT_EXERCISE_RUST
        for key in (f"{module}::{case}", case,
                    f"{module}::{name}", f"{module}::{name.removesuffix('_async')}",
                    name, name.removesuffix("_async"))
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
