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

from rust_adapter import RustChatAdapter, RustJSONAdapter  # noqa: E402

# Upstream tests whose features this crate has not written yet, with the reason. Delete a line
# once Rust renders that case; the strict xfail will fail the run if you forget.
NOT_YET_IMPLEMENTED = {
    # The crate's JsonAdapter renders a one-line system message and ignores demos entirely,
    # where dspy's states each field, its type description and its schema, and shows demos.
    "test_json_adapter_with_tool": "JsonAdapter system message omits type descriptions",
    "test_json_adapter_with_code": "JsonAdapter system message omits type descriptions",
    "test_json_adapter_formats_image": "JsonAdapter does not render content blocks",
    "test_json_adapter_formats_image_with_few_shot_examples": "JsonAdapter ignores demos",
    "test_json_adapter_formats_image_with_few_shot_examples_with_nested_images": (
        "JsonAdapter ignores demos"
    ),
    "test_json_adapter_formats_conversation_history": "JsonAdapter does not replay history",
    # Its parse is stricter than upstream's, which repairs a reply before reading it and
    # reports a failure carrying the fields it did recover.
    "test_json_adapter_parse_raise_error_on_mismatch_fields": (
        "AdapterParseError carries no parsed_result"
    ),
    "test_json_adapter_on_pydantic_model": "JsonAdapter parse does not repair a reply",
    "test_json_adapter_native_reasoning": "JsonAdapter parse does not repair a reply",
    # Provider negotiation: which of structured outputs, JSON mode and native function calling
    # a model supports, and what to fall back to. The crate states one output mode and stops.
    "test_json_adapter_not_using_structured_outputs_when_not_supported_by_model": (
        "no structured-output capability negotiation"
    ),
    "test_json_adapter_json_mode_no_structured_outputs": "no JSON-mode fallback",
    "test_json_adapter_fallback_to_json_mode_on_structured_output_failure": (
        "no JSON-mode fallback"
    ),
    "test_json_adapter_toolcalls_native_function_calling": "no native function calling",
    "test_json_adapter_toolcalls_no_native_function_calling": "no native function calling",
    # Lives in the chat file but builds a `dspy.JSONAdapter`, so it is a JsonAdapter case.
    "test_chat_adapter_toolcalls_native_function_calling": "no native function calling",
}


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
