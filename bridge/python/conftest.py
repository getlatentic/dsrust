"""Point upstream's test suite at the Rust-backed adapter, and be honest about the gaps.

Every test here runs against Rust; nothing silently defers to the Python implementation. The
cases this crate cannot render yet are listed below and marked `xfail(strict=True)`, which
means two things: they are never counted as passes, and if one starts passing the run FAILS
until its name is deleted from the list. The list is therefore the to-do list, it cannot
drift out of date, and a green run means every case not named here genuinely runs on Rust.
"""

import dspy
import pytest

from rust_adapter import RustChatAdapter

# Upstream tests whose features this crate has not written yet, with the reason. Delete a line
# once Rust renders that case; the strict xfail will fail the run if you forget.
NOT_YET_IMPLEMENTED = {
    "test_chat_adapter_formats_image": "dspy.Image fields",
    "test_chat_adapter_formats_image_with_few_shot_examples": "dspy.Image fields and demos",
    "test_chat_adapter_formats_image_with_nested_images": "dspy.Image fields",
    "test_chat_adapter_formats_image_with_few_shot_examples_with_nested_images": (
        "dspy.Image fields and demos"
    ),
    "test_chat_adapter_with_tool": "dspy.Tool fields",
    "test_chat_adapter_toolcalls_vague_match": "dspy.ToolCalls parsing",
    "test_chat_adapter_with_code": "dspy.Code fields",
    "test_code_output_field_omits_json_schema_in_prompt": "dspy.Code fields",
    "test_citations_output_field_keeps_json_schema_in_prompt": "dspy.Citations fields",
    "test_chat_adapter_formats_conversation_history": "dspy.History fields",
    "test_chat_adapter_parses_float_with_underscores": "python numeric literal parsing",
    "test_chat_adapter_exception_raised_on_failure": "AdapterParseError is not raised yet",
    "test_chat_adapter_respects_use_json_adapter_fallback_flag": (
        "the flag is not read through the bridge"
    ),
}


@pytest.fixture(autouse=True)
def _use_rust_adapter(monkeypatch):
    monkeypatch.setattr(dspy, "ChatAdapter", RustChatAdapter)
    monkeypatch.setattr("dspy.adapters.ChatAdapter", RustChatAdapter, raising=False)
    monkeypatch.setattr("dspy.adapters.chat_adapter.ChatAdapter", RustChatAdapter, raising=False)


def pytest_configure(config):
    """Upstream marks async tests with `@pytest.mark.asyncio`; honour it without editing them."""
    config.option.asyncio_mode = "auto"


def pytest_collection_modifyitems(items):
    for item in items:
        # Async variants share the sync name plus a suffix, and share the same gap.
        base = item.name.removesuffix("_async")
        reason = NOT_YET_IMPLEMENTED.get(item.name) or NOT_YET_IMPLEMENTED.get(base)
        if reason:
            item.add_marker(pytest.mark.xfail(strict=True, reason=f"not in Rust yet: {reason}"))
