"""Point upstream's test suite at the Rust-backed adapter.

Placed beside a checkout of dspy's own tests, this replaces `dspy.ChatAdapter` for the whole
run, so every `format_exact_messages_*` assertion in their file is checked against Rust.
"""

import dspy
import pytest

from rust_adapter import RustChatAdapter


@pytest.fixture(autouse=True)
def _use_rust_adapter(monkeypatch):
    monkeypatch.setattr(dspy, "ChatAdapter", RustChatAdapter)
    monkeypatch.setattr("dspy.adapters.ChatAdapter", RustChatAdapter, raising=False)
    monkeypatch.setattr("dspy.adapters.chat_adapter.ChatAdapter", RustChatAdapter, raising=False)


def pytest_configure(config):
    """Upstream marks async tests with `@pytest.mark.asyncio`; honour it without editing them."""
    config.option.asyncio_mode = "auto"
