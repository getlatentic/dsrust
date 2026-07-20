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

import rust_adapter  # noqa: E402
from rust_adapter import RustChatAdapter, RustJSONAdapter  # noqa: E402

# Upstream tests whose features this crate has not written yet, with the reason. Delete a line
# once Rust renders that case; the strict xfail will fail the run if you forget.
NOT_YET_IMPLEMENTED = {
}


# Upstream tests that pass without the crate rendering or parsing anything, with the reason.
# They are not conformance: they exercise dspy's own Python — a type's `__str__`, a helper — and
# would read as green whatever this crate did. Naming them keeps the passing count honest, and
# anything not named here must cross into Rust or the run fails.
DOES_NOT_EXERCISE_RUST = {
    # Calls dspy's private `_call_postprocess` with outputs already parsed, so it exercises
    # dspy's own plumbing around an adapter rather than anything the adapter renders.
    "test_tool_call_with_null_content_does_not_raise": "dspy-internal postprocessing",
}


@pytest.fixture(autouse=True)
def _require_a_crossing(request):
    """Fail a test that passed without the crate doing anything.

    A test can construct dspy's own adapter, or assert on a Python type directly, and never
    reach Rust. It then passes for reasons this crate has no part in, which is the one way a
    conformance suite can lie about its coverage.
    """
    before = rust_adapter.CROSSINGS
    yield
    if rust_adapter.CROSSINGS > before:
        return
    name = request.node.name.split("[")[0]
    if name in DOES_NOT_EXERCISE_RUST or name.removesuffix("_async") in DOES_NOT_EXERCISE_RUST:
        return
    pytest.fail(
        "this test passed without the crate rendering or parsing anything, so it says nothing "
        "about conformance; give it a line in DOES_NOT_EXERCISE_RUST if that is expected"
    )


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
