# Upstream conformance fixtures

Each fixture is one case lifted from Python DSPy's own adapter tests
(`tests/adapters/test_chat_adapter.py`, the `format_exact_messages_*` family).
The `expected` messages are copied verbatim from upstream's assertions, so a
passing fixture means this crate renders the prompt Python DSPy renders.

Regenerate or add cases with `python3 scripts/pull_dspy_fixtures.py`, which reads
the upstream test file at a pinned commit and writes the JSON here.
