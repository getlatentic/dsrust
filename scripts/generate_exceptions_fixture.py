"""What dspy's error types render, and what they classify, captured by running them.

`tests/utils/test_exceptions.py` is excused from the bridge because most of it constructs Python
exception objects and asserts `isinstance` against a fourteen-class tree. What that excuse hid is
that nine of its ten tests assert on things a Rust type has too: the stable code, the retryability,
the metadata, and the exact rendered string. Two of those strings were wrong here and nothing said
so, because nothing compared them to dspy.

    .venv/bin/python scripts/generate_exceptions_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import dspy
from dspy.utils.exceptions import (
    AdapterParseError,
    ContextWindowExceededError,
    is_retryable_lm_error,
)

from pins import require

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "lm"
PINNED = require("dspy")

#: Every LM error class, by the code this crate's `LmErrorKind` answers with.
KINDS = {
    "transport": dspy.LMTransportError,
    "configuration": dspy.LMConfigurationError,
    "not_configured": dspy.LMNotConfiguredError,
    "unsupported_feature": dspy.LMUnsupportedFeatureError,
    "provider": dspy.LMProviderError,
    "unexpected": dspy.LMUnexpectedError,
    "auth": dspy.LMAuthError,
    "billing": dspy.LMBillingError,
    "rate_limit": dspy.LMRateLimitError,
    "invalid_request": dspy.LMInvalidRequestError,
    "unsupported_model": dspy.LMUnsupportedModelError,
    "timeout": dspy.LMTimeoutError,
    "server": dspy.LMServerError,
}

#: Statuses at every boundary of `_lm_error_class_from_status`, plus "no status at all".
STATUSES = [None, 400, 401, 402, 403, 404, 408, 422, 429, 499, 500, 503, 599]


def by_kind() -> list[dict]:
    """Each class's code, its retryability, and whether it is a provider or configuration error."""
    return [
        {
            "code": code,
            "class_code": cls().code,
            "retryable": is_retryable_lm_error(cls()),
            "from_provider": issubclass(cls, dspy.LMProviderError),
            "configuration": issubclass(cls, dspy.LMConfigurationError),
        }
        for code, cls in KINDS.items()
    ]


def by_status() -> list[dict]:
    """The status map, read out of dspy rather than transcribed from it."""
    from dspy.clients.lm import _lm_error_class_from_status

    return [
        {"status": status, "code": _lm_error_class_from_status(status)().code}
        for status in STATUSES
    ]


def rendered() -> list[dict]:
    """`str(error)` for the shapes the crate builds, which is what a caller reads first."""
    cases = [
        ("rate limit with a model", dspy.LMRateLimitError(
            "rate limited", model="openai/gpt-4o", provider="openai",
            status=429, request_id="req-123", retry_after=2.5,
        )),
        ("a message with no model", dspy.LMAuthError("Incorrect API key provided")),
        ("context window, defaulted", ContextWindowExceededError()),
        ("context window with a model", ContextWindowExceededError(model="openai/gpt-4o")),
        ("context window with both", ContextWindowExceededError(
            model="openai/gpt-4o", message="Input is 200k tokens, limit is 128k"
        )),
        ("context window, message only", ContextWindowExceededError(message="Too many tokens")),
    ]
    return [
        {
            "label": label,
            "code": error.code,
            "model": error.model,
            "status": error.status,
            "retry_after": error.retry_after,
            "request_id": error.request_id,
            "rendered": str(error),
        }
        for label, error in cases
    ]


def parse_errors() -> list[dict]:
    """`AdapterParseError`, whose rendered text names the fields the reply was missing."""
    signature = dspy.make_signature("question->answer1, answer2")
    cases = [
        ("no message, no parsed result", AdapterParseError(
            adapter_name="ChatAdapter", signature=signature,
            lm_response="[[ ## answer1 ## ]]\nanswer1",
        )),
        ("with a message", AdapterParseError(
            adapter_name="ChatAdapter", signature=signature,
            lm_response="[[ ## answer1 ## ]]\nanswer1", message="Failed to parse",
        )),
        ("with a parsed result", AdapterParseError(
            adapter_name="ChatAdapter", signature=signature,
            lm_response="[[ ## answer1 ## ]]\nanswer1", parsed_result={"answer1": "answer1"},
        )),
    ]
    return [
        {
            "label": label,
            "code": error.code,
            "adapter_name": error.adapter_name,
            "lm_response": error.lm_response,
            "expected_fields": list(signature.output_fields.keys()),
            "rendered": str(error),
        }
        for label, error in cases
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = {
        "_source": f"dspy {PINNED} utils/exceptions.py, via {pathlib.Path(__file__).name}",
        "kinds": by_kind(),
        "statuses": by_status(),
        "rendered": rendered(),
        "parse_errors": parse_errors(),
    }
    path = OUT / "exceptions.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parent.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
