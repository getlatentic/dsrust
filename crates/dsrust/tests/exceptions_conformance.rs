//! The crate's error types against dspy's, rendered string for rendered string.
//!
//! `tests/utils/test_exceptions.py` is excused from the bridge because most of it asserts
//! `isinstance` against a fourteen-class Python tree. That excuse covered more than it should
//! have: nine of its ten tests assert on a code, a retryability, a metadata field, or an exact
//! rendered string — all of which a Rust type has too. Two of those strings were wrong here and
//! nothing noticed, because nothing compared them to dspy.
//!
//! `scripts/generate_exceptions_fixture.py` captures them by running the pinned dspy.

use std::path::Path;

use dsrust::adapter::parse::FieldMismatch;
use dsrust::lm::{ContextWindowExceeded, LmErrorKind, LmFailure};
use serde_json::{Value, json};

fn golden() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/lm/exceptions.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("the golden is committed"))
        .expect("the golden parses")
}

/// Every kind's code, its retryability, and which subtree it sits in.
#[test]
fn each_kind_matches_dspys_class() {
    let all = [
        LmErrorKind::Transport,
        LmErrorKind::Configuration,
        LmErrorKind::NotConfigured,
        LmErrorKind::UnsupportedFeature,
        LmErrorKind::Provider,
        LmErrorKind::Unexpected,
        LmErrorKind::Auth,
        LmErrorKind::Billing,
        LmErrorKind::RateLimit,
        LmErrorKind::InvalidRequest,
        LmErrorKind::UnsupportedModel,
        LmErrorKind::Timeout,
        LmErrorKind::Server,
    ];
    let recorded = golden();
    let cases = recorded["kinds"].as_array().expect("kinds");
    assert_eq!(cases.len(), all.len(), "every class is covered");

    for case in cases {
        let code = case["code"].as_str().expect("a code");
        let kind = all
            .iter()
            .find(|kind| kind.code() == code)
            .unwrap_or_else(|| panic!("no kind for {code}"));
        assert_eq!(
            case["class_code"],
            json!(code),
            "dspy's own default_code, {code}"
        );
        assert_eq!(kind.is_retryable(), case["retryable"], "retryable, {code}");
        assert_eq!(
            kind.is_from_provider(),
            case["from_provider"],
            "provider subtree, {code}"
        );
        assert_eq!(
            kind.is_configuration(),
            case["configuration"],
            "config subtree, {code}"
        );
    }
}

/// The status map, at every boundary — read out of dspy rather than transcribed from it.
#[test]
fn each_status_maps_to_dspys_class() {
    for case in golden()["statuses"].as_array().expect("statuses") {
        let status = case["status"].as_u64().map(|status| status as u16);
        let expected = case["code"].as_str().expect("a code");
        assert_eq!(
            LmErrorKind::from_status(status).code(),
            expected,
            "status {status:?}"
        );
    }
}

/// `[model] message`, and the two ways it degrades.
#[test]
fn a_failure_renders_as_dspy_renders_it() {
    for case in golden()["rendered"].as_array().expect("rendered") {
        let label = case["label"].as_str().expect("a label");
        let expected = case["rendered"].as_str().expect("the rendered text");
        let model = case["model"].as_str().unwrap_or_default();

        // The context-window class is its own type here, since a module branches on it by name.
        let rendered = if case["code"] == json!("context_window_exceeded") {
            let message = expected
                .strip_prefix(&format!("[{model}] "))
                .unwrap_or(expected);
            ContextWindowExceeded {
                model: model.to_owned(),
                message: match message == ContextWindowExceeded::DEFAULT_MESSAGE {
                    true => String::new(),
                    false => message.to_owned(),
                },
            }
            .to_string()
        } else {
            let message = expected
                .strip_prefix(&format!("[{model}] "))
                .unwrap_or(expected);
            let mut failure = LmFailure::new(LmErrorKind::Unexpected, message);
            if !model.is_empty() {
                failure = failure.on_model(model);
            }
            failure.to_string()
        };
        assert_eq!(rendered, expected, "{label}");
    }
}

/// `AdapterParseError`'s text, whitespace included — the line a caller reads when a reply does not
/// parse, and the one that said something entirely different here until it was compared.
#[test]
fn a_parse_failure_renders_as_dspy_renders_it() {
    for case in golden()["parse_errors"].as_array().expect("parse_errors") {
        let label = case["label"].as_str().expect("a label");
        // Upstream prefixes a caller-supplied message; the crate has no such argument, so the two
        // cases that carry one are compared against the body after it.
        let expected = case["rendered"]
            .as_str()
            .expect("the rendered text")
            .strip_prefix("Failed to parse\n\n")
            .unwrap_or_else(|| case["rendered"].as_str().expect("the rendered text"));
        let parsed = match label == "with a parsed result" {
            true => json!({ "answer1": "answer1" }),
            false => json!({}),
        };
        let mismatch = FieldMismatch {
            parsed,
            adapter_name: case["adapter_name"].as_str().expect("a name").to_owned(),
            lm_response: case["lm_response"].as_str().expect("a response").to_owned(),
            expected_fields: case["expected_fields"]
                .as_array()
                .expect("fields")
                .iter()
                .map(|name| name.as_str().expect("a name").to_owned())
                .collect(),
        };
        assert_eq!(mismatch.to_string(), expected, "{label}");
    }
}
