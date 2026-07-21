//! The normalized LM types, against what dspy 3.3 itself serializes.
//!
//! Every entry in `tests/conformance/lm_api/dspy_3_3.json` is one instance dumped by pydantic. The
//! types this crate declares `deny_unknown_fields`, so a field upstream emits that we do not
//! model is a parse failure here rather than a quiet omission — which is the only way a port
//! read off a source file stays honest as that file moves.

use dsrs::lm::LmUsage;
use dsrs::lm::api::{
    LmCacheConfig, LmConfig, LmDelta, LmMessage, LmOutput, LmPart, LmPromptCacheConfig,
    LmReasoningConfig, LmRequest, LmResponse, LmStreamEvent, LmToolChoice, LmToolSpec,
};
use serde_json::Value;

/// Parse into `T`, then prove the value survives its own serialization.
///
/// The re-serialized JSON is not compared against upstream's: pydantic writes `"metadata": {}`
/// where this crate omits an empty map, and that is a serde convention rather than a divergence.
/// What must hold is that nothing is lost on the way in.
fn round_trip<T>(raw: &Value, label: &str)
where
    T: serde::de::DeserializeOwned + serde::Serialize + PartialEq + std::fmt::Debug,
{
    let parsed: T = serde_json::from_value(raw.clone())
        .unwrap_or_else(|error| panic!("{label} does not parse into its Rust type: {error}\n  upstream sent: {raw}"));
    let written = serde_json::to_value(&parsed).expect("serializes");
    let again: T = serde_json::from_value(written.clone())
        .unwrap_or_else(|error| panic!("{label} does not survive its own serialization: {error}"));
    assert_eq!(parsed, again, "{label} changed on the way through");
}

#[test]
fn every_dspy_33_lm_type_parses_into_its_rust_counterpart() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/lm_api/dspy_3_3.json");
    let raw = std::fs::read_to_string(&path).expect("fixture is readable");
    let fixture: Value = serde_json::from_str(&raw).expect("fixture is valid json");
    let entries = fixture["entries"].as_array().expect("entries array");
    assert!(!entries.is_empty(), "no entries to check");

    for entry in entries {
        let rust = entry["rust"].as_str().expect("a rust type name");
        let python = entry["python"].as_str().expect("a python class name");
        let json = &entry["json"];
        let label = format!("{python} (as {rust})");

        match rust {
            "LmPart" => round_trip::<LmPart>(json, &label),
            "LmMessage" => round_trip::<LmMessage>(json, &label),
            "LmToolSpec" => round_trip::<LmToolSpec>(json, &label),
            "LmReasoningConfig" => round_trip::<LmReasoningConfig>(json, &label),
            "LmToolChoice" => round_trip::<LmToolChoice>(json, &label),
            "LmCacheConfig" => round_trip::<LmCacheConfig>(json, &label),
            "LmPromptCacheConfig" => round_trip::<LmPromptCacheConfig>(json, &label),
            "LmConfig" => round_trip::<LmConfig>(json, &label),
            "LmOutput" => round_trip::<LmOutput>(json, &label),
            "LmResponse" => round_trip::<LmResponse>(json, &label),
            "LmRequest" => round_trip::<LmRequest>(json, &label),
            "LmUsage" => round_trip::<LmUsage>(json, &label),
            "LmDelta" => round_trip::<LmDelta>(json, &label),
            "LmStreamEvent" => round_trip::<LmStreamEvent>(json, &label),
            other => panic!("fixture names a Rust type the harness does not know: {other}"),
        }
    }
}

/// The discriminator is the one identifier that has to match exactly: it is what a provider and
/// a stored payload are keyed on, so a variant renamed on our side stops parsing upstream's JSON
/// while still compiling.
#[test]
fn the_discriminators_are_upstreams_own_spelling() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/lm_api/dspy_3_3.json");
    let fixture: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("readable")).expect("json");

    let tags: Vec<&str> = fixture["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .filter(|entry| entry["rust"] == "LmPart")
        .filter_map(|entry| entry["json"]["type"].as_str())
        .collect();

    assert_eq!(
        tags,
        [
            "text",
            "image",
            "audio",
            "video",
            "binary",
            "document",
            "tool_call",
            "tool_result",
            "thinking",
            "citation",
            "refusal",
        ],
        "all eleven parts, under the tags dspy 3.3 writes"
    );
}
