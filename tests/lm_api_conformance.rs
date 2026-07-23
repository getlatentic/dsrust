//! The normalized LM types, against what dspy 3.3 itself serializes.
//!
//! Every entry in `tests/conformance/lm_api/dspy_3_3.json` is one instance dumped by pydantic, and
//! what must hold is that nothing upstream said is lost. Parsing alone does not show that: serde
//! ignores a field it does not know, so a type that dropped one would still round-trip cleanly
//! against itself. So the re-serialized value is checked back against upstream's own JSON, which
//! is the only way a port read off a source file stays honest as that file moves.
//!
//! `deny_unknown_fields` covers the same ground where it can, but it cannot be declared on the
//! variants that `flatten` their payload — `LmPart` among them — and those are exactly the types
//! most likely to grow a field.

use dsrs::lm::LmUsage;
use dsrs::lm::api::{
    LmCacheConfig, LmConfig, LmDelta, LmMessage, LmOutput, LmPart, LmPromptCacheConfig,
    LmReasoningConfig, LmRequest, LmResponse, LmStreamEvent, LmToolChoice, LmToolSpec,
};
use serde_json::Value;

/// Parse into `T`, prove the value survives its own serialization, and prove it still says
/// everything upstream said.
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
    if let Some(lost) = dropped(raw, &written, String::new()) {
        panic!("{label} drops what upstream sent at `{lost}`\n  upstream: {raw}\n  ours: {written}");
    }
}

/// Where the re-serialized value stops saying what upstream's JSON said, if anywhere.
///
/// pydantic writes a field it holds no value for — `"metadata": {}`, `"logprobs": null` — where
/// serde omits it, and that is a spelling convention rather than something lost. Anything the
/// fixture states, though, has to come back out.
fn dropped(upstream: &Value, ours: &Value, at: String) -> Option<String> {
    match (upstream, ours) {
        (Value::Object(theirs), Value::Object(mine)) => theirs.iter().find_map(|(key, value)| {
            let at = match at.is_empty() {
                true => key.clone(),
                false => format!("{at}.{key}"),
            };
            match mine.get(key) {
                Some(mine) => dropped(value, mine, at),
                None => says_nothing(value).then_some(()).map_or(Some(at), |()| None),
            }
        }),
        (Value::Array(theirs), Value::Array(mine)) if theirs.len() == mine.len() => theirs
            .iter()
            .zip(mine)
            .enumerate()
            .find_map(|(index, (value, mine))| dropped(value, mine, format!("{at}[{index}]"))),
        _ => None,
    }
}

/// Whether a value is the default its field would hold anyway, and so may go unwritten.
///
/// These types omit a field holding its default throughout — `Option::is_none`, `Map::is_empty`,
/// `Vec::is_empty`, `is_false` — where pydantic writes every field it declares. What that leaves
/// unsaid is recoverable; a field stating anything else is not.
fn says_nothing(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => true,
        Value::Object(fields) => fields.is_empty(),
        Value::Array(items) => items.is_empty(),
        _ => false,
    }
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
