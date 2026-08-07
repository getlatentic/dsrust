//! Comparing a demo against the one dspy recorded, field for field.
//!
//! Shared because the two golden suites that read demos — [`super::conformance`] for
//! `BootstrapFewShot` and [`super::mipro::demos`] for MIPROv2's Step 1 — each projected a demo
//! down to its question and answer. That projection is what hid `augmented` from both: the field
//! was dropped on the Rust side and on dspy's together, so the two agreed about a marker neither
//! was looking at, while the proposer that gathers on it was shown nothing.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::example::Example;

/// Every field a demo carries, as text.
pub(super) fn fields(demo: &Example) -> BTreeMap<String, String> {
    demo.fields()
        .map(|(name, value)| (name.to_owned(), rendered(value)))
        .collect()
}

/// Every field of a demo the golden recorded.
pub(super) fn recorded_fields(demo: &Value) -> BTreeMap<String, String> {
    demo.as_object()
        .expect("a demo")
        .iter()
        .map(|(name, value)| (name.clone(), rendered(value)))
        .collect()
}

/// A field value as text, keeping a bool distinguishable from an absent string — `augmented` is a
/// bool on both sides, and mapping it through `as_str` would flatten it to the empty string and
/// compare equal to a demo that has no such key at all.
fn rendered(value: &Value) -> String {
    match value {
        Value::Bool(flag) => flag.to_string(),
        other => other.as_str().unwrap_or_default().to_owned(),
    }
}
