//! Python's `json.dumps` text — the form dspy writes every JSON value it puts in a prompt.

use serde_json::{Value, json};

/// `", "` between items and `": "` after a key, the spacing `json.dumps` emits by default.
/// serde_json's own `Display` emits neither, so a value rendered through it differs from
/// upstream on every object and array a model ever reads.
pub(crate) fn json_dumps(value: &Value) -> String {
    match value {
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(json_dumps).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| format!("{}: {}", json!(key), json_dumps(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        scalar => scalar.to_string(),
    }
}
