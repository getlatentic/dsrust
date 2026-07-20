//! Python's spelling of a value, which serde_json does not share.
//!
//! Two gaps push a rendered prompt off dspy's bytes: `json.dumps` writes a space after every
//! comma and colon where serde_json's `Display` writes none, and Python's `str` spells the
//! three JSON keywords `True`, `False` and `None`. They are one concern — how Python would
//! have printed this value — so every field the crate renders goes through
//! [`format_field_value`] here rather than through a per-module copy.

use serde_json::{Value, json};

/// dspy's `format_field_value`: a dict or a list is `json.dumps` text, and anything else is
/// what Python's `str` would print. A string is therefore bare, since quoting it would change
/// what the model reads, and upstream is careful to avoid that.
///
/// The two spellings meet inside a structure: `json.dumps` keeps JSON's own `true` and `null`
/// for a nested scalar, so only a scalar at the top level is ever spelled Python's way.
pub(crate) fn format_field_value(value: &Value) -> String {
    match value {
        Value::Object(_) | Value::Array(_) => json_dumps(value),
        Value::String(text) => text.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        number => number.to_string(),
    }
}

/// Python's `json.dumps` spacing — `", "` between items, `": "` after a key. Escaping is left
/// to serde_json, which already agrees with the `ensure_ascii=False` upstream passes: both
/// emit a non-ASCII character as itself.
fn json_dumps(value: &Value) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_string_renders_bare_and_a_number_as_itself() {
        assert_eq!(format_field_value(&json!("plain string")), "plain string");
        assert_eq!(format_field_value(&json!(1.5)), "1.5");
        assert_eq!(format_field_value(&json!(3)), "3");
    }

    #[test]
    fn a_top_level_scalar_takes_pythons_spelling() {
        // `str(True)` and `str(None)`, which is what upstream reaches for once the value is
        // neither a dict nor a list.
        assert_eq!(format_field_value(&json!(true)), "True");
        assert_eq!(format_field_value(&json!(false)), "False");
        assert_eq!(format_field_value(&Value::Null), "None");
    }

    #[test]
    fn a_structure_keeps_json_spelling_inside_pythons_spacing() {
        // The nested keywords stay `true`/`null`: they are written by `json.dumps`, which
        // never reaches Python's `str`.
        assert_eq!(
            format_field_value(&json!([1, "two", true, null])),
            r#"[1, "two", true, null]"#
        );
        assert_eq!(
            format_field_value(&json!({ "a": null, "b": true })),
            r#"{"a": null, "b": true}"#
        );
    }

    #[test]
    fn nesting_is_spaced_at_every_depth() {
        assert_eq!(
            format_field_value(&json!({ "a": 1, "b": { "c": [1, 2] } })),
            r#"{"a": 1, "b": {"c": [1, 2]}}"#
        );
    }

    #[test]
    fn an_empty_structure_has_no_padding() {
        assert_eq!(format_field_value(&json!({})), "{}");
        assert_eq!(format_field_value(&json!([])), "[]");
    }

    #[test]
    fn escaping_follows_pythons_json_dumps() {
        // `ensure_ascii=False` leaves the accent alone, and neither side escapes a slash.
        assert_eq!(
            format_field_value(&json!({ "café": "a/b", "q": "he said \"hi\"" })),
            r#"{"café": "a/b", "q": "he said \"hi\""}"#
        );
    }
}
