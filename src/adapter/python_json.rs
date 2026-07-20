//! Python's spelling of a value, which serde_json does not share.
//!
//! Two gaps push a rendered prompt off dspy's bytes: `json.dumps` writes a space after every
//! comma and colon where serde_json's `Display` writes none, and Python's `str` spells the
//! three JSON keywords `True`, `False` and `None`. They are one concern — how Python would
//! have printed this value — so every field the crate renders goes through
//! [`format_value`] here rather than through a per-module copy.

use serde_json::{Value, json};

use crate::signature::FieldKind;

/// Python's spelling of a bare value: a dict or a list is `json.dumps` text, and anything else is
/// what Python's `str` would print. A string is therefore bare, since quoting it would change
/// what the model reads, and upstream is careful to avoid that.
///
/// The two spellings meet inside a structure: `json.dumps` keeps JSON's own `true` and `null`
/// for a nested scalar, so only a scalar at the top level is ever spelled Python's way.
pub fn format_value(value: &Value) -> String {
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

/// dspy `format_field_value`: a field's value as the model reads it.
///
/// A list handed to a `str` field is laid out as a numbered list rather than as JSON, because
/// upstream reads that field as prose the model wrote or will write — several retrieved
/// passages, say — and a JSON array of them reads as one blob. Every other field defers to
/// [`format_value`].
pub fn format_field_value(kind: &FieldKind, value: &Value) -> String {
    match (kind, value) {
        (FieldKind::Str, Value::Array(items)) => input_list(items),
        _ => format_value(value),
    }
}

/// dspy `_format_input_list_field_value`: nothing, one blob, or a numbered run of them.
fn input_list(items: &[Value]) -> String {
    match items {
        [] => "N/A".to_owned(),
        [only] => blob(&format_value(only)),
        many => many
            .iter()
            .enumerate()
            .map(|(index, item)| format!("[{}] {}", index + 1, blob(&format_value(item))))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// dspy `_format_blob`: guillemets around one entry, so the model can tell where each ends.
///
/// An entry that runs over lines, or that already carries a guillemet, takes the tripled form
/// with its lines indented — the single form could not be told from the text inside it.
fn blob(text: &str) -> String {
    if !text.contains('\n') && !text.contains('«') && !text.contains('»') {
        return format!("«{text}»");
    }
    format!("«««\n    {}\n»»»", text.replace('\n', "\n    "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_string_renders_bare_and_a_number_as_itself() {
        assert_eq!(format_value(&json!("plain string")), "plain string");
        assert_eq!(format_value(&json!(1.5)), "1.5");
        assert_eq!(format_value(&json!(3)), "3");
    }

    #[test]
    fn a_top_level_scalar_takes_pythons_spelling() {
        // `str(True)` and `str(None)`, which is what upstream reaches for once the value is
        // neither a dict nor a list.
        assert_eq!(format_value(&json!(true)), "True");
        assert_eq!(format_value(&json!(false)), "False");
        assert_eq!(format_value(&Value::Null), "None");
    }

    #[test]
    fn a_structure_keeps_json_spelling_inside_pythons_spacing() {
        // The nested keywords stay `true`/`null`: they are written by `json.dumps`, which
        // never reaches Python's `str`.
        assert_eq!(
            format_value(&json!([1, "two", true, null])),
            r#"[1, "two", true, null]"#
        );
        assert_eq!(
            format_value(&json!({ "a": null, "b": true })),
            r#"{"a": null, "b": true}"#
        );
    }

    #[test]
    fn nesting_is_spaced_at_every_depth() {
        assert_eq!(
            format_value(&json!({ "a": 1, "b": { "c": [1, 2] } })),
            r#"{"a": 1, "b": {"c": [1, 2]}}"#
        );
    }

    #[test]
    fn an_empty_structure_has_no_padding() {
        assert_eq!(format_value(&json!({})), "{}");
        assert_eq!(format_value(&json!([])), "[]");
    }

    #[test]
    fn escaping_follows_pythons_json_dumps() {
        // `ensure_ascii=False` leaves the accent alone, and neither side escapes a slash.
        assert_eq!(
            format_value(&json!({ "café": "a/b", "q": "he said \"hi\"" })),
            r#"{"café": "a/b", "q": "he said \"hi\""}"#
        );
    }

    #[test]
    fn a_list_on_a_str_field_is_laid_out_as_a_numbered_run() {
        // Copied from `dspy.ChatAdapter().format(...)` on 3.2.1. A `str` field holding several
        // entries is prose the model reads one at a time, so upstream numbers them and fences
        // each rather than handing over one JSON blob.
        assert_eq!(
            format_field_value(&FieldKind::Str, &json!(["first", "second"])),
            "[1] «first»\n[2] «second»"
        );
    }

    #[test]
    fn one_entry_is_fenced_without_a_number_and_none_reads_as_absent() {
        assert_eq!(
            format_field_value(&FieldKind::Str, &json!(["only"])),
            "«only»"
        );
        assert_eq!(format_field_value(&FieldKind::Str, &json!([])), "N/A");
    }

    #[test]
    fn an_entry_that_runs_over_lines_takes_the_tripled_fence() {
        // The single fence could not be told from a guillemet inside the text, so an entry
        // carrying either a newline or a guillemet is fenced the other way and indented.
        assert_eq!(
            format_field_value(&FieldKind::Str, &json!(["one\ntwo", "b"])),
            "[1] «««\n    one\n    two\n»»»\n[2] «b»"
        );
        assert_eq!(
            format_field_value(&FieldKind::Str, &json!(["a «quoted» one"])),
            "«««\n    a «quoted» one\n»»»"
        );
    }

    #[test]
    fn a_list_on_any_other_field_stays_json() {
        // The layout is the `str` field's alone: a structured field's list is its value.
        assert_eq!(
            format_field_value(&FieldKind::opaque_json(), &json!(["first", "second"])),
            r#"["first", "second"]"#
        );
    }
}
