//! Reading a model's reply back into the signature's output fields.
//!
//! Each wire format has its own reader: marker sections for `ChatAdapter`, a JSON object for
//! `JsonAdapter`. Both are lenient about what surrounds the answer — prose, code fences,
//! unknown headers — and strict about the answer itself, since a reply that does not speak
//! the format at all is a parse failure rather than a guess.

use anyhow::{Result, anyhow};
use serde_json::{Map, Value};

use crate::signature::{FieldKind, OutField, Signature};

mod repair;

/// DSPy ChatAdapter's parser: split the reply into sections at `[[ ## name ## ]]` headers,
/// keep the first section seen for each declared output field, ignore prose outside any
/// section and unknown headers (`completed` among them).
pub(super) fn parse_markers(signature: &Signature, raw: &str) -> Result<Value> {
    let mut sections: Vec<(&str, Vec<&str>)> = Vec::new();
    for line in raw.lines() {
        if let Some((name, rest)) = split_header(line) {
            let seed = if rest.is_empty() { vec![] } else { vec![rest] };
            sections.push((name, seed));
        } else if let Some(section) = sections.last_mut() {
            section.1.push(line);
        }
    }
    let mut fields = Map::new();
    for (name, lines) in sections {
        let Some(field) = signature.outputs.iter().find(|field| field.name == name) else {
            continue;
        };
        if fields.contains_key(name) {
            continue;
        }
        let joined = lines.join("\n");
        fields.insert(name.to_owned(), section_value(field, joined.trim()));
    }
    if fields.is_empty() {
        return Err(anyhow!("reply has no [[ ## field ## ]] sections"));
    }
    Ok(Value::Object(fields))
}

/// A section's text as the value it denotes. dspy runs every section through json-repair
/// before validating it, so a `Json` field answered in Python's literal syntax — single
/// quotes, `True`/`False`/`None`, digit-group underscores — lands as its declared type
/// rather than as the text that spells it. Every other section stays text for
/// [`Signature::coerce`], which is also where strict JSON is still read.
fn section_value(field: &OutField, text: &str) -> Value {
    match field.kind {
        FieldKind::Json(_) => repair::python_literal(text).unwrap_or_else(|| Value::from(text)),
        _ => Value::from(text),
    }
}

/// A section header at the start of a line: `[[ ## name ## ]]` with a word-character name,
/// keeping any trailing text on the line as that section's first content.
fn split_header(line: &str) -> Option<(&str, &str)> {
    let after_open = line.trim_start().strip_prefix("[[ ## ")?;
    let (name, rest) = after_open.split_once(" ## ]]")?;
    let word = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    word.then_some((name, rest.trim()))
}

/// A JSON object anywhere in the reply. Providers in JSON mode return the bare object;
/// models that ignore the mode wrap it in prose or code fences, so the outermost braces
/// are the recovery path (DSPy's JSONAdapter recovers with a regex the same way).
pub(super) fn parse_json(raw: &str) -> Result<Value> {
    if let Ok(value) = serde_json::from_str(raw) {
        return Ok(value);
    }
    if let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}'))
        && start < end
        && let Ok(value) = serde_json::from_str(&raw[start..=end])
    {
        return Ok(value);
    }
    Err(anyhow!("model returned invalid JSON"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::InField;
    use serde_json::json;

    fn signature() -> Signature {
        Signature::single_input(
            "Pick a color.",
            vec![
                OutField {
                    name: "color".into(),
                    desc: "the chosen color".into(),
                    kind: FieldKind::Str,
                    values: Some(vec!["red".into(), "blue".into()]),
                    schema: None,
                },
                OutField {
                    name: "why".into(),
                    desc: "one short sentence".into(),
                    kind: FieldKind::Str,
                    values: None,
                    schema: None,
                },
            ],
        )
    }

    fn json_signature() -> Signature {
        let mut signature = Signature::single_input(
            "Suggest ideas.",
            vec![OutField {
                name: "ideas".into(),
                desc: "three concrete ideas".into(),
                kind: FieldKind::opaque_json(),
                values: None,
                schema: Some(json!({ "type": "array", "items": { "type": "string" } })),
            }],
        );
        signature.inputs = vec![InField {
            name: "recipient".into(),
            desc: "who the gift is for".into(),
            kind: FieldKind::opaque_json(),
        }];
        signature
    }

    #[test]
    fn parse_markers_extracts_fields_and_tolerates_prose() {
        let raw = "Sure, here you go:\n\n[[ ## color ## ]]\nred\n\n[[ ## why ## ]]\nIt is calm.\nVery calm.\n\n[[ ## completed ## ]]\n";
        let value = parse_markers(&signature(), raw).expect("parses");
        assert_eq!(
            value,
            json!({ "color": "red", "why": "It is calm.\nVery calm." })
        );
    }

    #[test]
    fn parse_markers_keeps_first_occurrence_and_same_line_content() {
        let raw = "[[ ## color ## ]] red\n[[ ## color ## ]]\nblue\n[[ ## why ## ]]\ncalm";
        let value = parse_markers(&signature(), raw).expect("parses");
        assert_eq!(value["color"], "red");
    }

    #[test]
    fn parse_markers_leaves_missing_fields_to_validation() {
        let raw = "[[ ## color ## ]]\nred";
        let value = parse_markers(&signature(), raw).expect("parses");
        assert_eq!(value, json!({ "color": "red" }));
    }

    #[test]
    fn parse_markers_rejects_a_reply_with_no_sections() {
        assert!(parse_markers(&signature(), "red, because it is calm").is_err());
    }

    /// Upstream's `test_chat_adapter_parses_float_with_underscores` sends exactly this reply
    /// for a field declared as a model with one float, and expects 123456.789.
    #[test]
    fn parse_markers_reads_a_json_field_written_as_a_python_literal() {
        let raw = "[[ ## ideas ## ]]\n{'score': 123_456.789}\n[[ ## completed ## ]]";
        let value = parse_markers(&json_signature(), raw).expect("parses");
        assert_eq!(value["ideas"], json!({ "score": 123_456.789 }));
        assert_eq!(value["ideas"]["score"], json!(123456.789));
    }

    #[test]
    fn parse_markers_leaves_a_strict_json_field_as_text_to_coerce() {
        let raw = "[[ ## ideas ## ]]\n[\"a\", \"b\"]";
        let value = parse_markers(&json_signature(), raw).expect("parses");
        assert_eq!(value["ideas"], json!("[\"a\", \"b\"]"));
    }

    #[test]
    fn parse_json_accepts_bare_and_prose_wrapped_objects() {
        let bare = parse_json(r#"{ "color": "red" }"#).expect("bare");
        assert_eq!(bare["color"], "red");
        let wrapped =
            parse_json("Here it is:\n```json\n{ \"color\": \"blue\" }\n```").expect("wrapped");
        assert_eq!(wrapped["color"], "blue");
        assert!(parse_json("no json here").is_err());
    }
}
