//! dspy 3.3.1's nested XML for a structured output: the elements it is written as in an assistant
//! turn, the sketch of them the system prompt and the request show, and which fields get either.
use serde_json::{Map, Value};

use crate::adapter::python_json::json_dumps;
use crate::signature::{FieldKind, OutField, annotation};

/// `_uses_nested_xml(field.annotation)`: the spelling decides where it can, else the field's
/// schema — a class with members is a model or a TypedDict, anything else is written as text.
pub(super) fn uses_nested_xml(field: &OutField) -> bool {
    let FieldKind::Json(json) = &field.kind else {
        return false;
    };
    annotation::nested_xml_shape(&json.annotation).unwrap_or_else(|| {
        field.schema.as_ref().is_some_and(|schema| {
            schema.get("properties").is_some() || schema.get("$ref").is_some()
        })
    })
}

/// `_value_to_xml`: a list as `<item>` children, a mapping as named children — or `<entry key="…">`
/// where the key could not name an element — and a scalar as escaped text. `None` and an empty
/// list are the empty element.
pub(super) fn value_to_xml(value: &Value, tag: &str, key: Option<&str>) -> String {
    let attributes = key
        .map(|key| format!(" key={}", quoteattr(key)))
        .unwrap_or_default();
    match value {
        Value::Array(items) => {
            let children: String = items
                .iter()
                .map(|item| value_to_xml(item, "item", None))
                .collect();
            match children.is_empty() {
                true => format!("<{tag}{attributes} />"),
                false => format!("<{tag}{attributes}>{children}</{tag}>"),
            }
        }
        Value::Object(fields) => {
            let children: String = fields
                .iter()
                .map(|(name, child)| match names_an_element(name) {
                    true => value_to_xml(child, name, None),
                    false => value_to_xml(child, "entry", Some(name)),
                })
                .collect();
            format!("<{tag}{attributes}>{children}</{tag}>")
        }
        Value::Null => format!("<{tag}{attributes} />"),
        scalar => format!("<{tag}{attributes}>{}</{tag}>", escape(&python_str(scalar))),
    }
}

/// `str(value)` for a scalar: Python's own spellings.
fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        other => json_dumps(other),
    }
}

/// The two replacements upstream makes in text it writes between tags.
pub(super) fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;")
}

/// `(name[:1] + name[1:].replace("-", "_").replace(".", "_")).isidentifier()`.
fn names_an_element(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || unicode_ident::is_xid_start(first))
        && chars.all(|c| matches!(c, '-' | '.') || unicode_ident::is_xid_continue(c))
}

/// `xml.sax.saxutils.quoteattr`: escaped, then double-quoted unless the text holds a double quote
/// and no single one.
fn quoteattr(text: &str) -> String {
    let escaped = text
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\n', "&#10;")
        .replace('\r', "&#13;")
        .replace('\t', "&#9;");
    match (escaped.contains('"'), escaped.contains('\'')) {
        (true, false) => format!("'{escaped}'"),
        (true, true) => format!("\"{}\"", escaped.replace('"', "&quot;")),
        _ => format!("\"{escaped}\""),
    }
}

/// `_xml_schema(tag, annotation)`: the field's JSON schema as a sketch of elements.
pub(super) fn xml_schema(tag: &str, field: &OutField) -> String {
    let schema = field.property_schema();
    let definitions = schema
        .get("$defs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    schema_to_xml(tag, &schema, &definitions, &[])
}

/// `_schema_to_xml`: a reference followed once, a union read by its first non-null arm, an array
/// as one `<item>`, a class as its members, and anything else as `...`.
fn schema_to_xml(
    tag: &str,
    schema: &Value,
    definitions: &Map<String, Value>,
    seen: &[&str],
) -> String {
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        let name = reference.rsplit('/').next().unwrap_or(reference);
        return match definitions.get(name).filter(|_| !seen.contains(&name)) {
            Some(definition) => {
                let mut seen = seen.to_vec();
                seen.push(name);
                schema_to_xml(tag, definition, definitions, &seen)
            }
            None => format!("<{tag}>...</{tag}>"),
        };
    }
    if let Some(choices) = schema
        .get("anyOf")
        .and_then(Value::as_array)
        .filter(|c| !c.is_empty())
    {
        let chosen = choices
            .iter()
            .find(|choice| choice.get("type") != Some(&Value::from("null")))
            .unwrap_or(&choices[0]);
        return schema_to_xml(tag, chosen, definitions, seen);
    }
    if schema.get("type") == Some(&Value::from("array")) {
        let items = schema
            .get("items")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        return format!(
            "<{tag}>{}</{tag}>",
            schema_to_xml("item", &items, definitions, seen)
        );
    }
    let children: String = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .iter()
                .map(|(name, child)| schema_to_xml(name, child, definitions, seen))
                .collect()
        })
        .unwrap_or_default();
    match children.is_empty() {
        true => format!("<{tag}>...</{tag}>"),
        false => format!("<{tag}>{children}</{tag}>"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `dspy.XMLAdapter.format_assistant_message_content` on dspy 3.3.1, one field at a time.
    #[test]
    fn values_are_written_as_upstream_writes_them() {
        assert_eq!(
            value_to_xml(&json!(["x", "y & z"]), "tags", None),
            "<tags><item>x</item><item>y &amp; z</item></tags>"
        );
        assert_eq!(value_to_xml(&json!([]), "tags", None), "<tags />");
        assert_eq!(
            value_to_xml(&json!({"a": 1, "s": "x<y"}), "m", None),
            "<m><a>1</a><s>x&lt;y</s></m>"
        );
        assert_eq!(
            value_to_xml(
                &json!({"k": 1, "two words": 2, "a-b": 3, "1x": 4, "": 5}),
                "d",
                None
            ),
            "<d><k>1</k><entry key=\"two words\">2</entry><a-b>3</a-b><entry key=\"1x\">4</entry><entry key=\"\">5</entry></d>"
        );
        assert_eq!(value_to_xml(&json!({}), "dk", None), "<dk></dk>");
        assert_eq!(
            value_to_xml(&json!([{"k": "v"}]), "ld", None),
            "<ld><item><k>v</k></item></ld>"
        );
        assert_eq!(
            value_to_xml(&json!({"k": [1, 2]}), "dl", None),
            "<dl><k><item>1</item><item>2</item></k></dl>"
        );
        assert_eq!(
            value_to_xml(&json!({"k": 1.0, "j": 2.5e-7, "i": 1e21}), "d", None),
            "<d><k>1.0</k><j>2.5e-07</j><i>1e+21</i></d>"
        );
        assert_eq!(
            value_to_xml(&json!([true, false]), "lb", None),
            "<lb><item>True</item><item>False</item></lb>"
        );
        assert_eq!(
            value_to_xml(&json!({"k": null, "j": 3}), "dn", None),
            "<dn><k /><j>3</j></dn>"
        );
    }

    /// `xml.sax.saxutils.quoteattr` on Python 3.13.
    #[test]
    fn keys_are_quoted_as_saxutils_quotes_them() {
        assert_eq!(quoteattr("plain"), "\"plain\"");
        assert_eq!(quoteattr("say \"hi\""), "'say \"hi\"'");
        assert_eq!(quoteattr("it's"), "\"it's\"");
        assert_eq!(quoteattr("both ' and \""), "\"both ' and &quot;\"");
        assert_eq!(quoteattr("a&b<c>d"), "\"a&amp;b&lt;c&gt;d\"");
        assert_eq!(
            quoteattr("line\nbreak\ttab\rcr"),
            "\"line&#10;break&#9;tab&#13;cr\""
        );
    }

    #[test]
    fn a_key_names_an_element_when_python_calls_it_an_identifier() {
        assert!(
            names_an_element("k")
                && names_an_element("a-b")
                && names_an_element("a.b")
                && names_an_element("_x1")
                && names_an_element("réponse")
        );
        assert!(
            !names_an_element("two words")
                && !names_an_element("1x")
                && !names_an_element("")
                && !names_an_element("-a")
        );
    }
}
