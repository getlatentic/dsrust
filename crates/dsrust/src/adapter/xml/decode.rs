//! dspy 3.3.1's `XMLAdapter.parse`: the reply as an element tree, each declared field read by its
//! own JSON schema and cast, the defaults every adapter now applies, and the refusals in
//! upstream's words.
use anyhow::Result;
use serde_json::{Map, Value, json};

use super::tokenizer::parse_root;
use super::tree::Element;
use crate::adapter::parse::{FieldMismatch, apply_output_field_defaults, section_value};
use crate::signature::{OutField, Signature};

pub(super) fn parse_xml(signature: &Signature, raw: &str) -> Result<Value> {
    let refusal = |message: Option<String>, parsed: Value, reports_parsed: bool| {
        anyhow::Error::new(FieldMismatch {
            parsed,
            adapter_name: "XMLAdapter".to_owned(),
            lm_response: raw.to_owned(),
            expected_fields: signature.outputs.iter().map(|f| f.name.clone()).collect(),
            signature: signature.clone(),
            message,
            reports_parsed,
        })
    };
    let root = parse_root(raw).map_err(|error| {
        refusal(
            Some(format!("Failed to parse XML: {error}")),
            Value::Null,
            false,
        )
    })?;
    let grouped = group_children(&root);
    let mut fields = Map::new();
    for field in &signature.outputs {
        let Some(elements) = lookup(&grouped, &field.name) else {
            continue;
        };
        let value = read_field(signature, field, elements)
            .map_err(|message| refusal(Some(message), Value::Null, false))?;
        fields.insert(field.name.clone(), value);
    }
    let fields = apply_output_field_defaults(signature, fields);
    if fields.len() != signature.outputs.len() {
        return Err(refusal(None, Value::Object(fields), true));
    }
    Ok(Value::Object(fields))
}

type Grouped<'a> = Vec<(&'a str, Vec<&'a Element>)>;

/// `_group_children`: the direct children by tag, in order of first appearance, an
/// `<entry key="k">` filed under its key.
fn group_children(element: &Element) -> Grouped<'_> {
    let mut groups: Grouped<'_> = Vec::new();
    for child in element.elements() {
        let key = match child.name.as_str() {
            "entry" => child
                .attributes
                .iter()
                .find(|(name, _)| name == "key")
                .map_or("entry", |(_, value)| value.as_str()),
            name => name,
        };
        match groups.iter_mut().find(|(name, _)| *name == key) {
            Some((_, items)) => items.push(child),
            None => groups.push((key, vec![child])),
        }
    }
    groups
}

fn lookup<'a>(groups: &'a Grouped<'a>, name: &str) -> Option<&'a [&'a Element]> {
    groups
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, items)| items.as_slice())
}

/// Upstream tries the field's schema and then each `anyOf` arm in turn, reading the elements by
/// the candidate and casting what that reads; the first cast that fits is the value. When none
/// fits, the refusal names the field, the last value read and the last complaint.
fn read_field(
    signature: &Signature,
    field: &OutField,
    elements: &[&Element],
) -> Result<Value, String> {
    let schema = field.property_schema();
    let definitions = schema
        .get("$defs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let arms: Vec<Value> = schema
        .get("anyOf")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut failure: Option<(Value, anyhow::Error)> = None;
    for candidate in std::iter::once(&schema).chain(arms.iter()) {
        let read = elements_to_value(elements, candidate, &definitions);
        match cast(signature, field, read.clone()) {
            Ok(value) => return Ok(value),
            Err(error) => failure = Some((read, error)),
        }
    }
    let Some((value, error)) = failure else {
        unreachable!("the field's own schema is always tried");
    };
    Err(format!(
        "Failed to parse field {} with value {}: {error}",
        field_info(field),
        crate::python::text(&value)
    ))
}

/// `parse_value(value, annotation)` as this crate casts at parse time: the field's own reader over
/// text, then the scalar casts — under which a `str` field handed several elements reads as
/// `str(list)`, as upstream's does.
fn cast(signature: &Signature, field: &OutField, read: Value) -> Result<Value> {
    let typed = match read {
        Value::String(text) => section_value(field, &text),
        other => other,
    };
    let mut object = json!({ field.name.clone(): typed });
    signature.coerce(&mut object)?;
    Ok(object[&field.name].take())
}

/// `str(field)` for the `FieldInfo` upstream names in the message: pydantic's `annotation=…
/// required=True json_schema_extra={…}`, the extra in the order dspy builds it — what the caller
/// passed first, then `__dspy_field_type`, then the prefix and description it infers. A caller
/// who passed both is taken to have written the description first.
fn field_info(field: &OutField) -> String {
    let quoted = |text: &str| crate::python::repr(&Value::String(text.to_owned()));
    let default_desc = format!("${{{}}}", field.name);
    let given_desc = !field.desc.is_empty() && field.desc != default_desc;
    let mut extra: Vec<(&str, String)> = Vec::new();
    if given_desc {
        extra.push(("desc", quoted(&field.desc)));
    }
    if let Some(prefix) = &field.prefix {
        extra.push(("prefix", quoted(prefix)));
    }
    extra.push(("__dspy_field_type", quoted("output")));
    if field.prefix.is_none() {
        extra.push((
            "prefix",
            quoted(&crate::signature::infer_prefix(&field.name)),
        ));
    }
    if !given_desc {
        extra.push(("desc", quoted(&default_desc)));
    }
    let extra: Vec<String> = extra
        .iter()
        .map(|(key, value)| format!("{}: {value}", quoted(key)))
        .collect();
    format!(
        "annotation={} required=True json_schema_extra={{{}}}",
        field.annotation(),
        extra.join(", ")
    )
}

/// `_elements_to_value`: the elements under one name, read by the JSON schema they should fit.
fn elements_to_value(
    elements: &[&Element],
    schema: &Value,
    definitions: &Map<String, Value>,
) -> Value {
    let resolved = resolve(schema, definitions);
    let mut schema = resolved;
    let first = elements[0];
    let has_children = first.elements().next().is_some();
    if let Some(choices) = schema
        .get("anyOf")
        .and_then(Value::as_array)
        .filter(|c| !c.is_empty())
    {
        let null = json!({ "type": "null" });
        if choices.contains(&null) && !has_children && first.text().trim().is_empty() {
            return Value::Null;
        }
        let typed: Vec<&Value> = choices.iter().filter(|choice| **choice != null).collect();
        let mut chosen = typed.first().copied().unwrap_or(schema);
        if has_children
            && let Some(structured) = typed
                .iter()
                .find(|choice| choice.get("type") != Some(&Value::from("string")))
        {
            chosen = structured;
        }
        schema = chosen;
    }
    if schema.get("type") == Some(&Value::from("array")) {
        if elements.len() == 1 && !has_children {
            let text = first.text().trim();
            if text.is_empty() {
                return Value::Array(Vec::new());
            }
            if text.starts_with('[') {
                return Value::String(text.to_owned());
            }
        }
        let items = schema
            .get("items")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        let grouped = group_children(first);
        let listed: &[&Element] = match (elements.len(), lookup(&grouped, "item")) {
            (1, Some(items_of)) => items_of,
            _ => elements,
        };
        return Value::Array(
            listed
                .iter()
                .map(|element| elements_to_value(&[element], &items, definitions))
                .collect(),
        );
    }
    if schema.get("type") == Some(&Value::from("string")) && has_children {
        return Value::String(first.inner_xml());
    }
    let children = group_children(first);
    if children.is_empty() {
        if schema.get("type") == Some(&Value::from("object")) && first.text().trim().is_empty() {
            return Value::Object(Map::new());
        }
        let mut texts: Vec<Value> = elements
            .iter()
            .map(|element| Value::String(element.text().trim().to_owned()))
            .collect();
        return match texts.len() {
            1 => texts.remove(0),
            _ => Value::Array(texts),
        };
    }
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let additional = match schema.get("additionalProperties") {
        Some(Value::Object(object)) => Value::Object(object.clone()),
        _ => Value::Object(Map::new()),
    };
    Value::Object(
        children
            .into_iter()
            .map(|(name, items)| {
                let child_schema = properties.get(name).unwrap_or(&additional);
                (
                    name.to_owned(),
                    elements_to_value(&items, child_schema, definitions),
                )
            })
            .collect(),
    )
}

/// `definitions.get(schema.get("$ref", "").rsplit("/", 1)[-1], schema)`.
fn resolve<'a>(schema: &'a Value, definitions: &'a Map<String, Value>) -> &'a Value {
    schema
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| definitions.get(reference.rsplit('/').next().unwrap_or(reference)))
        .unwrap_or(schema)
}
