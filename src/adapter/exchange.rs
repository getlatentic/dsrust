//! One solved exchange: the user turn an example would have sent, and the assistant turn it
//! produced.
//!
//! Few-shot demos and conversation history are the same shape to a model — a request already
//! answered — and dspy renders both through one pair of functions, varying only the prefix and
//! the stand-in for a field the example never carried.

use serde_json::Value;

use crate::adapter::python_json::{format_field_value, format_value};
use crate::example::Example;
use crate::lm::ChatTurn;
use crate::signature::{FieldKind, Signature};

use super::{marker, section};

/// How an adapter writes one field and its value. dspy calls this `format_field_with_value`
/// and each adapter overrides it: marker sections, XML tags, a JSON member.
pub(super) type Wrap = fn(&str, &str) -> String;

/// How an adapter writes one input's value, given the field it belongs to. Most write every
/// value the one way dspy's `format_field_value` does; a format that lays some values out
/// differently decides that here, so a demo and a live request agree on it.
pub(super) type Render = fn(&Signature, &str, &Value) -> String;

/// dspy `format_field_value`: the value as the field it belongs to has it read.
///
/// The field is looked up rather than passed because the callers hold a name and a value —
/// which is the shape dspy's own `format_field_value(field_info, value)` has, once its caller
/// has resolved the name.
pub(super) fn plain(signature: &Signature, name: &str, value: &Value) -> String {
    match kind_of(signature, name) {
        Some(kind) => format_field_value(kind, value),
        None => format_value(value),
    }
}

/// The declared kind of a field, whichever half of the signature declares it.
pub(super) fn kind_of<'a>(signature: &'a Signature, name: &str) -> Option<&'a FieldKind> {
    let input = signature
        .inputs
        .iter()
        .find(|f| f.name == name)
        .map(|f| &f.kind);
    input.or_else(|| {
        signature
            .outputs
            .iter()
            .find(|f| f.name == name)
            .map(|f| &f.kind)
    })
}

/// The user turn an example would have sent. Only the inputs it carries appear: dspy leaves a
/// missing input out entirely rather than marking it, since the prefix already says so.
pub(super) fn ask(
    signature: &Signature,
    example: &Example,
    prefix: Option<&str>,
    style: Style,
) -> ChatTurn {
    let sections = signature.inputs.iter().filter_map(|field| {
        let value = example.get(&field.name)?;
        Some((style.wrap)(
            &field.name,
            &(style.value)(signature, &field.name, value),
        ))
    });
    let parts: Vec<String> = prefix
        .map(str::to_owned)
        .into_iter()
        .chain(sections)
        .collect();
    ChatTurn::user(parts.join("\n\n").trim().to_owned())
}

/// The assistant turn an example produced. Every output field earns a marker even when the
/// example lacks it, so the model always reads the full set of sections it is asked to produce.
pub(super) fn answer(signature: &Signature, example: &Example, missing: Option<&str>) -> ChatTurn {
    let sections: Vec<String> = signature
        .outputs
        .iter()
        .filter_map(|field| {
            let value = match example.get(&field.name) {
                Some(value) => format_field_value(&field.kind, value),
                None => missing?.to_owned(),
            };
            Some(section(&field.name, &value))
        })
        .collect();
    // dspy strips the field block before appending the marker, never after.
    ChatTurn::assistant(format!(
        "{}\n\n{}\n",
        sections.join("\n\n").trim(),
        marker("completed")
    ))
}

/// How an adapter writes the assistant half of an exchange.
pub(super) type Answer = fn(&Signature, &Example, Option<&str>) -> ChatTurn;

/// How one adapter writes an already-answered exchange: the field form its requests use, how a
/// value is laid out inside that form, and the shape its replies take. dspy spreads these across
/// `format_field_with_value`, `format_user_message_content` and
/// `format_assistant_message_content`; a demo and a history entry both need the set.
#[derive(Clone, Copy)]
pub(super) struct Style {
    pub wrap: Wrap,
    pub value: Render,
    pub answer: Answer,
}

/// The assistant turn as the JSON adapter writes it: the object the model would have returned,
/// rather than the marker sections the chat adapter reads back.
pub(super) fn json_answer(
    signature: &Signature,
    example: &Example,
    missing: Option<&str>,
) -> ChatTurn {
    let fields: serde_json::Map<String, Value> = signature
        .outputs
        .iter()
        .filter_map(|field| {
            let value = match example.get(&field.name) {
                Some(value) => typed_demo_value(field, value.clone()),
                None => Value::String(missing?.to_owned()),
            };
            Some((field.name.clone(), value))
        })
        .collect();
    ChatTurn::assistant(serde_json::to_string_pretty(&Value::Object(fields)).unwrap_or_default())
}

/// dspy dumps a demo's outputs into the JSON object as the types they were declared, so a float
/// reads `0.9` and not `"0.9"`. A value that reached this crate as text is read back to its
/// field's type; one that does not fit stays exactly as it arrived, which is what dspy would
/// print for it rather than an error a demo has no way to report.
fn typed_demo_value(field: &crate::signature::OutField, value: Value) -> Value {
    let mut typed = value.clone();
    match crate::signature::coerce_value(&field.kind, &field.name, &mut typed) {
        Ok(()) => typed,
        Err(_) => value,
    }
}
