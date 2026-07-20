//! One solved exchange: the user turn an example would have sent, and the assistant turn it
//! produced.
//!
//! Few-shot demos and conversation history are the same shape to a model — a request already
//! answered — and dspy renders both through one pair of functions, varying only the prefix and
//! the stand-in for a field the example never carried.

use serde_json::Value;

use crate::adapter::python_json::format_field_value;
use crate::example::Example;
use crate::lm::ChatTurn;
use crate::signature::Signature;

use super::{marker, section};

/// How an adapter writes one field and its value. dspy calls this `format_field_with_value`
/// and each adapter overrides it: marker sections, XML tags, a JSON member.
pub(super) type Wrap = fn(&str, &str) -> String;

/// How an adapter writes one input's value, given the field it belongs to. Most write every
/// value the one way dspy's `format_field_value` does; a format that lays some values out
/// differently decides that here, so a demo and a live request agree on it.
pub(super) type Render = fn(&Signature, &str, &Value) -> String;

/// dspy `format_field_value`, which reads nothing off the field but the value itself.
pub(super) fn plain(_: &Signature, _: &str, value: &Value) -> String {
    format_field_value(value)
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
                Some(value) => format_field_value(value),
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
                Some(value) => value.clone(),
                None => Value::String(missing?.to_owned()),
            };
            Some((field.name.clone(), value))
        })
        .collect();
    ChatTurn::assistant(serde_json::to_string_pretty(&Value::Object(fields)).unwrap_or_default())
}
