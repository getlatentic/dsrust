//! dspy `JSONAdapter`: marker sections for the request, one JSON object for the reply.
//!
//! The only format here that asks something of the provider — it engages native structured
//! output, so the schema constrains what the model can emit rather than merely describing it.
//! Inputs still arrive as the marker sections the chat adapter writes, because only the reply's
//! shape differs. It is also what a chat reply falls back to when it does not parse, and what
//! [`baml::BamlAdapter`](super::baml::BamlAdapter) builds on, trading the JSON schema this one
//! states for a compact notation of the same type.

use anyhow::Result;
use serde_json::Value;

use crate::example::Example;
use crate::lm::{ChatTurn, OutputMode};
use crate::signature::Signature;

use super::exchange::{Style, json_answer, plain};
use super::{Adapter, Input, blocks, conversation, live_inputs, output_slot, section};

/// The provider's native structured output, carrying the signature's JSON schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonAdapter {
    /// dspy `use_native_function_calling`: let the provider call tools itself when the model
    /// supports it, rather than asking for the calls as a rendered field.
    ///
    /// **On** by default here, unlike every other adapter — upstream's `JSONAdapter.__init__`
    /// takes `use_native_function_calling: bool = True` and says so: "JSONAdapter uses native
    /// function calling by default". A format that already asks the provider for structured
    /// output asks it for the calls too.
    pub use_native_function_calling: bool,
    /// dspy `parallel_tool_calls`: whether to ask the provider for parallel tool calls while
    /// native function calling is active. `None` leaves the provider option unset, which is
    /// upstream's default and not the same as `Some(false)`.
    pub parallel_tool_calls: Option<bool>,
}

impl Default for JsonAdapter {
    fn default() -> Self {
        Self {
            use_native_function_calling: true,
            parallel_tool_calls: None,
        }
    }
}

impl Adapter for JsonAdapter {
    fn native_function_calling(&self) -> super::NativeFunctionCalling {
        super::NativeFunctionCalling {
            enabled: self.use_native_function_calling,
            parallel: self.parallel_tool_calls,
        }
    }

    fn format(
        &self,
        signature: &Signature,
        demos: &[Example],
        inputs: &[Input<'_>],
    ) -> Result<(String, Vec<ChatTurn>)> {
        let (asked, mut turns) = conversation(signature, demos, inputs, JSON_STYLE, self.use_native_function_calling);
        turns.push(ChatTurn::user(json_user(
            &asked,
            &live_inputs(&asked, inputs),
        )));
        // dspy splits in the base `format`, which both adapters inherit, so a custom type
        // reaches a provider as blocks whichever wire format carries the rest of the request.
        Ok((json_system(signature), blocks::split_custom_types(turns)))
    }

    fn system_message(&self, signature: &Signature) -> Result<String> {
        Ok(json_system(signature))
    }

    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value> {
        super::parse::declared_fields(signature, super::parse::parse_json(raw)?)
    }

    fn output_mode<'a>(&self, schema: &'a Value) -> OutputMode<'a> {
        OutputMode::Json { schema }
    }
}

/// The JSON adapter's exchange: marker sections for the request, one object for the reply.
const JSON_STYLE: Style = Style {
    wrap: section,
    value: plain,
    answer: json_answer,
};

/// The JSON contract in prose, for the provider-native structured-output path.
fn json_system(signature: &Signature) -> String {
    // dspy names the two halves separately here, where the chat adapter runs them together:
    // inputs still arrive as marker sections, but the reply is one JSON object.
    let inputs = signature
        .inputs
        .iter()
        .map(|field| section(&field.name, &format!("{{{}}}", field.name)))
        .collect::<Vec<_>>()
        .join("\n\n");
    // Each slot is the same string the chat adapter puts after the marker, note and all,
    // carried as the field's value so the model reads the constraint where the value goes.
    let outputs: serde_json::Map<String, Value> = signature
        .outputs
        .iter()
        .map(|field| (field.name.clone(), Value::String(output_slot(field))))
        .collect();
    let structure = [
        "All interactions will be structured in the following way, with the appropriate values filled in.",
        "Inputs will have the following structure:",
        &inputs,
        "Outputs will be a JSON object with the following fields.",
        &serde_json::to_string_pretty(&Value::Object(outputs)).unwrap_or_default(),
    ]
    .join("\n\n");
    super::system_message(signature, &structure)
}

/// dspy `user_message_output_requirements` for JSON: the field order, each non-string naming
/// the Python type it must be formatted as.
///
/// The BAML adapter closes its own request with this, as upstream does by subclassing.
pub(super) fn json_output_requirements(signature: &Signature) -> String {
    let fields: Vec<String> = signature
        .outputs
        .iter()
        .map(|field| {
            format!("`{}`{}", field.name, super::output_hint(field))
        })
        .collect();
    format!(
        "Respond with a JSON object in the following order of fields: {}.",
        fields.join(", then ")
    )
}

/// The JSON adapter's user message: the same input sections the chat adapter renders, closed
/// by the reminder that the reply is one JSON object rather than marker blocks.
fn json_user(signature: &Signature, inputs: &[Input<'_>]) -> String {
    let mut parts: Vec<String> = inputs
        .iter()
        .map(|input| section(input.name, &plain(signature, input.name, &input.value)))
        .collect();
    parts.push(json_output_requirements(signature));
    parts.join("\n\n").trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{FieldKind, InField, OutField};
    use serde_json::json;

    fn signature() -> Signature {
        Signature::single_input(
            "Pick a color.",
            vec![
                OutField {
                    name: "color".into(),
                    desc: "the chosen color".into(),
                    values: Some(vec!["red".into(), "blue".into()]),
                    ..Default::default()
                },
                OutField {
                    name: "why".into(),
                    desc: "one short sentence".into(),
                    ..Default::default()
                },
            ],
        )
    }

    fn multi_signature() -> Signature {
        let mut signature = signature();
        signature.inputs = vec![
            InField {
                name: "room".into(),
                desc: "the room being painted".into(),
                ..Default::default()
            },
            InField {
                name: "mood".into(),
                desc: "the mood to set".into(),
                ..Default::default()
            },
        ];
        signature
    }

    #[test]
    fn json_user_renders_input_sections_then_the_json_reminder() {
        // dspy sends inputs as the same marker sections the chat adapter uses; only the
        // closing reminder differs, because only the reply's shape differs.
        let inputs = vec![
            Input::new("room", json!("the study")),
            Input::new("mood", json!("calm focus")),
        ];
        assert_eq!(
            json_user(&multi_signature(), &inputs),
            "[[ ## room ## ]]\nthe study\n\n[[ ## mood ## ]]\ncalm focus\n\n\
             Respond with a JSON object in the following order of fields: \
             `color` (must be formatted as a valid Python Literal['red', 'blue']), then `why`."
        );
    }

    #[test]
    fn a_non_string_output_names_the_python_type_it_must_be() {
        let mut signature = signature();
        signature.outputs[1].kind = FieldKind::Int;
        // A closed set is a `Literal`, which is not `str`, so it earns the hint too.
        assert!(
            json_output_requirements(&signature).ends_with(
                "`color` (must be formatted as a valid Python Literal['red', 'blue']), \
                 then `why` (must be formatted as a valid Python int)."
            ),
            "got: {}",
            json_output_requirements(&signature)
        );
    }

    #[test]
    fn json_system_states_the_input_sections_and_the_output_object() {
        let system = json_system(&multi_signature());
        assert!(system.contains("Inputs will have the following structure:"));
        assert!(system.contains("[[ ## room ## ]]\n{room}"));
        assert!(system.contains("Outputs will be a JSON object with the following fields."));
        // The slot carries the same note the chat template puts after the marker, so the
        // value runs on past the placeholder rather than closing at it.
        assert!(
            system.contains(
                "{\n  \"color\": \"{color}        # note: the value you produce must exactly \
                 match (no extra characters) one of: red; blue\",\n  \"why\": \"{why}\"\n}"
            ),
            "got: {system}"
        );
    }
}
