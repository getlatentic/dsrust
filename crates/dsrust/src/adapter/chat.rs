//! dspy `ChatAdapter`: every field in its own `[[ ## name ## ]]` section.
//!
//! DSPy's default and this crate's, because the format asks nothing of the provider — any model
//! that can write text can write a marker section. The field lists, the demos, the conversation
//! history and the objective come from the shared assembly; what this module decides is the
//! section itself: how a field is written, how a reply is read back, and the closing reminder
//! that names the output fields in order and ends at `[[ ## completed ## ]]`.

use anyhow::Result;
use serde_json::Value;

use crate::example::Example;
use crate::lm::ChatTurn;
use crate::signature::Signature;

use super::Input;
use super::exchange::{Style, answer, plain};
use super::{
    Adapter, JsonAdapter, blocks, conversation, live_inputs, marker, output_slot, section,
};

/// DSPy's default: every field in its own `[[ ## name ## ]]` section, readable by any model.
#[derive(Debug, Clone, Copy)]
pub struct ChatAdapter {
    /// Re-ask through [`JsonAdapter`] when a reply does not speak the marker format. On by
    /// default, matching dspy's `use_json_adapter_fallback`.
    pub use_json_adapter_fallback: bool,
    /// dspy `use_native_function_calling`: let the provider call tools itself when the model
    /// supports it. Off by default, as upstream has it.
    pub use_native_function_calling: bool,
    /// dspy `parallel_tool_calls`: `None` leaves the provider option unset, which is upstream's
    /// default and not the same as `Some(false)`.
    pub parallel_tool_calls: Option<bool>,
}

impl Default for ChatAdapter {
    fn default() -> Self {
        Self {
            use_json_adapter_fallback: true,
            use_native_function_calling: false,
            parallel_tool_calls: None,
        }
    }
}

impl ChatAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// A parse failure becomes final rather than a second ask in JSON.
    pub fn without_json_fallback() -> Self {
        Self {
            use_json_adapter_fallback: false,
            ..Self::default()
        }
    }

    /// Let the provider call tools itself where the model supports it — dspy's
    /// `use_native_function_calling`.
    pub fn use_native_function_calling(mut self, native: bool) -> Self {
        self.use_native_function_calling = native;
        self
    }

    /// Ask the provider for parallel tool calls while native function calling is active. Leaving
    /// this unset is not the same as setting it false: dspy only sends the option when it is set.
    pub fn parallel_tool_calls(mut self, parallel: Option<bool>) -> Self {
        self.parallel_tool_calls = parallel;
        self
    }

    /// dspy `_make_json_adapter_fallback`: the JSON adapter a failed parse re-asks through, or
    /// none where the fallback is off. It carries this adapter's native-function-calling settings,
    /// so the second attempt asks the provider exactly as the first did.
    pub fn json_fallback_adapter(&self) -> Option<JsonAdapter> {
        self.use_json_adapter_fallback.then_some(JsonAdapter {
            use_native_function_calling: self.use_native_function_calling,
            parallel_tool_calls: self.parallel_tool_calls,
        })
    }
}

impl Adapter for ChatAdapter {
    fn name(&self) -> &'static str {
        "ChatAdapter"
    }

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
        let (asked, mut turns) = conversation(
            signature,
            demos,
            inputs,
            MARKER_STYLE,
            self.use_native_function_calling,
        );
        turns.push(ChatTurn::user(chat_user(
            &asked,
            &live_inputs(&asked, inputs),
        )));
        Ok((
            self.system_message(signature)?,
            blocks::split_custom_types(turns),
        ))
    }

    fn system_message(&self, signature: &Signature) -> Result<String> {
        Ok(chat_system(signature))
    }

    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value> {
        super::parse::parse_markers(signature, raw)
    }

    fn json_fallback(&self) -> Option<Box<dyn Adapter>> {
        self.json_fallback_adapter()
            .map(|adapter| Box::new(adapter) as Box<dyn Adapter>)
    }
}

/// The chat adapter's exchange: marker sections both ways.
pub(super) const MARKER_STYLE: Style = Style {
    wrap: section,
    value: plain,
    answer,
};

/// DSPy ChatAdapter's system message: numbered input and output field lists, the
/// marker-structured exchange template ending at `[[ ## completed ## ]]`, then the task
/// objective.
fn chat_system(signature: &Signature) -> String {
    // dspy `format_field_structure`: the template blocks join with a blank line and the whole
    // section is stripped, so the trailing newline after `completed` never survives.
    let block = |slots: Vec<(&str, String)>| -> String {
        slots
            .iter()
            .map(|(name, slot)| section(name, slot))
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    let inputs = signature
        .inputs
        .iter()
        // dspy `translate_field_type` returns an empty note for every input: the model reads
        // input values, it does not produce them, so there is nothing to constrain.
        .map(|field| (field.name.as_str(), format!("{{{}}}", field.name)))
        .collect();
    let outputs = signature
        .outputs
        .iter()
        .map(|field| (field.name.as_str(), output_slot(field)))
        .collect();
    let structure = [
        "All interactions will be structured in the following way, with the appropriate values filled in.".to_owned(),
        block(inputs),
        block(outputs),
        format!("{}\n", marker("completed")),
    ]
    .join("\n\n");

    super::system_message(signature, &structure)
}

/// DSPy ChatAdapter's user message: each input in its own marker section, then the recap of
/// the output field order.
fn chat_user(signature: &Signature, inputs: &[Input<'_>]) -> String {
    // dspy `format_user_message_content`: input sections and the reminder are one list joined
    // by a blank line and stripped, rather than sections each carrying their own trailing gap.
    let mut parts: Vec<String> = inputs
        .iter()
        .map(|input| section(input.name, &plain(signature, input.name, &input.value)))
        .collect();
    parts.push(output_requirements(signature));
    parts.join("\n\n").trim().to_owned()
}

/// dspy `user_message_output_requirements`: the closing reminder of field order, where every
/// non-string output repeats its Python type so a long conversation cannot drift off-format.
/// dspy branches on the annotation rather than the wire type, so a closed set earns the hint —
/// it is a `Literal[...]`, not a plain `str`.
fn output_requirements(signature: &Signature) -> String {
    let fields: Vec<String> = signature
        .outputs
        .iter()
        .map(|field| format!("`{}`{}", marker(&field.name), super::output_hint(field)))
        .collect();
    format!(
        "Respond with the corresponding output fields, starting with the field {}, and then \
         ending with the marker for `{}`.",
        fields.join(", then "),
        marker("completed"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{FieldKind, InField, JsonType, OutField, TypeDescription};
    use serde_json::json;

    /// A structured output whose annotation names a custom type, the way `Citations` or `Code`
    /// reach the crate across the bridge.
    fn custom_output(annotation: &str, description: &str) -> Signature {
        custom_output_with(annotation, description, false)
    }

    /// `replaces_schema` is the property dspy reads off `dspy.Code` and no other type.
    fn custom_output_with(annotation: &str, description: &str, replaces_schema: bool) -> Signature {
        Signature::single_input(
            "Answer.",
            vec![OutField {
                name: "answer".into(),
                kind: FieldKind::Json(JsonType {
                    annotation: annotation.to_owned(),
                    descriptions: vec![TypeDescription {
                        name: annotation.to_owned(),
                        text: description.to_owned(),
                        replaces_schema,
                    }],
                    reflection: None,
                }),
                schema: Some(json!({ "type": "object" })),
                ..Default::default()
            }],
        )
    }

    #[test]
    fn a_custom_type_states_its_description_under_the_field_line() {
        let system = chat_system(&custom_output("Citations", "Citations with quoted text."));
        assert!(
            system.contains(
                "1. `answer` (Citations): \n    Type description of Citations: Citations with quoted text."
            ),
            "got: {system}"
        );
    }

    #[test]
    fn every_custom_type_in_one_annotation_earns_its_own_line() {
        // dspy walks the annotation and appends a line per type it finds, so a nested one is
        // announced as fully as a bare one.
        let mut signature = custom_output("Citations", "Quoted text.");
        let FieldKind::Json(json) = &mut signature.outputs[0].kind else {
            unreachable!("built as a structured field")
        };
        json.descriptions.push(TypeDescription {
            name: "Document".to_owned(),
            text: "A source.".to_owned(),
            replaces_schema: false,
        });
        let system = chat_system(&signature);
        assert!(system.contains("\n    Type description of Citations: Quoted text."));
        assert!(system.contains("\n    Type description of Document: A source."));
    }

    #[test]
    fn a_type_with_no_prose_adds_no_line() {
        let mut signature = custom_output("Citations", "");
        let system = chat_system(&signature);
        assert!(!system.contains("Type description"), "got: {system}");
        // And the same field with prose does add one, so the emptiness is what silenced it.
        signature = custom_output("Citations", "Quoted text.");
        assert!(chat_system(&signature).contains("Type description"));
    }

    #[test]
    fn code_states_its_contract_once_rather_than_repeating_its_schema() {
        // dspy drops the schema note for `Code` alone: its type description already says what
        // the field must contain, and the schema block is large.
        let system = chat_system(&custom_output_with(
            "Code",
            "Code represented in a string.",
            true,
        ));
        assert!(
            system.contains("Type description of Code:"),
            "got: {system}"
        );
        assert!(
            !system.contains("must adhere to the JSON schema"),
            "got: {system}"
        );
    }

    #[test]
    fn a_type_merely_named_code_keeps_its_schema() {
        // dspy asks whether the annotation *is* a `dspy.Code`, so a look-alike is an ordinary
        // custom type. Deciding on the printed name would drop this field's schema.
        let system = chat_system(&custom_output("Code", "Some unrelated type."));
        assert!(
            system.contains("must adhere to the JSON schema"),
            "got: {system}"
        );
    }

    #[test]
    fn another_custom_type_keeps_the_schema_that_steers_its_reply() {
        let system = chat_system(&custom_output("Citations", "Citations with quoted text."));
        assert!(
            system.contains("must adhere to the JSON schema"),
            "got: {system}"
        );
    }

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

    fn typed_signature() -> Signature {
        let mut signature = Signature::single_input(
            "Size the gift.",
            vec![
                OutField {
                    name: "amount".into(),
                    desc: "amount in MON".into(),
                    kind: FieldKind::Float,
                    ..Default::default()
                },
                OutField {
                    name: "double".into(),
                    desc: "double it".into(),
                    kind: FieldKind::Bool,
                    ..Default::default()
                },
            ],
        );
        signature.inputs = vec![InField {
            name: "age".into(),
            desc: "the age turned".into(),
            kind: FieldKind::Int,
            ..Default::default()
        }];
        signature
    }

    fn json_signature() -> Signature {
        let mut signature = Signature::single_input(
            "Suggest ideas.",
            vec![OutField {
                name: "ideas".into(),
                desc: "three concrete ideas".into(),
                kind: FieldKind::opaque_json(),
                schema: Some(json!({ "type": "array", "items": { "type": "string" } })),
                ..Default::default()
            }],
        );
        signature.inputs = vec![InField {
            name: "recipient".into(),
            desc: "who the gift is for".into(),
            kind: FieldKind::opaque_json(),
            ..Default::default()
        }];
        signature
    }

    fn single_request(value: &str) -> Vec<Input<'static>> {
        vec![Input::new("request", Value::String(value.to_owned()))]
    }

    #[test]
    fn chat_system_lists_fields_structure_and_objective() {
        let system = chat_system(&signature());
        assert!(system.contains("Your input fields are:\n1. `request` (str): the request"));
        // dspy types a closed set as `Literal[...]` and prints it as the annotation, with
        // nothing trailing the description.
        assert!(system.contains("1. `color` (Literal['red', 'blue']): the chosen color"));
        assert!(system.contains("2. `why` (str): one short sentence"));
        assert!(system.contains("[[ ## request ## ]]\n{request}"));
        assert!(system.contains("[[ ## color ## ]]\n{color}"));
        assert!(system.contains("[[ ## completed ## ]]"));
        // dspy indents the instruction onto its own line, so the sentence ends in a space.
        assert!(system.ends_with(
            "In adhering to this structure, your objective is: \n        Pick a color."
        ));
    }

    /// Copied from `dspy.ChatAdapter().format(...)` over the same signature: a closed set is a
    /// `Literal[...]` in both the numbered line and the closing reminder, because dspy branches
    /// on the annotation and a `Literal` is not `str`.
    #[test]
    fn a_closed_set_renders_as_dspys_literal_annotation() {
        let signature = signature();
        assert!(
            chat_system(&signature).starts_with(
                "Your input fields are:\n\
                 1. `request` (str): the request\n\
                 Your output fields are:\n\
                 1. `color` (Literal['red', 'blue']): the chosen color\n\
                 2. `why` (str): one short sentence\n"
            ),
            "got: {}",
            chat_system(&signature)
        );
        assert_eq!(
            output_requirements(&signature),
            "Respond with the corresponding output fields, starting with the field \
             `[[ ## color ## ]]` (must be formatted as a valid Python Literal['red', 'blue']), \
             then `[[ ## why ## ]]`, and then ending with the marker for \
             `[[ ## completed ## ]]`."
        );
    }

    #[test]
    fn numbered_lines_annotate_each_field_with_its_kind() {
        let system = chat_system(&typed_signature());
        assert!(system.contains("1. `age` (int): the age turned"));
        assert!(system.contains("1. `amount` (float): amount in MON"));
        assert!(system.contains("2. `double` (bool): double it"));
    }

    #[test]
    fn a_json_fields_schema_reaches_the_prompt_through_its_slot_alone() {
        // Upstream states the schema once. `get_field_description_string` stops at the
        // description, and the slot carries the note — spaced the way `json.dumps` writes it.
        let system = chat_system(&json_signature());
        assert!(system.contains("1. `recipient` (json): who the gift is for\n"));
        assert!(
            system.contains("1. `ideas` (json): three concrete ideas\n"),
            "got: {system}"
        );
        assert!(!system.contains("json matching schema"), "got: {system}");
        assert!(
            system.contains(
                "{ideas}        # note: the value you produce must adhere to the JSON schema: \
                 {\"type\": \"array\", \"items\": {\"type\": \"string\"}}"
            ),
            "got: {system}"
        );
    }

    #[test]
    fn chat_system_numbers_every_input_field_before_the_template() {
        let system = chat_system(&multi_signature());
        assert!(system.contains(
            "Your input fields are:\n1. `room` (str): the room being painted\n2. `mood` (str): the mood to set"
        ));
        let template = "[[ ## room ## ]]\n{room}\n\n[[ ## mood ## ]]\n{mood}\n\n[[ ## color ## ]]";
        assert!(system.contains(template));
    }

    #[test]
    fn chat_user_carries_input_and_field_order() {
        let user = chat_user(&signature(), &single_request("Recipient: Dad"));
        assert!(user.starts_with("[[ ## request ## ]]\nRecipient: Dad"));
        assert!(user.contains("starting with the field `[[ ## color ## ]]`"));
        assert!(user.contains(", then `[[ ## why ## ]]`"));
        assert!(user.contains("ending with the marker for `[[ ## completed ## ]]`"));
    }

    #[test]
    fn chat_user_renders_each_input_as_its_own_section_then_recaps_outputs() {
        let inputs = vec![
            Input::new("room", json!("the study")),
            Input::new("mood", json!("calm focus")),
        ];
        let user = chat_user(&multi_signature(), &inputs);
        assert!(user.starts_with(
            "[[ ## room ## ]]\nthe study\n\n[[ ## mood ## ]]\ncalm focus\n\nRespond with"
        ));
        assert!(user.contains("starting with the field `[[ ## color ## ]]`"));
        assert!(!user.contains("`[[ ## room ## ]]`"));
    }
}
