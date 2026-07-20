use anyhow::Result;
use serde_json::Value;

use crate::example::Example;
use crate::lm::{ChatTurn, DynChatModel, OutputMode};
use crate::signature::{FieldKind, Signature};

pub mod baml;
mod blocks;
mod two_step;
pub use two_step::{TwoStepAdapter, extractor_signature};
mod demos;
mod exchange;
mod history;
pub mod parse;
mod prompt;
pub mod python_json;
pub mod xml;

use demos::demo_turns;
use prompt::{marker, output_slot, section, system_message};
use python_json::format_field_value;

/// How a signature travels over the wire.
///
/// Mirrors DSPy's `Adapter` base class: implement it to teach the crate a new wire format.
/// [`ChatAdapter`] speaks `[[ ## field ## ]]` marker sections any model can produce with no
/// provider support, and is DSPy's default and ours; [`xml::XmlAdapter`] wraps the same fields
/// in tags, which some models follow more reliably. [`JsonAdapter`] engages the provider's
/// native structured output, and [`baml::BamlAdapter`] builds on it, trading the JSON schema it
/// states for a compact notation of the same type.
///
/// Like DSPy, a parse failure is final: there is no silent retry in another format, because a
/// caller who chose an adapter chose the wire contract it implies.
///
/// Formatting and parsing live here; the model call lives in the module that owns the
/// conversation. That split keeps this trait object-safe, so a caller can hold
/// `Box<dyn Adapter>` and swap wire formats at run time.
pub trait Adapter: Send + Sync {
    /// The whole conversation to send, with no model call: the system message, then the
    /// turns. Mirrors `Adapter.format`, which returns a message list for the same reason —
    /// a demo or a conversation history expands into several turns, not one.
    ///
    /// `demos` are the solved examples that precede the real request. An optimizer's whole
    /// output is a set of these, so an adapter that cannot render them cannot run a compiled
    /// program.
    fn format(
        &self,
        signature: &Signature,
        demos: &[Example],
        inputs: &[(&str, Value)],
    ) -> (String, Vec<ChatTurn>);

    /// Extract the signature's fields from a raw reply. A reply that does not speak this
    /// adapter's format at all fails here; a reply missing individual fields parses and
    /// leaves those gaps for the signature's own validation, whose failure carries feedback
    /// into a retry.
    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value>;

    /// What this adapter tells the model before the conversation starts: the fields, the shape
    /// of an interaction, and the objective. dspy exposes the same method, and its `format`
    /// builds the exchange around it.
    fn system_message(&self, signature: &Signature) -> String;

    /// How the provider should be asked to shape its reply. Text by default, since a format
    /// carried entirely in the prompt needs nothing from the provider.
    fn output_mode<'a>(&self, _schema: &'a Value) -> OutputMode<'a> {
        OutputMode::Text
    }

    /// The adapter to re-ask through when a reply fails to parse, if any.
    ///
    /// dspy's `ChatAdapter.__call__` catches a parse failure and retries the whole exchange
    /// through `JSONAdapter`, which its `use_json_adapter_fallback` flag disables. Most
    /// adapters have no second opinion to offer, so the default is none.
    fn json_fallback(&self) -> Option<Box<dyn Adapter>> {
        None
    }

    /// A second exchange this adapter needs before its reply carries the signature's fields.
    ///
    /// dspy's `TwoStepAdapter` lets the main model answer in prose, then asks a second model to
    /// pull the fields out of that prose. The second ask is a model call, which this trait
    /// cannot make and stay object-safe — so, exactly as [`Adapter::json_fallback`] hands back
    /// an adapter for the module to re-ask through, this hands back everything the module needs
    /// to run the extraction itself. Most adapters read their own replies and answer none.
    fn extraction(&self, _signature: &Signature) -> Option<Extraction<'_>> {
        None
    }
}

/// The second ask an adapter cannot make for itself: what to ask, how to render it, and which
/// model to ask.
pub struct Extraction<'a> {
    /// dspy's `_create_extractor_signature`: `text` in, the original outputs out.
    pub signature: Signature,
    /// The wire format the extraction speaks, which is dspy's `ChatAdapter`.
    pub adapter: &'a dyn Adapter,
    /// The model asked to do the extracting — a smaller one than answered the task.
    pub model: &'a dyn DynChatModel,
}

/// DSPy's default: every field in its own `[[ ## name ## ]]` section, readable by any model.
#[derive(Debug, Clone, Copy)]
pub struct ChatAdapter {
    /// Re-ask through [`JsonAdapter`] when a reply does not speak the marker format. On by
    /// default, matching dspy's `use_json_adapter_fallback`.
    pub use_json_adapter_fallback: bool,
}

impl Default for ChatAdapter {
    fn default() -> Self {
        Self {
            use_json_adapter_fallback: true,
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
        }
    }
}

/// The provider's native structured output, carrying the signature's JSON schema.
#[derive(Debug, Clone, Copy, Default)]
pub struct JsonAdapter;

impl Adapter for ChatAdapter {
    fn format(
        &self,
        signature: &Signature,
        demos: &[Example],
        inputs: &[(&str, Value)],
    ) -> (String, Vec<ChatTurn>) {
        let (asked, mut turns) = conversation(signature, demos, inputs, MARKER_STYLE);
        turns.push(ChatTurn::user(chat_user(
            &asked,
            &live_inputs(&asked, inputs),
        )));
        (
            self.system_message(signature),
            blocks::split_custom_types(turns),
        )
    }

    fn system_message(&self, signature: &Signature) -> String {
        chat_system(signature)
    }

    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value> {
        parse::parse_markers(signature, raw)
    }

    fn json_fallback(&self) -> Option<Box<dyn Adapter>> {
        self.use_json_adapter_fallback
            .then(|| Box::new(JsonAdapter) as Box<dyn Adapter>)
    }
}

impl Adapter for JsonAdapter {
    fn format(
        &self,
        signature: &Signature,
        demos: &[Example],
        inputs: &[(&str, Value)],
    ) -> (String, Vec<ChatTurn>) {
        let (asked, mut turns) = conversation(signature, demos, inputs, JSON_STYLE);
        turns.push(ChatTurn::user(json_user(
            &asked,
            &live_inputs(&asked, inputs),
        )));
        // dspy splits in the base `format`, which both adapters inherit, so a custom type
        // reaches a provider as blocks whichever wire format carries the rest of the request.
        (json_system(signature), blocks::split_custom_types(turns))
    }

    fn system_message(&self, signature: &Signature) -> String {
        json_system(signature)
    }

    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value> {
        parse::declared_fields(signature, parse::parse_json(raw)?)
    }

    fn output_mode<'a>(&self, schema: &'a Value) -> OutputMode<'a> {
        OutputMode::Json { schema }
    }
}

/// A rejected reply carried into the retry turn: the model sees its own previous output
/// and a precise statement of what was wrong and what is required.
pub struct Feedback {
    pub previous: String,
    pub error: String,
}

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

    system_message(signature, &structure)
}

/// The chat adapter's exchange: marker sections both ways.
const MARKER_STYLE: exchange::Style = exchange::Style {
    wrap: section,
    value: exchange::plain,
    answer: exchange::answer,
};

/// The JSON adapter's: marker sections for the request, one object for the reply.
const JSON_STYLE: exchange::Style = exchange::Style {
    wrap: section,
    value: exchange::plain,
    answer: exchange::json_answer,
};

/// Everything before the live request: the demos, then any conversation history, and the
/// signature the request itself is rendered against.
///
/// dspy assembles this in its base `format` for both adapters, which is why a history field is
/// replayed and hidden the same way whichever wire format carries the request. Only the
/// assistant half of each exchange differs, so that renderer is the parameter.
fn conversation(
    signature: &Signature,
    demos: &[Example],
    inputs: &[(&str, Value)],
    style: exchange::Style,
) -> (Signature, Vec<ChatTurn>) {
    let mut turns = demo_turns(signature, demos, style);
    let asked = match history::field_name(signature) {
        None => signature.clone(),
        Some(name) => {
            let stripped = history::without_field(signature, name);
            if let Some((_, value)) = inputs.iter().find(|(field, _)| *field == name) {
                turns.extend(history::turns(&stripped, value, style));
            }
            stripped
        }
    };
    (asked, turns)
}

/// The inputs the request renders: everything the asked-for signature still declares, which
/// drops the history field once its exchanges have been replayed.
fn live_inputs<'a>(asked: &Signature, inputs: &[(&'a str, Value)]) -> Vec<(&'a str, Value)> {
    inputs
        .iter()
        .filter(|(field, _)| asked.inputs.iter().any(|input| input.name == *field))
        .cloned()
        .collect()
}

/// DSPy ChatAdapter's user message: each input in its own marker section, then the recap of
/// the output field order.
fn chat_user(signature: &Signature, inputs: &[(&str, Value)]) -> String {
    // dspy `format_user_message_content`: input sections and the reminder are one list joined
    // by a blank line and stripped, rather than sections each carrying their own trailing gap.
    let mut parts: Vec<String> = inputs
        .iter()
        .map(|(name, value)| section(name, &format_field_value(value)))
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
        .map(|field| {
            let annotation = field.annotation();
            let hint = match annotation == FieldKind::Str.annotation() {
                true => String::new(),
                false => format!(" (must be formatted as a valid Python {annotation})"),
            };
            format!("`{}`{hint}", marker(&field.name))
        })
        .collect();
    format!(
        "Respond with the corresponding output fields, starting with the field {}, and then \
         ending with the marker for `{}`.",
        fields.join(", then "),
        marker("completed"),
    )
}

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
    system_message(signature, &structure)
}

/// dspy `user_message_output_requirements` for JSON: the field order, each non-string naming
/// the Python type it must be formatted as.
fn json_output_requirements(signature: &Signature) -> String {
    let fields: Vec<String> = signature
        .outputs
        .iter()
        .map(|field| {
            let annotation = field.annotation();
            let hint = match annotation == FieldKind::Str.annotation() {
                true => String::new(),
                false => format!(" (must be formatted as a valid Python {annotation})"),
            };
            format!("`{}`{hint}", field.name)
        })
        .collect();
    format!(
        "Respond with a JSON object in the following order of fields: {}.",
        fields.join(", then ")
    )
}

/// The JSON adapter's user message: the same input sections the chat adapter renders, closed
/// by the reminder that the reply is one JSON object rather than marker blocks.
fn json_user(signature: &Signature, inputs: &[(&str, Value)]) -> String {
    let mut parts: Vec<String> = inputs
        .iter()
        .map(|(name, value)| section(name, &format_field_value(value)))
        .collect();
    parts.push(json_output_requirements(signature));
    parts.join("\n\n").trim().to_owned()
}

/// The turns a module sends for one attempt: whatever the adapter rendered, plus the rejected
/// reply and its error when this is a feedback retry.
pub fn turns_for(mut turns: Vec<ChatTurn>, feedback: Option<&Feedback>) -> Vec<ChatTurn> {
    if let Some(feedback) = feedback {
        turns.push(ChatTurn::assistant(feedback.previous.clone()));
        turns.push(ChatTurn::user(format!(
            "Your previous reply was rejected: {}. Send the corrected reply now, in the same \
             format, with every output field present and valid.",
            feedback.error
        )));
    }
    turns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{InField, JsonType, OutField, TypeDescription};
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
                desc: String::new(),
                kind: FieldKind::Json(JsonType {
                    annotation: annotation.to_owned(),
                    descriptions: vec![TypeDescription {
                        name: annotation.to_owned(),
                        text: description.to_owned(),
                        replaces_schema,
                    }],
                    reflection: None,
                }),
                values: None,
                schema: Some(json!({ "type": "object" })),
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

    fn multi_signature() -> Signature {
        let mut signature = signature();
        signature.inputs = vec![
            InField {
                name: "room".into(),
                desc: "the room being painted".into(),
                kind: FieldKind::Str,
                values: None,
            },
            InField {
                name: "mood".into(),
                desc: "the mood to set".into(),
                kind: FieldKind::Str,
                values: None,
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
                    values: None,
                    schema: None,
                },
                OutField {
                    name: "double".into(),
                    desc: "double it".into(),
                    kind: FieldKind::Bool,
                    values: None,
                    schema: None,
                },
            ],
        );
        signature.inputs = vec![InField {
            name: "age".into(),
            desc: "the age turned".into(),
            kind: FieldKind::Int,
            values: None,
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
                values: None,
                schema: Some(json!({ "type": "array", "items": { "type": "string" } })),
            }],
        );
        signature.inputs = vec![InField {
            name: "recipient".into(),
            desc: "who the gift is for".into(),
            kind: FieldKind::opaque_json(),
            values: None,
        }];
        signature
    }

    fn single_request(value: &str) -> Vec<(&'static str, Value)> {
        vec![("request", Value::String(value.to_owned()))]
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
        let inputs = vec![("room", json!("the study")), ("mood", json!("calm focus"))];
        let user = chat_user(&multi_signature(), &inputs);
        assert!(user.starts_with(
            "[[ ## room ## ]]\nthe study\n\n[[ ## mood ## ]]\ncalm focus\n\nRespond with"
        ));
        assert!(user.contains("starting with the field `[[ ## color ## ]]`"));
        assert!(!user.contains("`[[ ## room ## ]]`"));
    }

    #[test]
    fn json_user_renders_input_sections_then_the_json_reminder() {
        // dspy sends inputs as the same marker sections the chat adapter uses; only the
        // closing reminder differs, because only the reply's shape differs.
        let inputs = vec![("room", json!("the study")), ("mood", json!("calm focus"))];
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

    #[test]
    fn conversation_appends_previous_output_and_error_on_retry() {
        let feedback = Feedback {
            previous: "[[ ## color ## ]]\ngreen".into(),
            error: "color must be one of red, blue; got \"green\"".into(),
        };
        let turns = turns_for(vec![ChatTurn::user("draft it")], Some(&feedback));
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[1].content.text().unwrap(), "[[ ## color ## ]]\ngreen");
        assert!(
            turns[2]
                .content
                .text()
                .unwrap()
                .contains("color must be one of red, blue")
        );
        assert!(turns_for(vec![ChatTurn::user("draft it")], None).len() == 1);
    }
}
