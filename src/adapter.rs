use anyhow::Result;
use serde_json::Value;

use crate::example::Example;
use crate::lm::{ChatTurn, OutputMode};
use crate::signature::{FieldKind, Signature, wire_forms};

mod parse;
pub(crate) mod python_json;

use python_json::json_dumps;

/// How a signature travels over the wire.
///
/// Mirrors DSPy's `Adapter` base class: implement it to teach the crate a new wire format.
/// The two shipped implementations are [`ChatAdapter`], which speaks `[[ ## field ## ]]`
/// marker sections any model can produce with no provider support, and [`JsonAdapter`],
/// which engages the provider's native structured output. Chat is DSPy's default and ours.
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
        inputs: &[(&str, String)],
    ) -> (String, Vec<ChatTurn>);

    /// Extract the signature's fields from a raw reply. A reply that does not speak this
    /// adapter's format at all fails here; a reply missing individual fields parses and
    /// leaves those gaps for the signature's own validation, whose failure carries feedback
    /// into a retry.
    fn parse(&self, signature: &Signature, raw: &str) -> Result<Value>;

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
        inputs: &[(&str, String)],
    ) -> (String, Vec<ChatTurn>) {
        let mut turns: Vec<ChatTurn> = demos
            .iter()
            .flat_map(|demo| chat_demo_turns(signature, demo))
            .collect();
        turns.push(ChatTurn::user(chat_user(signature, inputs)));
        (chat_system(signature), turns)
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
        _demos: &[Example],
        inputs: &[(&str, String)],
    ) -> (String, Vec<ChatTurn>) {
        (
            json_system(signature),
            vec![ChatTurn::user(json_user(inputs))],
        )
    }

    fn parse(&self, _signature: &Signature, raw: &str) -> Result<Value> {
        parse::parse_json(raw)
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

fn marker(name: &str) -> String {
    format!("[[ ## {name} ## ]]")
}

/// dspy `get_field_description_string`, one line: the number, the field name, its Python
/// annotation, and the description. A closed set says itself through the annotation
/// (`Literal['a', 'b']`), so nothing is appended after the description.
fn numbered_line(index: usize, name: &str, annotation: &str, desc: &str) -> String {
    format!("{}. `{name}` ({annotation}): {desc}", index + 1)
}

/// dspy `get_field_description_string`: join the numbered lines with a newline, then strip the
/// block. The strip matters — an empty description leaves `": "` with a trailing space, and
/// upstream drops it from the last line only, which its exact-message tests pin.
fn numbered_block(lines: Vec<String>) -> String {
    lines.join("\n").trim().to_owned()
}

fn numbered_input_lines(signature: &Signature) -> String {
    let lines: Vec<String> = signature
        .inputs
        .iter()
        .enumerate()
        .map(|(index, field)| numbered_line(index, &field.name, &field.annotation(), &field.desc))
        .collect();
    numbered_block(lines)
}

/// A `Json` field's schema does not appear here: upstream states it once, in the field's own
/// slot in the template below, which [`output_slot`] renders.
fn numbered_output_lines(signature: &Signature) -> String {
    let lines: Vec<String> = signature
        .outputs
        .iter()
        .enumerate()
        .map(|(index, field)| numbered_line(index, &field.name, &field.annotation(), &field.desc))
        .collect();
    numbered_block(lines)
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
            .map(|(name, slot)| format!("{}\n{slot}", marker(name)))
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

    format!(
        "Your input fields are:\n{}\n\
         Your output fields are:\n{}\n\
         {}\n\
         {}",
        numbered_input_lines(signature),
        numbered_output_lines(signature),
        structure.trim(),
        task_description(signature),
    )
}

/// dspy `translate_field_type`: an output slot carries a note telling the model what shape the
/// value must take. `str` says nothing, since a string needs no constraint; everything else
/// earns a note on the same line, indented eight spaces as a comment.
fn output_slot(field: &crate::signature::OutField) -> String {
    let note = match &field.kind {
        FieldKind::Str => match &field.values {
            Some(values) => format!(
                "must exactly match (no extra characters) one of: {}",
                wire_forms(values, "; ")
            ),
            None => String::new(),
        },
        FieldKind::Bool => "must be True or False".to_owned(),
        FieldKind::Int => "must be a single int value".to_owned(),
        FieldKind::Float => "must be a single float value".to_owned(),
        FieldKind::Json(_) => match &field.schema {
            Some(schema) => format!("must adhere to the JSON schema: {}", json_dumps(schema)),
            None => String::new(),
        },
    };
    match note.is_empty() {
        true => format!("{{{}}}", field.name),
        false => format!(
            "{{{}}}{}# note: the value you produce {note}",
            field.name,
            " ".repeat(8)
        ),
    }
}

/// dspy `format_task_description`: the instruction is dedented, then every line is pushed onto
/// its own 8-space-indented line — including the first, which is why the objective sentence
/// ends in a space and the instruction starts on the next line.
fn task_description(signature: &Signature) -> String {
    let objective: String = std::iter::once("")
        .chain(signature.instructions.lines())
        .collect::<Vec<_>>()
        .join("\n        ");
    format!("In adhering to this structure, your objective is: {objective}")
}

/// DSPy ChatAdapter's user message: each input in its own marker section, then the recap of
/// the output field order.
fn chat_user(signature: &Signature, inputs: &[(&str, String)]) -> String {
    // dspy `format_user_message_content`: input sections and the reminder are one list joined
    // by a blank line and stripped, rather than sections each carrying their own trailing gap.
    let mut parts: Vec<String> = inputs
        .iter()
        .map(|(name, value)| format!("{}\n{value}", marker(name)))
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

/// dspy `format_demos`: a demo becomes the user turn it would have been, then the assistant
/// turn it produced. The user turn carries no output-requirements reminder — the answer is
/// already there — and the assistant turn closes with the completed marker.
fn chat_demo_turns(signature: &Signature, demo: &Example) -> Vec<ChatTurn> {
    let section = |name: &str, value: String| format!("{}\n{value}", marker(name));
    let ask = signature
        .inputs
        .iter()
        .filter_map(|field| {
            Some(section(
                &field.name,
                demo.get(&field.name).map(crate::example::render)?,
            ))
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut answer = signature
        .outputs
        .iter()
        .filter_map(|field| {
            Some(section(
                &field.name,
                demo.get(&field.name).map(crate::example::render)?,
            ))
        })
        .collect::<Vec<_>>();
    answer.push(format!("{}\n", marker("completed")));
    vec![
        ChatTurn::user(ask),
        ChatTurn::assistant(answer.join("\n\n")),
    ]
}

/// The JSON contract in prose, for the provider-native structured-output path.
fn json_system(signature: &Signature) -> String {
    format!("{} {}", signature.instructions, signature.output_clause())
}

/// The JSON adapter's user message: each input as a labeled line.
fn json_user(inputs: &[(&str, String)]) -> String {
    inputs
        .iter()
        .map(|(name, value)| format!("{name}: {value}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{InField, OutField};
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

    fn single_request(value: &str) -> Vec<(&'static str, String)> {
        vec![("request", value.to_owned())]
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
            ("room", "the study".to_owned()),
            ("mood", "calm focus".to_owned()),
        ];
        let user = chat_user(&multi_signature(), &inputs);
        assert!(user.starts_with(
            "[[ ## room ## ]]\nthe study\n\n[[ ## mood ## ]]\ncalm focus\n\nRespond with"
        ));
        assert!(user.contains("starting with the field `[[ ## color ## ]]`"));
        assert!(!user.contains("`[[ ## room ## ]]`"));
    }

    #[test]
    fn json_user_labels_every_input_line() {
        let inputs = vec![
            ("room", "the study".to_owned()),
            ("mood", "calm focus".to_owned()),
        ];
        assert_eq!(json_user(&inputs), "room: the study\nmood: calm focus");
        assert_eq!(json_user(&single_request("hi")), "request: hi");
    }

    #[test]
    fn conversation_appends_previous_output_and_error_on_retry() {
        let feedback = Feedback {
            previous: "[[ ## color ## ]]\ngreen".into(),
            error: "color must be one of red, blue; got \"green\"".into(),
        };
        let turns = turns_for(vec![ChatTurn::user("draft it")], Some(&feedback));
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[1].content, "[[ ## color ## ]]\ngreen");
        assert!(turns[2].content.contains("color must be one of red, blue"));
        assert!(turns_for(vec![ChatTurn::user("draft it")], None).len() == 1);
    }
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
