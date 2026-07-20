use anyhow::{Result, anyhow};
use serde_json::{Map, Value};

use crate::lm::{ChatModel, ChatTurn, OutputMode};
use crate::signature::{FieldKind, Signature};

/// How a signature travels over the wire, mirroring DSPy's adapter pair: `Chat` speaks
/// `[[ ## field ## ]]` marker sections any model can produce with no provider support,
/// `Json` engages the provider's native structured output. Each is the other's fallback
/// when a reply fails to parse. Chat is DSPy's default and ours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Adapter {
    #[default]
    Chat,
    Json,
}

/// A rejected reply carried into the retry turn: the model sees its own previous output
/// and a precise statement of what was wrong and what is required.
pub struct Feedback {
    pub previous: String,
    pub error: String,
}

impl Adapter {
    pub fn fallback(self) -> Self {
        match self {
            Adapter::Chat => Adapter::Json,
            Adapter::Json => Adapter::Chat,
        }
    }

    /// The system and opening user message this adapter renders, with no model call.
    ///
    /// Mirrors `Adapter.format` in Python DSPy, which exists for the same reason: the prompt
    /// is the thing worth asserting on. The upstream conformance fixtures compare this output
    /// byte for byte against DSPy's own `format_exact_messages_*` expectations.
    pub fn format(self, signature: &Signature, inputs: &[(&str, String)]) -> (String, String) {
        match self {
            Adapter::Chat => (chat_system(signature), chat_user(signature, inputs)),
            Adapter::Json => (json_system(signature), json_user(inputs)),
        }
    }

    /// Format the conversation for this adapter, ask the model, hand back the raw reply.
    /// `inputs` carries the signature's input values as name/value pairs in signature order.
    pub async fn ask(
        self,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        signature: &Signature,
        inputs: &[(&str, String)],
        feedback: Option<&Feedback>,
    ) -> Result<String> {
        let schema = signature.schema();
        let (system, opening) = self.format(signature, inputs);
        let mode = match self {
            Adapter::Chat => OutputMode::Text,
            Adapter::Json => OutputMode::Json { schema: &schema },
        };
        lm.chat(http, &system, &conversation(opening, feedback), &mode)
            .await
    }

    /// Extract the signature's fields from a raw reply. An error here means the reply does
    /// not speak this adapter's format at all and triggers the adapter fallback; a reply
    /// missing individual fields parses and leaves those gaps for the signature's own
    /// validation, whose failure carries feedback into a retry.
    pub fn parse(self, signature: &Signature, raw: &str) -> Result<Value> {
        match self {
            Adapter::Chat => parse_markers(signature, raw),
            Adapter::Json => parse_json(raw),
        }
    }
}

fn conversation(opening: String, feedback: Option<&Feedback>) -> Vec<ChatTurn> {
    let mut turns = vec![ChatTurn::user(opening)];
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

fn marker(name: &str) -> String {
    format!("[[ ## {name} ## ]]")
}

fn numbered_line(
    index: usize,
    name: &str,
    kind: FieldKind,
    desc: &str,
    values: Option<&[&str]>,
    shape: Option<String>,
) -> String {
    let mut line = format!(
        "{}. `{}` ({}): {}",
        index + 1,
        name,
        kind.annotation(),
        desc
    );
    if let Some(shape) = shape {
        line.push_str(&format!(" ({shape})"));
    }
    if let Some(values) = values {
        line.push_str(&format!(" (must be one of: {})", values.join(", ")));
    }
    line
}

/// dspy `get_field_description_string`: join the numbered lines with a newline, then strip the
/// block. The strip matters — an empty description leaves `": "` with a trailing space, and
/// upstream drops it from the last line only, which its exact-message tests pin.
fn numbered_block(lines: Vec<String>) -> String {
    lines.join("\n").trim().to_owned()
}

/// Inputs never carry a shape note: the model reads their values, it does not produce them.
fn numbered_input_lines(signature: &Signature) -> String {
    let lines: Vec<String> = signature
        .inputs
        .iter()
        .enumerate()
        .map(|(index, field)| numbered_line(index, field.name, field.kind, &field.desc, None, None))
        .collect();
    numbered_block(lines)
}

fn numbered_output_lines(signature: &Signature) -> String {
    let lines: Vec<String> = signature
        .outputs
        .iter()
        .enumerate()
        .map(|(index, field)| {
            numbered_line(
                index,
                field.name,
                field.kind,
                &field.desc,
                field.values.as_deref(),
                field.schema_suffix(),
            )
        })
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
        .map(|field| (field.name, format!("{{{}}}", field.name)))
        .collect();
    let outputs = signature
        .outputs
        .iter()
        .map(|field| (field.name, output_slot(field)))
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
    let note = match field.kind {
        FieldKind::Str => match &field.values {
            Some(values) => format!(
                "must exactly match (no extra characters) one of: {}",
                values.join("; ")
            ),
            None => String::new(),
        },
        FieldKind::Bool => "must be True or False".to_owned(),
        FieldKind::Int => "must be a single int value".to_owned(),
        FieldKind::Float => "must be a single float value".to_owned(),
        FieldKind::Json => match &field.schema {
            Some(schema) => format!("must adhere to the JSON schema: {schema}"),
            None => String::new(),
        },
    };
    match note.is_empty() {
        true => format!("{{{}}}", field.name),
        false => format!("{{{}}}{}# note: the value you produce {note}", field.name, " ".repeat(8)),
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
fn output_requirements(signature: &Signature) -> String {
    let fields: Vec<String> = signature
        .outputs
        .iter()
        .map(|field| {
            let hint = match field.kind {
                FieldKind::Str => String::new(),
                kind => format!(
                    " (must be formatted as a valid Python {})",
                    kind.annotation()
                ),
            };
            format!("`{}`{hint}", marker(field.name))
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

/// DSPy ChatAdapter's parser: split the reply into sections at `[[ ## name ## ]]` headers,
/// keep the first section seen for each declared output field, ignore prose outside any
/// section and unknown headers (`completed` among them).
fn parse_markers(signature: &Signature, raw: &str) -> Result<Value> {
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
        let declared = signature.outputs.iter().any(|field| field.name == name);
        if declared && !fields.contains_key(name) {
            fields.insert(name.to_owned(), Value::from(lines.join("\n").trim()));
        }
    }
    if fields.is_empty() {
        return Err(anyhow!("reply has no [[ ## field ## ]] sections"));
    }
    Ok(Value::Object(fields))
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
fn parse_json(raw: &str) -> Result<Value> {
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
    use crate::signature::{InField, OutField};
    use serde_json::json;

    fn signature() -> Signature {
        Signature::single_input(
            "Pick a color.",
            vec![
                OutField {
                    name: "color",
                    desc: "the chosen color".into(),
                    kind: FieldKind::Str,
                    values: Some(vec!["red", "blue"]),
                    schema: None,
                },
                OutField {
                    name: "why",
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
                name: "room",
                desc: "the room being painted".into(),
                kind: FieldKind::Str,
            },
            InField {
                name: "mood",
                desc: "the mood to set".into(),
                kind: FieldKind::Str,
            },
        ];
        signature
    }

    fn typed_signature() -> Signature {
        let mut signature = Signature::single_input(
            "Size the gift.",
            vec![
                OutField {
                    name: "amount",
                    desc: "amount in MON".into(),
                    kind: FieldKind::Float,
                    values: None,
                    schema: None,
                },
                OutField {
                    name: "double",
                    desc: "double it".into(),
                    kind: FieldKind::Bool,
                    values: None,
                    schema: None,
                },
            ],
        );
        signature.inputs = vec![InField {
            name: "age",
            desc: "the age turned".into(),
            kind: FieldKind::Int,
        }];
        signature
    }

    fn json_signature() -> Signature {
        let mut signature = Signature::single_input(
            "Suggest ideas.",
            vec![OutField {
                name: "ideas",
                desc: "three concrete ideas".into(),
                kind: FieldKind::Json,
                values: None,
                schema: Some(json!({ "type": "array", "items": { "type": "string" } })),
            }],
        );
        signature.inputs = vec![InField {
            name: "recipient",
            desc: "who the gift is for".into(),
            kind: FieldKind::Json,
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
        assert!(system.contains("1. `color` (str): the chosen color (must be one of: red, blue)"));
        assert!(system.contains("2. `why` (str): one short sentence"));
        assert!(system.contains("[[ ## request ## ]]\n{request}"));
        assert!(system.contains("[[ ## color ## ]]\n{color}"));
        assert!(system.contains("[[ ## completed ## ]]"));
        // dspy indents the instruction onto its own line, so the sentence ends in a space.
        assert!(system.ends_with(
            "In adhering to this structure, your objective is: \n        Pick a color."
        ));
    }

    #[test]
    fn numbered_lines_annotate_each_field_with_its_kind() {
        let system = chat_system(&typed_signature());
        assert!(system.contains("1. `age` (int): the age turned"));
        assert!(system.contains("1. `amount` (float): amount in MON"));
        assert!(system.contains("2. `double` (bool): double it"));
    }

    #[test]
    fn numbered_lines_append_the_schema_to_json_outputs_only() {
        let signature = json_signature();
        let system = chat_system(&signature);
        assert!(system.contains("1. `recipient` (json): who the gift is for\n"));
        let expected = format!(
            "1. `ideas` (json): three concrete ideas (json matching schema: {})\n",
            signature.outputs[0].schema.as_ref().expect("json schema")
        );
        assert!(system.contains(&expected), "got: {system}");
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

    #[test]
    fn parse_json_accepts_bare_and_prose_wrapped_objects() {
        let bare = parse_json(r#"{ "color": "red" }"#).expect("bare");
        assert_eq!(bare["color"], "red");
        let wrapped =
            parse_json("Here it is:\n```json\n{ \"color\": \"blue\" }\n```").expect("wrapped");
        assert_eq!(wrapped["color"], "blue");
        assert!(parse_json("no json here").is_err());
    }

    #[test]
    fn fallback_swaps_the_pair() {
        assert_eq!(Adapter::Chat.fallback(), Adapter::Json);
        assert_eq!(Adapter::Json.fallback(), Adapter::Chat);
        assert_eq!(Adapter::default(), Adapter::Chat);
    }

    #[test]
    fn conversation_appends_previous_output_and_error_on_retry() {
        let feedback = Feedback {
            previous: "[[ ## color ## ]]\ngreen".into(),
            error: "color must be one of red, blue; got \"green\"".into(),
        };
        let turns = conversation("draft it".into(), Some(&feedback));
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[1].content, "[[ ## color ## ]]\ngreen");
        assert!(turns[2].content.contains("color must be one of red, blue"));
        assert!(conversation("draft it".into(), None).len() == 1);
    }
}
