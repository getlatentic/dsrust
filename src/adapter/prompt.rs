//! How a signature reads in a prompt: the numbered field lists, the slot each output is written
//! into, the objective, and the frame that holds them.
//!
//! Every adapter opens by telling the model the same three things — what the fields are, how an
//! interaction is laid out, and what the task is — and dspy builds all three in its base
//! `format_system_message`. Only the middle section varies with the wire format, so it arrives
//! as a parameter and the rest is shared. The marker vocabulary belongs here for the same
//! reason: a `[[ ## name ## ]]` section is how a field is written wherever one appears, in the
//! template, in a request, and in an already-answered exchange.

use super::python_json::json_dumps;
use crate::signature::{FieldKind, InField, JsonType, OutField, Signature, wire_forms};

pub(super) fn marker(name: &str) -> String {
    format!("[[ ## {name} ## ]]")
}

/// A field on the wire: its marker, then its value on the next line.
pub(super) fn section(name: &str, value: &str) -> String {
    format!("{}\n{value}", marker(name))
}

/// The parts of a field a numbered line reads. dspy describes an input and an output the same
/// way, so both arrive here rather than each growing its own copy of the format.
struct Described<'a> {
    name: &'a str,
    annotation: String,
    kind: &'a FieldKind,
    desc: &'a str,
    constraints: Option<&'a str>,
}

impl<'a> From<&'a InField> for Described<'a> {
    fn from(field: &'a InField) -> Self {
        Self {
            name: &field.name,
            annotation: field.annotation(),
            kind: &field.kind,
            desc: &field.desc,
            constraints: field.constraints.as_deref(),
        }
    }
}

impl<'a> From<&'a OutField> for Described<'a> {
    fn from(field: &'a OutField) -> Self {
        Self {
            name: &field.name,
            annotation: field.annotation(),
            kind: &field.kind,
            desc: &field.desc,
            constraints: field.constraints.as_deref(),
        }
    }
}

/// dspy `get_field_description_string`, one field: the number, the field name, its Python
/// annotation and the description on the numbered line, then a line apiece for what the
/// annotation's custom types and the field's own constraints say. A closed set states itself
/// through the annotation (`Literal['a', 'b']`) and adds no line of its own.
fn numbered_line(index: usize, field: &Described<'_>) -> String {
    format!(
        "{}. `{}` ({}): {}{}{}",
        index + 1,
        field.name,
        field.annotation,
        field.desc,
        type_descriptions(field.kind),
        constraint_line(field.constraints),
    )
}

/// dspy states a field's constraints on a line of their own, unindented, after the description
/// and any type descriptions. The prose is pydantic's, already rendered where the signature was
/// declared; an empty string is no constraints, the way Python's own truth test reads it.
fn constraint_line(constraints: Option<&str>) -> String {
    match constraints.unwrap_or_default() {
        "" => String::new(),
        text => format!("\nConstraints: {text}"),
    }
}

/// dspy appends one indented line per custom type the annotation names, before the field's own
/// description is placed after the colon. A type with empty prose contributes nothing.
fn type_descriptions(kind: &FieldKind) -> String {
    let FieldKind::Json(json) = kind else {
        return String::new();
    };
    json.descriptions
        .iter()
        .filter(|described| !described.text.is_empty())
        .map(|described| {
            format!(
                "\n    Type description of {}: {}",
                described.name, described.text
            )
        })
        .collect()
}

/// dspy `get_field_description_string`: join the numbered lines with a newline, then strip the
/// block. The strip matters — an empty description leaves `": "` with a trailing space, and
/// upstream drops it from the last line only, which its exact-message tests pin.
fn numbered_block<'a>(fields: impl Iterator<Item = Described<'a>>) -> String {
    fields
        .enumerate()
        .map(|(index, field)| numbered_line(index, &field))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned()
}

pub(crate) fn numbered_input_lines(signature: &Signature) -> String {
    numbered_block(signature.inputs.iter().map(Described::from))
}

/// A `Json` field's schema does not appear here: upstream states it once, in the field's own
/// slot in the interaction template, which [`output_slot`] renders.
pub(crate) fn numbered_output_lines(signature: &Signature) -> String {
    numbered_block(signature.outputs.iter().map(Described::from))
}

/// Whether any custom type in this annotation has already said what its schema would say.
fn states_its_own_contract(json: &JsonType) -> bool {
    json.descriptions
        .iter()
        .any(|described| described.replaces_schema && !described.text.is_empty())
}

/// dspy `translate_field_type`: an output slot carries a note telling the model what shape the
/// value must take. `str` says nothing, since a string needs no constraint; everything else
/// earns a note on the same line, indented eight spaces as a comment.
pub(super) fn output_slot(field: &OutField) -> String {
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
        // A type whose description already states its contract does not repeat it as a schema,
        // which would spend a large block of the prompt saying the same thing twice. Every
        // other custom type keeps the schema, which is what steers a structured reply.
        // dspy asks for one of the members' values, and prints the type's own name above.
        FieldKind::Enum(_) => match &field.values {
            Some(values) => format!("must be one of: {}", wire_forms(values, "; ")),
            None => String::new(),
        },
        FieldKind::Json(json) if states_its_own_contract(json) => String::new(),
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

/// dspy `format_field_description`: what the fields are, ahead of how they are laid out.
///
/// Upstream states this section on its own, and its tests read it on its own, so it is a
/// function here rather than two lines inside the frame.
pub fn field_description(signature: &Signature) -> String {
    format!(
        "Your input fields are:\n{}\nYour output fields are:\n{}",
        numbered_input_lines(signature),
        numbered_output_lines(signature)
    )
}

/// The frame every adapter shares: what the fields are, how an interaction is laid out, and what
/// the task is. dspy builds all three in its base `format_system_message`, and only the middle
/// section — `structure` — tells the model which wire format it is answering in.
pub(super) fn system_message(signature: &Signature, structure: &str) -> String {
    format!(
        "{}\n\
         {}\n\
         {}",
        field_description(signature),
        structure.trim(),
        task_description(signature),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::TypeDescription;

    fn constrained(name: &str, desc: &str, kind: FieldKind, constraints: &str) -> InField {
        InField {
            name: name.into(),
            desc: desc.into(),
            kind,
            constraints: Some(constraints.into()),
            ..Default::default()
        }
    }

    /// The signature of upstream's `test_field_constraints`, in the crate's own spelling.
    fn constrained_signature() -> Signature {
        Signature {
            instructions: "Test signature with constrained fields.".into(),
            inputs: vec![
                constrained("text", "Input text", FieldKind::Str, "minimum length: 5"),
                constrained(
                    "number",
                    "A number",
                    FieldKind::Int,
                    "greater than or equal to: 0, less than or equal to: 10",
                ),
            ],
            outputs: vec![OutField {
                name: "out".into(),
                desc: "Out".into(),
                constraints: Some("maximum length: 10".into()),
                ..Default::default()
            }],
        }
    }

    #[test]
    fn constraints_read_on_their_own_unindented_line_under_the_field() {
        assert_eq!(
            field_description(&constrained_signature()),
            "Your input fields are:\n\
             1. `text` (str): Input text\n\
             Constraints: minimum length: 5\n\
             2. `number` (int): A number\n\
             Constraints: greater than or equal to: 0, less than or equal to: 10\n\
             Your output fields are:\n\
             1. `out` (str): Out\n\
             Constraints: maximum length: 10"
        );
    }

    #[test]
    fn a_field_without_constraints_gains_no_line() {
        let mut signature = constrained_signature();
        signature.inputs[0].constraints = None;
        // An empty string is what dspy would leave behind for a field it found nothing to say
        // about, and its own truth test drops that too.
        signature.inputs[1].constraints = Some(String::new());
        signature.outputs[0].constraints = None;
        assert_eq!(
            field_description(&signature),
            "Your input fields are:\n\
             1. `text` (str): Input text\n\
             2. `number` (int): A number\n\
             Your output fields are:\n\
             1. `out` (str): Out"
        );
    }

    #[test]
    fn constraints_follow_the_type_descriptions_of_the_annotation() {
        let signature = Signature {
            instructions: String::new(),
            inputs: vec![constrained(
                "code",
                "the code",
                FieldKind::Json(JsonType {
                    annotation: "Code".into(),
                    descriptions: vec![TypeDescription {
                        name: "Code".into(),
                        text: "a code block".into(),
                        replaces_schema: true,
                    }],
                    reflection: None,
                }),
                "minimum length: 1",
            )],
            outputs: vec![OutField {
                name: "out".into(),
                ..Default::default()
            }],
        };
        assert!(
            field_description(&signature).starts_with(
                "Your input fields are:\n\
                 1. `code` (Code): the code\n    \
                 Type description of Code: a code block\n\
                 Constraints: minimum length: 1\n"
            ),
            "got: {}",
            field_description(&signature)
        );
    }
}
