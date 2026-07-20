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
use crate::signature::{FieldKind, JsonType, OutField, Signature, wire_forms};

pub(super) fn marker(name: &str) -> String {
    format!("[[ ## {name} ## ]]")
}

/// A field on the wire: its marker, then its value on the next line.
pub(super) fn section(name: &str, value: &str) -> String {
    format!("{}\n{value}", marker(name))
}

/// dspy `get_field_description_string`, one line: the number, the field name, its Python
/// annotation, and the description. A closed set says itself through the annotation
/// (`Literal['a', 'b']`), so nothing is appended after the description.
fn numbered_line(
    index: usize,
    name: &str,
    kind: &FieldKind,
    annotation: &str,
    desc: &str,
) -> String {
    format!(
        "{}. `{name}` ({annotation}): {desc}{}",
        index + 1,
        type_descriptions(kind)
    )
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
fn numbered_block(lines: Vec<String>) -> String {
    lines.join("\n").trim().to_owned()
}

fn numbered_input_lines(signature: &Signature) -> String {
    let lines: Vec<String> = signature
        .inputs
        .iter()
        .enumerate()
        .map(|(index, field)| {
            numbered_line(
                index,
                &field.name,
                &field.kind,
                &field.annotation(),
                &field.desc,
            )
        })
        .collect();
    numbered_block(lines)
}

/// A `Json` field's schema does not appear here: upstream states it once, in the field's own
/// slot in the interaction template, which [`output_slot`] renders.
fn numbered_output_lines(signature: &Signature) -> String {
    let lines: Vec<String> = signature
        .outputs
        .iter()
        .enumerate()
        .map(|(index, field)| {
            numbered_line(
                index,
                &field.name,
                &field.kind,
                &field.annotation(),
                &field.desc,
            )
        })
        .collect();
    numbered_block(lines)
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

/// The frame every adapter shares: what the fields are, how an interaction is laid out, and what
/// the task is. dspy builds all three in its base `format_system_message`, and only the middle
/// section — `structure` — tells the model which wire format it is answering in.
pub(super) fn system_message(signature: &Signature, structure: &str) -> String {
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
