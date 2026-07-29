//! What `ReAct` asks: the signature it builds, and the `finish` tool it adds to the caller's.
//!
//! dspy's `ReAct.__init__` assembles both, and the second signature too — the extraction pass that
//! reads a finished trajectory and produces the task's real outputs. The turn signature never
//! carries those outputs, which is why the extraction exists at all.

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use super::tool::Tool;
use super::{FINISH, backticked};
use crate::adapter::types::tool::format_tool;
use crate::signature::{FieldKind, InField, JsonType, LiteralValue, OutField, Signature};

/// dspy puts `finish` in the tool dict itself, so stopping has the same shape as any other
/// choice the model makes: a name, a description naming the outputs it unblocks, and an
/// argument object that happens to be empty.
pub(super) struct Finish {
    description: String,
    args: Value,
}

impl Finish {
    pub(super) fn new(outputs: &str) -> Self {
        Self {
            description: format!(
                "Marks the task as complete. That is, signals that all information for \
                 producing the outputs, i.e. {outputs}, are now available to be extracted."
            ),
            args: json!({}),
        }
    }
}

impl Tool for Finish {
    fn name(&self) -> &str {
        FINISH
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn args(&self) -> &Value {
        &self.args
    }

    /// dspy's finish is `lambda: "Completed."`, so arguments it never declared are a call
    /// error the model reads back in the trajectory rather than a silent success.
    fn call(&self, args: &Value) -> Result<String> {
        match args.as_object().is_none_or(|given| given.is_empty()) {
            true => Ok("Completed.".to_owned()),
            false => Err(anyhow!("{FINISH} takes no arguments")),
        }
    }
}

/// dspy's `instr` list, joined by newlines. The blocks that end in `\n` are the ones that
/// become blank-line separated in the prompt; the rest run on consecutive lines.
pub(super) fn react_instructions(signature: &Signature, tools: &[Box<dyn Tool>]) -> String {
    let inputs = backticked(signature.inputs.iter().map(|field| field.name.as_str()));
    let outputs = backticked(signature.outputs.iter().map(|field| field.name.as_str()));

    // dspy drops the task's own block entirely when the signature carries no instructions,
    // rather than opening the prompt with a blank line.
    let task = match signature.instructions.is_empty() {
        true => Vec::new(),
        false => vec![format!("{}\n", signature.instructions)],
    };

    let preamble = [
        format!(
            "You are an Agent. In each episode, you will be given the fields {inputs} as \
             input. And you can see your past trajectory so far."
        ),
        format!(
            "Your goal is to use one or more of the supplied tools to collect any necessary \
             information for producing {outputs}.\n"
        ),
        "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in \
         each turn, and also when finishing the task."
            .to_owned(),
        "After each tool call, you receive a resulting observation, which gets appended to \
         your trajectory.\n"
            .to_owned(),
        "When writing next_thought, you may reason about the current situation and plan for \
         future steps."
            .to_owned(),
        "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n"
            .to_owned(),
    ];

    let catalogue = tools.iter().enumerate().map(|(index, tool)| {
        format!(
            "({}) {}",
            index + 1,
            format_tool(tool.name(), tool.description(), tool.args())
        )
    });

    task.into_iter()
        .chain(preamble)
        .chain(catalogue)
        .chain([
            "When providing `next_tool_args`, the value inside the field must be in JSON format"
                .to_owned(),
        ])
        .collect::<Vec<_>>()
        .join("\n")
}

/// dspy appends `trajectory` with a bare `dspy.InputField()`, which carries no description of
/// its own: the instructions already say what the trajectory is.
pub(super) fn trajectory_field() -> InField {
    InField {
        name: "trajectory".to_owned(),
        ..Default::default()
    }
}

/// dspy types `next_tool_name` as `Literal[tuple(tools.keys())]`, which the chat adapter turns
/// into the closed set the model must match exactly.
pub(super) fn tool_name_set(tools: &[Box<dyn Tool>]) -> Vec<LiteralValue> {
    tools.iter().map(|tool| tool.name().into()).collect()
}

pub(super) fn out_field(name: &str, values: Option<Vec<LiteralValue>>, kind: FieldKind) -> OutField {
    OutField {
        name: name.to_owned(),
        kind,
        values,
        ..Default::default()
    }
}

/// The per-turn signature: the task's inputs, the trajectory so far, and the three fields the
/// model fills to take its next action.
pub(super) fn react_signature(signature: &Signature, tools: &[Box<dyn Tool>]) -> Signature {
    let mut inputs = signature.inputs.clone();
    inputs.push(trajectory_field());

    Signature {
        instructions: react_instructions(signature, tools),
        inputs,
        outputs: vec![
            out_field("next_thought", None, FieldKind::Str),
            out_field("next_tool_name", Some(tool_name_set(tools)), FieldKind::Str),
            // dspy types the argument object `dict[str, Any]`, and prints that Python type
            // beside the field name; pydantic turns the same type into the slot's schema note.
            OutField {
                schema: Some(json!({ "type": "object", "additionalProperties": true })),
                ..out_field(
                    "next_tool_args",
                    None,
                    FieldKind::Json(JsonType::plain("dict[str, Any]")),
                )
            },
        ],
    }
}

/// The final pass. dspy runs a `ChainOfThought` over the task's own signature plus the
/// trajectory, so the instructions carry through untouched and the model reasons in a leading
/// `reasoning` field before it fills in the outputs the caller asked for.
pub(super) fn extract_signature(signature: &Signature) -> Signature {
    let mut inputs = signature.inputs.clone();
    inputs.push(trajectory_field());

    let mut outputs = vec![out_field("reasoning", None, FieldKind::Str)];
    outputs.extend(signature.outputs.iter().cloned());

    Signature {
        instructions: signature.instructions.clone(),
        inputs,
        outputs,
    }
}

