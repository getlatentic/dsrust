//! What `ReActV2` asks: the signature it builds, and the `submit` tool it adds to the caller's.
//!
//! dspy's `ReActV2.__init__` assembles both. The signature is the task's inputs plus a `history`
//! the loop threads and a `tools` description, with the outputs moved onto `submit` — so the model
//! answers by *calling* a tool rather than by filling fields, which is the whole difference from
//! [`ReAct`](crate::react::ReAct).

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use crate::adapter::ToolCalls;
use crate::react::backticked;
use crate::react::tool::Tool;

use super::SUBMIT;
use crate::signature::{FieldKind, InField, JsonType, OutField, Signature, TypeDescription};

/// One tool as the `tools` input carries it: the name, description and argument schema an adapter
/// renders, or a native request formats into a provider function call.
pub(super) fn tool_descriptor(tool: &dyn Tool) -> Value {
    json!({ "name": tool.name(), "desc": tool.description(), "args": tool.args() })
}

/// dspy `_make_react_signature`: the task's inputs, plus the conversation and the tool list coming
/// in, answering with a thought and the calls to make.
pub(super) fn react_signature(task: &Signature, tools: &[Box<dyn Tool>]) -> Signature {
    let mut inputs: Vec<InField> = task
        .inputs
        .iter()
        .map(|field| InField {
            name: field.name.clone(),
            desc: field.desc.clone(),
            kind: widened(&field.kind),
            ..Default::default()
        })
        .collect();
    inputs.push(input("history", "History"));
    inputs.push(input("tools", "list[Tool]"));

    let outputs = vec![
        OutField {
            name: "next_thought".into(),
            kind: FieldKind::Reasoning,
            ..Default::default()
        },
        OutField {
            name: "tool_calls".into(),
            // dspy carries `ToolCalls`'s own description on the field's line and its JSON schema in
            // the note under the marker — the type describes itself, so every field of it reads the
            // same. `JsonType::plain` would state neither.
            kind: FieldKind::Json(JsonType {
                annotation: "ToolCalls".into(),
                descriptions: vec![TypeDescription {
                    name: "ToolCalls".into(),
                    text: ToolCalls::description().to_owned(),
                    replaces_schema: false,
                }],
                reflection: None,
            }),
            schema: Some(ToolCalls::output_schema()),
            ..Default::default()
        },
    ];
    Signature {
        instructions: react_instructions(task, tools),
        inputs,
        outputs,
    }
}

/// dspy `_optional_annotation`: each task input is widened to `X | None`, since a continuation turn
/// omits it. `get_annotation_name` renders that union `UnionType[X, NoneType]`, which is the only
/// change — the value still reads as its own type, so a scalar renders identically. (The rendering
/// omits an absent input already, so the loop needs nothing more from the widening than its name.)
fn widened(kind: &FieldKind) -> FieldKind {
    FieldKind::Json(JsonType::plain(format!(
        "UnionType[{}, NoneType]",
        kind.annotation()
    )))
}

/// An input field carrying a custom type's annotation, which is how the history and tools fields
/// are recognised — by the type they name, not their field name.
fn input(name: &str, annotation: &str) -> InField {
    InField {
        name: name.to_owned(),
        kind: FieldKind::Json(JsonType::plain(annotation)),
        ..Default::default()
    }
}

/// dspy's `instructions` for the per-turn signature: the task's own, then how to act, then the
/// tools it may call by name. Joined by newlines and stripped, so a task without instructions does
/// not open with a blank line.
fn react_instructions(task: &Signature, tools: &[Box<dyn Tool>]) -> String {
    let inputs = backticked(task.inputs.iter().map(|field| field.name.as_str()));
    let outputs = backticked(task.outputs.iter().map(|field| field.name.as_str()));
    let tool_names = backticked(tools.iter().map(|tool| tool.name()));
    [
        task.instructions.clone(),
        format!("You are an Agent. Use the supplied tools to produce {outputs} from {inputs}."),
        "Call tools when more information is needed.".to_owned(),
        format!("When the final answer is ready, call `submit` with {outputs}."),
        format!("The available tools are: {tool_names}."),
    ]
    .join("\n")
    .trim()
    .to_owned()
}

/// dspy `_make_submit_tool`: the reserved tool that ends the task, its arguments the signature's
/// own output fields.
pub(super) fn submit_tool(task: &Signature) -> Box<dyn Tool> {
    let output_names = task
        .outputs
        .iter()
        .map(|field| field.name.clone())
        .collect();
    let args = Value::Object(
        task.outputs
            .iter()
            .map(|field| (field.name.clone(), schema_for_kind(&field.kind)))
            .collect(),
    );
    Box::new(Submit {
        output_names,
        args,
        description: "Submit the final outputs for the task.".to_owned(),
    })
}

/// dspy `_json_schema_for_annotation`: the argument's JSON schema, or the string fallback upstream
/// uses for a type pydantic could not describe.
fn schema_for_kind(kind: &FieldKind) -> Value {
    match kind {
        FieldKind::Bool => json!({ "type": "boolean" }),
        FieldKind::Int => json!({ "type": "integer" }),
        FieldKind::Float => json!({ "type": "number" }),
        FieldKind::Json(json_type) => json_type
            .reflection
            .clone()
            .unwrap_or_else(|| json!({ "type": "string" })),
        FieldKind::Str | FieldKind::Reasoning | FieldKind::Enum(_) => json!({ "type": "string" }),
    }
}

/// dspy's `submit`: the reserved final-output tool. Called with the task's outputs, it returns them
/// as the mapping the loop reads back; a missing output is the error the model is shown.
struct Submit {
    output_names: Vec<String>,
    args: Value,
    description: String,
}

impl Tool for Submit {
    fn name(&self) -> &str {
        SUBMIT
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn args(&self) -> &Value {
        &self.args
    }

    /// The string form is the mapping as JSON, for a caller on the text path; the loop reads the
    /// structured result through [`Tool::call_value`].
    fn call(&self, args: &Value) -> Result<String> {
        self.call_value(args).map(|value| value.to_string())
    }

    fn call_value(&self, args: &Value) -> Result<Value> {
        let given = args.as_object().cloned().unwrap_or_default();
        let missing: Vec<&str> = self
            .output_names
            .iter()
            .map(String::as_str)
            .filter(|name| !given.contains_key(*name))
            .collect();
        if !missing.is_empty() {
            return Err(anyhow!(
                "Missing required final output field(s): {}",
                missing.join(", ")
            ));
        }
        Ok(Value::Object(
            self.output_names
                .iter()
                .map(|name| (name.clone(), given[name].clone()))
                .collect(),
        ))
    }
}

/// `react_v2!("question -> answer", tools)` — a [`ReActV2`] agent over a signature and its tools,
/// the module form of `ReActV2::new(signature!(...), tools)`. `max_iters = N` caps the loop.
///
/// ```
/// use dsrust::{react_v2, FnTool, Tool};
/// use serde_json::{json, Value};
///
/// let tools: Vec<Box<dyn Tool>> = vec![Box::new(FnTool::new(
///     "lookup",
///     "look something up",
///     json!({ "query": { "type": "string" } }),
///     |args: &Value| Ok(format!("found {}", args["query"].as_str().unwrap_or_default())),
/// ))];
/// let agent = react_v2!("question -> answer", tools, max_iters = 8);
/// assert_eq!(agent.max_iters, 8);
/// ```
#[macro_export]
macro_rules! react_v2 {
    ($signature:literal, $tools:expr $(,)?) => {
        $crate::ReActV2::new($crate::signature!($signature), $tools)
    };
    ($signature:literal, $tools:expr, max_iters = $max:expr $(,)?) => {
        $crate::ReActV2::new($crate::signature!($signature), $tools).with_max_iters($max)
    };
}
