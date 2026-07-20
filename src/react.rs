//! Reasoning and acting: a module that calls tools until it can answer.
//!
//! dspy's `ReAct` wraps any signature. Each turn the model emits a thought, a tool name and
//! that tool's arguments; the tool runs, its observation joins the trajectory, and the loop
//! repeats until the model calls `finish` or the iteration budget runs out. A second pass then
//! reads the trajectory and produces the signature's real outputs.
//!
//! The Rust shape differs in one place worth naming: dspy takes any Python callable and
//! inspects it for a name and an argument schema. Here a tool is a trait, so its name,
//! description and argument schema are declared rather than derived, and the compiler checks
//! the implementation.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use crate::example::{Example, Prediction};
use crate::module::{Module, NamedPredictor};
use crate::predict::Predict;
use crate::signature::{FieldKind, InField, OutField, Signature, json_field_schema};

/// Something the agent can call. dspy derives these from a callable's signature; declaring
/// them keeps the argument contract visible to both the model and the compiler.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;

    /// Shown to the model when it chooses. A tool nobody can tell apart from another will not
    /// be chosen correctly, so this earns its place in the prompt.
    fn description(&self) -> &str;

    /// dspy's `Tool.args`: a JSON object mapping each argument name to that argument's JSON
    /// Schema. It is rendered into the instructions, so a model that has never seen this tool
    /// still knows what to send. A tool that takes nothing returns an empty object.
    ///
    /// Required rather than defaulted: a tool whose arguments go undeclared is one the model
    /// can only guess at, which is the failure this trait exists to prevent.
    fn args(&self) -> &Value;

    /// Run with the arguments the model supplied, returning the observation it will read.
    fn call(&self, args: &Value) -> Result<String>;
}

/// The `args` map for a tool whose arguments are the fields of `T`, so the schema comes from a
/// Rust type instead of a hand-written literal that can drift from the code reading it. dspy
/// reads the same map off a Python function's type hints.
///
/// Carries per-argument schemas only, matching dspy's `Tool.args`: which arguments are
/// optional shows up as a `default` on the argument, never as a separate `required` list.
pub fn tool_args<T: schemars::JsonSchema>() -> Value {
    json_field_schema::<T>()
        .get("properties")
        .cloned()
        .unwrap_or_else(|| json!({}))
}

/// The name the model uses to say it is done. dspy adds this tool itself, so the model always
/// has a way to stop that is indistinguishable from any other choice it makes.
pub const FINISH: &str = "finish";

/// One turn of the loop, kept so the next turn can read what already happened.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub thought: String,
    pub tool: String,
    pub args: Value,
    pub observation: String,
}

/// What the agent did, in order. A failed tool call stays in the trajectory rather than being
/// dropped: the model needs to see the error to recover from it, which is the whole point of
/// interleaving observations with reasoning.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trajectory {
    pub steps: Vec<Step>,
}

impl Trajectory {
    /// The trajectory as prompt text, one labelled block per field, matching how dspy renders
    /// it through the adapter's own field formatting.
    pub fn rendered(&self) -> String {
        let mut blocks = Vec::new();
        for (index, step) in self.steps.iter().enumerate() {
            blocks.push(format!("[[ ## thought_{index} ## ]]\n{}", step.thought));
            blocks.push(format!("[[ ## tool_name_{index} ## ]]\n{}", step.tool));
            blocks.push(format!("[[ ## tool_args_{index} ## ]]\n{}", step.args));
            blocks.push(format!(
                "[[ ## observation_{index} ## ]]\n{}",
                step.observation
            ));
        }
        blocks.join("\n\n")
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// An agent that interleaves reasoning with tool calls over any signature.
pub struct ReAct {
    /// The task's real signature: what the caller asked for.
    pub signature: Signature,
    tools: BTreeMap<String, Box<dyn Tool>>,
    /// One turn of the loop: pick a thought, a tool, and its arguments.
    react: Predict,
    /// The final pass: read the trajectory and produce the signature's outputs.
    extract: Predict,
    /// dspy defaults to 20. A budget is not optional — a model that never calls `finish`
    /// would otherwise loop against a paid provider forever.
    pub max_iters: usize,
}

impl ReAct {
    pub fn new(signature: Signature, tools: Vec<Box<dyn Tool>>) -> Self {
        let named: BTreeMap<String, Box<dyn Tool>> = tools
            .into_iter()
            .map(|tool| (tool.name().to_owned(), tool))
            .collect();
        let react = Predict::new(react_signature(&signature, &named));
        let extract = Predict::new(extract_signature(&signature));
        Self {
            signature,
            tools: named,
            react,
            extract,
            max_iters: 20,
        }
    }

    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    /// Run one tool, turning a failure into an observation rather than an error.
    ///
    /// dspy does the same: a tool that raises reports its error into the trajectory so the
    /// model can try something else. Aborting the episode would throw away the reasoning that
    /// got this far.
    fn observe(&self, name: &str, args: &Value) -> String {
        if name == FINISH {
            return "Completed.".to_owned();
        }
        match self.tools.get(name) {
            None => format!("Execution error: no tool named {name}"),
            Some(tool) => match tool.call(args) {
                Ok(observation) => observation,
                Err(error) => format!("Execution error in {name}: {error:#}"),
            },
        }
    }
}

/// One line of the tool catalogue, matching dspy's `Tool.__str__`: the name, the description
/// in `<desc>` tags, and the argument schema the model has to fill.
fn describe(name: &str, description: &str, args: &Value) -> String {
    let desc = match description.is_empty() {
        true => ".".to_owned(),
        // dspy flattens newlines so a multi-line description cannot break the numbered list.
        false => format!(", whose description is <desc>{description}</desc>.").replace('\n', "  "),
    };
    format!("{name}{desc} It takes arguments {}.", python_repr(args))
}

/// Render a JSON value the way Python's `repr` prints a dict, because that is literally what
/// dspy interpolates into the instructions — `str(tool)` formats `self.args`, a dict.
///
/// The difference is visible to the model: `{'city': {'type': 'string'}}` rather than
/// `{"city":{"type":"string"}}`. Matching it keeps the prompt bytes identical, which is the
/// standard the conformance fixtures hold everything else to.
fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(text) => format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(python_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| format!("'{key}': {}", python_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        number => number.to_string(),
    }
}

/// dspy appends `finish` to the tool dict itself, so stopping has the same shape as any other
/// choice the model makes: a name, a description, and an argument object that happens to be
/// empty.
fn finish_entry(outputs: &str) -> String {
    describe(
        FINISH,
        &format!(
            "Marks the task as complete. That is, signals that all information for producing \
             the outputs, i.e. {outputs}, are now available to be extracted."
        ),
        &json!({}),
    )
}

/// The numbered tool list dspy renders into the ReAct instructions, `finish` last.
fn tool_catalogue(tools: &BTreeMap<String, Box<dyn Tool>>, outputs: &str) -> String {
    tools
        .values()
        .map(|tool| describe(tool.name(), tool.description(), tool.args()))
        .chain(std::iter::once(finish_entry(outputs)))
        .enumerate()
        .map(|(index, entry)| format!("({}) {entry}", index + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The per-turn signature: the task's inputs, the trajectory so far, and the three fields the
/// model fills to take its next action.
fn react_signature(signature: &Signature, tools: &BTreeMap<String, Box<dyn Tool>>) -> Signature {
    let inputs = signature
        .inputs
        .iter()
        .map(|field| field.name)
        .collect::<Vec<_>>()
        .join(", ");
    let outputs = signature
        .outputs
        .iter()
        .map(|field| field.name)
        .collect::<Vec<_>>()
        .join(", ");
    let instructions = format!(
        "{}\n\nYou are an Agent. Each turn you are given the fields {inputs} and the \
         trajectory so far. Use the tools to collect what you need to produce {outputs}.\n\n\
         Interleave next_thought, next_tool_name and next_tool_args each turn. After each \
         tool call you receive an observation, which joins the trajectory.\n\n\
         The tool must be one of:\n\n{}\n\
         When providing `next_tool_args`, the value inside the field must be in JSON format",
        signature.instructions,
        tool_catalogue(tools, &outputs),
    );

    let mut fields: Vec<InField> = signature
        .inputs
        .iter()
        .map(|field| InField {
            name: field.name,
            desc: field.desc.clone(),
            kind: field.kind,
        })
        .collect();
    fields.push(InField {
        name: "trajectory",
        desc: "what has happened so far".into(),
        kind: FieldKind::Str,
    });

    Signature {
        instructions,
        inputs: fields,
        outputs: vec![
            OutField {
                name: "next_thought",
                desc: "reasoning about the current situation".into(),
                kind: FieldKind::Str,
                values: None,
                schema: None,
            },
            OutField {
                name: "next_tool_name",
                desc: "the tool to call".into(),
                kind: FieldKind::Str,
                values: None,
                schema: None,
            },
            OutField {
                name: "next_tool_args",
                desc: "arguments for that tool, as a JSON object".into(),
                kind: FieldKind::Json,
                values: None,
                schema: None,
            },
        ],
    }
}

/// The final pass: the task's own outputs, read from the trajectory.
fn extract_signature(signature: &Signature) -> Signature {
    let mut inputs: Vec<InField> = signature
        .inputs
        .iter()
        .map(|field| InField {
            name: field.name,
            desc: field.desc.clone(),
            kind: field.kind,
        })
        .collect();
    inputs.push(InField {
        name: "trajectory",
        desc: "what has happened so far".into(),
        kind: FieldKind::Str,
    });
    Signature {
        instructions: format!(
            "{}\n\nRead the trajectory and produce the requested fields.",
            signature.instructions
        ),
        inputs,
        outputs: signature
            .outputs
            .iter()
            .map(|field| OutField {
                name: field.name,
                desc: field.desc.clone(),
                kind: field.kind,
                values: field.values.clone(),
                schema: field.schema.clone(),
            })
            .collect(),
    }
}

impl Module for ReAct {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let mut trajectory = Trajectory::default();
            for _ in 0..self.max_iters {
                let mut turn_inputs = inputs.clone();
                turn_inputs.set("trajectory", Value::String(trajectory.rendered()));

                let step = self.react.forward(turn_inputs).await?;
                let thought = string_field(&step, "next_thought");
                let tool = string_field(&step, "next_tool_name");
                let args = step
                    .get("next_tool_args")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));

                let observation = self.observe(&tool, &args);
                let finished = tool == FINISH;
                trajectory.steps.push(Step {
                    thought,
                    tool,
                    args,
                    observation,
                });
                if finished {
                    break;
                }
            }

            let mut final_inputs = inputs;
            final_inputs.set("trajectory", Value::String(trajectory.rendered()));
            self.extract.forward(final_inputs).await
        })
    }

    /// Both predictors are visible to a compiler: the per-turn one decides which tools get
    /// called, and demos for it are what teach a model to use them well.
    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        let mut found = Vec::new();
        for (name, predictor) in [("react", &mut self.react), ("extract", &mut self.extract)] {
            for mut inner in predictor.named_predictors() {
                inner.name = name.to_owned();
                found.push(inner);
            }
        }
        found
    }
}

fn string_field(prediction: &Prediction, name: &str) -> String {
    match prediction.get(name) {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// A tool built from a closure, for callers who do not want to declare a type per tool.
pub struct FnTool<F> {
    pub name: String,
    pub description: String,
    /// dspy's `Tool.args`: argument name to that argument's JSON Schema. Build it from a type
    /// with [`tool_args`], or write the object out for a one-argument tool.
    pub args: Value,
    pub call: F,
}

impl<F> FnTool<F>
where
    F: Fn(&Value) -> Result<String> + Send + Sync,
{
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        args: Value,
        call: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            args,
            call,
        }
    }
}

impl<F> Tool for FnTool<F>
where
    F: Fn(&Value) -> Result<String> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn args(&self) -> &Value {
        &self.args
    }

    fn call(&self, args: &Value) -> Result<String> {
        (self.call)(args)
    }
}

/// Read a required string argument, with an error the model can act on.
pub fn arg_str<'a>(args: &'a Value, name: &str) -> Result<&'a str> {
    args.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing string argument `{name}`"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather() -> Box<dyn Tool> {
        Box::new(FnTool::new(
            "get_weather",
            "look up the weather for a city",
            json!({ "city": { "type": "string" } }),
            |args: &Value| {
                Ok(format!(
                    "The weather in {} is sunny.",
                    arg_str(args, "city")?
                ))
            },
        ))
    }

    fn task() -> Signature {
        Signature::single_input(
            "Answer the question.",
            vec![OutField {
                name: "answer",
                desc: "the answer".into(),
                kind: FieldKind::Str,
                values: None,
                schema: None,
            }],
        )
    }

    #[test]
    fn the_turn_signature_lists_every_tool_and_finish() {
        let react = ReAct::new(task(), vec![weather()]);
        let instructions = &react.react.signature.instructions;
        assert!(instructions.contains(
            "(1) get_weather, whose description is <desc>look up the weather for a city</desc>. \
             It takes arguments {'city': {'type': 'string'}}."
        ));
        assert!(
            instructions.contains("(2) finish, whose description is <desc>Marks the task as"),
            "the model needs a way to stop"
        );
        assert!(
            instructions.contains("Answer the question."),
            "the task survives"
        );
    }

    #[test]
    fn finish_is_described_as_a_tool_taking_no_arguments() {
        // dspy builds `finish` with `args={}`, so the model is never invited to invent any.
        let react = ReAct::new(task(), vec![weather()]);
        assert!(react.react.signature.instructions.contains(
            "i.e. answer, are now available to be extracted.</desc>. \
                          It takes arguments {}."
        ));
    }

    #[test]
    fn the_catalogue_tells_the_model_the_argument_field_is_json() {
        let react = ReAct::new(task(), vec![weather()]);
        assert!(react.react.signature.instructions.ends_with(
            "When providing `next_tool_args`, the value inside the field must be in JSON format"
        ));
    }

    #[test]
    fn a_description_spanning_lines_stays_on_one_catalogue_line() {
        // dspy replaces newlines so a wrapped docstring cannot break the numbered list apart.
        let entry = describe("noisy", "first\nsecond", &json!({}));
        assert_eq!(
            entry,
            "noisy, whose description is <desc>first  second</desc>. It takes arguments {}."
        );
    }

    #[test]
    fn a_tool_with_no_description_is_rendered_without_the_desc_tags() {
        assert_eq!(
            describe("bare", "", &json!({})),
            "bare. It takes arguments {}."
        );
    }

    #[test]
    fn tool_args_reads_the_argument_schema_off_a_rust_type() {
        // The point of the helper: the schema cannot drift from the struct the tool parses.
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct WeatherArgs {
            city: String,
            days: u8,
        }
        assert_eq!(
            tool_args::<WeatherArgs>(),
            json!({
                "city": { "type": "string" },
                "days": { "type": "integer", "format": "uint8", "minimum": 0, "maximum": 255 },
            })
        );
    }

    #[test]
    fn the_turn_signature_asks_for_thought_tool_and_args() {
        let react = ReAct::new(task(), vec![weather()]);
        let names: Vec<&str> = react
            .react
            .signature
            .outputs
            .iter()
            .map(|field| field.name)
            .collect();
        assert_eq!(names, ["next_thought", "next_tool_name", "next_tool_args"]);
    }

    #[test]
    fn the_extract_signature_produces_the_tasks_own_outputs() {
        let react = ReAct::new(task(), vec![weather()]);
        let names: Vec<&str> = react
            .extract
            .signature
            .outputs
            .iter()
            .map(|field| field.name)
            .collect();
        assert_eq!(names, ["answer"]);
        assert!(
            react
                .extract
                .signature
                .inputs
                .iter()
                .any(|field| field.name == "trajectory")
        );
    }

    #[test]
    fn a_tool_error_becomes_an_observation_rather_than_ending_the_episode() {
        // dspy reports the error into the trajectory so the model can recover; aborting would
        // throw away the reasoning that got this far.
        let react = ReAct::new(task(), vec![weather()]);
        let observation = react.observe("get_weather", &json!({}));
        assert!(observation.starts_with("Execution error in get_weather"));
        assert!(observation.contains("missing string argument `city`"));
    }

    #[test]
    fn an_unknown_tool_is_reported_the_same_way() {
        let react = ReAct::new(task(), vec![weather()]);
        assert_eq!(
            react.observe("teleport", &json!({})),
            "Execution error: no tool named teleport"
        );
    }

    #[test]
    fn a_working_tool_returns_its_observation() {
        let react = ReAct::new(task(), vec![weather()]);
        assert_eq!(
            react.observe("get_weather", &json!({ "city": "Tokyo" })),
            "The weather in Tokyo is sunny."
        );
    }

    #[test]
    fn the_trajectory_renders_each_step_as_labelled_blocks() {
        let trajectory = Trajectory {
            steps: vec![Step {
                thought: "I should look it up".to_owned(),
                tool: "get_weather".to_owned(),
                args: json!({ "city": "Tokyo" }),
                observation: "sunny".to_owned(),
            }],
        };
        let rendered = trajectory.rendered();
        assert!(rendered.contains("[[ ## thought_0 ## ]]\nI should look it up"));
        assert!(rendered.contains("[[ ## tool_name_0 ## ]]\nget_weather"));
        assert!(rendered.contains("[[ ## observation_0 ## ]]\nsunny"));
    }

    #[test]
    fn both_predictors_are_visible_to_a_compiler() {
        // An optimizer improves tool use by putting demos on the per-turn predictor, so it
        // has to be reachable — and distinguishable from the extract one.
        let mut react = ReAct::new(task(), vec![weather()]);
        let names: Vec<String> = react
            .named_predictors()
            .into_iter()
            .map(|predictor| predictor.name)
            .collect();
        assert_eq!(names, ["react", "extract"]);
    }
}
