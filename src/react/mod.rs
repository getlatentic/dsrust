//! Reasoning and acting: a module that calls tools until it can answer.
//!
//! dspy's `ReAct` wraps any signature. Each turn the model emits a thought, a tool name and
//! that tool's arguments; the tool runs, its observation joins the trajectory, and the loop
//! repeats until the model calls `finish` or the iteration budget runs out. A second pass then
//! reads the trajectory and produces the signature's real outputs.

pub mod mcp;
mod tool;
mod trajectory;

use anyhow::{Result, anyhow};
use serde_json::{Value, json};

use crate::example::{Example, Prediction};
use crate::module::{Module, NamedPredictor, TraceStep, relabel};
use crate::predict::Predict;
use crate::signature::{FieldKind, InField, JsonType, LiteralValue, OutField, Signature};
use tool::describe;

pub use mcp::{mcp_tool, mcp_tool_args, mcp_tool_result};
pub use tool::{FINISH, FnTool, Tool, arg_str, tool_args};
pub use trajectory::{Step, Trajectory};

/// An agent that interleaves reasoning with tool calls over any signature.
pub struct ReAct {
    /// The task's real signature: what the caller asked for.
    pub signature: Signature,
    /// Every tool the model may pick, `finish` included, in the order it will be numbered.
    tools: Vec<Box<dyn Tool>>,
    /// One turn of the loop: pick a thought, a tool, and its arguments.
    react: Predict,
    /// The final pass: read the trajectory and produce the signature's outputs.
    extract: Predict,
    /// dspy defaults to 20. A budget is not optional — a model that never calls `finish`
    /// would otherwise loop against a paid provider forever.
    pub max_iters: usize,
}

impl ReAct {
    /// The signature one turn of the loop is asked with — the task's inputs plus `trajectory`,
    /// answering with a thought, a tool name and its arguments.
    ///
    /// dspy reaches the same thing as `react.react.signature`. It is not the signature handed in:
    /// this module rewrites it before asking, which is most of what it does.
    pub fn turn_signature(&self) -> &Signature {
        &self.react.signature
    }

    pub fn new(signature: Signature, tools: Vec<Box<dyn Tool>>) -> Self {
        let finish = Finish::new(&backticked(
            signature.outputs.iter().map(|field| field.name.as_str()),
        ));
        let tools = as_dict(tools.into_iter().chain([Box::new(finish) as Box<dyn Tool>]));
        let react = Predict::from_signature(react_signature(&signature, &tools));
        let extract = Predict::from_signature(extract_signature(&signature));
        Self {
            signature,
            tools,
            react,
            extract,
            max_iters: 20,
        }
    }

    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Every tool the model may pick, `finish` included, in the order they are numbered.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|tool| tool.name()).collect()
    }

    fn tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|tool| tool.name() == name)
            .map(Box::as_ref)
    }

    /// Run one tool, turning a failure into an observation rather than an error.
    ///
    /// dspy does the same: a tool that raises reports its error into the trajectory so the
    /// model can try something else. Aborting the episode would throw away the reasoning that
    /// got this far.
    fn observe(&self, name: &str, args: &Value) -> String {
        match self.tool(name) {
            None => format!("Execution error in {name}: no such tool"),
            Some(tool) => match tool.call(args) {
                Ok(observation) => observation,
                Err(error) => format!("Execution error in {name}: {error:#}"),
            },
        }
    }
}

/// dspy keeps its tools in `{tool.name: tool for tool in tools}`, so the catalogue is numbered
/// in the order the caller supplied and a repeated name replaces the earlier tool without
/// moving it. Sorting instead would renumber the prompt the model reads.
fn as_dict(tools: impl Iterator<Item = Box<dyn Tool>>) -> Vec<Box<dyn Tool>> {
    let mut ordered: Vec<Box<dyn Tool>> = Vec::new();
    for tool in tools {
        match ordered.iter().position(|held| held.name() == tool.name()) {
            Some(index) => ordered[index] = tool,
            None => ordered.push(tool),
        }
    }
    ordered
}

/// The field-name list dspy interpolates into the instructions, each name in backticks.
fn backticked<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// dspy puts `finish` in the tool dict itself, so stopping has the same shape as any other
/// choice the model makes: a name, a description naming the outputs it unblocks, and an
/// argument object that happens to be empty.
struct Finish {
    description: String,
    args: Value,
}

impl Finish {
    fn new(outputs: &str) -> Self {
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
fn react_instructions(signature: &Signature, tools: &[Box<dyn Tool>]) -> String {
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
            describe(tool.name(), tool.description(), tool.args())
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
fn trajectory_field() -> InField {
    InField {
        name: "trajectory".to_owned(),
        ..Default::default()
    }
}

/// dspy types `next_tool_name` as `Literal[tuple(tools.keys())]`, which the chat adapter turns
/// into the closed set the model must match exactly.
fn tool_name_set(tools: &[Box<dyn Tool>]) -> Vec<LiteralValue> {
    tools.iter().map(|tool| tool.name().into()).collect()
}

fn out_field(name: &str, values: Option<Vec<LiteralValue>>, kind: FieldKind) -> OutField {
    OutField {
        name: name.to_owned(),
        kind,
        values,
        ..Default::default()
    }
}

/// The per-turn signature: the task's inputs, the trajectory so far, and the three fields the
/// model fills to take its next action.
fn react_signature(signature: &Signature, tools: &[Box<dyn Tool>]) -> Signature {
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
fn extract_signature(signature: &Signature) -> Signature {
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

impl ReAct {
    /// The episode itself, written once because [`Module::forward`] and
    /// [`Module::forward_traced`] differ only in whether anyone keeps what the trace records.
    async fn run(&self, inputs: Example, trace: &mut Vec<TraceStep>) -> Result<Prediction> {
        {
            let mut trajectory = Trajectory::default();
            for _ in 0..self.max_iters {
                let mut turn_inputs = inputs.clone();
                turn_inputs.set("trajectory", Value::String(trajectory.rendered()));

                let mark = trace.len();
                let step = self.react.forward_traced(turn_inputs, trace).await?;
                relabel(trace, mark, "react");
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
            let mark = trace.len();
            let extracted = self.extract.forward_traced(final_inputs, trace).await?;
            relabel(trace, mark, "extract");

            // dspy returns `Prediction(trajectory=trajectory, **extract)`: what the agent did
            // travels back beside what it concluded, so a caller can inspect the episode.
            let mut example = Example::new([("trajectory", trajectory.as_value())]);
            for (name, value) in extracted.example.fields() {
                example.set(name, value.clone());
            }
            Ok(Prediction::new(example, extracted.raw))
        }
    }
}

impl Module for ReAct {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let mut discarded = Vec::new();
            self.run(inputs, &mut discarded).await
        })
    }

    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(self.run(inputs, trace))
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
                name: "answer".into(),
                desc: "the answer".into(),
                ..Default::default()
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
            "i.e. `answer`, are now available to be extracted.</desc>. \
                          It takes arguments {}."
        ));
    }

    #[test]
    fn finish_refuses_arguments_it_never_declared() {
        let react = ReAct::new(task(), vec![weather()]);
        assert_eq!(react.observe(FINISH, &json!({})), "Completed.");
        assert_eq!(
            react.observe(FINISH, &json!({ "answer": "sunny" })),
            "Execution error in finish: finish takes no arguments"
        );
    }

    #[test]
    fn the_catalogue_tells_the_model_the_argument_field_is_json() {
        let react = ReAct::new(task(), vec![weather()]);
        assert!(react.react.signature.instructions.ends_with(
            "When providing `next_tool_args`, the value inside the field must be in JSON format"
        ));
    }

    #[test]
    fn the_turn_signature_asks_for_thought_tool_and_args() {
        let react = ReAct::new(task(), vec![weather()]);
        let names: Vec<&str> = react
            .react
            .signature
            .outputs
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        assert_eq!(names, ["next_thought", "next_tool_name", "next_tool_args"]);
    }

    #[test]
    fn the_extract_signature_reasons_before_the_tasks_own_outputs() {
        // dspy's extract pass is a ChainOfThought, so `reasoning` leads the output fields.
        let react = ReAct::new(task(), vec![weather()]);
        let names: Vec<&str> = react
            .extract
            .signature
            .outputs
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        assert_eq!(names, ["reasoning", "answer"]);
        let inputs: Vec<&str> = react
            .extract
            .signature
            .inputs
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        assert_eq!(inputs, ["request", "trajectory"]);
    }

    #[test]
    fn the_extract_signature_leaves_the_tasks_instructions_alone() {
        let react = ReAct::new(task(), vec![weather()]);
        assert_eq!(react.extract.signature.instructions, "Answer the question.");
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
        // dspy indexes its tool dict and lets the KeyError land in the same `Execution error
        // in {name}:` shape every other tool failure takes.
        let react = ReAct::new(task(), vec![weather()]);
        assert_eq!(
            react.observe("teleport", &json!({})),
            "Execution error in teleport: no such tool"
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
    fn the_tool_catalogue_keeps_the_order_the_caller_supplied() {
        // dspy numbers a dict built from the tool list, so the caller's order is the prompt's.
        let zebra = Box::new(FnTool::new("zebra", "z", json!({}), |_: &Value| {
            Ok(String::new())
        })) as Box<dyn Tool>;
        let alpha = Box::new(FnTool::new("alpha", "a", json!({}), |_: &Value| {
            Ok(String::new())
        })) as Box<dyn Tool>;
        let react = ReAct::new(task(), vec![zebra, alpha]);
        assert_eq!(react.tool_names(), ["zebra", "alpha", "finish"]);
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
