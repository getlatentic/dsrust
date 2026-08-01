//! Reasoning and acting: a module that calls tools until it can answer.
//!
//! dspy's `ReAct` wraps any signature. Each turn the model emits a thought, a tool name and
//! that tool's arguments; the tool runs, its observation joins the trajectory, and the loop
//! repeats until the model calls `finish` or the iteration budget runs out. A second pass then
//! reads the trajectory and produces the signature's real outputs.

pub mod mcp;
mod signature;
mod tool;
mod trajectory;
mod v2;

use anyhow::{Result, bail};
use serde_json::{Value, json};

use crate::example::{Example, Prediction};
use crate::lm::ContextWindowExceeded;
use crate::module::{Module, NamedPredictor, TraceStep, relabel};
use crate::predict::Predict;
use crate::signature::Signature;
use signature::{Finish, extract_signature, react_signature};

/// dspy tries three times before giving up on a trajectory that will not fit.
const TRUNCATION_ATTEMPTS: usize = 3;

pub use mcp::{mcp_tool, mcp_tool_args, mcp_tool_result};
pub use tool::{FINISH, FnTool, Tool, arg_str, tool_args};
pub use trajectory::{Step, Trajectory};
pub use v2::ReActV2;

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

    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Ask both inner predictors — the loop's `react` turn and the final `extract` — through this
    /// model. Without it they reach for the globally configured one, the way dspy's do.
    pub fn set_lm(mut self, lm: std::sync::Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.react = self.react.set_lm(lm.clone());
        self.extract = self.extract.set_lm(lm);
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
    fn observe(&self, name: &str, args: &Value) -> Value {
        match self.tool(name) {
            None => json!(format!("Execution error in {name}: no such tool")),
            Some(tool) => match crate::observe::tool_call(tool, args) {
                Ok(observation) => observation,
                Err(error) => json!(format!("Execution error in {name}: {error:#}")),
            },
        }
    }
}

/// dspy keeps its tools in `{tool.name: tool for tool in tools}`, so the catalogue is numbered
/// in the order the caller supplied and a repeated name replaces the earlier tool without
/// moving it. Sorting instead would renumber the prompt the model reads.
pub(super) fn as_dict(tools: impl Iterator<Item = Box<dyn Tool>>) -> Vec<Box<dyn Tool>> {
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
pub(super) fn backticked<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

impl ReAct {
    /// dspy `_call_with_potential_trajectory_truncation`: ask, and where the prompt was too long
    /// for the model, drop the oldest tool call and ask again — three times, then give up.
    ///
    /// This is the one place the crate branches on an error's *identity* rather than its message.
    /// Every other failure means the call will not work; this one means the call will not work
    /// *as it stands*, and there is something to do about it. Without it a long agent run ends
    /// where upstream's carries on.
    async fn asked(
        &self,
        predictor: &Predict,
        inputs: Example,
        trajectory: &mut Trajectory,
        trace: &mut Vec<TraceStep>,
    ) -> Result<Prediction> {
        for _ in 0..TRUNCATION_ATTEMPTS {
            let mut asking = inputs.clone();
            asking.set("trajectory", Value::String(trajectory.rendered()));
            match predictor.forward_traced(asking, trace).await {
                Ok(step) => return Ok(step),
                Err(error) if error.is::<ContextWindowExceeded>() => {
                    tracing::warn!(
                        "Trajectory exceeded the context window, truncating the oldest tool call information."
                    );
                    trajectory.truncate_oldest()?;
                }
                Err(error) => return Err(error),
            }
        }
        bail!("The context window was exceeded even after 3 attempts to truncate the trajectory.")
    }

    /// The episode itself, written once because [`Module::forward`] and
    /// [`Module::forward_traced`] differ only in whether anyone keeps what the trace records.
    async fn run(&self, inputs: Example, trace: &mut Vec<TraceStep>) -> Result<Prediction> {
        {
            let mut trajectory = Trajectory::default();
            for _ in 0..self.max_iters {
                let mut turn_inputs = inputs.clone();
                turn_inputs.set("trajectory", Value::String(trajectory.rendered()));

                let mark = trace.len();
                let step = self
                    .asked(&self.react, turn_inputs, &mut trajectory, trace)
                    .await?;
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

            let final_inputs = inputs;
            let mark = trace.len();
            let extracted = self
                .asked(&self.extract, final_inputs, &mut trajectory, trace)
                .await?;
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
            let span = crate::observe::module_shown("ReAct", &inputs);
            let mut discarded = Vec::new();
            crate::observe::watching(span, self.run(inputs, &mut discarded)).await
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

/// `ReAct!("question -> answer", tools)` — a [`ReAct`] agent over a signature and its tools, the
/// module form of `ReAct::new(make_signature!(...), tools)`. `max_iters = N` caps the loop.
///
/// ```
/// use dsrust::{ReAct, FnTool, Tool};
/// use serde_json::{json, Value};
///
/// let tools: Vec<Box<dyn Tool>> = vec![Box::new(FnTool::new(
///     "get_weather",
///     "look up the weather for a city",
///     json!({ "city": { "type": "string" } }),
///     |args: &Value| Ok(format!("sunny in {}", args["city"].as_str().unwrap_or_default())),
/// ))];
/// let agent = ReAct!("question -> answer", tools, max_iters = 5);
/// assert_eq!(agent.max_iters, 5);
/// ```
#[macro_export]
macro_rules! ReAct {
    ($signature:literal, $tools:expr $(,)?) => {
        $crate::ReAct::new($crate::make_signature!($signature), $tools)
    };
    ($signature:literal, $tools:expr, max_iters = $max:expr $(,)?) => {
        $crate::ReAct::new($crate::make_signature!($signature), $tools).max_iters($max)
    };
    ($task:ty, $tools:expr $(,)?) => {
        $crate::ReAct::new(
            <$task as $crate::signature::SignatureSpec>::signature(),
            $tools,
        )
    };
    ($task:ty, $tools:expr, max_iters = $max:expr $(,)?) => {
        $crate::ReAct::new(
            <$task as $crate::signature::SignatureSpec>::signature(),
            $tools,
        )
        .max_iters($max)
    };
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
    use crate::signature::OutField;
    use serde_json::json;

    pub(super) fn weather() -> Box<dyn Tool> {
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

    pub(super) fn task() -> Signature {
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
        assert_eq!(react.observe(FINISH, &json!({})), json!("Completed."));
        assert_eq!(
            react.observe(FINISH, &json!({ "answer": "sunny" })),
            json!("Execution error in finish: finish takes no arguments")
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
        let observation = observation.as_str().expect("an error observation");
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
            json!("Execution error in teleport: no such tool")
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

#[cfg(test)]
mod truncation_tests {
    use super::tests::{task, weather};
    use super::*;
    use crate::lm::api::{LmRequest, LmResponse};
    use crate::lm::{Capabilities, DynChatModel};
    use anyhow::anyhow;
    use serde_json::json;
    use std::sync::Mutex;

    /// A model that refuses while the trajectory is longer than it can read, and answers once it
    /// fits.
    ///
    /// The whole point of the typed error is that this failure is *recoverable*: the same request
    /// with less trajectory in it succeeds. A model that always refused would prove only that the
    /// loop gives up. `budget` is a step count — `Trajectory::rendered` renumbers from zero, so a
    /// prompt still naming `thought_{budget}` is one still carrying more steps than that.
    struct TooLong {
        budget: usize,
        refusals: Mutex<usize>,
        reply: String,
    }

    impl DynChatModel for TooLong {
        fn forward_dyn<'a>(
            &'a self,
            request: &'a LmRequest,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<LmResponse>> + Send + 'a>> {
            let asked: String = request.messages.iter().filter_map(|m| m.text()).collect();
            let too_long = asked.contains(&format!("thought_{}", self.budget));
            Box::pin(std::future::ready(match too_long {
                true => {
                    *self.refusals.lock().expect("refusals") += 1;
                    Err(crate::lm::ContextWindowExceeded {
                        model: "test".to_owned(),
                        message: "maximum context length is 8192 tokens".to_owned(),
                    }
                    .into())
                }
                false => Ok(LmResponse::completions(vec![self.reply.clone()])),
            }))
        }

        fn capabilities_dyn<'a>(
            &'a self,
        ) -> std::pin::Pin<Box<dyn Future<Output = Capabilities> + Send + 'a>> {
            Box::pin(std::future::ready(Capabilities::default()))
        }

        fn native_reasoning_usable_dyn(&self) -> bool {
            false
        }

        fn native_citations_usable_dyn(&self) -> bool {
            false
        }

        /// A scripted double states nothing, which is the `null` dspy writes for an unpinned predictor.
        fn dump_state_dyn(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
            None
        }
    }

    fn long_trajectory(steps: usize) -> Trajectory {
        Trajectory {
            steps: (0..steps)
                .map(|n| Step {
                    thought: format!("thinking about step {n} at some length to fill the window"),
                    tool: "get_weather".to_owned(),
                    args: json!({ "city": "Paris" }),
                    observation: json!(format!("observation {n}, also reasonably wordy")),
                })
                .collect(),
        }
    }

    /// The oldest tool call goes, and only that one.
    #[test]
    fn truncating_drops_the_oldest_step() {
        let mut trajectory = long_trajectory(3);
        trajectory
            .truncate_oldest()
            .expect("three steps can lose one");
        assert_eq!(trajectory.steps.len(), 2);
        assert!(
            trajectory.steps[0].thought.contains("step 1"),
            "the oldest went"
        );
    }

    /// A trajectory of one cannot be shortened, and says so rather than emptying itself — an empty
    /// one would ask again with nothing learned and fail the same way, three times over.
    #[test]
    fn a_single_step_cannot_be_truncated() {
        let refused = long_trajectory(1)
            .truncate_oldest()
            .expect_err("nothing to drop");
        assert!(
            refused.to_string().contains("only has one tool call"),
            "got: {refused}"
        );
    }

    /// The recovery end to end: a prompt too long is trimmed and asked again, and the run finishes.
    #[tokio::test]
    async fn a_prompt_too_long_is_trimmed_and_asked_again() {
        let model = std::sync::Arc::new(TooLong {
            budget: 3,
            refusals: Mutex::new(0),
            reply: "[[ ## reasoning ## ]]\nit was sunny\n\n[[ ## answer ## ]]\nsunny\n\n[[ ## completed ## ]]".to_owned(),
        });
        let react = ReAct::new(task(), vec![weather()]).set_lm(model.clone());
        // Five steps against a three-step budget: refuse, trim, refuse, trim, fits — the third of
        // the three attempts upstream allows. Six would never get there, upstream included.
        let mut trajectory = long_trajectory(5);
        let mut trace = Vec::new();

        let answered = react
            .asked(
                &react.extract,
                Example::default(),
                &mut trajectory,
                &mut trace,
            )
            .await
            .expect("the trimmed trajectory fits");

        assert_eq!(answered.get("answer"), Some(&json!("sunny")));
        assert_eq!(
            *model.refusals.lock().expect("refusals"),
            2,
            "it really did refuse first"
        );
        assert_eq!(
            trajectory.steps.len(),
            3,
            "and the trajectory really was trimmed"
        );
    }

    /// Three refusals is the end of it, with dspy's wording.
    #[tokio::test]
    async fn a_prompt_that_never_fits_gives_up_after_three_tries() {
        let model = std::sync::Arc::new(TooLong {
            budget: 0,
            refusals: Mutex::new(0),
            reply: String::new(),
        });
        let react = ReAct::new(task(), vec![weather()]).set_lm(model.clone());
        let mut trajectory = long_trajectory(8);

        let refused = react
            .asked(
                &react.extract,
                Example::default(),
                &mut trajectory,
                &mut Vec::new(),
            )
            .await
            .expect_err("nothing fits");
        assert!(
            refused
                .to_string()
                .contains("even after 3 attempts to truncate"),
            "got: {refused}"
        );
        assert_eq!(
            *model.refusals.lock().expect("refusals"),
            3,
            "three tries, not more"
        );
    }

    /// Any other refusal is passed straight up. Trimming does not fix an expired key, and trying
    /// would spend three more calls before failing anyway.
    #[tokio::test]
    async fn another_failure_is_not_retried() {
        struct Broken;
        impl DynChatModel for Broken {
            fn forward_dyn<'a>(
                &'a self,
                _request: &'a LmRequest,
            ) -> std::pin::Pin<Box<dyn Future<Output = Result<LmResponse>> + Send + 'a>>
            {
                Box::pin(std::future::ready(Err(anyhow!(
                    "Incorrect API key provided"
                ))))
            }
            fn capabilities_dyn<'a>(
                &'a self,
            ) -> std::pin::Pin<Box<dyn Future<Output = Capabilities> + Send + 'a>> {
                Box::pin(std::future::ready(Capabilities::default()))
            }
            fn native_reasoning_usable_dyn(&self) -> bool {
                false
            }

            fn native_citations_usable_dyn(&self) -> bool {
                false
            }

            /// A scripted double states nothing, which is the `null` dspy writes for an unpinned predictor.
            fn dump_state_dyn(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
                None
            }
        }

        let react = ReAct::new(task(), vec![weather()]).set_lm(std::sync::Arc::new(Broken));
        let mut trajectory = long_trajectory(4);
        let refused = react
            .asked(
                &react.extract,
                Example::default(),
                &mut trajectory,
                &mut Vec::new(),
            )
            .await
            .expect_err("the key is wrong");
        assert!(
            refused.to_string().contains("Incorrect API key"),
            "got: {refused}"
        );
        assert_eq!(trajectory.steps.len(), 4, "and nothing was trimmed over it");
    }
}
