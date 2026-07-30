//! dspy `predict/react_v2.py`: an agent that calls tools until it can submit the task's answer.
//!
//! Where [`ReAct`](super::ReAct) renders a thought, a tool name and its arguments as fields and
//! reads an observation back into a growing trajectory, `ReActV2` asks the provider to call tools
//! itself. Each turn the model emits a thought and a set of tool calls; the calls run, their
//! results join the conversation [`History`], and the loop repeats until the model calls the
//! reserved `submit` tool with the task's outputs — or the budget runs out and one last turn is
//! forced to submit.

use std::pin::Pin;

use anyhow::{Result, anyhow};
use serde_json::{Map, Value, json};

use crate::adapter::{Adapter, History, ToolCallResults, ToolCalls};
use crate::example::{Example, Prediction};
use crate::module::{Module, NamedPredictor, TraceStep};
use crate::predict::{Predict, Steering};
use crate::signature::Signature;

use super::as_dict;
use super::tool::Tool;

mod signature;
use signature::{react_signature, submit_tool, tool_descriptor};

/// The name dspy reserves for the final-output tool, which a user tool may not take.
const SUBMIT: &str = "submit";

/// An agent that reasons and calls tools natively until it can submit the task's outputs.
pub struct ReActV2 {
    /// The task's real signature: what the caller asked for.
    pub signature: Signature,
    /// Every tool the model may call, `submit` last, in the order dspy's dict holds them.
    tools: Vec<Box<dyn Tool>>,
    /// One turn: emit a thought and the tool calls to make, over the conversation so far.
    react: Predict,
    /// dspy defaults to 20. A budget is not optional — a model that never submits would loop
    /// against a paid provider forever.
    pub max_iters: usize,
    /// The `tools` input value, the same every turn: each tool as `{name, desc, args}`.
    tool_list: Value,
}

impl ReActV2 {
    /// Wrap a task in an agent over the given tools. dspy dedups the tools into a dict keyed by
    /// name, reserves `submit` for the final output, and builds the per-turn predictor.
    ///
    /// A tool named `submit` collides with the reserved final-output tool, which upstream refuses;
    /// it is a construction-time misuse, not runtime data, so it panics with upstream's message.
    pub fn new(signature: Signature, tools: Vec<Box<dyn Tool>>) -> Self {
        let mut ordered = as_dict(tools.into_iter());
        assert!(
            !ordered.iter().any(|tool| tool.name() == SUBMIT),
            "`submit` is reserved by ReActV2 as the final-output tool."
        );
        ordered.push(submit_tool(&signature));

        let react = Predict::from_signature(react_signature(&signature, &ordered));
        let tool_list = Value::Array(
            ordered
                .iter()
                .map(|tool| tool_descriptor(tool.as_ref()))
                .collect(),
        );
        Self {
            signature,
            tools: ordered,
            react,
            max_iters: 20,
            tool_list,
        }
    }

    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// The adapter the per-turn predictor asks through — a native one turns the tool calls into
    /// provider function calls rather than a rendered field.
    pub fn with_adapter(mut self, adapter: impl Adapter + 'static) -> Self {
        self.react = self.react.with_adapter(adapter);
        self
    }

    /// The model the per-turn predictor asks, when it is not the configured one. dspy reaches the
    /// context LM; here a module can be handed its own, which is what an optimizer varies and what
    /// a test scripts.
    pub fn with_lm(mut self, lm: std::sync::Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.react = self.react.with_lm(lm);
        self
    }

    /// The signature one turn is asked with — the task's inputs plus `history` and `tools`,
    /// answering with `next_thought` and `tool_calls`. dspy's `react.react.signature`.
    pub fn turn_signature(&self) -> &Signature {
        &self.react.signature
    }

    /// The `tools` input value the per-turn predictor is handed each turn — each tool as
    /// `{name, desc, args}`, the same list a native request formats into provider function calls.
    pub fn turn_tools(&self) -> &Value {
        &self.tool_list
    }

    /// Every tool the model may call, `submit` included, in the order they are held.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|tool| tool.name()).collect()
    }

    fn tool(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|tool| tool.name() == name)
            .map(Box::as_ref)
    }

    /// The episode, written once because [`Module::forward`] and [`Module::forward_traced`] differ
    /// only in whether anyone keeps what the trace records.
    async fn run(&self, input_args: Example, trace: &mut Vec<TraceStep>) -> Result<Prediction> {
        let max_iters = input_args
            .get("max_iters")
            .and_then(Value::as_u64)
            .map_or(self.max_iters, |iters| iters as usize);
        let mut history = coerce_history(input_args.get("history"));
        let mut pending = self.pending_inputs(&input_args);

        let mut break_reason = "max_iters";
        for turn_index in 0..max_iters {
            let asked = self
                .ask_turn(&history, &pending, &Steering::default(), trace)
                .await;
            let (pred, calls) = match asked {
                Ok(pair) => pair,
                // dspy catches a parse or validation failure and ends the loop, so a recoverable
                // slip becomes a forced submit rather than an aborted episode.
                Err(_) => {
                    break_reason = "parse_error";
                    break;
                }
            };
            if calls.tool_calls.is_empty() {
                break_reason = "empty_tool_calls";
                break;
            }

            let calls = ensure_ids(calls, turn_index);
            let (results, final_outputs) = self.execute_tool_calls(&calls);
            let event = self.history_event(&pending, &pred, calls, results, final_outputs.as_ref());
            record(&mut history, event);
            pending = Example::default();

            if let Some(final_outputs) = final_outputs {
                return Ok(submitted(final_outputs, &history, "submit"));
            }
        }

        self.forced_submit(history, pending, break_reason, max_iters, trace)
            .await
    }

    /// The task's own input fields that were supplied. dspy carries these on the first turn only;
    /// once recorded into the history they are cleared, so a continuation omits them.
    fn pending_inputs(&self, input_args: &Example) -> Example {
        Example::new(self.signature.inputs.iter().filter_map(|field| {
            input_args
                .get(&field.name)
                .map(|value| (field.name.clone(), value.clone()))
        }))
    }

    /// One turn of the loop: ask the per-turn predictor, steered as the caller asks, and read its
    /// tool calls back. The forced submit steers `submit` and clears reasoning; a normal turn
    /// steers nothing, so native reasoning turns on wherever the model allows it.
    async fn ask_turn(
        &self,
        history: &History,
        pending: &Example,
        steering: &Steering,
        trace: &mut Vec<TraceStep>,
    ) -> Result<(Prediction, ToolCalls)> {
        let mut inputs = pending.clone();
        inputs.set("history", history.to_value());
        inputs.set("tools", self.tool_list.clone());

        let pred = self
            .react
            .forward_with_steering(inputs.clone(), steering)
            .await?;
        trace.push(TraceStep {
            predictor: "react".to_owned(),
            inputs,
            outputs: pred.example.clone(),
        });
        let calls = coerce_tool_calls(&pred)?;
        Ok((pred, calls))
    }

    /// dspy `_execute_tool_calls`: run each call, turning an unknown tool or a raised error into an
    /// observation the model reads rather than an abort. `submit` returning a mapping is the task's
    /// final output.
    fn execute_tool_calls(
        &self,
        calls: &ToolCalls,
    ) -> (ToolCallResults, Option<Map<String, Value>>) {
        let mut values = Vec::new();
        let mut is_errors = Vec::new();
        let mut final_outputs = None;

        for call in &calls.tool_calls {
            let Some(tool) = self.tool(&call.name) else {
                values.push(json!(format!("Unknown tool: {}", call.name)));
                is_errors.push(true);
                continue;
            };
            match crate::observe::tool_call(tool, &Value::Object(call.args.clone())) {
                Ok(value) => {
                    if call.name == SUBMIT && value.is_object() {
                        final_outputs = value.as_object().cloned();
                    }
                    values.push(value);
                    is_errors.push(false);
                }
                Err(error) => {
                    values.push(json!(format!(
                        "Execution error in {}: {error:#}",
                        call.name
                    )));
                    is_errors.push(true);
                }
            }
        }

        let results =
            ToolCallResults::from_tool_calls_and_values(&calls.tool_calls, values, Some(is_errors))
                .unwrap_or_default();
        (results, final_outputs)
    }

    /// dspy `_history_event`: what this turn contributes to the conversation — the inputs it was
    /// given, the thought it produced, and the calls it made with their results. The calls are kept
    /// [with their ids](ToolCalls::to_value_with_ids) so a native replay can pair each result back.
    fn history_event(
        &self,
        pending: &Example,
        pred: &Prediction,
        calls: ToolCalls,
        results: ToolCallResults,
        final_outputs: Option<&Map<String, Value>>,
    ) -> Map<String, Value> {
        let mut event: Map<String, Value> = pending
            .fields()
            .map(|(name, value)| (name.to_owned(), value.clone()))
            .collect();
        if let Some(thought) = pred.get("next_thought").filter(|value| !value.is_null()) {
            event.insert("next_thought".to_owned(), thought.clone());
        }
        if !calls.tool_calls.is_empty() {
            let calls = match results.tool_call_results.is_empty() {
                true => calls,
                false => calls.with_results(results),
            };
            event.insert("tool_calls".to_owned(), calls.to_value_with_ids());
        }
        if let Some(final_outputs) = final_outputs {
            for (name, value) in final_outputs {
                event.insert(name.clone(), value.clone());
            }
        }
        event
    }

    /// dspy `_forced_submit`: the loop ended without a submit, so ask once more — steered to pin the
    /// provider to `submit` and turn native reasoning off, as upstream's
    /// `config={"tool_choice": submit, "reasoning_effort": None}` does — and take only a submit call.
    /// Anything else, or a failure, ends the episode carrying the reason it stopped.
    async fn forced_submit(
        &self,
        mut history: History,
        pending: Example,
        break_reason: &str,
        turn_index: usize,
        trace: &mut Vec<TraceStep>,
    ) -> Result<Prediction> {
        let steering = Steering {
            reasoning_effort: crate::adapter::native_reasoning::ReasoningEffort::Off,
            forced_tool: Some(SUBMIT.to_owned()),
            ..Steering::default()
        };
        let Ok((pred, calls)) = self.ask_turn(&history, &pending, &steering, trace).await else {
            return Ok(failed(&history, break_reason));
        };
        let calls = ensure_ids(calls, turn_index);
        let submit_calls = ToolCalls::new(
            calls
                .tool_calls
                .into_iter()
                .filter(|call| call.name == SUBMIT)
                .collect(),
        );
        if submit_calls.tool_calls.is_empty() {
            return Ok(failed(&history, break_reason));
        }

        let (results, final_outputs) = self.execute_tool_calls(&submit_calls);
        let event = self.history_event(
            &pending,
            &pred,
            submit_calls,
            results,
            final_outputs.as_ref(),
        );
        record(&mut history, event);

        match final_outputs {
            Some(final_outputs) => Ok(submitted(final_outputs, &history, "forced_submit")),
            None => Ok(failed(&history, break_reason)),
        }
    }
}

impl Module for ReActV2 {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let span = crate::observe::module_shown("ReActV2", &inputs);
            let mut discarded = Vec::new();
            crate::observe::watching(span, self.run(inputs, &mut discarded)).await
        })
    }

    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(self.run(inputs, trace))
    }

    /// The per-turn predictor is the one an optimizer improves: demos on it are what teach a model
    /// to call the right tools. dspy names it `react`.
    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        self.react
            .named_predictors()
            .into_iter()
            .map(|mut predictor| {
                predictor.name = "react".to_owned();
                predictor
            })
            .collect()
    }
}

/// dspy `_append_history_event`: an empty event is not recorded, so a turn that produced nothing
/// does not leave a blank message in the conversation.
fn record(history: &mut History, event: Map<String, Value>) {
    if !event.is_empty() {
        history.push(event);
    }
}

/// dspy `Prediction(**final_outputs, history=history, termination_reason=...)`: the task's outputs
/// beside the conversation that produced them and why the loop stopped.
fn submitted(final_outputs: Map<String, Value>, history: &History, reason: &str) -> Prediction {
    let mut example = Example::new(final_outputs);
    example.set("history", history.to_value());
    example.set("termination_reason", json!(reason));
    Prediction::new(example, String::new())
}

/// The episode ended without a submit: no outputs, only the history and the reason it stopped.
fn failed(history: &History, reason: &str) -> Prediction {
    let mut example = Example::default();
    example.set("history", history.to_value());
    example.set("termination_reason", json!(reason));
    Prediction::new(example, String::new())
}

/// dspy `_coerce_history`: nothing supplied is an empty conversation; a serialized mapping reads
/// back into one.
fn coerce_history(value: Option<&Value>) -> History {
    match value {
        None | Some(Value::Null) => History::default(),
        Some(value) => serde_json::from_value(value.clone()).unwrap_or_default(),
    }
}

/// dspy `_coerce_tool_calls`: a predictor that stated no calls made none; one that stated some has
/// them read back into the type.
fn coerce_tool_calls(pred: &Prediction) -> Result<ToolCalls> {
    match pred.get("tool_calls").filter(|value| !value.is_null()) {
        None => Ok(ToolCalls::default()),
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|error| anyhow!("could not read `tool_calls`: {error}")),
    }
}

/// dspy `_ensure_tool_call_ids`: a call the model left unidentified is numbered by its turn and
/// position, so a result can be paired to it — `call_{turn}_{index}`.
fn ensure_ids(calls: ToolCalls, turn_index: usize) -> ToolCalls {
    let ensured = calls
        .tool_calls
        .into_iter()
        .enumerate()
        .map(|(index, mut call)| {
            if call.id.is_none() {
                call.id = Some(format!("call_{turn_index}_{index}"));
            }
            call
        })
        .collect();
    ToolCalls::new(ensured)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::react::FnTool;

    fn lookup() -> Box<dyn Tool> {
        Box::new(FnTool::new(
            "lookup",
            "look something up",
            json!({ "query": { "type": "string" } }),
            |args: &Value| {
                Ok(format!(
                    "found {}",
                    args["query"].as_str().unwrap_or_default()
                ))
            },
        ))
    }

    fn task() -> Signature {
        "question -> answer".parse().expect("a signature")
    }

    /// dspy `test_react_v2_submit_tool_returns_original_output_fields`: submit hands back the task's
    /// outputs, and the per-turn signature never carries a `tool_call_results` input.
    #[test]
    fn the_submit_tool_returns_the_tasks_output_fields() {
        let agent = ReActV2::new(task(), vec![]);
        let submit = agent.tool(SUBMIT).expect("submit exists");
        assert_eq!(
            submit
                .call_value(&json!({ "answer": "Paris" }))
                .expect("submits"),
            json!({ "answer": "Paris" })
        );
        assert!(
            !agent
                .turn_signature()
                .inputs
                .iter()
                .any(|field| field.name == "tool_call_results")
        );
    }

    /// submit refuses to end the task without every output, and the message is what the model reads.
    #[test]
    fn the_submit_tool_refuses_a_missing_output() {
        let agent = ReActV2::new(task(), vec![]);
        let error = agent
            .tool(SUBMIT)
            .expect("submit")
            .call_value(&json!({}))
            .expect_err("refused");
        assert!(
            error
                .to_string()
                .contains("Missing required final output field(s): answer")
        );
    }

    /// The per-turn signature carries the task's inputs, then `history` and `tools`, answering with
    /// `next_thought` and `tool_calls` — recognised by the types they name.
    #[test]
    fn the_turn_signature_has_dspys_fields() {
        let agent = ReActV2::new(task(), vec![lookup()]);
        let inputs: Vec<&str> = agent
            .turn_signature()
            .inputs
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        assert_eq!(inputs, ["question", "history", "tools"]);
        let outputs: Vec<&str> = agent
            .turn_signature()
            .outputs
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        assert_eq!(outputs, ["next_thought", "tool_calls"]);
        assert_eq!(agent.tool_names(), ["lookup", "submit"]);
    }

    /// The instructions name the task, the agent's job, and the tools it may call — dspy's list,
    /// joined and stripped.
    #[test]
    fn the_instructions_name_the_task_and_the_tools() {
        let agent = ReActV2::new(task(), vec![lookup()]);
        let instructions = &agent.turn_signature().instructions;
        assert!(
            instructions.starts_with("Given the fields `question`, produce the fields `answer`.")
        );
        assert!(instructions.contains("call `submit` with `answer`."));
        assert!(instructions.ends_with("The available tools are: `lookup`, `submit`."));
    }

    /// A user tool cannot take the reserved final-output name.
    #[test]
    #[should_panic(expected = "`submit` is reserved")]
    fn a_user_submit_tool_is_refused() {
        let clashing = Box::new(FnTool::new("submit", "no", json!({}), |_: &Value| {
            Ok(String::new())
        })) as Box<dyn Tool>;
        ReActV2::new(task(), vec![clashing]);
    }
}
