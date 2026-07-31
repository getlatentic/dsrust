//! dspy `predict/rlm.py`: the Recursive Language Model.
//!
//! An inference strategy for context too long to hand a model directly: the input stays in a REPL
//! as a variable, and the model writes Python to explore it — printing samples, slicing, and
//! calling sub-LLMs over the pieces it cares about — until it can submit an answer. What runs the
//! code is the caller's [`CodeInterpreter`], and what the
//! model is shown of the session is [`ReplHistory`].

mod fences;
mod signatures;
mod submission;
mod tools;

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use serde_json::{Value, json};

use crate::adapter::Type;
use crate::adapter::python_json::json_dumps;
use crate::adapter::types::base::{Formatted, to_field_value};
use crate::example::{Example, Prediction};
use crate::interpreter::{
    CodeInterpreter, DenoInterpreter, Executed, OutputField, ReplEntry, ReplHistory, ReplVariable,
    SandboxSerializable, sandbox, with_constraints,
};
use crate::module::{Module, NamedPredictor, TraceStep, relabel};
use crate::react::Tool;
use crate::signature::{Signature, python_name};

use fences::strip_code_fences;
use signatures::signatures;

pub use tools::llm_query_tools;

use super::{Dynamic, Predict};

/// dspy's default sub-LLM call budget.
const DEFAULT_MAX_LLM_CALLS: usize = 50;

/// The recorded run of dspy's own RLM, which the fence and signature conformance tests are held to.
/// `RLM!("question -> answer")` — the model explores a value in the sandbox rather than reading it in a prompt.
///
/// Takes a string signature or a task declared with `#[derive(Signature)]`, as every other module
/// macro does; the declared form carries its doc comment as the signature's instructions.
/// `max_iterations = N` caps the loop.
#[macro_export]
macro_rules! RLM {
    ($signature:literal $(,)?) => {
        $crate::Rlm::new($crate::make_signature!($signature))
    };
    ($signature:literal, max_iterations = $max:expr $(,)?) => {
        $crate::Rlm::new($crate::make_signature!($signature)).max_iterations($max)
    };
    ($task:ty $(,)?) => {
        $crate::Rlm::new(<$task as $crate::signature::SignatureSpec>::signature())
    };
    ($task:ty, max_iterations = $max:expr $(,)?) => {
        $crate::Rlm::new(<$task as $crate::signature::SignatureSpec>::signature())
            .max_iterations($max)
    };
}

#[cfg(test)]
fn golden() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/predict/rlm.json");
    let text = std::fs::read_to_string(&path).expect("the golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

/// dspy's `RLM`: a model that explores its input in a REPL rather than being handed it.
pub struct Rlm {
    /// The task's real signature: what the caller asked for.
    pub signature: Signature,
    /// How many snippets the model may run before the extract ask happens anyway.
    pub max_iterations: usize,
    /// The sub-LLM call budget the model is told about.
    pub max_llm_calls: usize,
    /// How much of one output reaches the next prompt.
    pub max_output_chars: usize,
    generate_action: Predict<Dynamic>,
    extract: Predict<Dynamic>,
    tools: Vec<Arc<dyn Tool>>,
    interpreter: Arc<dyn CodeInterpreter>,
    /// Inputs that live in the sandbox rather than crossing as JSON, by field name. See
    /// [`Self::with_sandbox_input`].
    sandboxed: BTreeMap<String, Arc<dyn SandboxSerializable>>,
}

impl Rlm {
    /// dspy's `interpreter=None`: the Deno/Pyodide sandbox, which is what upstream defaults to.
    pub fn new(signature: Signature) -> Self {
        Self::interpreter(signature, Arc::new(DenoInterpreter::new()))
    }

    /// The same, running code somewhere the caller chose.
    pub fn interpreter(signature: Signature, interpreter: Arc<dyn CodeInterpreter>) -> Self {
        Self::with_tools(signature, Vec::new(), interpreter)
    }

    pub fn with_tools(
        signature: Signature,
        tools: Vec<Arc<dyn Tool>>,
        interpreter: Arc<dyn CodeInterpreter>,
    ) -> Self {
        let (action, extract) = signatures(&signature, &tools, DEFAULT_MAX_LLM_CALLS);
        Self {
            signature,
            max_iterations: 20,
            max_llm_calls: DEFAULT_MAX_LLM_CALLS,
            max_output_chars: crate::interpreter::repl::MAX_OUTPUT_CHARS,
            generate_action: Predict::from_signature(action),
            extract: Predict::from_signature(extract),
            tools,
            interpreter,
            sandboxed: BTreeMap::new(),
        }
    }

    /// Hand this input to the sandbox as its own reconstruction rather than as JSON.
    ///
    /// dspy decides this by type — an input that `isinstance`s as `SandboxSerializable` takes the
    /// other path. An [`Example`] holds JSON, so a Rust caller names the field instead, and the
    /// value is serialized, rebuilt in the sandbox before the first turn, and described to the
    /// model by [`build_repl_variable`](crate::interpreter::build_repl_variable) rather than previewed.
    pub fn with_sandbox_input(
        mut self,
        name: impl Into<String>,
        value: Arc<dyn SandboxSerializable>,
    ) -> Self {
        self.sandboxed.insert(name.into(), value);
        self
    }

    /// dspy `_get_output_fields_info`: the signature's outputs, as the sandbox's `SUBMIT` takes
    /// them. The type travels only where Python can spell it in a generated signature.
    fn output_fields(&self) -> Vec<OutputField> {
        self.signature
            .outputs
            .iter()
            .map(|field| OutputField {
                name: field.name.clone(),
                python_type: python_name(&field.kind).map(str::to_owned),
            })
            .collect()
    }

    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// The budget the model is told about, which is stated in the action instructions — so
    /// changing it rebuilds them.
    pub fn max_llm_calls(mut self, max_llm_calls: usize) -> Self {
        self.max_llm_calls = max_llm_calls;
        let (action, _) = signatures(&self.signature, &self.tools, max_llm_calls);
        self.generate_action = Predict::from_signature(action);
        self
    }

    /// The signature each REPL turn is asked with. dspy reaches the same thing as
    /// `rlm.generate_action.signature`.
    pub fn action_signature(&self) -> &Signature {
        &self.generate_action.signature
    }

    /// Ask both steps of this model.
    pub fn with_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.generate_action = self.generate_action.with_lm(lm.clone());
        self.extract = self.extract.with_lm(lm);
        self
    }

    /// Ask the REPL turns of this model, leaving the extract step on whatever it had.
    ///
    /// The two steps are separable because upstream's are: `rlm.generate_action` and `rlm.extract`
    /// are attributes its own tests replace one at a time, and a caller wanting a cheaper model to
    /// read back a finished session wants the same seam.
    pub fn with_action_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.generate_action = self.generate_action.with_lm(lm);
        self
    }

    /// Ask the extract step of this model. See [`Self::with_action_lm`].
    pub fn with_extract_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.extract = self.extract.with_lm(lm);
        self
    }

    async fn run(&self, inputs: Example, trace: &mut Vec<TraceStep>) -> Result<Prediction> {
        let missing: Vec<&str> = self
            .signature
            .inputs
            .iter()
            // A sandbox-held input is registered on the module rather than passed in the
            // example, since an example carries JSON and this value is not JSON.
            .filter(|field| {
                inputs.get(&field.name).is_none() && !self.sandboxed.contains_key(&field.name)
            })
            .map(|field| field.name.as_str())
            .collect();
        if !missing.is_empty() {
            bail!("Missing required inputs: {missing:?}");
        }

        self.interpreter.define_tools(&self.tools)?;
        // dspy passes `output_fields` when it builds the interpreter, so the sandbox gets a typed
        // `SUBMIT(answer, …)`. Without it the default single-argument one answers under `output`
        // and every declared field reads as missing — the model submits correctly and is refused.
        self.interpreter.define_outputs(&self.output_fields())?;
        // dspy `_prepare_serializable_vars`: a sandbox-held value is rebuilt in the sandbox once,
        // before the first turn, and is not among the values bound on each `execute` after that.
        self.interpreter.start()?;
        for (name, value) in &self.sandboxed {
            let (code, bound) = sandbox::injection(value.as_ref(), name);
            self.interpreter.execute(&code, &bound)?;
        }

        let variables = self.variables(&inputs);
        // dspy binds the caller's *remaining* inputs on every `execute`, which is what lets the
        // model's code reach them by name — `SUBMIT(sum(numbers))` sees `numbers`.
        let bound: serde_json::Map<String, Value> = self
            .signature
            .inputs
            .iter()
            .filter(|field| !self.sandboxed.contains_key(&field.name))
            .filter_map(|field| {
                inputs
                    .get(&field.name)
                    .map(|value| (field.name.clone(), value.clone()))
            })
            .collect();
        let mut history = ReplHistory::new(self.max_output_chars);

        for iteration in 0..self.max_iterations {
            let asked = Example::new([
                ("variables_info", json!(variables)),
                ("repl_history", to_field_value(&history)),
                (
                    "iteration",
                    json!(format!("{}/{}", iteration + 1, self.max_iterations)),
                ),
            ]);
            let mark = trace.len();
            let action = self.generate_action.forward_traced(asked, trace).await?;
            relabel(trace, mark, "generate_action");

            let reasoning = string_field(&action.example, "reasoning");
            let written = string_field(&action.example, "code");
            // A fence the parser refuses is not run: the refusal itself is what the next turn
            // reads, which is how the model learns to write Python.
            let (code, refused) = match strip_code_fences(&written) {
                Ok(code) => (code, None),
                Err(error) => (written.clone(), Some(format!("[Error] {error}"))),
            };
            let outcome = match refused {
                Some(error) => Err(error),
                None => self
                    .interpreter
                    .execute(&code, &bound)
                    .map_err(|error| format!("[Error] {error}")),
            };

            match outcome {
                // An error is an observation, not the end of the episode.
                Err(error) => history = history.append(ReplEntry::new(reasoning, code, error)),
                Ok(Executed::Submitted(value)) => {
                    match self.submitted(&value) {
                        // A malformed submission is fed back so the model can submit again.
                        Err(error) => {
                            history = history.append(ReplEntry::new(reasoning, code, error))
                        }
                        Ok(outputs) => {
                            let final_history = history.append(ReplEntry::new(
                                reasoning.clone(),
                                code,
                                format!("FINAL: {}", json_dumps(&Value::Object(outputs.clone()))),
                            ));
                            return Ok(self.answered(outputs, &final_history, reasoning));
                        }
                    }
                }
                Ok(Executed::Printed(printed)) => {
                    history =
                        history.append(ReplEntry::new(reasoning, code, printed_output(&printed)))
                }
            }
        }

        // Out of iterations: the extract ask reads the outputs off the session instead.
        tracing::warn!("RLM reached max iterations, using extract to get final output");
        let asked = Example::new([
            ("variables_info", json!(variables)),
            ("repl_history", to_field_value(&history)),
        ]);
        let mark = trace.len();
        let extracted = self.extract.forward_traced(asked, trace).await?;
        relabel(trace, mark, "extract");
        let outputs = extracted
            .example
            .fields()
            .map(|(name, value)| (name.to_owned(), value.clone()))
            .collect();
        Ok(self.answered(outputs, &history, "Extract forced final output".to_owned()))
    }

    /// dspy `_build_variables`: what the model is told about each input it can reach.
    fn variables(&self, inputs: &Example) -> Vec<String> {
        self.signature
            .inputs
            .iter()
            .filter_map(|field| {
                let constraints = field.constraints.clone().unwrap_or_default();
                let variable = match self.sandboxed.get(&field.name) {
                    Some(held) => {
                        with_constraints(held.as_ref(), &field.name, &field.desc, &constraints)
                    }
                    None => {
                        let mut built =
                            ReplVariable::from_value(&field.name, inputs.get(&field.name)?);
                        built.desc = field.desc.clone();
                        built.constraints = constraints;
                        built
                    }
                };
                match Type::format(&variable) {
                    Formatted::Text(rendered) => Some(rendered),
                    Formatted::Blocks(_) => None,
                }
            })
            .collect()
    }

    /// dspy `_process_final_output`, in [`submission`]: what a `SUBMIT()` must be, and what the
    /// model is told when it is not — fed back rather than raised, so it can submit again.
    fn submitted(&self, value: &Value) -> Result<serde_json::Map<String, Value>, String> {
        submission::process(&self.signature, value)
    }

    /// dspy returns the outputs beside the trajectory and the reasoning that ended the run.
    fn answered(
        &self,
        outputs: serde_json::Map<String, Value>,
        history: &ReplHistory,
        final_reasoning: String,
    ) -> Prediction {
        let mut example = Example::new(outputs);
        example.set("trajectory", json!(history.entries));
        example.set("final_reasoning", json!(final_reasoning));
        Prediction::new(example, String::new())
    }
}

/// dspy `_format_output`: silence is reported as such, since a turn that printed nothing is
/// almost always a turn that forgot to.
fn printed_output(printed: &Value) -> String {
    let output = match printed {
        Value::Null => String::new(),
        // dspy joins a list of output lines with newlines.
        Value::Array(lines) => lines
            .iter()
            .map(|line| match line {
                Value::String(text) => text.clone(),
                other => json_dumps(other),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::String(text) => text.clone(),
        other => json_dumps(other),
    };
    match output.is_empty() {
        true => "(no output - did you forget to print?)".to_owned(),
        false => output,
    }
}

fn string_field(example: &Example, name: &str) -> String {
    example
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

impl Module for Rlm {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let span = crate::observe::module_shown("RLM", &inputs);
            let mut discarded = Vec::new();
            crate::observe::watching(span, self.run(inputs, &mut discarded)).await
        })
    }

    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move { self.run(inputs, trace).await })
    }

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        let mut predictors = Vec::new();
        for mut predictor in self.generate_action.named_predictors() {
            predictor.name = format!("generate_action.{}", predictor.name);
            predictors.push(predictor);
        }
        for mut predictor in self.extract.named_predictors() {
            predictor.name = format!("extract.{}", predictor.name);
            predictors.push(predictor);
        }
        predictors
    }
}

#[cfg(test)]
mod loop_tests {
    use super::*;
    use crate::example;
    use crate::interpreter::tests::Scripted as ScriptedInterpreter;
    use crate::predict::scripted::Scripted;
    use crate::signature::OutField;

    fn task() -> Signature {
        "context -> answer".parse().expect("parses")
    }

    fn action(reasoning: &str, code: &str) -> String {
        format!(
            "[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## code ## ]]\n{code}\n\n[[ ## completed ## ]]"
        )
    }

    fn rlm(interpreter: Arc<ScriptedInterpreter>, replies: &[&'static str]) -> Rlm {
        let model = Arc::new(Scripted::new(replies));
        let mut rlm = Rlm::interpreter(task(), interpreter);
        rlm.generate_action = rlm.generate_action.with_lm(model.clone());
        rlm.extract = rlm.extract.with_lm(model);
        rlm
    }

    /// A `SUBMIT()` carrying every output field ends the run, and the trajectory records the turn
    /// that did it.
    /// The sandbox must be told the signature's outputs, or it keeps a single-argument `SUBMIT`
    /// whose result arrives under `output` — and every declared field then reads as missing.
    ///
    /// Found against a live model, not here: gemma answered `SUBMIT("Lagos")` correctly, was told
    /// `Missing output fields: ["answer"]` twice, burned its iterations and fell through to the
    /// forced extraction. Nothing in 51 passing upstream RLM tests noticed, because the bridge
    /// builds dspy's own interpreter and dspy registers them itself.
    #[tokio::test]
    async fn the_signatures_outputs_are_registered_with_the_sandbox() {
        let interpreter = Arc::new(ScriptedInterpreter::new([Ok(Executed::Submitted(
            json!({ "answer": "Lagos", "count": 3 }),
        ))]));
        let rlm = Rlm::interpreter(
            "question -> answer: str, count: int"
                .parse()
                .expect("parses"),
            interpreter.clone(),
        );
        let model = Scripted::new(&[
            "[[ ## reasoning ## ]]\nlook\n\n[[ ## code ## ]]\nSUBMIT(answer=\"Lagos\", count=3)\n\n[[ ## completed ## ]]",
        ]);
        let _ = rlm
            .with_lm(Arc::new(model))
            .forward(example! { question: "which city?" })
            .await;

        let registered = interpreter
            .outputs
            .lock()
            .expect("the output fields")
            .clone();
        let named: Vec<(&str, Option<&str>)> = registered
            .iter()
            .map(|field| (field.name.as_str(), field.python_type.as_deref()))
            .collect();
        assert_eq!(named, vec![("answer", Some("str")), ("count", Some("int"))]);
    }

    #[tokio::test]
    async fn a_submission_ends_the_run() {
        let interpreter = Arc::new(ScriptedInterpreter::new([Ok(Executed::Submitted(
            json!({ "answer": "42" }),
        ))]));
        let rlm = rlm(
            interpreter.clone(),
            &[&*Box::leak(
                action("submit it", "```python\nSUBMIT(answer='42')\n```").into_boxed_str(),
            )],
        );

        let prediction = rlm
            .forward(example! { context: "a long document" })
            .await
            .expect("answers");
        assert_eq!(prediction.get("answer"), Some(&json!("42")));
        assert_eq!(prediction.get("final_reasoning"), Some(&json!("submit it")));
        // The fence was stripped before the code reached the interpreter.
        assert_eq!(
            *interpreter.ran.lock().expect("ran"),
            ["SUBMIT(answer='42')"]
        );
        let trajectory = prediction.get("trajectory").expect("a trajectory");
        assert_eq!(trajectory.as_array().expect("entries").len(), 1);
        assert!(
            trajectory[0]["output"]
                .as_str()
                .expect("output")
                .starts_with("FINAL: ")
        );
    }

    /// Printed output becomes the next turn's history rather than ending anything, and silence is
    /// reported as such.
    #[tokio::test]
    async fn printed_output_becomes_history() {
        let interpreter = Arc::new(ScriptedInterpreter::new([
            Ok(Executed::Printed(json!("1000 lines"))),
            Ok(Executed::Printed(Value::Null)),
            Ok(Executed::Submitted(json!({ "answer": "done" }))),
        ]));
        let look =
            Box::leak(action("look", "```python\nprint(len(context))\n```").into_boxed_str());
        let quiet = Box::leak(action("quiet", "```python\nx = 1\n```").into_boxed_str());
        let finish =
            Box::leak(action("finish", "```python\nSUBMIT(answer='done')\n```").into_boxed_str());
        let rlm = rlm(interpreter.clone(), &[look, quiet, finish]);

        let prediction = rlm
            .forward(example! { context: "doc" })
            .await
            .expect("answers");
        let trajectory = prediction.get("trajectory").expect("a trajectory");
        assert_eq!(trajectory[0]["output"], json!("1000 lines"));
        assert_eq!(
            trajectory[1]["output"],
            json!("(no output - did you forget to print?)")
        );
    }

    /// A failing run, and a fence the parser refuses, both reach the model as the turn's output
    /// rather than ending the episode.
    #[tokio::test]
    async fn failures_are_fed_back_as_observations() {
        let interpreter = Arc::new(ScriptedInterpreter::new([
            Err("NameError: name 'x' is not defined".to_owned()),
            Ok(Executed::Submitted(json!({ "answer": "ok" }))),
        ]));
        let broken = Box::leak(action("try", "```python\nprint(x)\n```").into_boxed_str());
        let finish =
            Box::leak(action("done", "```python\nSUBMIT(answer='ok')\n```").into_boxed_str());
        let rlm = rlm(interpreter, &[broken, finish]);

        let prediction = rlm
            .forward(example! { context: "doc" })
            .await
            .expect("answers");
        let trajectory = prediction.get("trajectory").expect("a trajectory");
        assert_eq!(
            trajectory[0]["output"],
            json!("[Error] NameError: name 'x' is not defined")
        );
    }

    /// A submission missing a field is refused with dspy's wording and fed back, so the model can
    /// submit again rather than the run ending wrong.
    #[tokio::test]
    async fn an_incomplete_submission_is_fed_back() {
        let mut signature = task();
        signature.outputs.push(OutField {
            name: "count".to_owned(),
            ..Default::default()
        });
        let interpreter = Arc::new(ScriptedInterpreter::new([
            Ok(Executed::Submitted(json!({ "answer": "42" }))),
            Ok(Executed::Submitted(json!({ "answer": "42", "count": "1" }))),
        ]));
        let first =
            Box::leak(action("partial", "```python\nSUBMIT(answer='42')\n```").into_boxed_str());
        let second = Box::leak(
            action("full", "```python\nSUBMIT(answer='42', count=1)\n```").into_boxed_str(),
        );
        let model = Arc::new(Scripted::new(&[first, second]));
        let mut rlm = Rlm::interpreter(signature, interpreter);
        rlm.generate_action = rlm.generate_action.with_lm(model.clone());
        rlm.extract = rlm.extract.with_lm(model);

        let prediction = rlm
            .forward(example! { context: "doc" })
            .await
            .expect("answers");
        let trajectory = prediction.get("trajectory").expect("a trajectory");
        assert_eq!(
            trajectory[0]["output"],
            json!("[Error] Missing output fields: [\"count\"]. Use SUBMIT(answer, count)")
        );
        assert_eq!(prediction.get("count"), Some(&json!("1")));
    }

    /// Out of iterations, the extract ask reads the outputs off the session instead.
    #[tokio::test]
    async fn it_extracts_when_the_iterations_run_out() {
        let interpreter = Arc::new(ScriptedInterpreter::new([Ok(Executed::Printed(json!(
            "still looking"
        )))]));
        let look = Box::leak(action("look", "```python\nprint(1)\n```").into_boxed_str());
        let extracted = "[[ ## answer ## ]]\nfrom the trajectory\n\n[[ ## completed ## ]]";
        let rlm = rlm(interpreter, &[look, extracted]).max_iterations(1);

        let prediction = rlm
            .forward(example! { context: "doc" })
            .await
            .expect("answers");
        assert_eq!(
            prediction.get("answer"),
            Some(&json!("from the trajectory"))
        );
        assert_eq!(
            prediction.get("final_reasoning"),
            Some(&json!("Extract forced final output"))
        );
    }

    /// The caller's inputs are bound in the sandbox on every turn, which is what lets the model's
    /// code reach them by name rather than by having them pasted into the prompt.
    #[tokio::test]
    async fn the_inputs_are_bound_in_the_sandbox_each_turn() {
        let interpreter = Arc::new(ScriptedInterpreter::new([
            Err("NameError".to_owned()),
            Ok(Executed::Submitted(json!({ "answer": "ok" }))),
        ]));
        let look = Box::leak(action("look", "```python\nprint(context)\n```").into_boxed_str());
        let finish =
            Box::leak(action("done", "```python\nSUBMIT(answer='ok')\n```").into_boxed_str());
        let rlm = rlm(interpreter.clone(), &[look, finish]);

        rlm.forward(example! { context: "a long document" })
            .await
            .expect("answers");

        let bound = interpreter.bound.lock().expect("bound").clone();
        assert_eq!(bound.len(), 2, "both turns bind");
        for turn in &bound {
            assert_eq!(turn["context"], json!("a long document"));
        }
    }

    /// A submission whose value is not its field's type is refused with dspy's wording and fed
    /// back, so the model corrects it rather than the run ending on a wrong answer.
    #[tokio::test]
    async fn a_wrongly_typed_submission_is_fed_back() {
        let mut signature = task();
        signature.outputs[0].values =
            Some(vec![crate::signature::LiteralValue::Str("yes".to_owned())]);
        let interpreter = Arc::new(ScriptedInterpreter::new([
            Ok(Executed::Submitted(json!({ "answer": "maybe" }))),
            Ok(Executed::Submitted(json!({ "answer": "yes" }))),
        ]));
        let wrong =
            Box::leak(action("guess", "```python\nSUBMIT(answer='maybe')\n```").into_boxed_str());
        let right =
            Box::leak(action("fix", "```python\nSUBMIT(answer='yes')\n```").into_boxed_str());
        let model = Arc::new(Scripted::new(&[wrong, right]));
        let mut rlm = Rlm::interpreter(signature, interpreter);
        rlm.generate_action = rlm.generate_action.with_lm(model.clone());
        rlm.extract = rlm.extract.with_lm(model);

        let prediction = rlm
            .forward(example! { context: "doc" })
            .await
            .expect("answers");
        let trajectory = prediction.get("trajectory").expect("a trajectory");
        assert_eq!(
            trajectory[0]["output"],
            json!("[Type Error] answer: expected Literal, got str: 'maybe' is not one of ('yes',)")
        );
        assert_eq!(prediction.get("answer"), Some(&json!("yes")));
    }

    /// An input the signature declares but the caller did not pass is refused, with dspy's wording.
    #[tokio::test]
    async fn it_refuses_missing_inputs() {
        let interpreter = Arc::new(ScriptedInterpreter::new([]));
        let rlm = rlm(interpreter, &[]);
        let error = rlm
            .forward(Example::new([("other", json!(1))]))
            .await
            .expect_err("refuses");
        assert!(
            error.to_string().contains("Missing required inputs"),
            "got: {error}"
        );
    }
}

#[cfg(test)]
mod sandbox_tests {
    use super::*;
    use crate::interpreter::tests::Scripted as ScriptedInterpreter;
    use crate::predict::scripted::Scripted;

    /// A value the sandbox rebuilds rather than one the prompt carries.
    struct Corpus(usize);

    impl SandboxSerializable for Corpus {
        fn sandbox_setup(&self) -> String {
            "import json".to_owned()
        }

        fn to_sandbox(&self) -> Vec<u8> {
            format!("{{\"documents\": {}}}", self.0).into_bytes()
        }

        fn sandbox_assignment(&self, var_name: &str, data_expr: &str) -> String {
            format!("{var_name} = json.loads({data_expr})")
        }

        fn rlm_preview(&self, _max_chars: usize) -> String {
            format!("Corpus of {} documents", self.0)
        }

        fn type_name(&self) -> &str {
            "Corpus"
        }
    }

    fn action(reasoning: &str, code: &str) -> String {
        format!(
            "[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## code ## ]]\n{code}\n\n[[ ## completed ## ]]"
        )
    }

    /// The value is rebuilt in the sandbox before the first turn, described rather than previewed,
    /// and left out of the per-turn bindings — it is already there under its own name.
    #[tokio::test]
    async fn a_sandbox_input_is_injected_once_and_described_not_previewed() {
        let interpreter = Arc::new(ScriptedInterpreter::new([
            Ok(Executed::Printed(json!("ok"))),
            Ok(Executed::Submitted(json!({ "answer": "done" }))),
        ]));
        let submit =
            Box::leak(action("finish", "```python\nSUBMIT(answer='done')\n```").into_boxed_str());
        let model = Arc::new(Scripted::new(&[submit]));
        let rlm = Rlm::interpreter(
            "corpus -> answer".parse().expect("parses"),
            interpreter.clone(),
        )
        .with_sandbox_input("corpus", Arc::new(Corpus(12)))
        .with_lm(model);

        let prediction = rlm.forward(Example::default()).await.expect("answers");
        assert_eq!(prediction.get("answer"), Some(&json!("done")));

        let ran = interpreter.ran.lock().expect("ran").clone();
        assert_eq!(
            ran[0], "import json\ncorpus = json.loads(_raw_corpus)",
            "the value is rebuilt before the first turn"
        );
        let bound = interpreter.bound.lock().expect("bound").clone();
        assert_eq!(bound[0]["_raw_corpus"], json!("{\"documents\": 12}"));
        assert!(
            bound[1].is_empty(),
            "a sandbox-held input is not rebound per turn: {:?}",
            bound[1]
        );
    }

    /// The model is told what the value *is*, under the heading dspy uses, rather than shown a
    /// slice of it.
    #[test]
    fn the_model_is_told_what_the_sandbox_holds() {
        let interpreter = Arc::new(ScriptedInterpreter::new([]));
        let mut signature: Signature = "corpus -> answer".parse().expect("parses");
        signature.inputs[0].desc = "everything we have".to_owned();
        let rlm = Rlm::interpreter(signature, interpreter)
            .with_sandbox_input("corpus", Arc::new(Corpus(12)));

        let described = rlm.variables(&Example::default());
        assert_eq!(described.len(), 1);
        assert!(
            described[0].contains("Type: Corpus"),
            "got: {}",
            described[0]
        );
        assert!(
            described[0].contains(
                "Description: everything we have\nSandbox imports available:\nimport json"
            ),
            "got: {}",
            described[0]
        );
        assert!(described[0].contains("Preview:\n```\nCorpus of 12 documents\n```"));
    }
}
