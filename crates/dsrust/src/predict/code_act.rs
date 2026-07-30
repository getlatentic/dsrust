//! dspy `predict/code_act.py`: an agent whose actions are Python, not tool calls.
//!
//! Where [`ReAct`](crate::react::ReAct) picks one tool per turn, this writes a snippet per turn and
//! runs it, so a turn can call several tools, loop, or compute. Each turn appends what it wrote and
//! what came back to a trajectory the next turn reads, until the model marks itself `finished` or
//! the iterations run out; a final ask reads the task's outputs off the trajectory.
//!
//! What runs the code is the caller's [`CodeInterpreter`], and the tools reach it through
//! [`define_tools`](CodeInterpreter::define_tools) — see that method for why Rust takes upstream's
//! host-callback route rather than its source-injection one.

use std::sync::Arc;

use anyhow::Result;
use serde_json::{Map, Value, json};

use crate::adapter::types::tool::format_tool;
use crate::example::{Example, Prediction};
use crate::interpreter::{CodeInterpreter, DenoInterpreter};
use crate::module::{Module, NamedPredictor, TraceStep, relabel};
use crate::react::Tool;
use crate::signature::{FieldKind, InField, OutField, Signature};

use super::chain_of_thought::ChainOfThought;
use super::{Dynamic, Predict};

/// dspy's `CodeAct`: write code, run it, and keep going until the task is answered.
pub struct CodeAct {
    /// The task's real signature: what the caller asked for.
    pub signature: Signature,
    /// How many snippets the model may write before the final ask happens anyway.
    pub max_iters: usize,
    tools: Vec<Arc<dyn Tool>>,
    codeact: Predict<Dynamic>,
    extractor: ChainOfThought,
    interpreter: Arc<dyn CodeInterpreter>,
}

impl CodeAct {
    /// dspy's `interpreter=None`: the Deno/Pyodide sandbox, which is what upstream defaults to.
    pub fn new(signature: Signature, tools: Vec<Arc<dyn Tool>>) -> Self {
        Self::with_interpreter(signature, tools, Arc::new(DenoInterpreter::new()))
    }

    /// The same, running code somewhere the caller chose.
    pub fn with_interpreter(
        signature: Signature,
        tools: Vec<Arc<dyn Tool>>,
        interpreter: Arc<dyn CodeInterpreter>,
    ) -> Self {
        Self {
            codeact: Predict::from_signature(codeact_signature(&signature, &tools)),
            extractor: ChainOfThought::from_signature(extract_signature(&signature)),
            signature,
            max_iters: 5,
            tools,
            interpreter,
        }
    }

    pub fn with_max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Ask both steps — the per-turn code and the final extraction — of this model.
    pub fn with_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.codeact = self.codeact.with_lm(lm.clone());
        self.extractor = self.extractor.with_lm(lm);
        self
    }

    /// The signature each turn is asked with — the task's inputs, the trajectory, and the code and
    /// `finished` flag the model answers with. dspy reaches the same thing as `codeact.signature`.
    pub fn turn_signature(&self) -> &Signature {
        &self.codeact.signature
    }

    /// The tools the model is told it may call, in the order they are numbered.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|tool| tool.name()).collect()
    }

    async fn run(&self, inputs: Example, trace: &mut Vec<TraceStep>) -> Result<Prediction> {
        // dspy makes the tools available in the sandbox before the first turn.
        self.interpreter.define_tools(&self.tools)?;

        let mut trajectory = Map::new();
        for turn in 0..self.max_iters {
            let mut asked = inputs.clone();
            asked.set("trajectory", Value::Object(trajectory.clone()));
            let mark = trace.len();
            let written = self.codeact.forward_traced(asked, trace).await?;
            relabel(trace, mark, "codeact");

            let (code, error) = super::program_of_thought::parse_generated_code(&written.example);
            if let Some(error) = error {
                trajectory.insert(
                    format!("observation_{turn}"),
                    json!(format!("Failed to parse the generated code: {error}")),
                );
                continue;
            }

            trajectory.insert(format!("generated_code_{turn}"), json!(code));
            match self.interpreter.execute(&code, &Map::new()) {
                Ok(executed) => {
                    let output = crate::adapter::python_json::json_dumps(executed.value());
                    trajectory.insert(format!("code_output_{turn}"), json!(output));
                }
                Err(error) => {
                    trajectory.insert(
                        format!("observation_{turn}"),
                        json!(format!("Failed to execute the generated code: {error}")),
                    );
                }
            }

            if written
                .example
                .get("finished")
                .and_then(Value::as_bool)
                .unwrap_or_default()
            {
                break;
            }
        }

        let mut asked = inputs;
        asked.set("trajectory", Value::Object(trajectory.clone()));
        let mark = trace.len();
        let extracted = self.extractor.forward_traced(asked, trace).await;
        relabel(trace, mark, "extractor");
        self.interpreter.shutdown();
        let extracted = extracted?;

        // dspy returns `Prediction(trajectory=trajectory, **extract)`: what the agent did travels
        // back beside what it concluded.
        let mut example = Example::new([("trajectory", Value::Object(trajectory))]);
        for (name, value) in extracted.example.fields() {
            example.set(name, value.clone());
        }
        Ok(Prediction::new(example, extracted.raw))
    }
}

impl Module for CodeAct {
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
        Box::pin(async move { self.run(inputs, trace).await })
    }

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        let mut predictors = Vec::new();
        for mut predictor in self.codeact.named_predictors() {
            predictor.name = format!("codeact.{}", predictor.name);
            predictors.push(predictor);
        }
        for mut predictor in self.extractor.named_predictors() {
            predictor.name = format!("extractor.{}", predictor.name);
            predictors.push(predictor);
        }
        predictors
    }
}

/// The signature one turn is asked with: the task's inputs and the trajectory, answering with the
/// next snippet and whether the agent is done.
fn codeact_signature(signature: &Signature, tools: &[Arc<dyn Tool>]) -> Signature {
    let mut inputs = signature.inputs.clone();
    inputs.push(InField {
        name: "trajectory".to_owned(),
        ..Default::default()
    });
    Signature {
        instructions: instructions(signature, tools),
        inputs,
        outputs: vec![
            OutField {
                name: "generated_code".to_owned(),
                desc: "Python code that when executed, produces output relevant to answering the \
                       question"
                    .to_owned(),
                ..Default::default()
            },
            OutField {
                name: "finished".to_owned(),
                desc: "a boolean flag to determine if the process is done".to_owned(),
                kind: FieldKind::Bool,
                ..Default::default()
            },
        ],
    }
}

/// The final ask: the task's own signature, plus the trajectory to read the answer out of.
fn extract_signature(signature: &Signature) -> Signature {
    let mut inputs = signature.inputs.clone();
    inputs.push(InField {
        name: "trajectory".to_owned(),
        ..Default::default()
    });
    Signature {
        instructions: signature.instructions.clone(),
        inputs,
        outputs: signature.outputs.clone(),
    }
}

/// dspy `_build_instructions`: what the agent is told, and the tools it may call.
fn instructions(signature: &Signature, tools: &[Arc<dyn Tool>]) -> String {
    let mut lines = Vec::new();
    // dspy leads with the task's own instructions and a blank line, where it has any.
    if !signature.instructions.is_empty() {
        lines.push(format!("{}\n", signature.instructions));
    }
    let inputs = backticked(signature.inputs.iter().map(|field| field.name.as_str()));
    let outputs = backticked(signature.outputs.iter().map(|field| field.name.as_str()));
    lines.push(format!(
        "You are an intelligent agent. For each episode, you will receive the fields {inputs} as \
         input.\n\
         Your goal is to generate executable Python code that collects any necessary information \
         for producing {outputs}.\n\
         For each iteration, you will generate a code snippet that either solves the task or \
         progresses towards the solution.\n\
         Ensure any output you wish to extract from the code is printed to the console. The code \
         should be enclosed in a fenced code block.\n\
         When all information for producing the outputs ({outputs}) are available to be extracted, \
         mark `finished=True` besides the final Python code.\n\
         You have access to the Python Standard Library and the following functions:"
    ));
    for (index, tool) in tools.iter().enumerate() {
        lines.push(format!(
            "({}) {}",
            index + 1,
            format_tool(tool.name(), tool.description(), tool.args())
        ));
    }
    lines.join("\n")
}

fn backticked<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `CodeAct!("question -> answer", tools)` — code that calls the tools you supply, run in the sandbox.
///
/// Takes a string signature or a task declared with `#[derive(Signature)]`, as every other module
/// macro does; the declared form carries its doc comment as the signature's instructions.
/// `max_iters = N` caps the loop.
#[macro_export]
macro_rules! CodeAct {
    ($signature:literal, $tools:expr $(,)?) => {
        $crate::CodeAct::new($crate::make_signature!($signature), $tools)
    };
    ($signature:literal, $tools:expr, max_iters = $max:expr $(,)?) => {
        $crate::CodeAct::new($crate::make_signature!($signature), $tools).with_max_iters($max)
    };
    ($task:ty, $tools:expr $(,)?) => {
        $crate::CodeAct::new(
            <$task as $crate::signature::SignatureSpec>::signature(),
            $tools,
        )
    };
    ($task:ty, $tools:expr, max_iters = $max:expr $(,)?) => {
        $crate::CodeAct::new(
            <$task as $crate::signature::SignatureSpec>::signature(),
            $tools,
        )
        .with_max_iters($max)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use crate::interpreter::Executed;
    use crate::interpreter::tests::Scripted as ScriptedInterpreter;
    use crate::react::{FnTool, tool_args};

    fn task() -> Signature {
        "question -> answer".parse().expect("parses")
    }

    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Args {
        /// the number to factor
        n: u32,
    }

    fn tools() -> Vec<Arc<dyn Tool>> {
        vec![Arc::new(FnTool::new(
            "factorial",
            "Compute the factorial.",
            tool_args::<Args>(),
            |_args| Ok("120".to_owned()),
        ))]
    }

    /// The turn signature carries the trajectory and answers with the code and the flag.
    #[test]
    fn the_turn_signature_asks_for_code_and_a_finished_flag() {
        let signature = codeact_signature(&task(), &tools());
        let inputs: Vec<&str> = signature.inputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(inputs, ["question", "trajectory"]);
        let outputs: Vec<&str> = signature.outputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(outputs, ["generated_code", "finished"]);
        assert_eq!(signature.outputs[1].kind, FieldKind::Bool);
    }

    /// The final ask is the task's own signature plus the trajectory.
    #[test]
    fn the_extract_signature_is_the_task_plus_the_trajectory() {
        let signature = extract_signature(&task());
        let inputs: Vec<&str> = signature.inputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(inputs, ["question", "trajectory"]);
        let outputs: Vec<&str> = signature.outputs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(outputs, ["answer"]);
    }

    /// A turn's code and its output land in the trajectory under the turn's number, and a
    /// `finished` flag ends the loop.
    #[tokio::test]
    async fn it_records_each_turn_and_stops_when_finished() {
        let interpreter = Arc::new(ScriptedInterpreter::new([Ok(Executed::Printed(json!(
            "120"
        )))]));
        let model = super::super::scripted::Scripted::new(&[
            "[[ ## generated_code ## ]]\nprint(factorial(5))\n\n[[ ## finished ## ]]\ntrue\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nread it\n\n[[ ## answer ## ]]\n120\n\n[[ ## completed ## ]]",
        ]);
        let model = Arc::new(model);
        let mut act = CodeAct::with_interpreter(task(), tools(), interpreter.clone());
        act.codeact = act.codeact.with_lm(model.clone());
        act.extractor = act.extractor.with_lm(model);

        let prediction = act
            .forward(example! { question: "5!" })
            .await
            .expect("answers");
        assert_eq!(prediction.get("answer"), Some(&json!("120")));
        let trajectory = prediction.get("trajectory").expect("a trajectory");
        assert_eq!(trajectory["generated_code_0"], json!("print(factorial(5))"));
        assert_eq!(trajectory["code_output_0"], json!("\"120\""));
        // One turn only: the flag stopped the loop before the second.
        assert_eq!(interpreter.ran.lock().expect("ran").len(), 1);
        assert_eq!(*interpreter.shutdowns.lock().expect("shutdowns"), 1);
    }

    /// Code that will not run is reported into the trajectory and the loop carries on, which is
    /// what lets the next turn see the failure and try something else.
    #[tokio::test]
    async fn a_failed_run_becomes_an_observation_and_the_loop_continues() {
        let interpreter = Arc::new(ScriptedInterpreter::new([
            Err("NameError: name 'x' is not defined".to_owned()),
            Ok(Executed::Printed(json!("120"))),
        ]));
        let reply = "[[ ## generated_code ## ]]\nprint(x)\n\n[[ ## finished ## ]]\nfalse\n\n[[ ## completed ## ]]";
        let done = "[[ ## generated_code ## ]]\nprint(factorial(5))\n\n[[ ## finished ## ]]\ntrue\n\n[[ ## completed ## ]]";
        let model = Arc::new(super::super::scripted::Scripted::new(&[
            reply,
            done,
            "[[ ## reasoning ## ]]\nr\n\n[[ ## answer ## ]]\n120\n\n[[ ## completed ## ]]",
        ]));
        let mut act = CodeAct::with_interpreter(task(), tools(), interpreter.clone());
        act.codeact = act.codeact.with_lm(model.clone());
        act.extractor = act.extractor.with_lm(model);

        let prediction = act
            .forward(example! { question: "5!" })
            .await
            .expect("answers");
        let trajectory = prediction.get("trajectory").expect("a trajectory");
        assert_eq!(
            trajectory["observation_0"],
            json!("Failed to execute the generated code: NameError: name 'x' is not defined")
        );
        assert_eq!(trajectory["code_output_1"], json!("\"120\""));
    }
}

/// CodeAct's two signatures against dspy's own.
///
/// The turn signature's instructions embed the tool catalogue, which is where a port drifts: the
/// numbering, each tool's string form with its newlines flattened, and the blank line after the
/// task's own instructions — which lead the prompt even when they are the default dspy writes for
/// a signature nobody described. The golden (`tests/conformance/predict/code_act.json`, see
/// `scripts/generate_codeact_fixture.py`) is what upstream built.
#[cfg(test)]
mod conformance {
    use super::*;
    use crate::react::FnTool;
    use serde_json::Value;

    fn golden() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/predict/code_act.json");
        let text = std::fs::read_to_string(&path).expect("the golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    /// The fixture's two tools, mirrored — including the two-line docstring, whose newlines
    /// upstream flattens into the catalogue line.
    ///
    /// The argument maps are written out rather than derived from a Rust type: `schemars` spells
    /// an integer `{"type": "integer", "format": "int64"}` where pydantic spells it
    /// `{"type": "integer"}`, and that difference belongs to schema generation, which has its own
    /// tests. What is under test here is the catalogue line CodeAct builds around them.
    fn tool(name: &str) -> Arc<dyn Tool> {
        match name {
            "factorial" => Arc::new(FnTool::new(
                "factorial",
                "Compute the factorial of n.",
                json!({ "n": { "type": "integer" } }),
                |_| Ok(String::new()),
            )),
            _ => Arc::new(FnTool::new(
                "lookup",
                // The trailing newline is what Python's docstring carries, and upstream flattens
                // it into the two trailing spaces the golden records.
                "Look a thing up.\n\nSpans two lines, so the newline flattening in a tool's string \
                 form is exercised.\n",
                json!({ "name": { "type": "string" }, "year": { "type": "integer" } }),
                |_| Ok(String::new()),
            )),
        }
    }

    fn described(fields: &Value) -> Vec<(String, String)> {
        fields
            .as_array()
            .expect("fields")
            .iter()
            .map(|field| {
                (
                    field["name"].as_str().expect("a name").to_owned(),
                    field["desc"].as_str().expect("a desc").to_owned(),
                )
            })
            .collect()
    }

    fn ours(signature: &Signature) -> (Vec<(String, String)>, Vec<(String, String)>) {
        (
            signature
                .inputs
                .iter()
                .map(|f| (f.name.clone(), f.desc.clone()))
                .collect(),
            signature
                .outputs
                .iter()
                .map(|f| (f.name.clone(), f.desc.clone()))
                .collect(),
        )
    }

    #[test]
    fn it_builds_the_signatures_dspy_builds() {
        for case in golden()["cases"].as_array().expect("cases") {
            let label = case["label"].as_str().expect("a label");
            // The task signature: parsed from its spelling, with the instructions dspy recorded
            // (a described signature's docstring is not in the spelling).
            let mut task: Signature = match case["task"].as_str().expect("a task") {
                "Described" => "question -> answer".parse().expect("parses"),
                spelling => spelling.parse().expect("parses"),
            };
            task.instructions = case["task_instructions"]
                .as_str()
                .expect("instructions")
                .to_owned();
            let tools: Vec<Arc<dyn Tool>> = case["tools"]
                .as_array()
                .expect("tools")
                .iter()
                .map(|listed| {
                    let listed = listed.as_str().expect("a tool");
                    tool(listed.split([',', '.']).next().unwrap_or_default())
                })
                .collect();

            // Every tool's catalogue line, which is what the instructions embed.
            for (ours, theirs) in tools.iter().zip(case["tools"].as_array().expect("tools")) {
                assert_eq!(
                    format_tool(ours.name(), ours.description(), ours.args()),
                    theirs.as_str().expect("a tool"),
                    "tool line for {label}"
                );
            }

            let codeact = codeact_signature(&task, &tools);
            assert_eq!(
                codeact.instructions,
                case["codeact"]["instructions"]
                    .as_str()
                    .expect("instructions"),
                "codeact instructions for {label}"
            );
            let (inputs, outputs) = ours(&codeact);
            assert_eq!(
                inputs,
                described(&case["codeact"]["inputs"]),
                "codeact inputs for {label}"
            );
            assert_eq!(
                outputs,
                described(&case["codeact"]["outputs"]),
                "codeact outputs for {label}"
            );

            // The extract signature reaches the model through ChainOfThought, which prepends the
            // reasoning field — so it is that module's signature the golden recorded.
            let extract = ChainOfThought::from_signature(extract_signature(&task));
            assert_eq!(
                extract.signature().instructions,
                case["extract"]["instructions"]
                    .as_str()
                    .expect("instructions"),
                "extract instructions for {label}"
            );
            let (inputs, outputs) = ours(extract.signature());
            assert_eq!(
                inputs,
                described(&case["extract"]["inputs"]),
                "extract inputs for {label}"
            );
            assert_eq!(
                outputs,
                described(&case["extract"]["outputs"]),
                "extract outputs for {label}"
            );
        }
    }
}
