//! dspy `predict/program_of_thought.py`: answer by writing a program and running it.
//!
//! Three asks, not one. The first writes code for the task; if the code will not parse or will not
//! run, the second is shown the failure and writes it again, up to `max_iters`; the third is shown
//! the working code and its output and states the task's real outputs. Each is a
//! [`ChainOfThought`] over a signature built from the caller's, which is most of what this module
//! does — the loop around them is short.
//!
//! What runs the code is the caller's [`CodeInterpreter`]; see that trait for why the crate ships
//! no sandbox of its own.

use std::sync::Arc;

use anyhow::{Result, bail};
use serde_json::{Map, Value};

use crate::example::{Example, Prediction};
use crate::interpreter::{CodeInterpreter, DenoInterpreter, InterpreterFactory, Lease};
use crate::module::{Module, NamedPredictor, TraceStep, relabel};
use crate::signature::Signature;

use super::chain_of_thought::ChainOfThought;
use signatures::mode_signature;

mod code;
mod signatures;

pub(super) use code::parse_generated_code;

/// Which of the three asks a signature is being built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Generate,
    Regenerate,
    Answer,
}

/// dspy's `ProgramOfThought`: write a program, run it, and read the answer out of what it printed.
pub struct ProgramOfThought {
    /// The task's real signature: what the caller asked for.
    pub signature: Signature,
    /// How many times the code may be written before the module gives up. dspy counts the first
    /// attempt, so `max_iters = 3` allows the first and two rewrites.
    pub max_iters: usize,
    generate: ChainOfThought,
    regenerate: ChainOfThought,
    answer: ChainOfThought,
    /// dspy's `interpreter_factory`: one sandbox is built per forward pass and shut down after.
    ///
    /// Held for the module's lifetime until dspy 3.3.0, which is a different program rather than a
    /// different spelling — a sandbox is a child process. It also could not survive a second call:
    /// `run` shut the held interpreter down on its way out, so a module asked twice ran the second
    /// pass against a dead one.
    interpreter_factory: InterpreterFactory,
}

impl ProgramOfThought {
    /// dspy's `interpreter=None`: the Deno/Pyodide sandbox, which is what upstream defaults to.
    ///
    /// Ask for another with [`Self::interpreter_factory`] — a caller who wants their own kind of
    /// environment, or a test that scripts one.
    /// dspy `ProgramOfThought("question -> answer")`: the task named by its fields.
    ///
    /// `ProgramOfThought!` checks a spelling written in the source while the caller compiles;
    /// this is for a signature that is only a string at run time.
    ///
    /// ```
    /// # fn wrapper() -> anyhow::Result<()> {
    /// let module = dsrust::ProgramOfThought::parse("question -> answer")?;
    /// # let _ = module;
    /// # Ok(())
    /// # }
    /// ```
    pub fn parse(signature: &str) -> anyhow::Result<Self> {
        Ok(Self::new(signature.parse()?))
    }

    pub fn new(signature: Signature) -> Self {
        Self::interpreter_factory(signature, crate::interpreter::factory(DenoInterpreter::new))
    }

    /// The same, building the caller's own kind of sandbox for each pass.
    ///
    /// dspy's `interpreter_factory`, and its docstring's warning applies here too: the callable
    /// "may be invoked concurrently", which is why the type is `Send + Sync`. A caller who wants
    /// *one* interpreter across several calls hands it to [`Self::ask_in`] instead — that one is
    /// theirs and is not shut down.
    pub fn interpreter_factory(
        signature: Signature,
        interpreter_factory: InterpreterFactory,
    ) -> Self {
        Self {
            generate: ChainOfThought::from_signature(mode_signature(&signature, Mode::Generate)),
            regenerate: ChainOfThought::from_signature(mode_signature(
                &signature,
                Mode::Regenerate,
            )),
            answer: ChainOfThought::from_signature(mode_signature(&signature, Mode::Answer)),
            signature,
            max_iters: 3,
            interpreter_factory,
        }
    }

    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Ask all three steps — write, rewrite and answer — of this model.
    pub fn set_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.generate = self.generate.set_lm(lm.clone());
        self.regenerate = self.regenerate.set_lm(lm.clone());
        self.answer = self.answer.set_lm(lm);
        self
    }

    /// Ask only the first write of this model — dspy's `self.code_generate` predictor.
    ///
    /// Per stage rather than [`Self::set_lm`]'s all-three, for the reason `Rlm` has `action_lm`
    /// beside `extract_lm`: upstream's tests stub the module's *predictors* one at a time, and a
    /// bridge honouring that needs each stage to be its own seam.
    pub fn generate_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.generate = self.generate.set_lm(lm);
        self
    }

    /// Ask only the rewrite step — dspy's `self.code_regenerate`. See [`Self::generate_lm`].
    pub fn regenerate_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.regenerate = self.regenerate.set_lm(lm);
        self
    }

    /// Ask only the final answer step — dspy's `self.generate_output`. See [`Self::generate_lm`].
    pub fn answer_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.answer = self.answer.set_lm(lm);
        self
    }

    /// The signature of one of the three asks, as dspy's `_generate_signature` builds it.
    pub fn mode_signature_for(&self, mode: &str) -> Option<Signature> {
        let mode = match mode {
            "generate" => Mode::Generate,
            "regenerate" => Mode::Regenerate,
            "answer" => Mode::Answer,
            _ => return None,
        };
        Some(mode_signature(&self.signature, mode))
    }

    /// dspy `_execute_code`: run it, and answer with the output as JSON text or with the error.
    ///
    /// The output reaches the third ask as a field the model reads, so it is `json.dumps` of the
    /// result exactly as upstream sends it — a submitted value unwrapped from its `FinalOutput`.
    /// Fallible, because not every failure is one the model can be asked to fix. dspy 3.3.0 splits
    /// the code's own failure — fed back as the error to correct, which is this loop — from the
    /// interpreter's, which is terminal and propagates. Returning only the feedback pair meant a
    /// dead sandbox was handed to the model as something to rewrite, and the run continued asking
    /// an interpreter that was gone.
    fn execute(
        interpreter: &Arc<dyn CodeInterpreter>,
        code: &str,
    ) -> Result<(Option<String>, Option<String>)> {
        if code.is_empty() {
            return Ok((None, Some("Error: Empty code before execution.".to_owned())));
        }
        Ok(match interpreter.execute(code, &Map::new()) {
            Ok(executed) => (
                Some(crate::adapter::python_json::json_dumps(executed.value())),
                None,
            ),
            // The interpreter's own failure ends the run; the code's is the next prompt.
            Err(error)
                if matches!(
                    error.downcast_ref::<crate::interpreter::InterpreterFailure>(),
                    Some(crate::interpreter::InterpreterFailure::Session(_))
                ) =>
            {
                return Err(error);
            }
            Err(error) => (None, Some(format!("{error}"))),
        })
    }

    async fn run(&self, inputs: Example, trace: &mut Vec<TraceStep>) -> Result<Prediction> {
        self.run_in(inputs, trace, None).await
    }

    /// The pass itself, in the caller's interpreter or in one built for it.
    ///
    /// The `Lease` decides which and shuts down only what it made — upstream's
    /// `_interpreter_context`. Because that shutdown is `Drop`, every way out of this function
    /// releases the process: the `?` on a model call, the max-hops `bail!`, and the ordinary
    /// Run one pass in an interpreter the caller owns — dspy's positional
    /// `forward(interpreter, /, **kwargs)`.
    ///
    /// Its own method because Rust has no positional-optional first argument, and because the
    /// ownership is the point: this interpreter is **not** shut down when the pass ends, so a
    /// caller can carry state across several calls or hand the same sandbox to several modules.
    /// Tools and output fields are still injected into it, as upstream injects them "even for
    /// user-provided interpreters" — each pass gets fresh ones with a fresh call counter.
    pub async fn ask_in(
        &self,
        interpreter: Arc<dyn CodeInterpreter>,
        inputs: Example,
    ) -> Result<Prediction> {
        let mut discarded = Vec::new();
        self.run_in(inputs, &mut discarded, Some(interpreter)).await
    }

    /// return. The explicit calls this replaced covered two of those three.
    async fn run_in(
        &self,
        inputs: Example,
        trace: &mut Vec<TraceStep>,
        caller: Option<Arc<dyn CodeInterpreter>>,
    ) -> Result<Prediction> {
        let lease = Lease::open(&self.interpreter_factory, caller)?;
        let interpreter = lease.get();
        // dspy passes only the task's own inputs on, so a stray field cannot reach the ask.
        let mut asked = self.task_inputs(&inputs);

        let mark = trace.len();
        let written = self.generate.forward_traced(asked.clone(), trace).await?;
        relabel(trace, mark, "code_generate");
        let (mut code, mut error) = parse_generated_code(&written.example);
        let mut output = None;
        if error.is_none() {
            (output, error) = Self::execute(interpreter, &code)?;
        }

        let mut hop = 1;
        while let Some(reported) = error.clone() {
            tracing::error!(error = %reported, "error in code execution");
            if hop == self.max_iters {
                bail!("Max hops reached. Failed to run ProgramOfThought: {reported}");
            }
            asked.set("previous_code", Value::String(code.clone()));
            asked.set("error", Value::String(reported));
            let mark = trace.len();
            let written = self.regenerate.forward_traced(asked.clone(), trace).await?;
            relabel(trace, mark, "code_regenerate");
            (code, error) = parse_generated_code(&written.example);
            if error.is_none() {
                (output, error) = Self::execute(interpreter, &code)?;
            }
            hop += 1;
        }

        asked.set("final_generated_code", Value::String(code));
        asked.set("code_output", Value::String(output.unwrap_or_default()));
        let mark = trace.len();
        let answered = self.answer.forward_traced(asked, trace).await;
        relabel(trace, mark, "generate_output");
        answered
    }

    /// The task's own input fields, dropping anything the caller passed beside them.
    fn task_inputs(&self, inputs: &Example) -> Example {
        Example::new(self.signature.inputs.iter().filter_map(|field| {
            inputs
                .get(&field.name)
                .map(|value| (field.name.clone(), value.clone()))
        }))
    }
}

impl Module for ProgramOfThought {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let span = crate::observe::module_shown("ProgramOfThought", &inputs, self.callbacks());
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
        for (name, module) in [
            ("code_generate", &mut self.generate),
            ("code_regenerate", &mut self.regenerate),
            ("generate_output", &mut self.answer),
        ] {
            for mut predictor in module.named_predictors() {
                predictor.name = format!("{name}.{}", predictor.name);
                predictors.push(predictor);
            }
        }
        predictors
    }
}

// `call!` on this module. See `Ask` for why the trait is written per type rather than blanket.
crate::asks_with_a_prediction!(ProgramOfThought);

#[cfg(test)]
mod tests {
    use super::super::scripted::Scripted;
    use super::*;
    use crate::example;
    use crate::interpreter::Executed;
    use crate::interpreter::tests::Scripted as ScriptedInterpreter;
    use serde_json::json;

    fn task() -> Signature {
        "question -> answer".parse().expect("parses")
    }

    /// The three asks carry the fields dspy composes, in upstream's order.
    #[test]
    fn each_mode_adds_the_fields_dspy_adds() {
        let names = |fields: &Signature, inputs: bool| -> Vec<String> {
            match inputs {
                true => fields.inputs.iter().map(|f| f.name.clone()).collect(),
                false => fields.outputs.iter().map(|f| f.name.clone()).collect(),
            }
        };
        let generate = mode_signature(&task(), Mode::Generate);
        assert_eq!(names(&generate, true), ["question"]);
        assert_eq!(names(&generate, false), ["generated_code"]);

        let regenerate = mode_signature(&task(), Mode::Regenerate);
        assert_eq!(
            names(&regenerate, true),
            ["question", "previous_code", "error"]
        );
        assert_eq!(names(&regenerate, false), ["generated_code"]);

        let answer = mode_signature(&task(), Mode::Answer);
        assert_eq!(
            names(&answer, true),
            ["question", "final_generated_code", "code_output"]
        );
        assert_eq!(names(&answer, false), ["answer"]);
    }

    /// The happy path: code is written, runs, and the third ask states the task's outputs. The
    /// interpreter is shut down on the way out.
    #[tokio::test]
    async fn it_writes_runs_and_answers() {
        let interpreter = Arc::new(ScriptedInterpreter::new([Ok(Executed::Submitted(
            json!({ "answer": "2" }),
        ))]));
        let model = Scripted::new(&[
            "[[ ## reasoning ## ]]\nadd\n\n[[ ## generated_code ## ]]\nSUBMIT({'answer': '2'})\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nread it\n\n[[ ## answer ## ]]\n2\n\n[[ ## completed ## ]]",
        ]);
        let pot = with_model(
            ProgramOfThought::interpreter_factory(
                task(),
                crate::interpreter::handing_back(interpreter.clone()),
            ),
            Arc::new(model),
        );

        let prediction = pot
            .forward(example! { question: "1+1?" })
            .await
            .expect("answers");
        assert_eq!(prediction.get("answer"), Some(&json!("2")));
        assert_eq!(interpreter.ran.lock().expect("ran").len(), 1);
        assert_eq!(*interpreter.shutdowns.lock().expect("shutdowns"), 1);
    }

    /// A failing run is fed back for a rewrite, and the rewrite's code is what runs next.
    #[tokio::test]
    async fn a_failed_run_is_rewritten_with_the_error_in_hand() {
        let interpreter = Arc::new(ScriptedInterpreter::new([
            Err("NameError: name 'x' is not defined".to_owned()),
            Ok(Executed::Printed(json!("2"))),
        ]));
        let model = Scripted::new(&[
            "[[ ## reasoning ## ]]\nfirst\n\n[[ ## generated_code ## ]]\nprint(x)\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nfix\n\n[[ ## generated_code ## ]]\nprint(2)\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nread\n\n[[ ## answer ## ]]\n2\n\n[[ ## completed ## ]]",
        ]);
        let pot = with_model(
            ProgramOfThought::interpreter_factory(
                task(),
                crate::interpreter::handing_back(interpreter.clone()),
            ),
            Arc::new(model),
        );

        let prediction = pot
            .forward(example! { question: "1+1?" })
            .await
            .expect("answers");
        assert_eq!(prediction.get("answer"), Some(&json!("2")));
        assert_eq!(
            *interpreter.ran.lock().expect("ran"),
            ["print(x)", "print(2)"]
        );
    }

    /// dspy gives up at `max_iters` and says so, shutting the interpreter down first.
    #[tokio::test]
    async fn it_gives_up_after_max_iters() {
        let interpreter = Arc::new(ScriptedInterpreter::new([
            Err("boom".to_owned()),
            Err("boom".to_owned()),
            Err("boom".to_owned()),
        ]));
        let reply = "[[ ## reasoning ## ]]\nr\n\n[[ ## generated_code ## ]]\nprint(x)\n\n[[ ## completed ## ]]";
        let model = Scripted::new(&[reply, reply, reply, reply]);
        let pot = with_model(
            ProgramOfThought::interpreter_factory(
                task(),
                crate::interpreter::handing_back(interpreter.clone()),
            )
            .max_iters(2),
            Arc::new(model),
        );

        let error = pot
            .forward(example! { question: "1+1?" })
            .await
            .expect_err("gives up");
        assert!(
            error.to_string().starts_with("Max hops reached."),
            "got: {error}"
        );
        assert_eq!(*interpreter.shutdowns.lock().expect("shutdowns"), 1);
    }

    /// Point all three asks at one model.
    fn with_model(
        mut pot: ProgramOfThought,
        model: Arc<dyn crate::lm::DynChatModel>,
    ) -> ProgramOfThought {
        pot.generate = pot.generate.set_lm(model.clone());
        pot.regenerate = pot.regenerate.set_lm(model.clone());
        pot.answer = pot.answer.set_lm(model);
        pot
    }
}

/// ProgramOfThought's derived signatures and code parsing, against dspy's own.
///
/// The three signatures decide three prompts, and `_parse_code` is a pair of regexes over whatever
/// the model wrote — both drift quietly when reimplemented. The golden
/// (`tests/conformance/predict/program_of_thought.json`, see `scripts/generate_pot_fixture.py`) is
/// what upstream produced, for inputs chosen where a hand-written matcher parts company with the
/// regexes: a fence with no language line, a two-backtick close, a `---` tail that keeps its
/// newline, a trailing assignment echoed only when there is more than one line, and a fence whose
/// body is empty and so is left unparsed.
#[cfg(test)]
mod conformance {
    use super::*;
    use crate::example;
    use serde_json::Value;

    fn golden() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/predict/program_of_thought.json");
        let text = std::fs::read_to_string(&path).expect("the golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
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

    /// Every field, its description and the instructions, for each of the three asks.
    #[test]
    fn it_derives_the_signatures_dspy_derives() {
        for case in golden()["signatures"].as_array().expect("cases") {
            let task: Signature = case["task"]
                .as_str()
                .expect("a task")
                .parse()
                .expect("parses");
            for (name, mode) in [
                ("generate", Mode::Generate),
                ("regenerate", Mode::Regenerate),
                ("answer", Mode::Answer),
            ] {
                let ours = mode_signature(&task, mode);
                let theirs = &case["modes"][name];
                let named = format!("{} / {name}", case["task"]);
                assert_eq!(
                    ours.instructions,
                    theirs["instructions"].as_str().expect("instructions"),
                    "instructions for {named}"
                );
                let inputs: Vec<(String, String)> = ours
                    .inputs
                    .iter()
                    .map(|f| (f.name.clone(), f.desc.clone()))
                    .collect();
                assert_eq!(inputs, described(&theirs["inputs"]), "inputs for {named}");
                let outputs: Vec<(String, String)> = ours
                    .outputs
                    .iter()
                    .map(|f| (f.name.clone(), f.desc.clone()))
                    .collect();
                assert_eq!(
                    outputs,
                    described(&theirs["outputs"]),
                    "outputs for {named}"
                );
            }
        }
    }

    /// Every code the fixture wrote, parsed to the same text and the same error.
    #[test]
    fn it_parses_the_code_dspy_parses() {
        for case in golden()["parse_code"].as_array().expect("cases") {
            let written = case["written"].as_str().expect("written");
            let (code, error) = parse_generated_code(&example! { generated_code: written });
            assert_eq!(
                code,
                case["code"].as_str().expect("code"),
                "code for {written:?}"
            );
            assert_eq!(
                error.as_deref(),
                case["error"].as_str(),
                "error for {written:?}"
            );
        }
    }
}
