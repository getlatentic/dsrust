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
use crate::interpreter::{CodeInterpreter, DenoInterpreter};
use crate::module::{Module, NamedPredictor, TraceStep, relabel};
use crate::signature::{FieldKind, InField, OutField, Signature};

use super::chain_of_thought::ChainOfThought;

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
    interpreter: Arc<dyn CodeInterpreter>,
}

impl ProgramOfThought {
    /// dspy's `interpreter=None`: the Deno/Pyodide sandbox, which is what upstream defaults to.
    ///
    /// Ask for another with [`Self::interpreter`] — a caller who wants their own environment,
    /// or a test that scripts one.
    pub fn new(signature: Signature) -> Self {
        Self::interpreter(signature, Arc::new(DenoInterpreter::new()))
    }

    /// The same, running code somewhere the caller chose.
    pub fn interpreter(signature: Signature, interpreter: Arc<dyn CodeInterpreter>) -> Self {
        Self {
            generate: ChainOfThought::from_signature(mode_signature(&signature, Mode::Generate)),
            regenerate: ChainOfThought::from_signature(mode_signature(
                &signature,
                Mode::Regenerate,
            )),
            answer: ChainOfThought::from_signature(mode_signature(&signature, Mode::Answer)),
            signature,
            max_iters: 3,
            interpreter,
        }
    }

    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// Ask all three steps — write, rewrite and answer — of this model.
    pub fn with_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.generate = self.generate.with_lm(lm.clone());
        self.regenerate = self.regenerate.with_lm(lm.clone());
        self.answer = self.answer.with_lm(lm);
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
    fn execute(&self, code: &str) -> (Option<String>, Option<String>) {
        if code.is_empty() {
            return (None, Some("Error: Empty code before execution.".to_owned()));
        }
        match self.interpreter.execute(code, &Map::new()) {
            Ok(executed) => (
                Some(crate::adapter::python_json::json_dumps(executed.value())),
                None,
            ),
            Err(error) => (None, Some(format!("{error}"))),
        }
    }

    async fn run(&self, inputs: Example, trace: &mut Vec<TraceStep>) -> Result<Prediction> {
        // dspy passes only the task's own inputs on, so a stray field cannot reach the ask.
        let mut asked = self.task_inputs(&inputs);

        let mark = trace.len();
        let written = self.generate.forward_traced(asked.clone(), trace).await?;
        relabel(trace, mark, "code_generate");
        let (mut code, mut error) = parse_generated_code(&written.example);
        let mut output = None;
        if error.is_none() {
            (output, error) = self.execute(&code);
        }

        let mut hop = 1;
        while let Some(reported) = error.clone() {
            tracing::error!(error = %reported, "error in code execution");
            if hop == self.max_iters {
                self.interpreter.shutdown();
                bail!("Max hops reached. Failed to run ProgramOfThought: {reported}");
            }
            asked.set("previous_code", Value::String(code.clone()));
            asked.set("error", Value::String(reported));
            let mark = trace.len();
            let written = self.regenerate.forward_traced(asked.clone(), trace).await?;
            relabel(trace, mark, "code_regenerate");
            (code, error) = parse_generated_code(&written.example);
            if error.is_none() {
                (output, error) = self.execute(&code);
            }
            hop += 1;
        }

        asked.set("final_generated_code", Value::String(code));
        asked.set("code_output", Value::String(output.unwrap_or_default()));
        let mark = trace.len();
        let answered = self.answer.forward_traced(asked, trace).await;
        relabel(trace, mark, "generate_output");
        // dspy shuts the interpreter down on the way out, including the way out that raises.
        self.interpreter.shutdown();
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
            let span = crate::observe::module_shown("ProgramOfThought", &inputs);
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

/// dspy `_generate_signature`: the task's inputs, plus the fields this ask adds.
fn mode_signature(signature: &Signature, mode: Mode) -> Signature {
    let mut inputs = signature.inputs.clone();
    let outputs = match mode {
        Mode::Generate => vec![generated_code()],
        Mode::Regenerate => {
            inputs.push(input_field(
                "previous_code",
                "previously-generated python code that errored",
            ));
            inputs.push(input_field(
                "error",
                "error message from previously-generated python code",
            ));
            vec![generated_code()]
        }
        Mode::Answer => {
            inputs.push(input_field(
                "final_generated_code",
                "python code that answers the question",
            ));
            inputs.push(input_field(
                "code_output",
                "output of previously-generated python code",
            ));
            signature.outputs.clone()
        }
    };
    Signature {
        instructions: instructions(signature, mode, &inputs, &outputs),
        inputs,
        outputs,
    }
}

fn input_field(name: &str, desc: &str) -> InField {
    InField {
        name: name.to_owned(),
        desc: desc.to_owned(),
        ..Default::default()
    }
}

fn generated_code() -> OutField {
    OutField {
        name: "generated_code".to_owned(),
        desc: "python code that answers the question".to_owned(),
        kind: FieldKind::Str,
        ..Default::default()
    }
}

/// dspy `_generate_instruction`: what each of the three asks is told to do.
fn instructions(
    signature: &Signature,
    mode: Mode,
    inputs: &[InField],
    outputs: &[OutField],
) -> String {
    let mode_inputs = backticked(inputs.iter().map(|field| field.name.as_str()));
    let mode_outputs = backticked(outputs.iter().map(|field| field.name.as_str()));
    let lines = match mode {
        Mode::Generate => {
            let final_outputs =
                backticked(signature.outputs.iter().map(|field| field.name.as_str()));
            vec![
                format!(
                    "You will be given {mode_inputs} and you will respond with {mode_outputs}."
                ),
                format!(
                    "Generating executable Python code that programmatically computes the correct \
                     {mode_outputs}."
                ),
                "After you're done with the computation and think you have the final output, make \
                 sure to submit your output by calling the preloaded function `SUBMIT()`."
                    .to_owned(),
                format!(
                    "You must structure your output in a dict, like {{\"field_a\": value_a, ...}}, \
                     with the correct value mapping for the field(s): {final_outputs}."
                ),
            ]
        }
        Mode::Regenerate => vec![
            format!("You are given {mode_inputs} due to an error in previous code."),
            "Your task is to correct the error and provide the new `generated_code`.".to_owned(),
        ],
        Mode::Answer => {
            vec![format!(
                "Given the final code {mode_inputs}, provide the final {mode_outputs}."
            )]
        }
    };
    lines.join("\n")
}

fn backticked<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// dspy `_parse_code`: the runnable code out of the field the model wrote, or why it is not
/// runnable.
///
/// Upstream cuts the field at the first `---` or blank-blank-line, prefers a fenced ```python
/// block if there is one, and — where the last line assigns a name — appends that name so the
/// value becomes the program's result.
pub(super) fn parse_generated_code(written: &Example) -> (String, Option<String>) {
    let code = written
        .get("generated_code")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let code = code.split("---").next().unwrap_or_default();
    let code = code.split("\n\n\n").next().unwrap_or_default();
    let block = fenced_python(code).unwrap_or(code);
    if block.is_empty() {
        return (
            code.to_owned(),
            Some("Error: Empty code after parsing.".to_owned()),
        );
    }
    if !block.contains('\n') && block.matches('=').count() > 1 {
        return (
            code.to_owned(),
            Some("Error: Code format is not correct.".to_owned()),
        );
    }
    let lines: Vec<&str> = block.split('\n').collect();
    let mut block = block.to_owned();
    if let Some(assigned) = assigned_name(lines.last().unwrap_or(&"").trim())
        && lines.len() > 1
    {
        block.push('\n');
        block.push_str(assigned);
    }
    (block, None)
}

/// The body of the first ```` ```python ```` block, matching upstream's
/// `` ```python[ \n](.*?)[ \n]```? `` — one space or newline after the opener, the shortest body,
/// and a closing fence of two backticks or three.
fn fenced_python(code: &str) -> Option<&str> {
    let opened = code.find("```python")? + "```python".len();
    let rest = &code[opened..];
    if !rest.starts_with([' ', '\n']) {
        return None;
    }
    let body = &rest[1..];
    // The shortest body: the earliest separator that is followed by a closing fence.
    let mut at = 0;
    while at < body.len() {
        let next = body[at..].find([' ', '\n'])? + at;
        if body[next + 1..].starts_with("``") {
            return Some(&body[..next]);
        }
        at = next + 1;
    }
    None
}

/// The name a line assigns to, matching upstream's `^(\w+)\s*=`. A `==` is not an assignment, and
/// upstream's regex agrees: `\s*=` matches the first `=`, leaving the second unread.
fn assigned_name(line: &str) -> Option<&str> {
    let name_end = line.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    if name_end == 0 {
        return None;
    }
    let (name, rest) = line.split_at(name_end);
    rest.trim_start().starts_with('=').then_some(name)
}

/// `ProgramOfThought!("question -> answer")` — the model writes code, is shown what it produced, and states the answer.
///
/// Takes a string signature or a task declared with `#[derive(Signature)]`, as every other module
/// macro does; the declared form carries its doc comment as the signature's instructions.
/// `max_iters = N` caps the loop.
#[macro_export]
macro_rules! ProgramOfThought {
    ($signature:literal $(,)?) => {
        $crate::ProgramOfThought::new($crate::make_signature!($signature))
    };
    ($signature:literal, max_iters = $max:expr $(,)?) => {
        $crate::ProgramOfThought::new($crate::make_signature!($signature)).max_iters($max)
    };
    ($task:ty $(,)?) => {
        $crate::ProgramOfThought::new(<$task as $crate::signature::SignatureSpec>::signature())
    };
    ($task:ty, max_iters = $max:expr $(,)?) => {
        $crate::ProgramOfThought::new(<$task as $crate::signature::SignatureSpec>::signature())
            .max_iters($max)
    };
}

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
            ProgramOfThought::interpreter(task(), interpreter.clone()),
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
            ProgramOfThought::interpreter(task(), interpreter.clone()),
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
            ProgramOfThought::interpreter(task(), interpreter.clone()).max_iters(2),
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
        pot.generate = pot.generate.with_lm(model.clone());
        pot.regenerate = pot.regenerate.with_lm(model.clone());
        pot.answer = pot.answer.with_lm(model);
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
