//! dspy `predict/rlm.py`: the Recursive Language Model.
//!
//! An inference strategy for context too long to hand a model directly: the input stays in a REPL
//! as a variable, and the model writes Python to explore it — printing samples, slicing, and
//! calling sub-LLMs over the pieces it cares about — until it can submit an answer. What runs the
//! code is the caller's [`CodeInterpreter`](crate::interpreter::CodeInterpreter), and what the
//! model is shown of the session is [`ReplHistory`](crate::interpreter::ReplHistory).

use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use serde_json::Value;

use serde_json::json;

use crate::adapter::python_json::json_dumps;
use crate::adapter::types::base::{Formatted, to_field_value};
use crate::adapter::Type;
use crate::example::{Example, Prediction};
use crate::interpreter::{CodeInterpreter, Executed, ReplEntry, ReplHistory, ReplVariable};
use crate::module::{Module, NamedPredictor, TraceStep, relabel};
use crate::react::Tool;
use crate::signature::{FieldKind, InField, JsonType, OutField, Signature};

use super::{Dynamic, Predict};

/// dspy's default sub-LLM call budget.
const DEFAULT_MAX_LLM_CALLS: usize = 50;

/// dspy `_PYTHON_FENCE_LANGS`: the language tags a fence may carry and still be Python. The empty
/// one is a bare ``` fence, which upstream reads as Python rather than refusing.
const PYTHON_FENCE_LANGS: [&str; 5] = ["python", "py", "python3", "py3", ""];

/// dspy `_strip_code_fences`: the Python out of whatever the model wrote around it.
///
/// Five rules, in upstream's order. Text with no fence is its own code. Decorative fence pairs
/// wrapping the whole thing are peeled off in a loop. The first fence in what remains is the one
/// read, so prose before it is skipped. A tag that is not Python is an error rather than a guess —
/// it reaches the model as the next turn's output, which is how it learns to write Python. And a
/// fence whose opener has no newline after it is left alone entirely, since there is no body to
/// take.
pub(crate) fn strip_code_fences(code: &str) -> Result<String> {
    let code = code.trim();
    if !code.contains("```") {
        return Ok(code.to_owned());
    }

    // Peel decorative pairs: ```\n```python\n…\n```\n``` and deeper.
    let mut lines: Vec<&str> = code.lines().collect();
    while lines.len() >= 2
        && lines[0].trim() == "```"
        && lines[lines.len() - 1].trim() == "```"
    {
        lines.remove(0);
        lines.pop();
    }
    let peeled = lines.join("\n");
    let code = peeled.trim();
    if !code.contains("```") {
        return Ok(code.to_owned());
    }

    let opened = code.find("```").expect("the fence just checked for") + 3;
    // No newline after the opener means no body, and upstream hands back what it was given.
    let Some((language_line, body)) = code[opened..].split_once('\n') else {
        return Ok(code.to_owned());
    };

    // The first word of the language line, lowercased; a line of only whitespace reads as bare.
    let language = language_line.split_whitespace().next().unwrap_or_default().to_lowercase();
    if !PYTHON_FENCE_LANGS.contains(&language.as_str()) {
        bail!(
            "Expected Python code but got ```{language} fence. Write Python code, not {language}."
        );
    }

    Ok(match body.find("```") {
        // An unterminated fence keeps everything after the opener.
        None => body.trim().to_owned(),
        Some(closed) => body[..closed].trim().to_owned(),
    })
}

/// dspy `ACTION_INSTRUCTIONS_TEMPLATE`: what the model is told about the REPL it is driving.
///
/// The four holes are the input names, the output-field list, the names `SUBMIT()` takes, and the
/// sub-LLM call budget.
const ACTION_INSTRUCTIONS: &str = "You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: {inputs} (your input data)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries: re, json, collections, math, etc.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output.";

/// dspy's extract instructions, verbatim — the indentation on the second line is the Python
/// literal's own, and it reaches the prompt.
const EXTRACT_INSTRUCTIONS: &str = "Based on the REPL trajectory, extract the final outputs now.

            Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs.";

/// dspy's `_build_signatures`: the two asks RLM makes.
///
/// The first drives the REPL — it is shown what variables exist, what has been run, and which
/// iteration this is, and answers with reasoning and the next snippet. The second reads the task's
/// real outputs off the session once the loop ends without a `SUBMIT()`.
pub(crate) fn signatures(
    signature: &Signature,
    tools: &[Arc<dyn Tool>],
    max_llm_calls: usize,
) -> (Signature, Signature) {
    // dspy appends two newlines to the task's own instructions wherever it has any, and every
    // signature has some — a string signature carries the default dspy writes for it.
    let task = match signature.instructions.is_empty() {
        true => String::new(),
        false => format!("{}\n\n", signature.instructions),
    };

    let inputs = backticked(signature.inputs.iter().map(|field| field.name.as_str()));
    // The names `SUBMIT()` takes are bare, unlike every other list in the template.
    let submits: Vec<&str> = signature.outputs.iter().map(|field| field.name.as_str()).collect();
    let output_fields = signature
        .outputs
        .iter()
        .map(|field| format!("- {}", crate::adapter::prompt::output_slot(field)))
        .collect::<Vec<_>>()
        .join("\n");

    let action_instructions = format!(
        "{task}{}{}",
        ACTION_INSTRUCTIONS
            .replace("{inputs}", &inputs)
            .replace("{output_fields}", &output_fields)
            .replace("{final_output_names}", &submits.join(", "))
            .replace("{max_llm_calls}", &max_llm_calls.to_string()),
        tool_docs(tools),
    );

    let action = Signature {
        instructions: action_instructions,
        inputs: vec![
            input("variables_info", "Metadata about the variables available in the REPL", FieldKind::Str),
            input(
                "repl_history",
                "Previous REPL code executions and their outputs",
                FieldKind::Json(JsonType::plain("REPLHistory")),
            ),
            input("iteration", "Current iteration number (1-indexed) out of max_iterations", FieldKind::Str),
        ],
        outputs: vec![
            output(
                "reasoning",
                "Think step-by-step: what do you know? What remains? Plan your next action.",
            ),
            output(
                "code",
                "Python code to execute. Use markdown code block format: ```python\\n<code>\\n```",
            ),
        ],
    };

    // The task's objective leads the extract instructions, so the model knows what it is
    // extracting for.
    let objective = match task.is_empty() {
        true => String::new(),
        false => format!("The trajectory was generated with the following objective: \n{task}\n"),
    };
    let extract = Signature {
        instructions: format!("{objective}{EXTRACT_INSTRUCTIONS}"),
        // dspy prepends `repl_history` and then `variables_info`, so the second lands first.
        inputs: vec![
            input("variables_info", "Metadata about the variables available in the REPL", FieldKind::Str),
            input(
                "repl_history",
                "Your REPL interactions so far",
                FieldKind::Json(JsonType::plain("REPLHistory")),
            ),
        ],
        outputs: signature.outputs.clone(),
    };

    (action, extract)
}

/// dspy `_format_tool_docs`: the caller's own tools, appended after the template. Nothing at all
/// where there are none, so the instructions end at the template.
fn tool_docs(tools: &[Arc<dyn Tool>]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut lines =
        vec!["\nAdditional tools available (use these instead of standard library equivalents):"
            .to_owned()];
    for tool in tools {
        let params = tool
            .args()
            .as_object()
            .map(|args| {
                args.iter()
                    .map(|(name, schema)| {
                        let kind = schema.get("type").and_then(Value::as_str).unwrap_or("Any");
                        format!("{name}: {kind}")
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        // dspy flattens a description's newlines so a multi-line one cannot break the list.
        let desc = match tool.description().is_empty() {
            true => "No description".to_owned(),
            false => tool.description().replace('\n', "  "),
        };
        lines.push(format!("- `{}({params})` - {desc}", tool.name()));
    }
    lines.join("\n")
}

fn input(name: &str, desc: &str, kind: FieldKind) -> InField {
    InField { name: name.to_owned(), desc: desc.to_owned(), kind, ..Default::default() }
}

fn output(name: &str, desc: &str) -> OutField {
    OutField { name: name.to_owned(), desc: desc.to_owned(), ..Default::default() }
}

fn backticked<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names.map(|name| format!("`{name}`")).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod conformance {
    use super::*;
    use serde_json::Value;

    /// Every shape dspy was given, to the same code or the same refusal.
    ///
    /// The golden (`tests/conformance/predict/rlm.json`, see `scripts/generate_rlm_fixture.py`) is
    /// what upstream returned for inputs chosen at the edges of its five rules — where a
    /// reimplementation guesses rather than agrees.
    fn golden() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/predict/rlm.json");
        let text = std::fs::read_to_string(&path).expect("the golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    #[test]
    fn it_strips_the_fences_dspy_strips() {
        for case in golden()["strip_code_fences"].as_array().expect("cases") {
            let written = case["written"].as_str().expect("written");
            match case["error"].as_str() {
                None => {
                    let code = strip_code_fences(written).expect("parses");
                    assert_eq!(code, case["code"].as_str().expect("code"), "code for {written:?}");
                }
                Some(error) => {
                    let refused = strip_code_fences(written).expect_err("refuses");
                    assert_eq!(refused.to_string(), error, "error for {written:?}");
                }
            }
        }
    }

    /// Both signatures — every field, its description and annotation, and the instructions.
    ///
    /// The action instructions are the largest byte-surface RLM has: a template with the input
    /// names, the output-field list (each carrying `translate_field_type`'s note), the bare names
    /// `SUBMIT()` takes, and the call budget, with the task's own instructions leading and the
    /// caller's tool docs trailing.
    #[test]
    fn it_builds_the_signatures_dspy_builds() {
        use crate::react::FnTool;
        use serde_json::json;

        let described = |fields: &Value| -> Vec<(String, String, String)> {
            fields
                .as_array()
                .expect("fields")
                .iter()
                .map(|field| {
                    (
                        field["name"].as_str().expect("name").to_owned(),
                        field["desc"].as_str().expect("desc").to_owned(),
                        field["annotation"].as_str().expect("annotation").to_owned(),
                    )
                })
                .collect()
        };
        let ours = |signature: &Signature| {
            (
                signature
                    .inputs
                    .iter()
                    .map(|f| (f.name.clone(), f.desc.clone(), f.annotation().to_owned()))
                    .collect::<Vec<_>>(),
                signature
                    .outputs
                    .iter()
                    .map(|f| (f.name.clone(), f.desc.clone(), f.annotation().to_owned()))
                    .collect::<Vec<_>>(),
            )
        };

        for case in golden()["signatures"].as_array().expect("cases") {
            let label = case["label"].as_str().expect("label");
            // The task signature, with the instructions dspy recorded (a docstring is not in the
            // spelling) and the typed outputs the notes are built from.
            let mut task: Signature = match label {
                "typed" => "context -> answer, count: int".parse().expect("parses"),
                "described" => "context -> answer".parse().expect("parses"),
                _ => case["task"].as_str().expect("task").parse().expect("parses"),
            };
            task.instructions = case["task_instructions"].as_str().expect("instructions").to_owned();
            if label == "typed" {
                task.outputs[1].desc = "how many".to_owned();
            }
            let tools: Vec<Arc<dyn Tool>> = case["tools"]
                .as_array()
                .expect("tools")
                .iter()
                .map(|_| {
                    Arc::new(FnTool::new(
                        "factorial",
                        "Compute the factorial of n.",
                        json!({ "n": { "type": "integer" } }),
                        |_| Ok(String::new()),
                    )) as Arc<dyn Tool>
                })
                .collect();
            let calls = case["max_llm_calls"].as_u64().expect("max_llm_calls") as usize;

            let (action, extract) = signatures(&task, &tools, calls);
            assert_eq!(
                action.instructions,
                case["action"]["instructions"].as_str().expect("instructions"),
                "action instructions for {label}"
            );
            let (inputs, outputs) = ours(&action);
            assert_eq!(inputs, described(&case["action"]["inputs"]), "action inputs for {label}");
            assert_eq!(outputs, described(&case["action"]["outputs"]), "action outputs for {label}");

            assert_eq!(
                extract.instructions,
                case["extract"]["instructions"].as_str().expect("instructions"),
                "extract instructions for {label}"
            );
            let (inputs, outputs) = ours(&extract);
            assert_eq!(inputs, described(&case["extract"]["inputs"]), "extract inputs for {label}");
            assert_eq!(outputs, described(&case["extract"]["outputs"]), "extract outputs for {label}");
        }
    }
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
}

impl Rlm {
    pub fn new(signature: Signature, interpreter: Arc<dyn CodeInterpreter>) -> Self {
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
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// The budget the model is told about, which is stated in the action instructions — so
    /// changing it rebuilds them.
    pub fn with_max_llm_calls(mut self, max_llm_calls: usize) -> Self {
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

    async fn run(&self, inputs: Example, trace: &mut Vec<TraceStep>) -> Result<Prediction> {
        let missing: Vec<&str> = self
            .signature
            .inputs
            .iter()
            .filter(|field| inputs.get(&field.name).is_none())
            .map(|field| field.name.as_str())
            .collect();
        if !missing.is_empty() {
            bail!("Missing required inputs: {missing:?}");
        }

        self.interpreter.define_tools(&self.tools)?;
        let variables = self.variables(&inputs);
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
                    .execute(&code)
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
                    history = history.append(ReplEntry::new(reasoning, code, printed_output(&printed)))
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
                let value = inputs.get(&field.name)?;
                let text = match value {
                    Value::String(text) => text.clone(),
                    other => json_dumps(other),
                };
                let mut variable = ReplVariable::new(&field.name, python_type_name(value), &text);
                variable.desc = field.desc.clone();
                variable.constraints = field.constraints.clone().unwrap_or_default();
                match Type::format(&variable) {
                    Formatted::Text(rendered) => Some(rendered),
                    Formatted::Blocks(_) => None,
                }
            })
            .collect()
    }

    /// dspy `_process_final_output`: a submission must be a mapping carrying every output field.
    /// What it is not is fed back to the model rather than raised, so it can submit again.
    fn submitted(&self, value: &Value) -> Result<serde_json::Map<String, Value>, String> {
        let Some(fields) = value.as_object() else {
            let names: Vec<&str> =
                self.signature.outputs.iter().map(|field| field.name.as_str()).collect();
            return Err(format!(
                "[Error] FINAL returned {}, expected dict with fields: {names:?}",
                python_type_name(value)
            ));
        };
        let mut missing: Vec<&str> = self
            .signature
            .outputs
            .iter()
            .map(|field| field.name.as_str())
            .filter(|name| !fields.contains_key(*name))
            .collect();
        if !missing.is_empty() {
            missing.sort_unstable();
            let names: Vec<&str> =
                self.signature.outputs.iter().map(|field| field.name.as_str()).collect();
            return Err(format!(
                "[Error] Missing output fields: {missing:?}. Use SUBMIT({})",
                names.join(", ")
            ));
        }
        let mut outputs = serde_json::Map::new();
        for field in &self.signature.outputs {
            outputs.insert(field.name.clone(), fields[&field.name].clone());
        }
        Ok(outputs)
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

/// The name Python would print for this value's type, which is what the model is shown.
fn python_type_name(value: &Value) -> &'static str {
    match value {
        Value::String(_) => "str",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_f64() => "float",
        Value::Number(_) => "int",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
        Value::Null => "NoneType",
    }
}

fn string_field(example: &Example, name: &str) -> String {
    example.get(name).and_then(Value::as_str).unwrap_or_default().to_owned()
}

impl Module for Rlm {
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

    fn task() -> Signature {
        "context -> answer".parse().expect("parses")
    }

    fn action(reasoning: &str, code: &str) -> String {
        format!("[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## code ## ]]\n{code}\n\n[[ ## completed ## ]]")
    }

    fn rlm(interpreter: Arc<ScriptedInterpreter>, replies: &[&'static str]) -> Rlm {
        let model = Arc::new(Scripted::new(replies));
        let mut rlm = Rlm::new(task(), interpreter);
        rlm.generate_action = rlm.generate_action.with_lm(model.clone());
        rlm.extract = rlm.extract.with_lm(model);
        rlm
    }

    /// A `SUBMIT()` carrying every output field ends the run, and the trajectory records the turn
    /// that did it.
    #[tokio::test]
    async fn a_submission_ends_the_run() {
        let interpreter = Arc::new(ScriptedInterpreter::new([Ok(Executed::Submitted(
            json!({ "answer": "42" }),
        ))]));
        let rlm = rlm(interpreter.clone(), &[&*Box::leak(
            action("submit it", "```python\nSUBMIT(answer='42')\n```").into_boxed_str(),
        )]);

        let prediction = rlm.forward(example! { context: "a long document" }).await.expect("answers");
        assert_eq!(prediction.get("answer"), Some(&json!("42")));
        assert_eq!(prediction.get("final_reasoning"), Some(&json!("submit it")));
        // The fence was stripped before the code reached the interpreter.
        assert_eq!(*interpreter.ran.lock().expect("ran"), ["SUBMIT(answer='42')"]);
        let trajectory = prediction.get("trajectory").expect("a trajectory");
        assert_eq!(trajectory.as_array().expect("entries").len(), 1);
        assert!(trajectory[0]["output"].as_str().expect("output").starts_with("FINAL: "));
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
        let look = Box::leak(action("look", "```python\nprint(len(context))\n```").into_boxed_str());
        let quiet = Box::leak(action("quiet", "```python\nx = 1\n```").into_boxed_str());
        let finish = Box::leak(action("finish", "```python\nSUBMIT(answer='done')\n```").into_boxed_str());
        let rlm = rlm(interpreter.clone(), &[look, quiet, finish]);

        let prediction = rlm.forward(example! { context: "doc" }).await.expect("answers");
        let trajectory = prediction.get("trajectory").expect("a trajectory");
        assert_eq!(trajectory[0]["output"], json!("1000 lines"));
        assert_eq!(trajectory[1]["output"], json!("(no output - did you forget to print?)"));
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
        let finish = Box::leak(action("done", "```python\nSUBMIT(answer='ok')\n```").into_boxed_str());
        let rlm = rlm(interpreter, &[broken, finish]);

        let prediction = rlm.forward(example! { context: "doc" }).await.expect("answers");
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
        signature.outputs.push(OutField { name: "count".to_owned(), ..Default::default() });
        let interpreter = Arc::new(ScriptedInterpreter::new([
            Ok(Executed::Submitted(json!({ "answer": "42" }))),
            Ok(Executed::Submitted(json!({ "answer": "42", "count": "1" }))),
        ]));
        let first = Box::leak(action("partial", "```python\nSUBMIT(answer='42')\n```").into_boxed_str());
        let second = Box::leak(action("full", "```python\nSUBMIT(answer='42', count=1)\n```").into_boxed_str());
        let model = Arc::new(Scripted::new(&[first, second]));
        let mut rlm = Rlm::new(signature, interpreter);
        rlm.generate_action = rlm.generate_action.with_lm(model.clone());
        rlm.extract = rlm.extract.with_lm(model);

        let prediction = rlm.forward(example! { context: "doc" }).await.expect("answers");
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
        let interpreter =
            Arc::new(ScriptedInterpreter::new([Ok(Executed::Printed(json!("still looking")))]));
        let look = Box::leak(action("look", "```python\nprint(1)\n```").into_boxed_str());
        let extracted = "[[ ## answer ## ]]\nfrom the trajectory\n\n[[ ## completed ## ]]";
        let rlm = rlm(interpreter, &[look, extracted]).with_max_iterations(1);

        let prediction = rlm.forward(example! { context: "doc" }).await.expect("answers");
        assert_eq!(prediction.get("answer"), Some(&json!("from the trajectory")));
        assert_eq!(
            prediction.get("final_reasoning"),
            Some(&json!("Extract forced final output"))
        );
    }

    /// An input the signature declares but the caller did not pass is refused, with dspy's wording.
    #[tokio::test]
    async fn it_refuses_missing_inputs() {
        let interpreter = Arc::new(ScriptedInterpreter::new([]));
        let rlm = rlm(interpreter, &[]);
        let error = rlm.forward(Example::new([("other", json!(1))])).await.expect_err("refuses");
        assert!(error.to_string().contains("Missing required inputs"), "got: {error}");
    }
}

/// The shared budget the two sub-LLM tools spend from.
struct CallBudget {
    spent: Mutex<usize>,
    max: usize,
}

impl CallBudget {
    /// dspy `_check_and_increment`: the whole batch is charged before any of it runs, so a batch
    /// that would overrun is refused rather than half-answered.
    fn charge(&self, calls: usize) -> Result<()> {
        let mut spent = self.spent.lock().expect("the budget lock");
        if *spent + calls > self.max {
            bail!(
                "LLM call limit exceeded: {spent} + {calls} > {}. Use Python code for aggregation \
                 instead of making more LLM calls.",
                self.max
            );
        }
        *spent += calls;
        Ok(())
    }
}

/// dspy's `llm_query`: one prompt to a sub-LLM, charged against the budget.
struct LlmQuery<A> {
    budget: Arc<CallBudget>,
    ask: Arc<A>,
    args: Value,
}

impl<A: Fn(&str) -> Result<String> + Send + Sync> Tool for LlmQuery<A> {
    fn name(&self) -> &str {
        "llm_query"
    }

    fn description(&self) -> &str {
        "Query the LLM with a prompt string."
    }

    fn args(&self) -> &Value {
        &self.args
    }

    fn call(&self, args: &Value) -> Result<String> {
        let prompt = args.get("prompt").and_then(Value::as_str).unwrap_or_default();
        if prompt.is_empty() {
            bail!("prompt cannot be empty");
        }
        self.budget.charge(1)?;
        (self.ask)(prompt)
    }
}

/// dspy's `llm_query_batched`: several prompts at once, answered in the order they were given.
struct LlmQueryBatched<A> {
    budget: Arc<CallBudget>,
    ask: Arc<A>,
    args: Value,
}

impl<A: Fn(&str) -> Result<String> + Send + Sync> Tool for LlmQueryBatched<A> {
    fn name(&self) -> &str {
        "llm_query_batched"
    }

    fn description(&self) -> &str {
        "Query the LLM with multiple prompts concurrently."
    }

    fn args(&self) -> &Value {
        &self.args
    }

    fn call(&self, args: &Value) -> Result<String> {
        Ok(json_dumps(&self.call_value(args)?))
    }

    /// The answers as a list, which is what the code that called this reads.
    ///
    /// dspy runs the prompts on a thread pool; that is a speed property rather than an observable
    /// one, since it reassembles the answers in the order the prompts were given either way. The
    /// ask here is synchronous — [`Tool::call`] is — so they run in that order to begin with.
    fn call_value(&self, args: &Value) -> Result<Value> {
        let Some(prompts) = args.get("prompts").and_then(Value::as_array) else {
            bail!("prompts must be a list");
        };
        // An empty batch is answered with an empty list and costs nothing.
        if prompts.is_empty() {
            return Ok(json!([]));
        }
        self.budget.charge(prompts.len())?;
        let answers: Vec<Value> = prompts
            .iter()
            .map(|prompt| {
                let prompt = prompt.as_str().unwrap_or_default();
                // dspy answers a prompt that failed with the error in place, so one bad prompt
                // does not lose the rest of the batch.
                match (self.ask)(prompt) {
                    Ok(answer) => json!(answer),
                    Err(error) => json!(format!("[ERROR] {error}")),
                }
            })
            .collect();
        Ok(Value::Array(answers))
    }
}

/// dspy `_make_llm_tools`: the `llm_query` pair the REPL code can call, sharing one budget.
///
/// `ask` is the caller's bridge to a sub-LLM, synchronous because [`Tool::call`] is — the same
/// contract [`mcp_tool`](crate::react::mcp_tool) states, and a caller driving an async model blocks
/// on it. Hand the pair to [`Rlm::with_tools`] and they reach the sandbox through
/// [`define_tools`](CodeInterpreter::define_tools) with the caller's own tools.
pub fn llm_query_tools<A>(max_llm_calls: usize, ask: A) -> Vec<Arc<dyn Tool>>
where
    A: Fn(&str) -> Result<String> + Send + Sync + 'static,
{
    let budget = Arc::new(CallBudget { spent: Mutex::new(0), max: max_llm_calls });
    let ask = Arc::new(ask);
    vec![
        Arc::new(LlmQuery {
            budget: budget.clone(),
            ask: ask.clone(),
            args: json!({ "prompt": { "type": "string" } }),
        }),
        Arc::new(LlmQueryBatched {
            budget,
            ask,
            args: json!({ "prompts": { "type": "array", "items": { "type": "string" } } }),
        }),
    ]
}

#[cfg(test)]
mod sub_llm_tests {
    use super::*;

    fn tools(max: usize) -> Vec<Arc<dyn Tool>> {
        llm_query_tools(max, |prompt| match prompt {
            "boom" => bail!("the sub-LLM failed"),
            other => Ok(format!("answered: {other}")),
        })
    }

    #[test]
    fn one_query_spends_one_call_and_an_empty_prompt_is_refused() {
        let tools = tools(2);
        let query = &tools[0];
        assert_eq!(query.name(), "llm_query");
        assert_eq!(query.call(&json!({ "prompt": "hi" })).expect("answers"), "answered: hi");
        assert!(query.call(&json!({ "prompt": "" })).is_err(), "an empty prompt is refused");
    }

    /// The two tools share one budget, and overrunning it says what to do instead.
    #[test]
    fn the_budget_is_shared_and_refuses_an_overrun() {
        let tools = tools(2);
        let (query, batched) = (&tools[0], &tools[1]);
        query.call(&json!({ "prompt": "one" })).expect("answers");
        // One spent, so a batch of two would overrun and is refused whole.
        let error = batched
            .call_value(&json!({ "prompts": ["a", "b"] }))
            .expect_err("refuses");
        assert!(error.to_string().starts_with("LLM call limit exceeded: 1 + 2 > 2."), "got: {error}");
        assert!(error.to_string().contains("Use Python code for aggregation"), "got: {error}");
        // The refused batch was not charged, so one call remains.
        query.call(&json!({ "prompt": "two" })).expect("the last call");
        assert!(query.call(&json!({ "prompt": "three" })).is_err(), "the budget is spent");
    }

    /// A batch answers in the order it was given, and one failed prompt does not lose the rest.
    #[test]
    fn a_batch_keeps_its_order_and_reports_a_failure_in_place() {
        let tools = tools(10);
        let batched = &tools[1];
        let answers = batched
            .call_value(&json!({ "prompts": ["a", "boom", "c"] }))
            .expect("answers");
        assert_eq!(answers[0], json!("answered: a"));
        assert_eq!(answers[2], json!("answered: c"));
        assert!(
            answers[1].as_str().expect("an error").starts_with("[ERROR] "),
            "got: {}",
            answers[1]
        );
        // An empty batch costs nothing and answers with nothing.
        assert_eq!(batched.call_value(&json!({ "prompts": [] })).expect("answers"), json!([]));
    }
}
