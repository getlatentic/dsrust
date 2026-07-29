//! dspy `RLM._build_signatures`: the two asks RLM makes, and the instructions they carry.

use std::sync::Arc;

use serde_json::Value;

use crate::react::Tool;
use crate::signature::{FieldKind, InField, JsonType, OutField, Signature};

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

/// The first signature drives the REPL — it is shown what variables exist, what has been run, and
/// which iteration this is, and answers with reasoning and the next snippet. The second reads the
/// task's real outputs off the session once the loop ends without a `SUBMIT()`.
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
    let submits: Vec<&str> = signature
        .outputs
        .iter()
        .map(|field| field.name.as_str())
        .collect();
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
            input(
                "variables_info",
                "Metadata about the variables available in the REPL",
                FieldKind::Str,
            ),
            input(
                "repl_history",
                "Previous REPL code executions and their outputs",
                FieldKind::Json(JsonType::plain("REPLHistory")),
            ),
            input(
                "iteration",
                "Current iteration number (1-indexed) out of max_iterations",
                FieldKind::Str,
            ),
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
            input(
                "variables_info",
                "Metadata about the variables available in the REPL",
                FieldKind::Str,
            ),
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
    let mut lines = vec![
        "\nAdditional tools available (use these instead of standard library equivalents):"
            .to_owned(),
    ];
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
    InField {
        name: name.to_owned(),
        desc: desc.to_owned(),
        kind,
        ..Default::default()
    }
}

fn output(name: &str, desc: &str) -> OutField {
    OutField {
        name: name.to_owned(),
        desc: desc.to_owned(),
        ..Default::default()
    }
}

fn backticked<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod conformance {
    use super::*;
    use crate::react::FnTool;
    use serde_json::json;

    /// Both signatures — every field, its description and annotation, and the instructions.
    ///
    /// The action instructions are the largest byte-surface RLM has: a template with the input
    /// names, the output-field list (each carrying `translate_field_type`'s note), the bare names
    /// `SUBMIT()` takes, and the call budget, with the task's own instructions leading and the
    /// caller's tool docs trailing.
    #[test]
    fn it_builds_the_signatures_dspy_builds() {
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

        for case in super::super::golden()["signatures"]
            .as_array()
            .expect("cases")
        {
            let label = case["label"].as_str().expect("label");
            // The task signature, with the instructions dspy recorded (a docstring is not in the
            // spelling) and the typed outputs the notes are built from.
            let mut task: Signature = match label {
                "typed" => "context -> answer, count: int".parse().expect("parses"),
                "described" => "context -> answer".parse().expect("parses"),
                _ => case["task"]
                    .as_str()
                    .expect("task")
                    .parse()
                    .expect("parses"),
            };
            task.instructions = case["task_instructions"]
                .as_str()
                .expect("instructions")
                .to_owned();
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
                case["action"]["instructions"]
                    .as_str()
                    .expect("instructions"),
                "action instructions for {label}"
            );
            let (inputs, outputs) = ours(&action);
            assert_eq!(
                inputs,
                described(&case["action"]["inputs"]),
                "action inputs for {label}"
            );
            assert_eq!(
                outputs,
                described(&case["action"]["outputs"]),
                "action outputs for {label}"
            );

            assert_eq!(
                extract.instructions,
                case["extract"]["instructions"]
                    .as_str()
                    .expect("instructions"),
                "extract instructions for {label}"
            );
            let (inputs, outputs) = ours(&extract);
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
