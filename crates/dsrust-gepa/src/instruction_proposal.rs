//! GEPA's reflection prompt (`strategies/instruction_proposal.py`): the `InstructionProposalSignature`
//! that both gepa's default proposer and dspy's GEPA adapter use to rewrite a component's instruction.
//! It renders the current instruction and a reflective dataset (per-example inputs, outputs, and
//! feedback) into one prompt string, and extracts the new instruction from the reflection LM's reply.
//!
//! The rendering is byte-sensitive — an adapter feeds the reflection LM this exact text — so this is a
//! faithful reproduction held to the gepa package (pin `gepa==0.1.1`). Reflective values are text
//! here; gepa 0.1.1's additive `Image` path (a value rendered as `[IMAGE-N …]`, turning the prompt
//! into a message list) is a multimodal boundary not yet built — dsrs reflects text instructions.

/// dspy's GEPA adapter and gepa's default proposer both call this with `prompt_template=None`.
/// gepa 0.1.1 renamed the placeholders from `<curr_instructions>`/`<inputs_outputs_feedback>`.
pub const DEFAULT_PROMPT_TEMPLATE: &str = r#"I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."#;

/// A value inside a reflective example: a leaf string, or a nested ordered map or list. GEPA renders
/// these to markdown, so their order is preserved (dspy inserts `Inputs`, `Generated Outputs`, then
/// `Feedback`, and a sorted map would render them the wrong way round).
pub enum Reflective {
    Text(String),
    Map(Vec<(String, Reflective)>),
    List(Vec<Reflective>),
}

impl Reflective {
    /// This value as JSON: text as a string, a map as an object in its recorded order, a list as
    /// an array.
    ///
    /// What a caller's own proposer needs. dspy hands `propose_new_texts` plain dicts and lists, and
    /// a proposer built from a module — the GEPA advanced page's `RAGInstructionImprover` — passes
    /// them straight into a signature field. This is the same data in the same shape; the order of
    /// a map's keys is the order dspy renders them in, and is kept.
    pub fn to_value(&self) -> serde_json::Value {
        match self {
            Reflective::Text(text) => serde_json::Value::String(text.clone()),
            Reflective::Map(entries) => serde_json::Value::Object(
                entries
                    .iter()
                    .map(|(name, value)| (name.clone(), value.to_value()))
                    .collect(),
            ),
            Reflective::List(items) => {
                serde_json::Value::Array(items.iter().map(Reflective::to_value).collect())
            }
        }
    }
}

/// One reflective example: an ordered map of section name to value (dspy's `{Inputs, Generated
/// Outputs, Feedback}`), rendered as a `# Example N` block.
pub type ReflectiveSample = Vec<(String, Reflective)>;

/// dspy `InstructionProposalSignature.prompt_renderer`: substitute the current instruction and the
/// rendered reflective dataset into the template. Both placeholders are replaced everywhere they
/// occur, current-instruction first (matching Python's two sequential `str.replace` calls).
pub fn render_prompt(
    current_instruction: &str,
    dataset: &[ReflectiveSample],
    template: Option<&str>,
) -> String {
    let template = template.unwrap_or(DEFAULT_PROMPT_TEMPLATE);
    let prompt = template.replace("<curr_param>", current_instruction);
    prompt.replace("<side_info>", &format_samples(dataset))
}

/// dspy `format_samples`: each example as a `# Example N` markdown block, joined by blank lines.
fn format_samples(samples: &[ReflectiveSample]) -> String {
    samples
        .iter()
        .enumerate()
        .map(|(index, sample)| convert_sample_to_markdown(sample, index + 1))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// dspy `convert_sample_to_markdown`: the example's sections at a fixed `##` depth, their values
/// rendered one level deeper.
fn convert_sample_to_markdown(sample: &ReflectiveSample, examplenum: usize) -> String {
    let mut s = format!("# Example {examplenum}\n");
    for (key, value) in sample {
        s.push_str(&format!("## {key}\n"));
        s.push_str(&render_value(value, 3));
    }
    s
}

/// dspy `render_value`: maps and lists become `#`-headed sections (their name or `Item N`) with values
/// rendered one header deeper (capped at `######`); a leaf is its stripped text and a blank line; an
/// empty map or list contributes a lone blank line.
fn render_value(value: &Reflective, level: usize) -> String {
    match value {
        Reflective::Map(entries) => {
            render_sections(entries.iter().map(|(k, v)| (k.clone(), v)), level)
        }
        Reflective::List(items) => render_sections(
            items
                .iter()
                .enumerate()
                .map(|(i, v)| (format!("Item {}", i + 1), v)),
            level,
        ),
        Reflective::Text(text) => format!("{}\n\n", text.trim()),
    }
}

/// The shared body of the map and list arms: a `#`-headed section per entry, then a lone blank line if
/// there were none.
fn render_sections<'a>(
    entries: impl Iterator<Item = (String, &'a Reflective)>,
    level: usize,
) -> String {
    let mut s = String::new();
    let mut any = false;
    for (heading, value) in entries {
        any = true;
        s.push_str(&format!("{} {heading}\n", "#".repeat(level)));
        s.push_str(&render_value(value, (level + 1).min(6)));
    }
    if !any {
        s.push('\n');
    }
    s
}

/// dspy `InstructionProposalSignature.output_extractor`: the new instruction is the text between the
/// first and last ``` fences (skipping an optional language line). A single or absent fence falls back
/// to stripping whatever partial fence is present.
pub fn extract_new_instruction(lm_out: &str) -> String {
    // The two `-1` defaults are dspy's sentinel and neither can be told from another value: `find`
    // and `rfind` answer `None` together — a string either contains the fence or it does not — and
    // in that case `start` is at least 2 while `end` is at most 1, so the branch below is taken
    // whatever the sentinels are. Recorded because a mutation run keeps offering both.
    let find = lm_out.find("```").map(|i| i as isize).unwrap_or(-1);
    let start = find + 3;
    let end = lm_out.rfind("```").map(|i| i as isize).unwrap_or(-1);

    if start >= end {
        return extract_from_partial_fence(lm_out);
    }

    let content = &lm_out[start as usize..end as usize];
    skip_language_line(content).trim().to_string()
}

/// dspy's `start >= end` path: no complete pair of fences.
fn extract_from_partial_fence(lm_out: &str) -> String {
    let stripped = lm_out.trim();
    if stripped.starts_with("```") {
        if let Some(after_fence) = match_opening_fence(lm_out) {
            return lm_out[after_fence..].trim().to_string();
        }
    } else if let Some(body) = stripped.strip_suffix("```") {
        return body.trim().to_string();
    }
    stripped.to_string()
}

/// dspy `re.match(r"^```\S*\n?", lm_out)`: the ``` fence, a maximal non-whitespace language tag, and an
/// optional single newline — anchored at the very start (so leading whitespace means no match).
fn match_opening_fence(lm_out: &str) -> Option<usize> {
    let rest = lm_out.strip_prefix("```")?;
    let tag_len = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let mut end = 3 + tag_len;
    // Stepping over the newline. Skipping it entirely is invisible — the caller trims, so a
    // leading newline goes either way — which is why the mutant that leaves `end` alone survives
    // where the one that walks it *backwards* is caught. Kept as dspy's regex has it.
    if lm_out[end..].starts_with('\n') {
        end += 1;
    }
    Some(end)
}

/// dspy `re.match(r"^\S*\n", content)`: skip a leading language line — a maximal non-whitespace run
/// terminated by a newline, anchored at the start. It matches only if the first whitespace is that
/// newline, so a space before the newline leaves the content untouched.
fn skip_language_line(content: &str) -> &str {
    match content.find('\n') {
        // As in `match_opening_fence`: keeping the newline rather than stepping over it cannot be
        // seen through the caller's trim, so `newline + 1` and `newline` are indistinguishable.
        // dspy's regex consumes it, so this does.
        Some(newline) if content[..newline].chars().all(|c| !c.is_whitespace()) => {
            &content[newline + 1..]
        }
        _ => content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn reflective_of(node: &Value) -> Reflective {
        let object = node.as_object().expect("a tagged node");
        if let Some(text) = object.get("text") {
            Reflective::Text(text.as_str().expect("text is a string").to_string())
        } else if let Some(entries) = object.get("map") {
            Reflective::Map(
                entries
                    .as_array()
                    .expect("map entries")
                    .iter()
                    .map(pair_of)
                    .collect(),
            )
        } else {
            let items = object
                .get("list")
                .expect("text, map, or list")
                .as_array()
                .expect("list items");
            Reflective::List(items.iter().map(reflective_of).collect())
        }
    }

    fn pair_of(entry: &Value) -> (String, Reflective) {
        let pair = entry.as_array().expect("a [key, node] pair");
        (
            pair[0].as_str().expect("key").to_string(),
            reflective_of(&pair[1]),
        )
    }

    fn sample_of(sample: &Value) -> ReflectiveSample {
        sample
            .as_array()
            .expect("a sample")
            .iter()
            .map(pair_of)
            .collect()
    }

    /// The reflection prompt and the extracted instruction gepa's own `InstructionProposalSignature`
    /// produces, over nested/empty/deep reflective datasets and every branch of the fence extractor —
    /// from `tests/conformance/reflection.json`, generated by running the real gepa package.
    #[test]
    fn renders_and_extracts_as_gepa_does() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/reflection.json");
        let text = std::fs::read_to_string(&path).expect("the reflection golden is committed");
        let fixture: Value = serde_json::from_str(&text).expect("the golden parses");

        assert_eq!(
            DEFAULT_PROMPT_TEMPLATE,
            fixture["default_template"]
                .as_str()
                .expect("default_template"),
            "the default template is byte-identical to gepa's"
        );

        for case in fixture["render_cases"].as_array().expect("render_cases") {
            let instruction = case["current_instruction"]
                .as_str()
                .expect("current_instruction");
            let dataset: Vec<ReflectiveSample> = case["dataset"]
                .as_array()
                .expect("dataset")
                .iter()
                .map(sample_of)
                .collect();
            let label = case["label"].as_str().unwrap_or("");
            assert_eq!(
                render_prompt(instruction, &dataset, None),
                case["prompt"].as_str().expect("prompt"),
                "render case {label}"
            );
        }

        for case in fixture["extract_cases"].as_array().expect("extract_cases") {
            let lm_out = case["lm_out"].as_str().expect("lm_out");
            assert_eq!(
                extract_new_instruction(lm_out),
                case["new_instruction"].as_str().expect("new_instruction"),
                "extract case {lm_out:?}"
            );
        }
    }
}
