//! dspy GroundedProposer's `GenerateModuleInstruction`: propose one instruction for one predictor.
//!
//! Optionally it first describes the whole program and the module under optimisation, then feeds
//! those plus the dataset summary, demos, past attempts and a tip to the instruction generator. The
//! three model calls are the byte-critical part, over the [`signatures`](super::signatures) this
//! module verifies. The assembly of `task_demos` from bootstrapped demo sets belongs to the
//! orchestration layer, which owns the `augmented` marker that gathering keys on, so it arrives here
//! as a string.

use std::sync::Arc;

use anyhow::Result;
use serde_json::{Value, json};

use super::signatures::{self, InstructionInputs};
use crate::example::Example;
use crate::lm::DynChatModel;
use crate::module::Module;
use crate::predict::Predict;
use crate::signature::Signature;

/// dspy `strip_prefix`: drop a leading label of up to five words ending in a colon
/// (`^[\*\s]*(([\w'\-]+\s+){0,4}[\w'\-]+):\s*`), then surrounding double quotes. Applied to every
/// instruction a model returns, so what a predictor ends up holding depends on it.
pub(super) fn strip_prefix(text: &str) -> String {
    without_label(text).trim_matches('"').to_owned()
}

/// The text past a leading `Word … Word:` label, or all of it when there is no such label. The
/// leading run of `*` and whitespace is part of the label and goes with it; a `*` after the colon
/// is not, matching upstream's regex.
fn without_label(text: &str) -> &str {
    let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '\'' || c == '-';
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut at = 0;
    while at < chars.len() && (chars[at].1 == '*' || chars[at].1.is_whitespace()) {
        at += 1;
    }
    let mut words = 0;
    while words < 5 {
        let word_start = at;
        while at < chars.len() && is_word(chars[at].1) {
            at += 1;
        }
        if at == word_start {
            break;
        }
        words += 1;
        if at < chars.len() && chars[at].1 == ':' {
            at += 1;
            while at < chars.len() && chars[at].1.is_whitespace() {
                at += 1;
            }
            return match chars.get(at) {
                Some(&(byte, _)) => &text[byte..],
                None => "",
            };
        }
        let space_start = at;
        while at < chars.len() && chars[at].1.is_whitespace() {
            at += 1;
        }
        if at == space_start {
            break;
        }
    }
    text
}

/// dspy's module summary line: `Predict(inputs) -> outputs`. The class is always `Predict` here, as
/// every predictor a program holds is one.
fn module_code(predictor: &Signature) -> String {
    let names = |fields: &[String]| fields.join(", ");
    let inputs = names(&predictor.inputs.iter().map(|f| f.name.clone()).collect::<Vec<_>>());
    let outputs = names(&predictor.outputs.iter().map(|f| f.name.clone()).collect::<Vec<_>>());
    format!("Predict({inputs}) -> {outputs}")
}

fn text_of(prediction: &crate::example::Prediction, field: &str) -> String {
    prediction.get(field).and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// dspy `GenerateModuleInstruction`: the three-call instruction proposer for one predictor.
pub(crate) struct GenerateModuleInstruction {
    describe_program: Predict,
    describe_module: Predict,
    generate: Predict,
    program_code: Option<String>,
    inputs: InstructionInputs,
}

impl GenerateModuleInstruction {
    /// Build the three predictors this proposer drives. `program_aware` only holds when there is
    /// program code to describe, so it is reconciled with `program_code` here — the same reconciliation
    /// dspy does when source introspection fails.
    pub(crate) fn new(
        program_code: Option<String>,
        mut inputs: InstructionInputs,
        model: Arc<dyn DynChatModel>,
    ) -> Self {
        inputs.program_aware = inputs.program_aware && program_code.is_some();
        let predict = |signature| Predict::from_signature(signature).with_lm(model.clone());
        Self {
            describe_program: predict(signatures::describe_program()),
            describe_module: predict(signatures::describe_module()),
            generate: predict(signatures::generate_module_instruction(inputs)),
            program_code,
            inputs,
        }
    }

    /// Propose an instruction, given the context the orchestration layer assembled: the predictor
    /// being optimised, its demos rendered to a string, the dataset summary, the history of past
    /// attempts, and a tip.
    pub(crate) async fn forward(
        &self,
        predictor: &Signature,
        task_demos: &str,
        data_summary: &str,
        previous_instructions: &str,
        tip: Option<&str>,
    ) -> Result<String> {
        let module = module_code(predictor);
        let (program_description, module_description) = self.describe(task_demos, &module).await?;

        let mut fields: Vec<(String, Value)> = Vec::new();
        if self.inputs.dataset_summary {
            fields.push(("dataset_description".into(), json!(data_summary)));
        }
        if self.inputs.program_aware {
            fields.push(("program_code".into(), json!(self.program_code.as_deref().unwrap_or_default())));
            fields.push(("program_description".into(), json!(program_description)));
            fields.push(("module".into(), json!(module)));
            fields.push(("module_description".into(), json!(module_description)));
        }
        fields.push(("task_demos".into(), json!(task_demos)));
        if self.inputs.instruct_history {
            fields.push(("previous_instructions".into(), json!(previous_instructions)));
        }
        fields.push(("basic_instruction".into(), json!(predictor.instructions)));
        if self.inputs.tip {
            fields.push(("tip".into(), json!(tip.unwrap_or_default())));
        }

        let proposed = self.generate.forward(Example::new(fields)).await?;
        Ok(strip_prefix(&text_of(&proposed, "proposed_instruction")))
    }

    /// The program-aware summaries. dspy strips a label from the program description but not the
    /// module one; both default to fixed strings when program code was not available.
    async fn describe(&self, task_demos: &str, module: &str) -> Result<(String, String)> {
        let Some(program_code) = self.program_code.as_deref().filter(|_| self.inputs.program_aware) else {
            return Ok(("Not available".into(), "Not provided".into()));
        };
        let described = self
            .describe_program
            .forward(Example::new([
                ("program_code".to_owned(), json!(program_code)),
                ("program_example".to_owned(), json!(task_demos)),
            ]))
            .await?;
        let program_description = strip_prefix(&text_of(&described, "program_description"));
        let described = self
            .describe_module
            .forward(Example::new([
                ("program_code".to_owned(), json!(program_code)),
                ("program_description".to_owned(), json!(program_description)),
                ("program_example".to_owned(), json!(task_demos)),
                ("module".to_owned(), json!(module)),
            ]))
            .await?;
        Ok((program_description, text_of(&described, "module_description")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{ChatModel, api};

    #[test]
    fn strip_prefix_matches_dspy() {
        // From `dspy.propose.utils.strip_prefix`, run on 3.2.1. A label of up to five words and a
        // colon goes, quotes go, and text with neither is left alone; a `*` after the colon stays.
        let cases = [
            ("Instruction: Answer the question.", "Answer the question."),
            ("Proposed Instruction: Do the thing.", "Do the thing."),
            ("\"Just quoted text\"", "Just quoted text"),
            ("No prefix here just text", "No prefix here just text"),
            ("**Bold Label:** content", "** content"),
            ("Task Description For The Model: solve it", "solve it"),
            ("One Two Three Four Five: kept?", "kept?"),
        ];
        for (input, expected) in cases {
            assert_eq!(strip_prefix(input), expected, "strip_prefix({input:?})");
        }
    }

    /// A model that plays all three roles by what its system message asks for, so one proposer run
    /// exercises the whole program-aware chain: describe the program, describe the module, propose.
    struct Proposer;

    impl ChatModel for Proposer {
        async fn forward(
            &self,
            _http: &reqwest::Client,
            request: &api::LmRequest,
        ) -> Result<api::LmResponse> {
            let system = request.system();
            let reply = |field: &str, value: &str| {
                format!("[[ ## {field} ## ]]\n{value}\n\n[[ ## completed ## ]]")
            };
            let content = if system.contains("describe what type of task this program") {
                reply("program_description", "A question answerer.")
            } else if system.contains("describe the purpose of one of the specified module") {
                reply("module_description", "It answers the question.")
            } else {
                reply("proposed_instruction", "Instruction: Answer precisely and briefly.")
            };
            Ok(api::LmResponse::text(content))
        }
    }

    #[tokio::test]
    async fn proposes_through_the_program_aware_chain() {
        let model = Arc::new(Proposer);
        let proposer = GenerateModuleInstruction::new(
            Some("class Program: ...".to_owned()),
            InstructionInputs { dataset_summary: true, program_aware: true, instruct_history: true, tip: true },
            model,
        );
        let predictor: Signature = "question -> answer".parse().expect("parses");

        let proposed = proposer
            .forward(&predictor, "No task demos provided.", "A QA dataset.", "", Some("Keep the instruction clear and concise."))
            .await
            .expect("proposes");

        // The generator's `Instruction:` label is stripped, which is why the stored instruction is
        // clean even though the model prefixed one.
        assert_eq!(proposed, "Answer precisely and briefly.");
    }
}
