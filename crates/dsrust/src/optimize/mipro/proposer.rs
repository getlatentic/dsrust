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
    // Every scan below is a `take_while` over the remaining slice rather than a hand-advanced
    // cursor, so progress belongs to the iterator. Under the `while at < len { at += 1 }` form
    // this replaces, four separate `+=` mutants each spun for the whole timeout — and a hang is
    // detection only in the sense that the suite never finished.
    let run = |from: usize, accepts: &dyn Fn(char) -> bool| {
        chars[from..]
            .iter()
            .take_while(|(_, c)| accepts(*c))
            .count()
    };

    let mut at = run(0, &|c: char| c == '*' || c.is_whitespace());
    for _word in 0..5 {
        let word = run(at, &is_word);
        if word == 0 {
            break;
        }
        at += word;
        if chars.get(at).is_some_and(|(_, c)| *c == ':') {
            at += 1 + run(at + 1, &char::is_whitespace);
            return match chars.get(at) {
                Some(&(byte, _)) => &text[byte..],
                None => "",
            };
        }
        let spaces = run(at, &char::is_whitespace);
        if spaces == 0 {
            break;
        }
        at += spaces;
    }
    text
}

/// dspy's module summary line: `Predict(inputs) -> outputs`. The class is always `Predict` here, as
/// every predictor a program holds is one.
fn module_code(predictor: &Signature) -> String {
    let names = |fields: &[String]| fields.join(", ");
    let inputs = names(
        &predictor
            .inputs
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>(),
    );
    let outputs = names(
        &predictor
            .outputs
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>(),
    );
    format!("Predict({inputs}) -> {outputs}")
}

fn text_of(prediction: &crate::example::Prediction, field: &str) -> String {
    prediction
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
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
    /// `sampling` is how dspy asks for the proposal itself: `prompt_model.copy(rollout_id=…,
    /// temperature=init_temperature)`. The rollout id misses the response cache and the temperature
    /// makes the proposals differ — without both, `num_candidates` proposals sharing a tip are one
    /// proposal replayed. It reaches only `generate`; the two description calls are dspy's plain
    /// `prompt_model`, and caching them is what makes a multi-predictor run affordable.
    pub(crate) fn new(
        program_code: Option<String>,
        mut inputs: InstructionInputs,
        model: Arc<dyn DynChatModel>,
        sampling: crate::lm::Sampling,
    ) -> Self {
        inputs.program_aware = inputs.program_aware && program_code.is_some();
        let predict = |signature| Predict::from_signature(signature).set_lm(model.clone());
        Self {
            describe_program: predict(signatures::describe_program()),
            describe_module: predict(signatures::describe_module()),
            generate: predict(signatures::generate_module_instruction(inputs)).config(sampling),
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
            fields.push((
                "program_code".into(),
                json!(self.program_code.as_deref().unwrap_or_default()),
            ));
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
        let Some(program_code) = self
            .program_code
            .as_deref()
            .filter(|_| self.inputs.program_aware)
        else {
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
        Ok((
            program_description,
            text_of(&described, "module_description"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{ChatModel, api};

    /// The word class includes `_`, `'` and `-`, so a label spelled with any of them still strips.
    /// The or-chain's mutants narrowed the class and a labelled line came through label and all.
    #[test]
    fn labels_spelled_with_underscores_and_apostrophes_still_strip() {
        assert_eq!(strip_prefix("my_label: keep this"), "keep this");
        assert_eq!(strip_prefix("Dave's Tip: keep this"), "keep this");
        assert_eq!(strip_prefix("well-known Advice: keep this"), "keep this");
    }

    /// dspy's module summary line, exactly: `Predict(inputs) -> outputs`. Replaced wholesale, the
    /// module description prompt described a nameless program and every test stayed green.
    #[test]
    fn module_code_spells_the_predict_line() {
        let signature: Signature = "question, context -> answer".parse().expect("parses");
        assert_eq!(
            module_code(&signature),
            "Predict(question, context) -> answer"
        );
    }

    /// `describe` asks twice and hands both descriptions on — the program's, label stripped, and
    /// the module's. Without program code it answers dspy's fixed strings and asks nothing. All
    /// four tuple-replacement mutants survived because nothing read either half.
    #[tokio::test]
    async fn describe_hands_on_both_descriptions_or_the_fixed_strings() {
        let lm = std::sync::Arc::new(crate::DummyLM::new([
            crate::example! { program_description: "Program Description: solves questions" },
            crate::example! { module_description: "answers directly" },
        ]));
        let proposer = GenerateModuleInstruction::new(
            Some("class Program: ...".to_owned()),
            InstructionInputs {
                dataset_summary: false,
                program_aware: true,
                instruct_history: false,
                tip: false,
            },
            lm.clone(),
            crate::lm::Sampling::default(),
        );
        let (program, module) = proposer
            .describe("No task demos provided.", "Predict(question) -> answer")
            .await
            .expect("the script answers");
        assert_eq!(program, "solves questions", "the label is stripped");
        assert_eq!(module, "answers directly");
        assert_eq!(lm.asked().len(), 2);

        let unaware = GenerateModuleInstruction::new(
            None,
            InstructionInputs {
                dataset_summary: false,
                program_aware: false,
                instruct_history: false,
                tip: false,
            },
            std::sync::Arc::new(crate::DummyLM::new([])),
            crate::lm::Sampling::default(),
        );
        let (program, module) = unaware
            .describe("No task demos provided.", "Predict(question) -> answer")
            .await
            .expect("no ask to fail");
        assert_eq!(program, "Not available");
        assert_eq!(module, "Not provided");
    }

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
        async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
            let system = request.system();
            let reply = |field: &str, value: &str| {
                format!("[[ ## {field} ## ]]\n{value}\n\n[[ ## completed ## ]]")
            };
            let content = if system.contains("describe what type of task this program") {
                reply("program_description", "A question answerer.")
            } else if system.contains("describe the purpose of one of the specified module") {
                reply("module_description", "It answers the question.")
            } else {
                reply(
                    "proposed_instruction",
                    "Instruction: Answer precisely and briefly.",
                )
            };
            Ok(api::LmResponse::text(content))
        }
    }

    /// dspy asks for a proposal through `prompt_model.copy(rollout_id=…, temperature=…)`. Both reach
    /// the request, and only the request that proposes: the two description calls are dspy's plain
    /// prompt model, and caching them is what makes a multi-predictor run affordable.
    #[tokio::test]
    async fn the_proposal_carries_its_rollout_and_temperature() {
        #[derive(Default)]
        struct Recording(std::sync::Mutex<Vec<api::LmConfig>>);
        impl crate::lm::ChatModel for Recording {
            async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
                self.0
                    .lock()
                    .expect("not poisoned")
                    .push(request.config.clone());
                Ok(api::LmResponse::text(
                    "[[ ## proposed_instruction ## ]]\nBe precise.",
                ))
            }
        }

        let model = Arc::new(Recording::default());
        let asked = Arc::clone(&model);
        let sampling = crate::lm::Sampling {
            temperature: Some(0.7),
            ..crate::lm::Sampling::rollout(4242)
        };
        let proposer = GenerateModuleInstruction::new(
            None,
            InstructionInputs {
                dataset_summary: false,
                program_aware: false,
                instruct_history: false,
                tip: false,
            },
            model,
            sampling,
        );
        let predictor: Signature = "question -> answer".parse().expect("parses");
        proposer
            .forward(&predictor, "", "", "", None)
            .await
            .expect("it proposes");

        let asks = asked.0.lock().expect("not poisoned");
        let proposing = asks.last().expect("the proposal itself");
        assert_eq!(proposing.temperature, Some(0.7));
        assert_eq!(
            proposing.rollout_id(),
            Some(&crate::lm::api::RolloutId::Number(4242)),
            "a fresh rollout is what misses the response cache"
        );
        assert!(
            asks[..asks.len() - 1]
                .iter()
                .all(|earlier| earlier.rollout_id().is_none()),
            "the description calls stay cacheable"
        );
    }

    #[tokio::test]
    async fn proposes_through_the_program_aware_chain() {
        let model = Arc::new(Proposer);
        let proposer = GenerateModuleInstruction::new(
            Some("class Program: ...".to_owned()),
            InstructionInputs {
                dataset_summary: true,
                program_aware: true,
                instruct_history: true,
                tip: true,
            },
            model,
            crate::lm::Sampling::default(),
        );
        let predictor: Signature = "question -> answer".parse().expect("parses");

        let proposed = proposer
            .forward(
                &predictor,
                "No task demos provided.",
                "A QA dataset.",
                "",
                Some("Keep the instruction clear and concise."),
            )
            .await
            .expect("proposes");

        // The generator's `Instruction:` label is stripped, which is why the stored instruction is
        // clean even though the model prefixed one.
        assert_eq!(proposed, "Answer precisely and briefly.");
    }
}
