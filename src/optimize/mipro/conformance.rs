//! MIPROv2 against dspy's own, end to end.
//!
//! The pieces are verified apart — the proposer signatures byte-for-byte, the demo sets, the TPE
//! sampler against optuna. This pins them together: given the same model, does the crate's MIPROv2
//! select the instruction dspy's does? The golden in `tests/conformance/optimize/mipro.json` is what
//! dspy compiled (see `scripts/generate_mipro_fixture.py`).
//!
//! The model is instruction-sensitive so the assertion means something: it answers the task
//! correctly only when the proposed instruction (carrying `GOOD`) is in force, so that proposal
//! scores against the original's zero and the search must actually select it — not merely keep the
//! baseline, which a score-tying model would let a broken search do.

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

use super::MIPROv2;
use crate::evaluate::exact_match;
use crate::example::Example;
use crate::lm::{ChatModel, api};
use crate::example;
use crate::predict::Predict;

/// dspy's `Coach` (`scripts/generate_mipro_fixture.py`), mirrored: propose an instruction carrying
/// `GOOD`, and answer the task correctly only when `GOOD` is the instruction in force.
struct Coach {
    table: Vec<(String, String)>,
    proposal: String,
}

impl ChatModel for Coach {
    async fn forward(
        &self,
        _http: &reqwest::Client,
        request: &api::LmRequest,
    ) -> Result<api::LmResponse> {
        let system = request.system();
        let last = request.messages.last().and_then(|message| message.text()).unwrap_or_default();
        let content = if system.contains("generate a new instruction that will be used") {
            format!("[[ ## proposed_instruction ## ]]\n{}\n\n[[ ## completed ## ]]", self.proposal)
        } else {
            let answer = self
                .table
                .iter()
                .find(|(question, _)| last.contains(question.as_str()))
                .filter(|_| system.contains("GOOD"))
                .map(|(_, answer)| answer.as_str())
                .unwrap_or("wrong");
            format!("[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]")
        };
        Ok(api::LmResponse::text(content))
    }
}

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/optimize/mipro.json");
    let text = std::fs::read_to_string(&path).expect("the mipro golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

fn rows(fixture: &Value) -> Vec<(String, String)> {
    fixture["trainset"]
        .as_array()
        .expect("trainset")
        .iter()
        .map(|row| {
            (
                row["question"].as_str().unwrap().to_owned(),
                row["answer"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

#[tokio::test]
async fn compiles_the_instruction_dspy_compiles() {
    let fixture = fixture();
    let table = rows(&fixture);
    let proposal = fixture["proposal"].as_str().expect("proposal").to_owned();
    let trainset: Vec<Example> = table
        .iter()
        .map(|(question, answer)| {
            example! { question: question.clone(), answer: answer.clone() }.with_inputs(["question"])
        })
        .collect();

    for case in fixture["cases"].as_array().expect("cases") {
        let model = Arc::new(Coach { table: table.clone(), proposal: proposal.clone() });
        let mut student = Predict::parse("question -> answer").expect("parses").with_lm(model.clone());

        MIPROv2::new(exact_match, model.clone())
            .with_candidates(case["num_candidates"].as_u64().expect("num_candidates") as usize)
            .with_trials(case["num_trials"].as_u64().expect("num_trials") as usize)
            .with_seed(case["seed"].as_u64().expect("seed"))
            .compile(&mut student, &trainset)
            .await
            .expect("compiles");

        let compiled = case["compiled"][0].as_str().expect("a compiled instruction");
        assert_eq!(
            student.signature.instructions, compiled,
            "compiled instruction for case {case}"
        );
    }
}
