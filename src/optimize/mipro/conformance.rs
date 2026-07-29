//! MIPROv2 against dspy's own, end to end — in the regime where the search dynamics decide.
//!
//! The pieces are verified apart — the proposer signatures byte-for-byte, the demo sets, the TPE
//! sampler against optuna. This pins them together: given the same model, does the crate's MIPROv2
//! run the trials dspy's runs and select what dspy selects? The golden in
//! `tests/conformance/optimize/mipro.json` is what dspy compiled (see
//! `scripts/generate_mipro_fixture.py`).
//!
//! Discrimination is the point. The model proposes a *distinct* instruction per proposal ask
//! (`GOOD-1`, `GOOD-2`, … in call order), each with its own score profile and a deliberate tie, and
//! most cases run fewer trials than candidates — so which candidates are tried, in which order,
//! decides the compiled instruction, and different seeds genuinely compile different instructions.
//! What is compared is the whole search path: every proposal in call order, every trial's chosen
//! candidate and score (the baseline first, as the optuna study records it), then the winner.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use serde_json::Value;

use super::MIPROv2;
use crate::evaluate::exact_match;
use crate::example;
use crate::example::Example;
use crate::lm::{ChatModel, api};
use crate::predict::Predict;

/// dspy's `Coach` (`scripts/generate_mipro_fixture.py`), mirrored: propose `GOOD-k` on the k-th
/// proposal ask, and answer question `q` correctly iff the instruction in force carries a marker
/// whose profile contains `q`. The call tally is shared state there (dspy's proposer runs each ask
/// on a shallow copy of the model) and an atomic here, for the same reason spelled differently.
struct Coach {
    table: Vec<(String, String)>,
    profiles: HashMap<u64, Vec<String>>,
    proposal_calls: AtomicUsize,
    proposals: std::sync::Mutex<Vec<String>>,
}

/// The `GOOD-<k>` marker in an instruction, if any. Written by hand because the crate carries no
/// regex; the marker is always `GOOD-` followed by digits.
fn marker(text: &str) -> Option<u64> {
    let start = text.find("GOOD-")? + "GOOD-".len();
    let digits: String = text[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

impl ChatModel for Coach {
    async fn forward(
        &self,
        _http: &reqwest::Client,
        request: &api::LmRequest,
    ) -> Result<api::LmResponse> {
        let system = request.system();
        let last = request
            .messages
            .last()
            .and_then(|message| message.text())
            .unwrap_or_default();
        let content = if system.contains("generate a new instruction that will be used") {
            let call = self.proposal_calls.fetch_add(1, Ordering::SeqCst) + 1;
            let proposal = format!("Answer with GOOD-{call} precision.");
            self.proposals
                .lock()
                .expect("proposals lock")
                .push(proposal.clone());
            format!("[[ ## proposed_instruction ## ]]\n{proposal}\n\n[[ ## completed ## ]]")
        } else {
            let solved = marker(system)
                .and_then(|k| self.profiles.get(&k))
                .map(Vec::as_slice)
                .unwrap_or_default();
            let answer = self
                .table
                .iter()
                .find(|(question, _)| last.contains(question.as_str()))
                .filter(|(question, _)| solved.contains(question))
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

fn profiles(fixture: &Value) -> HashMap<u64, Vec<String>> {
    fixture["profiles"]
        .as_object()
        .expect("profiles")
        .iter()
        .map(|(k, questions)| {
            let questions = questions
                .as_array()
                .expect("profile questions")
                .iter()
                .map(|q| q.as_str().unwrap().to_owned())
                .collect();
            (k.parse().expect("a numeric marker"), questions)
        })
        .collect()
}

#[tokio::test]
async fn runs_the_trials_dspy_runs_and_compiles_what_dspy_compiles() {
    let fixture = fixture();
    let table = rows(&fixture);
    let profiles = profiles(&fixture);
    let trainset: Vec<Example> = table
        .iter()
        .map(|(question, answer)| {
            example! { question: question.clone(), answer: answer.clone() }
                .with_inputs(["question"])
        })
        .collect();

    // The golden's case set compiles more than one distinct instruction — the property that lets
    // it tell a seeded search from one that ignores its seed. The generator refuses to write a
    // golden without it; this keeps a hand-edited one honest too.
    let distinct: std::collections::HashSet<&str> = fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| case["compiled"][0].as_str().expect("compiled"))
        .collect();
    assert!(
        distinct.len() > 1,
        "the golden is not discriminating: {distinct:?}"
    );

    for case in fixture["cases"].as_array().expect("cases") {
        let model = Arc::new(Coach {
            table: table.clone(),
            profiles: profiles.clone(),
            proposal_calls: AtomicUsize::new(0),
            proposals: std::sync::Mutex::new(Vec::new()),
        });
        let mut student = Predict::parse("question -> answer")
            .expect("parses")
            .with_lm(model.clone());

        let trials = MIPROv2::new(exact_match, model.clone())
            .with_candidates(case["num_candidates"].as_u64().expect("num_candidates") as usize)
            .with_trials(case["num_trials"].as_u64().expect("num_trials") as usize)
            .with_seed(case["seed"].as_u64().expect("seed"))
            .compile_traced(&mut student, &trainset)
            .await
            .expect("compiles");

        // Every proposal, in call order — pins the proposer's call count as well as its replies.
        let proposed: Vec<String> = model.proposals.lock().expect("proposals lock").clone();
        let expected: Vec<&str> = case["proposals"]
            .as_array()
            .expect("proposals")
            .iter()
            .map(|proposal| proposal.as_str().unwrap())
            .collect();
        assert_eq!(proposed, expected, "proposals for case {case}");

        // Every trial: the candidate index it chose and the score it earned, the baseline first —
        // the search path, not merely its destination.
        let recorded = case["trials"].as_array().expect("trials");
        assert_eq!(trials.len(), recorded.len(), "trial count for case {case}");
        for (index, (ours, theirs)) in trials.iter().zip(recorded).enumerate() {
            let instruction = theirs["params"]["0_predictor_instruction"]
                .as_u64()
                .expect("instruction") as usize;
            assert_eq!(
                ours.params,
                vec![instruction],
                "trial {index} params for case {case}"
            );
            let score = theirs["score"].as_f64().expect("score");
            assert_eq!(ours.score, score, "trial {index} score for case {case}");
        }

        let compiled = case["compiled"][0]
            .as_str()
            .expect("a compiled instruction");
        assert_eq!(
            student.signature.instructions, compiled,
            "compiled instruction for case {case}"
        );
    }
}
