//! GEPA end to end: the crate's optimizer against a scripted task model and reflection model.
//!
//! The sanity test drives the whole wiring — the [`gepa`] engine over the dsrust adapter — and checks
//! that GEPA reflects the seed instruction into the one that scores, accepts it on the minibatch, and
//! leaves the student holding it. The dspy comparison (`gepa_makes_the_decisions_dspy_makes`) replays
//! a run recorded from dspy's GEPA teleprompter and checks the candidate dsrust lands on matches.

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

use super::{Feedback, GEPA};
use crate::example;
use crate::example::{Example, Prediction};
use crate::lm::api::{self, LmResponse};
use crate::lm::ChatModel;
use crate::predict::Predict;

const TABLE: [(&str, &str); 3] =
    [("capital of France?", "Paris"), ("capital of Germany?", "Berlin"), ("capital of Spain?", "Madrid")];

const PROPOSAL: &str = "Answer with GOOD precision.";

/// The task model: it answers a question correctly only when the instruction in force carries `GOOD`
/// (which it reads from the system prompt), so a candidate carrying `GOOD` outscores the seed.
struct TaskCoach;

impl ChatModel for TaskCoach {
    async fn forward(&self, _http: &reqwest::Client, request: &api::LmRequest) -> Result<LmResponse> {
        let has_good = request.system().contains("GOOD");
        let last = request.messages.last().and_then(|message| message.text()).unwrap_or_default();
        let answer = TABLE
            .iter()
            .find(|(question, _)| last.contains(question))
            .map(|(_, correct)| if has_good { *correct } else { "wrong" })
            .unwrap_or("wrong");
        Ok(LmResponse::text(format!("[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]")))
    }
}

/// The reflection model: whatever it is shown, it proposes the instruction carrying `GOOD`, wrapped in
/// a fenced block for [`gepa::extract_new_instruction`].
struct Reflector;

impl ChatModel for Reflector {
    async fn forward(&self, _http: &reqwest::Client, _request: &api::LmRequest) -> Result<LmResponse> {
        Ok(LmResponse::text(format!("```\n{PROPOSAL}\n```")))
    }
}

/// A GEPA feedback metric: exact-match on the answer, with a word of feedback either way.
fn metric(gold: &Example, pred: &Prediction) -> Feedback {
    let correct = gold.get("answer") == pred.get("answer");
    if correct {
        Feedback::new(1.0, "Correct.")
    } else {
        Feedback::new(0.0, "Wrong answer; be more precise.")
    }
}

fn trainset() -> Vec<Example> {
    TABLE.iter().map(|(q, a)| example! { question: *q, answer: *a }.with_inputs(["question"])).collect()
}

/// GEPA reflects the seed into the `GOOD` instruction, which scores 100% against the seed's 0%, so the
/// search accepts it and the student is left holding it.
#[tokio::test]
async fn gepa_evolves_the_instruction_that_scores() {
    let task = Arc::new(TaskCoach);
    let mut student = Predict::parse("question -> answer").expect("parses").with_lm(task);
    student.signature.instructions = "Answer the question.".to_owned();

    GEPA::new(metric, Arc::new(Reflector))
        .with_max_metric_calls(20)
        .with_reflection_minibatch_size(2)
        .compile(&mut student, &trainset(), &trainset())
        .await
        .expect("compiles");

    assert_eq!(student.signature.instructions, PROPOSAL);
}

/// The candidate dspy's GEPA lands on, replayed: the same scripted task + reflection models drive
/// dspy's teleprompter (`scripts/generate_gepa_optimize_fixture.py`, `use_merge=False`), and the crate
/// reproduces its winning instruction.
#[tokio::test]
async fn gepa_makes_the_decisions_dspy_makes() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/optimize/gepa.json");
    let text = std::fs::read_to_string(&path).expect("the gepa golden is committed");
    let fixture: Value = serde_json::from_str(&text).expect("the golden parses");

    for case in fixture["cases"].as_array().expect("cases") {
        let seed = case["seed"].as_u64().expect("seed");
        let budget = case["max_metric_calls"].as_u64().expect("budget") as usize;
        let minibatch = case["minibatch_size"].as_u64().expect("minibatch") as usize;

        let mut student = Predict::parse("question -> answer").expect("parses").with_lm(Arc::new(TaskCoach));
        student.signature.instructions = case["seed_instruction"].as_str().expect("seed_instruction").to_owned();

        let outcome = GEPA::new(metric, Arc::new(Reflector))
            .with_max_metric_calls(budget)
            .with_reflection_minibatch_size(minibatch)
            .with_seed(seed)
            .compile(&mut student, &trainset(), &trainset())
            .await
            .expect("compiles");

        // The instruction GEPA lands on, and — proving the engine made dspy's decisions from the
        // adapter's scores — the same candidate count and metric-call total.
        assert_eq!(
            student.signature.instructions,
            case["compiled_instruction"].as_str().expect("compiled_instruction"),
            "seed {seed}: compiled instruction"
        );
        assert_eq!(
            outcome.candidates.len(),
            case["num_candidates"].as_u64().expect("num_candidates") as usize,
            "seed {seed}: candidate count"
        );
        assert_eq!(
            outcome.total_num_evals,
            case["total_metric_calls"].as_u64().expect("total_metric_calls") as usize,
            "seed {seed}: metric-call total"
        );
    }
}
