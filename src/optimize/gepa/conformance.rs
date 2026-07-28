//! GEPA end to end: the crate's optimizer against a scripted task model and reflection model.
//!
//! The sanity test drives the whole wiring — the [`gepa`] engine over the dsrust adapter — and checks
//! that GEPA reflects the seed instruction into the one that scores, accepts it on the minibatch, and
//! leaves the student holding it. The dspy comparison (`gepa_makes_the_decisions_dspy_makes`) replays
//! runs recorded from dspy's GEPA teleprompter in the regime where the search dynamics decide: the
//! reflection model proposes a *distinct* instruction per ask, each with its own score profile, so
//! whether a proposal survives its seeded minibatch decides what enters the pool — and different
//! seeds genuinely compile different instructions. What is compared is the whole evolution: every
//! candidate in discovery order, its parents, its validation score, the eval bookkeeping, the
//! reflection-call count, and the winner.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use serde_json::Value;

use super::{Feedback, GEPA};
use crate::example;
use crate::example::{Example, Prediction};
use crate::lm::ChatModel;
use crate::lm::api::{self, LmResponse};
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

/// The `GOOD-<k>` marker in an instruction, if any. Written by hand because the crate carries no
/// regex; the marker is always `GOOD-` followed by digits.
fn marker(text: &str) -> Option<u64> {
    let start = text.find("GOOD-")? + "GOOD-".len();
    let digits: String = text[start..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// dspy's `Coach` (`scripts/generate_gepa_optimize_fixture.py`), mirrored: answer question `q`
/// correctly iff the instruction in force carries a marker whose profile contains `q`.
struct ProfileCoach {
    profiles: HashMap<u64, Vec<String>>,
}

impl ChatModel for ProfileCoach {
    async fn forward(&self, _http: &reqwest::Client, request: &api::LmRequest) -> Result<LmResponse> {
        let system = request.system();
        let last = request.messages.last().and_then(|message| message.text()).unwrap_or_default();
        let solved = marker(system)
            .and_then(|k| self.profiles.get(&k))
            .map(Vec::as_slice)
            .unwrap_or_default();
        let answer = TABLE
            .iter()
            .find(|(question, _)| last.contains(question))
            .filter(|(question, _)| solved.iter().any(|q| q == question))
            .map(|(_, correct)| *correct)
            .unwrap_or("wrong");
        Ok(LmResponse::text(format!("[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]")))
    }
}

/// dspy's `Reflector`, mirrored: propose `GOOD-k`, fenced, on the k-th ask. The tally is shared
/// state there (a dict surviving the model's shallow copies) and an atomic here.
struct CountingReflector {
    calls: AtomicUsize,
    proposals: std::sync::Mutex<Vec<String>>,
}

impl ChatModel for CountingReflector {
    async fn forward(&self, _http: &reqwest::Client, _request: &api::LmRequest) -> Result<LmResponse> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let proposal = format!("Answer with GOOD-{call} precision.");
        self.proposals.lock().expect("proposals lock").push(proposal.clone());
        Ok(LmResponse::text(format!("```\n{proposal}\n```")))
    }
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

fn usizes(value: &Value) -> Vec<usize> {
    value.as_array().expect("an array").iter().map(|v| v.as_u64().expect("an index") as usize).collect()
}

/// The evolution dspy's GEPA runs, replayed decision for decision: candidates in discovery order,
/// their parents and validation scores, the eval bookkeeping, and the compiled winner.
#[tokio::test]
async fn gepa_makes_the_decisions_dspy_makes() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/optimize/gepa.json");
    let text = std::fs::read_to_string(&path).expect("the gepa golden is committed");
    let fixture: Value = serde_json::from_str(&text).expect("the golden parses");
    let profiles = profiles(&fixture);

    // The golden's case set compiles more than one distinct instruction — the property that lets
    // it tell a seeded search from one that ignores its seed.
    let distinct: std::collections::HashSet<&str> = fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| case["compiled_instruction"].as_str().expect("compiled_instruction"))
        .collect();
    assert!(distinct.len() > 1, "the golden is not discriminating: {distinct:?}");

    for case in fixture["cases"].as_array().expect("cases") {
        let seed = case["seed"].as_u64().expect("seed");
        let budget = case["max_metric_calls"].as_u64().expect("budget") as usize;
        let minibatch = case["minibatch_size"].as_u64().expect("minibatch") as usize;

        let mut student = Predict::parse("question -> answer")
            .expect("parses")
            .with_lm(Arc::new(ProfileCoach { profiles: profiles.clone() }));
        student.signature.instructions = case["seed_instruction"].as_str().expect("seed_instruction").to_owned();

        let reflector = Arc::new(CountingReflector {
            calls: AtomicUsize::new(0),
            proposals: std::sync::Mutex::new(Vec::new()),
        });
        let outcome = GEPA::new(metric, reflector.clone())
            .with_max_metric_calls(budget)
            .with_reflection_minibatch_size(minibatch)
            .with_seed(seed)
            .compile(&mut student, &trainset(), &trainset())
            .await
            .expect("compiles");

        // Every reflection ask, in call order — pins how often the engine reflected as well as
        // what came back.
        let proposed: Vec<String> = reflector.proposals.lock().expect("proposals lock").clone();
        let expected: Vec<&str> = case["proposals"]
            .as_array()
            .expect("proposals")
            .iter()
            .map(|proposal| proposal.as_str().unwrap())
            .collect();
        assert_eq!(proposed, expected, "seed {seed}: proposals");
        assert_eq!(
            reflector.calls.load(Ordering::SeqCst),
            case["reflection_calls"].as_u64().expect("reflection_calls") as usize,
            "seed {seed}: reflection calls"
        );

        // The pool, in discovery order: each candidate's instruction, its parents, its validation
        // score, and what its discovery had cost.
        let candidates: Vec<&str> = outcome
            .candidates
            .iter()
            .map(|candidate| candidate.values().next().expect("one component").as_str())
            .collect();
        let recorded: Vec<&str> = case["candidates"]
            .as_array()
            .expect("candidates")
            .iter()
            .map(|candidate| candidate.as_str().unwrap())
            .collect();
        assert_eq!(candidates, recorded, "seed {seed}: candidates");

        let parents: Vec<Vec<usize>> =
            case["parents"].as_array().expect("parents").iter().map(usizes).collect();
        assert_eq!(outcome.parents, parents, "seed {seed}: parents");

        let scores: Vec<f64> = case["val_aggregate_scores"]
            .as_array()
            .expect("scores")
            .iter()
            .map(|score| score.as_f64().expect("a score"))
            .collect();
        assert_eq!(outcome.val_aggregate_scores, scores, "seed {seed}: val scores");

        assert_eq!(
            outcome.best_idx,
            case["best_idx"].as_u64().expect("best_idx") as usize,
            "seed {seed}: best index"
        );
        assert_eq!(
            outcome.num_metric_calls_by_discovery,
            usizes(&case["discovery_eval_counts"]),
            "seed {seed}: discovery eval counts"
        );
        assert_eq!(
            outcome.num_full_ds_evals,
            case["num_full_val_evals"].as_u64().expect("num_full_val_evals") as usize,
            "seed {seed}: full valset evals"
        );
        assert_eq!(
            outcome.total_num_evals,
            case["total_metric_calls"].as_u64().expect("total_metric_calls") as usize,
            "seed {seed}: metric-call total"
        );

        assert_eq!(
            student.signature.instructions,
            case["compiled_instruction"].as_str().expect("compiled_instruction"),
            "seed {seed}: compiled instruction"
        );
    }
}
