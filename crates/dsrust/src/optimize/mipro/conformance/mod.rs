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
    async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
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

mod minibatch;

/// The proposer's tip texts, against dspy's own `TIPS` dict.
///
/// Transcribed constants are the one part of this module a golden was not holding: the strings were
/// right, and nothing would have said so if upstream reworded one. Order counts as much as the
/// text, since `random.choice` indexes into `list(TIPS.keys())` — so a tip inserted rather than
/// appended would change every proposal after it.
#[test]
fn the_proposers_tips_are_dspys() {
    let fixture = fixture();
    let recorded: Vec<&str> = fixture["tips"]
        .as_array()
        .expect("the golden records dspy's tips")
        .iter()
        .map(|tip| tip.as_str().expect("a tip"))
        .collect();
    assert_eq!(super::grounded::TIPS.as_slice(), recorded.as_slice());
}

use super::{Auto, Trial};

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
        let model = coach(&table, &profiles);
        let mut student = Predict::parse("question -> answer")
            .expect("parses")
            .set_lm(model.clone());

        let trials = MIPROv2::new(exact_match, model.clone())
            .num_candidates(case["num_candidates"].as_u64().expect("num_candidates") as usize)
            .num_trials(case["num_trials"].as_u64().expect("num_trials") as usize)
            // The golden's own `minibatch=False`. On by default in both, and these cases are the
            // non-minibatch regime; `minibatch.rs` carries the other one.
            .minibatch(false)
            .data_aware_proposer(false)
            // The golden's own `data_aware_proposer=False`.
            .data_aware_proposer(false)
            .seed(case["seed"].as_u64().expect("seed"))
            .max_bootstrapped_demos(case["max_bootstrapped_demos"].as_u64().expect("boot") as usize)
            .max_labeled_demos(case["max_labeled_demos"].as_u64().expect("labeled") as usize)
            .compile_traced(&mut student, &trainset, Some(&trainset))
            .await
            .expect("compiles");

        agrees_with(case, &model, &trials, &student);
    }
}

/// What every case asserts, whichever family it came from: the proposer's whole call sequence, the
/// whole trial sequence, and what the run left on the predictor.
///
/// The trial sequence is the part that matters. A minibatch case's trials include the interleaved
/// full evaluations, so comparing only the winner would pass on a port that never ran them.
fn agrees_with(case: &Value, model: &Coach, trials: &[Trial], student: &Predict) {
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
        // Interleaved per predictor — instruction then demos — which is the order upstream
        // suggests them in and therefore the order optuna's multivariate TPE draws in. A
        // zero-shot run suggests no demo parameter at all, so the vector is one wide.
        let named = |name: &str| {
            theirs["params"][name]
                .as_u64()
                .unwrap_or_else(|| panic!("param {name} for case {case}")) as usize
        };
        let mut params = vec![named("0_predictor_instruction")];
        if theirs["params"].get("0_predictor_demos").is_some() {
            params.push(named("0_predictor_demos"));
        }
        assert_eq!(ours.params, params, "trial {index} params for case {case}");
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

    // The demo set the winning trial left on the predictor. Choosing the set is half of what a
    // few-shot run does, and the instruction alone would not say which one was chosen.
    let wanted = case["compiled_demos"][0]
        .as_array()
        .expect("compiled demos");
    assert_eq!(
        student.demos.len(),
        wanted.len(),
        "compiled demo count for case {case}"
    );
    for (ours, theirs) in student.demos.iter().zip(wanted) {
        let fields = theirs.as_object().expect("a demo is an object");
        for (name, value) in fields {
            assert_eq!(
                ours.get(name),
                Some(value),
                "demo field {name} for case {case}"
            );
        }
    }
}

/// A fresh proposer-and-answerer for one case. Each case needs its own: the proposal tally is what
/// makes replies distinct, and sharing one across cases would carry the count over.
fn coach(table: &[(String, String)], profiles: &HashMap<u64, Vec<String>>) -> Arc<Coach> {
    Arc::new(Coach {
        table: table.to_vec(),
        profiles: profiles.clone(),
        proposal_calls: AtomicUsize::new(0),
        proposals: std::sync::Mutex::new(Vec::new()),
    })
}

/// dspy scopes `task_model` around Step 1 and Step 3 and leaves Step 2 to `prompt_model`, so the
/// program is bootstrapped and evaluated on one model while another writes the proposals.
///
/// Two models that answer differently is the only way to see the split: a run that evaluated on the
/// proposer's model would compile a different instruction, and one that proposed on the task model
/// would propose different text. Both are asserted.
#[tokio::test]
async fn the_task_model_runs_the_program_and_the_prompt_model_writes_the_proposals() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Records which system prompts it was shown, so a test can say what each model was asked for.
    struct Counting {
        proposals: AtomicUsize,
        programs: AtomicUsize,
    }

    impl ChatModel for Counting {
        async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
            if request
                .system()
                .contains("generate a new instruction that will be used")
            {
                self.proposals.fetch_add(1, Ordering::Relaxed);
                return Ok(api::LmResponse::text(
                    "[[ ## proposed_instruction ## ]]\nBe precise.\n\n[[ ## completed ## ]]",
                ));
            }
            self.programs.fetch_add(1, Ordering::Relaxed);
            Ok(api::LmResponse::text(
                "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]",
            ))
        }
    }

    fn counting() -> Arc<Counting> {
        Arc::new(Counting {
            proposals: AtomicUsize::new(0),
            programs: AtomicUsize::new(0),
        })
    }

    let prompt_model = counting();
    let task = counting();
    // A *third* model is configured, so nothing but the scope can send the program to `task`. With
    // the task model also configured this test would pass without any scoping at all — which is the
    // shape of assertion that proves nothing.
    let configured = counting();
    let _installed = crate::lm::global::install_for_test(configured.clone());
    let mut student = Predict::parse("question -> answer").expect("parses");

    let trainset = vec![
        example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
    ];
    MIPROv2::new(exact_match, prompt_model.clone())
        .num_candidates(2)
        .num_trials(1)
        .minibatch(false)
        // The summary would be three more calls on the prompt model, which this test counts.
        .data_aware_proposer(false)
        .seed(0)
        .max_bootstrapped_demos(0)
        .max_labeled_demos(0)
        .task_model(task.clone())
        .compile(&mut student, &trainset, Some(&trainset))
        .await
        .expect("compiles");

    assert!(
        prompt_model.proposals.load(Ordering::Relaxed) > 0,
        "the prompt model should have written the proposals"
    );
    assert_eq!(
        prompt_model.programs.load(Ordering::Relaxed),
        0,
        "the prompt model should never have run the program"
    );
    assert!(
        task.programs.load(Ordering::Relaxed) > 0,
        "the task model should have run the program"
    );
    assert_eq!(
        configured.programs.load(Ordering::Relaxed),
        0,
        "the configured model should have been scoped out of Steps 1 and 3"
    );
    assert_eq!(
        task.proposals.load(Ordering::Relaxed),
        0,
        "the task model should never have written a proposal"
    );
}
