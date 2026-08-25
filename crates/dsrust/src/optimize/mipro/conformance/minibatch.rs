//! `auto` and minibatch trials against dspy's own, on a valset big enough for either to bite.
//!
//! The sibling cases run `minibatch=False` over three examples, where every trial scores the whole
//! set and the RNG is read only by the bootstrap and the proposer. None of what this file covers is
//! visible there:
//!
//!   - `auto` subsamples the valset *before* Step 1 draws, so a preset moves every later draw;
//!   - a minibatch trial draws its own subsample, so the generator advances per trial — and does not
//!     when the batch covers the set, which is a different sequence, not a shorter one;
//!   - a full evaluation is added to the study as a trial, so it takes a trial number and shifts the
//!     schedule of every full evaluation after it, as well as feeding the sampler a score;
//!   - `auto` splits one candidate count in two, halving the instruction budget when demos are also
//!     searched — visible only as a proposal count, which the shared assertions compare.
//!
//! Two cases pass no valset at all, which is not "score on everything": upstream keeps the first 20%
//! of the trainset to bootstrap from and scores on the last 80%.

use super::{
    Auto, Coach, Example, MIPROv2, Predict, Value, agrees_with, coach, exact_match, example,
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::callback::{CallId, Callback, watched_by};
use crate::evaluate::Pass;

/// The 140-row set the minibatch cases run on, and its score profiles.
fn dataset(fixture: &Value) -> (Vec<(String, String)>, HashMap<u64, Vec<String>>) {
    let table = fixture["minibatch_trainset"]
        .as_array()
        .expect("minibatch trainset")
        .iter()
        .map(|row| {
            (
                row["question"].as_str().unwrap().to_owned(),
                row["answer"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    let profiles = fixture["minibatch_profiles"]
        .as_object()
        .expect("minibatch profiles")
        .iter()
        .map(|(marker, questions)| {
            let questions = questions
                .as_array()
                .expect("profile questions")
                .iter()
                .map(|question| question.as_str().unwrap().to_owned())
                .collect();
            (marker.parse().expect("a numeric marker"), questions)
        })
        .collect();
    (table, profiles)
}

/// dspy's `auto` strings, which the crate takes as a type.
fn preset(case: &Value) -> Option<Auto> {
    match case["auto"].as_str() {
        Some("light") => Some(Auto::Light),
        Some("medium") => Some(Auto::Medium),
        Some("heavy") => Some(Auto::Heavy),
        Some(other) => panic!("unknown auto preset {other}"),
        None => None,
    }
}

#[tokio::test]
async fn runs_the_minibatch_trials_dspy_runs() {
    let fixture = super::fixture();
    let (table, profiles) = dataset(&fixture);
    let trainset: Vec<Example> = table
        .iter()
        .map(|(question, answer)| {
            example! { question: question.clone(), answer: answer.clone() }
                .with_inputs(["question"])
        })
        .collect();

    let cases = fixture["minibatch_cases"]
        .as_array()
        .expect("minibatch cases");
    // Both regimes have to be present, or the file silently stops covering one of them: a preset
    // decides the counts and the valset, where an explicit `minibatch` leaves both alone.
    assert!(
        cases.iter().any(|case| preset(case).is_some())
            && cases.iter().any(|case| preset(case).is_none()),
        "the golden covers only one of auto and explicit minibatch"
    );

    for case in cases {
        let model = coach(&table, &profiles, &[]);
        let mut student = Predict::parse("question -> answer")
            .expect("parses")
            .set_lm(model.clone());

        let mut optimizer = MIPROv2::new(exact_match, model.clone() as Arc<Coach>)
            .seed(case["seed"].as_u64().expect("seed"))
            .minibatch_size(case["minibatch_size"].as_u64().expect("size") as usize)
            .minibatch_full_eval_steps(
                case["minibatch_full_eval_steps"].as_u64().expect("steps") as usize
            )
            .max_bootstrapped_demos(case["max_bootstrapped_demos"].as_u64().expect("boot") as usize)
            .max_labeled_demos(case["max_labeled_demos"].as_u64().expect("labeled") as usize)
            // The golden's own `data_aware_proposer=False`.
            .data_aware_proposer(false);
        // Mutually exclusive upstream, which raises on the pair — so a case carries one or the
        // other, and the generator's counts are what the non-preset cases get.
        optimizer = match preset(case) {
            Some(auto) => optimizer.auto(auto),
            None => optimizer.num_candidates(6).num_trials(9).minibatch(true),
        };

        let valset = case["valset_given"]
            .as_bool()
            .expect("valset_given")
            .then_some(trainset.as_slice());
        let trials = optimizer
            .compile_traced(&mut student, &trainset, valset)
            .await
            .expect("compiles");

        agrees_with(case, &model, &trials, &student);
    }
}

/// Every pass a watcher was told about, in the order the search ran them.
#[derive(Default)]
struct Passes(Mutex<Vec<Option<Pass>>>);

impl Callback for Passes {
    fn on_evaluate_start(&self, _call: &CallId, _rows: usize, _threads: usize, pass: Option<Pass>) {
        self.0.lock().expect("not poisoned").push(pass);
    }
}

/// dspy's `callback_metadata`: a watcher is told which pass each scoring is.
///
/// A minibatch search alternates two kinds of evaluation that mean different things — a subsample,
/// whose score moves no winner, and the whole valset, which does. Upstream tells a handler which is
/// which by filling `callback_metadata` with `eval_full` or `eval_minibatch`, and the distinction
/// is invisible from `rows` alone: a subsample the size of the valset is a full pass, and the
/// schedule decides when one happens rather than the count.
///
/// Run on the same 140-row set as the cases above, because the distinction only exists on a valset
/// big enough for a minibatch to be smaller than it.
#[tokio::test]
async fn a_search_tells_a_watcher_which_pass_each_scoring_is() {
    let fixture = super::fixture();
    let (table, profiles) = dataset(&fixture);
    let trainset: Vec<Example> = table
        .iter()
        .map(|(question, answer)| {
            example! { question: question.clone(), answer: answer.clone() }
                .with_inputs(["question"])
        })
        .collect();
    let model = coach(&table, &profiles, &[]);
    let mut student = Predict::parse("question -> answer")
        .expect("parses")
        .set_lm(model.clone());

    let passes = Arc::new(Passes::default());
    watched_by(passes.clone())
        .run(async {
            MIPROv2::new(exact_match, model.clone() as Arc<Coach>)
                .seed(0)
                .minibatch_size(10)
                .minibatch_full_eval_steps(2)
                .max_bootstrapped_demos(0)
                .max_labeled_demos(0)
                .data_aware_proposer(false)
                .num_candidates(2)
                .num_trials(4)
                .minibatch(true)
                .compile_traced(&mut student, &trainset, Some(trainset.as_slice()))
                .await
                .expect("compiles");
        })
        .await;

    let seen = passes.0.lock().expect("not poisoned").clone();
    assert!(
        seen.contains(&Some(Pass::Minibatch)),
        "a minibatch trial told nobody it was one: {seen:?}"
    );
    assert!(
        seen.contains(&Some(Pass::Full)),
        "no scoring announced itself as a full pass: {seen:?}"
    );
}
