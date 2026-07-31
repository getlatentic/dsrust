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
use std::sync::Arc;

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
        let model = coach(&table, &profiles);
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
            .max_labeled_demos(case["max_labeled_demos"].as_u64().expect("labeled") as usize);
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
