//! `BootstrapFewShotWithOptuna` against a dspy run whose sampler was pinned down.
//!
//! dspy creates its study with no sampler, so upstream's own runs are entropy-seeded and none of
//! them agree with each other. The golden's run injected `TPESampler(seed=0)` and changed nothing
//! else, which isolates the part there *is* an answer for: which demos are on offer, how a trial's
//! indices become a program, what that program scores, and which trial wins.

use std::sync::Arc;

use dsrust::optimize::BootstrapFewShotWithOptuna;
use dsrust::predict::Predict;
use dsrust::{DummyLM, Example, example};
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("conformance/optimize/optuna_bootstrap.json"))
        .expect("the optuna golden is valid JSON")
}

const CAPITALS: [(&str, &str); 6] = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Japan", "Tokyo"),
    ("Peru", "Lima"),
];

fn dataset() -> Vec<Example> {
    CAPITALS
        .iter()
        .map(|(country, capital)| {
            example! {
                question: format!("What is the capital of {country}?"),
                answer: *capital
            }
            .with_inputs(["question"])
        })
        .collect()
}

/// The scripted model the golden ran against.
///
/// It answers the capital of the last country **in [`CAPITALS`] order** that appears anywhere in
/// the prompt — which is the demo's country when that sorts after the question's, and the
/// question's otherwise. That is what makes a trial's score depend on the demo it kept: a demo from
/// early in the list leaves every later question answerable, and one from late in the list spoils
/// the earlier ones. Without something of the shape the search would have nothing to find.
struct LastCountryNamed;

impl dsrust::lm::ChatModel for LastCountryNamed {
    async fn forward(
        &self,
        request: &dsrust::lm::api::LmRequest,
    ) -> anyhow::Result<dsrust::lm::api::LmResponse> {
        let prompt: String = request
            .messages
            .iter()
            .filter_map(|message| message.text())
            .collect::<Vec<_>>()
            .join("\n");
        let answer = CAPITALS
            .iter()
            .rfind(|(country, _)| prompt.contains(&format!("of {country}?")))
            .map_or("unknown", |(_, capital)| *capital);
        Ok(dsrust::lm::api::LmResponse::completions([format!(
            "[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
        )]))
    }
}

#[tokio::test]
async fn the_trials_and_the_winner_are_dspys() {
    let golden = golden();
    let expected: Vec<Vec<usize>> = golden["trials"]
        .as_array()
        .expect("trials")
        .iter()
        .map(|indices| {
            indices
                .as_array()
                .expect("a trial")
                .iter()
                .map(|i| i.as_u64().expect("an index") as usize)
                .collect()
        })
        .collect();
    let scores: Vec<f64> = golden["scores"]
        .as_array()
        .expect("scores")
        .iter()
        .map(|s| s.as_f64().expect("a score"))
        .collect();

    let mut student = Predict::parse("question -> answer")
        .expect("parses")
        .set_lm(Arc::new(LastCountryNamed));
    let metric = |example: &Example, prediction: &dsrust::Prediction| -> f64 {
        f64::from(prediction.example.get("answer") == example.get("answer"))
    };
    let trials = BootstrapFewShotWithOptuna::new(&metric)
        .max_labeled_demos(0)
        .num_candidate_programs(expected.len())
        .compile(&mut student, 4, &dataset(), &dataset())
        .await
        .expect("compiles");

    assert_eq!(
        trials.iter().map(|t| t.indices.clone()).collect::<Vec<_>>(),
        expected,
        "the demo indices optuna proposed, in order"
    );
    assert_eq!(
        trials.iter().map(|t| t.score).collect::<Vec<_>>(),
        scores,
        "what each trial's program scored"
    );

    let best = golden["best_trial"].as_u64().expect("best_trial") as usize;
    let kept = golden["demos_kept"]["self"].as_array().expect("demos_kept");
    assert_eq!(
        student.demos.len(),
        kept.len(),
        "dspy kept {} demo(s) on the winning program",
        kept.len()
    );
    assert_eq!(
        student.demos[0].get("question"),
        kept[0].get("question"),
        "the winning trial ({best}) is the one whose demo is kept"
    );
}

/// A predictor the bootstrap taught nothing leaves optuna an empty range, which upstream reaches as
/// `suggest_int(name, 0, -1)` and optuna refuses.
#[tokio::test]
async fn a_predictor_with_no_demos_is_refused_by_name() {
    let mut student = Predict::parse("question -> answer")
        .expect("parses")
        .set_lm(Arc::new(DummyLM::new([example! { answer: "wrong" }])));
    let metric = |_: &Example, _: &dsrust::Prediction| 0.0;
    let refused = BootstrapFewShotWithOptuna::new(&metric)
        .max_labeled_demos(0)
        .compile(&mut student, 4, &dataset(), &dataset())
        .await
        .expect_err("nothing was learned, so there is no range to search");
    let message = format!("{refused}");
    assert!(
        message.contains("self") && message.contains("no demos"),
        "the refusal names the predictor: {message}"
    );
}
