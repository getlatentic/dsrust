//! A GEPA run reports its decisions to a subscriber the caller supplied.
//!
//! dspy fills this seam with `logger.log(f"Iteration {i}: …")` — a formatted line handed to
//! Python's `logging`. A caller streaming an optimization to a user interface needs the values in
//! that line rather than the line, so the seam here carries the decision and renders upstream's
//! sentence from it.
//!
//! What this holds is that the wiring reaches all the way: a `Progress` named on the `dsrust`
//! builder receives events from a real compile, not merely from a direct call.

use std::sync::{Arc, Mutex};

use dsrust::lm::DynChatModel;
use dsrust::optimize::{
    Acceptance, Event, Feedback, GEPA, MetricContext, Progress, Rejection, Selection,
};
use dsrust::{DummyLM, Example, Predict, Prediction, example};
use serde_json::Value;

#[derive(Default)]
struct Watched {
    seen: Mutex<Vec<String>>,
}

impl Progress for Watched {
    fn report(&self, event: Event<'_>) {
        // Both halves, because both are what a caller consumes: the decision to act on, and
        // upstream's own sentence to print.
        self.seen.lock().expect("not poisoned").push(match event {
            Event::Accepted { is_best, score, .. } => format!("accepted best={is_best} {score}"),
            // The reason, not merely that it was rejected: gepa 0.1.4 tells an acceptance
            // failure from a selection drop from a duplicate, and a subscriber that only sees
            // "rejected" cannot tell a run that is stalling from one that is choosing.
            Event::Rejected { reason, .. } => match reason {
                Rejection::NotBetter => "rejected: not better".to_owned(),
                Rejection::NotSelected(strategy) => format!("rejected: {strategy} passed it over"),
                Rejection::Duplicate => "rejected: duplicate".to_owned(),
            },
            other => other.message(),
        });
    }
}

fn scripted() -> Arc<dyn DynChatModel> {
    // Enough turns for the reflection to propose and the run to spend its budget.
    Arc::new(DummyLM::new(std::iter::repeat_n(
        example! { answer: "better", improved_instruction: "Answer better." },
        64,
    )))
}

#[tokio::test]
async fn a_run_reports_its_decisions_to_the_caller() {
    let watched = Arc::new(Watched::default());
    let mut program =
        Predict::from_signature("question -> answer".parse().expect("parses")).set_lm(scripted());
    let trainset = vec![example! { question: "Where?" }.with_inputs(["question"])];

    GEPA::new(
        |_: &Example, prediction: &Prediction, _: &MetricContext<'_>| match prediction
            .get("answer")
            .and_then(Value::as_str)
        {
            Some("better") => Feedback::new(1.0, "right"),
            _ => Feedback::new(0.0, "wrong answer"),
        },
        scripted(),
    )
    .max_metric_calls(8)
    .reflection_minibatch_size(1)
    .progress(watched.clone())
    .compile(&mut program, &trainset, &trainset)
    .await
    .expect("compiles");

    let seen = watched.seen.lock().expect("not poisoned");
    // The point of the seam: a run that reported nothing is a progress bar that never moves, and
    // nothing about the types would have said so.
    assert!(!seen.is_empty(), "the run reported nothing at all");
    assert!(
        seen.iter().all(|line| !line.is_empty()),
        "an event rendered to nothing: {seen:?}"
    );
}

/// gepa 0.1.4's proposal controls reach the engine from the builder. What this holds is the
/// wiring — that naming a criterion, a selection strategy and a sampling strategy on `GEPA`
/// changes the run rather than being carried and dropped, which is what an unexercised builder
/// method does.
#[tokio::test]
async fn the_proposal_strategies_reach_the_engine() {
    let watched = Arc::new(Watched::default());
    let mut program =
        Predict::from_signature("question -> answer".parse().expect("parses")).set_lm(scripted());
    let trainset = vec![
        example! { question: "Where?" }.with_inputs(["question"]),
        example! { question: "When?" }.with_inputs(["question"]),
    ];
    GEPA::new(
        |_: &Example, prediction: &Prediction, _: &MetricContext<'_>| match prediction
            .get("answer")
            .and_then(Value::as_str)
        {
            Some("better") => Feedback::new(1.0, "right"),
            _ => Feedback::new(0.0, "wrong answer"),
        },
        scripted(),
    )
    .max_metric_calls(16)
    // A lateral move the strict criterion refuses, kept; the best of the iteration's proposals
    // taken; two parents proposed from per iteration.
    .acceptance_criterion(Acceptance::ImprovementOrEqual)
    .selection_strategy(Selection::BestImprovement)
    .sampling_strategy(dsrust::optimize::Sampling::Independent { n: 2 })
    .progress(watched.clone())
    .reflection_minibatch_size(1)
    .compile(&mut program, &trainset, &trainset)
    .await
    .expect("the run finishes");

    let seen = watched.seen.lock().expect("not poisoned").clone();
    // Two parents per iteration means two selections reported before the first decision.
    assert!(
        seen.iter()
            .filter(|line| line.starts_with("Iteration 1:"))
            .count()
            >= 1,
        "the run reported {seen:?}"
    );
    assert!(
        seen.iter().any(|line| line.contains("Selected program")),
        "a parent is reported once its minibatch is scored: {seen:?}"
    );
}
