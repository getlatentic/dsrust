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
use dsrust::optimize::{Event, Feedback, GEPA, MetricContext, Progress};
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
            Event::Rejected { .. } => "rejected".to_owned(),
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
