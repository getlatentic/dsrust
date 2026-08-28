//! Where an evaluation stops being a score and becomes a failure, against dspy's own boundary.
//!
//! Two things are easy to get wrong here and this crate had both. Upstream's `ParallelExecutor`
//! cancels at `error_count >= max_errors`, so a cap of three tolerates *two* failing rows and gives
//! up on the third — the comparison is `>=`, not `>`. And the cancellation **raises**: it propagates
//! out of `Evaluate.__call__` rather than being folded into a partial score, because a run that blew
//! its error budget reporting what it managed to score hands back a number that reads as a result.
//!
//! The golden (`tests/conformance/evaluate/max_errors.json`, see
//! `scripts/generate_evaluate_errors_fixture.py`) was recorded by running the pinned dspy at and
//! around that boundary.

use dsrust::{Evaluate, Example, Prediction, example};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/evaluate/max_errors.json");
    let text = std::fs::read_to_string(&path).expect("the max_errors golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

/// Each case at the boundary: whether dspy raised, and what it scored when it did not.
#[tokio::test]
async fn it_gives_up_where_dspy_gives_up() {
    for case in golden()["cases"].as_array().expect("cases") {
        let cap = case["max_errors"].as_u64().expect("a cap") as usize;
        let failing = case["failing"].as_u64().expect("how many fail") as usize;
        let rows = case["rows"].as_u64().expect("how many rows") as usize;

        let devset: Vec<Example> = (0..rows)
            .map(|at| example! { question: at.to_string(), answer: "x" }.with_inputs(["question"]))
            .collect();
        let seen = AtomicUsize::new(0);
        let evaluation = Evaluate::new(
            devset,
            |_: Example| {
                let at = seen.fetch_add(1, Ordering::SeqCst) + 1;
                std::future::ready(match at <= failing {
                    true => Err(anyhow::anyhow!("boom")),
                    false => Ok(Prediction::new(example! { answer: "x" }, String::new())),
                })
            },
            |_: &Example, _: &Prediction| 1.0,
        )
        .max_errors(cap)
        .run()
        .await;

        let label = format!("max_errors={cap}, {failing} of {rows} rows fail");
        match case["raised"].as_bool().expect("a verdict") {
            true => {
                let refused = evaluation.expect_err(&format!("{label}: dspy raised"));
                assert_eq!(
                    refused.to_string(),
                    case["message"].as_str().expect("the message"),
                    "{label}: the words dspy uses"
                );
            }
            false => {
                let scored = evaluation.unwrap_or_else(|_| panic!("{label}: dspy returned"));
                assert_eq!(
                    scored.results.len(),
                    case["results"].as_u64().expect("results") as usize,
                    "{label}: a tolerated failure still appears in the results"
                );
                assert_eq!(
                    scored.score,
                    case["score"].as_f64().expect("score"),
                    "{label}: and still scores `failure_score`"
                );
            }
        }
    }
}
