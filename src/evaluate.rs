//! Score a program over a dataset.
//!
//! This is the layer that turns "the prompt seems fine" into a number, and it is the thing an
//! optimizer searches against — a compiler needs an objective before it can compile anything.
//!
//! dspy's `Evaluate` takes a devset, a metric, and a thread count, and returns an aggregate
//! score plus the per-example results. The same shape works here, with two Rust differences:
//! the metric is a closure rather than a duck-typed callable, and a failing example is an
//! `Err` on that row rather than an exception that may or may not stop the run.

use std::future::Future;

use crate::example::{Example, Prediction};

/// What one example scored, and what produced that score.
///
/// The prediction is kept alongside the number because a bare score cannot be debugged: when
/// a metric returns 0.0 the next question is always "what did the model actually say".
#[derive(Debug, Clone)]
pub struct Scored {
    pub example: Example,
    pub prediction: Result<Prediction, String>,
    pub score: f64,
}

impl Scored {
    pub fn failed(&self) -> bool {
        self.prediction.is_err()
    }
}

/// The outcome of a run: the aggregate, and every row behind it.
#[derive(Debug, Clone)]
pub struct Evaluation {
    pub results: Vec<Scored>,
    /// Mean score across the devset, matching dspy's headline number.
    pub score: f64,
}

impl Evaluation {
    /// Rows whose program call errored rather than merely scoring badly. Worth separating:
    /// a low score is a result, an error is usually a bug or an outage.
    pub fn failures(&self) -> impl Iterator<Item = &Scored> {
        self.results.iter().filter(|row| row.failed())
    }

    pub fn failure_count(&self) -> usize {
        self.failures().count()
    }
}

/// Score a program over a devset.
///
/// `program` receives each example's declared inputs and returns a [`Prediction`]. `metric`
/// scores that prediction against the same example's labels. A program error scores
/// `failure_score` rather than aborting the run, because one bad row should not discard the
/// evidence from every other row — dspy makes the same choice for the same reason.
pub struct Evaluate<P, M> {
    pub devset: Vec<Example>,
    pub program: P,
    pub metric: M,
    /// What an errored row scores. dspy's default is 0.0.
    pub failure_score: f64,
}

impl<P, M, F> Evaluate<P, M>
where
    P: Fn(Example) -> F,
    F: Future<Output = anyhow::Result<Prediction>>,
    M: Fn(&Example, &Prediction) -> f64,
{
    pub fn new(devset: Vec<Example>, program: P, metric: M) -> Self {
        Self {
            devset,
            program,
            metric,
            failure_score: 0.0,
        }
    }

    /// Score every example in order.
    ///
    /// Sequential on purpose for now: a parallel runner is worth having, but it needs a
    /// concurrency limit and a provider-aware backoff to be anything but a way to get rate
    /// limited, and inventing that before the optimizer that needs it would be guesswork.
    pub async fn run(&self) -> Evaluation {
        let mut results = Vec::with_capacity(self.devset.len());
        for example in &self.devset {
            let outcome = (self.program)(example.inputs()).await;
            let (prediction, score) = match outcome {
                Ok(prediction) => {
                    let score = (self.metric)(example, &prediction);
                    (Ok(prediction), score)
                }
                Err(error) => (Err(format!("{error:#}")), self.failure_score),
            };
            results.push(Scored {
                example: example.clone(),
                prediction,
                score,
            });
        }
        let score = match results.is_empty() {
            true => 0.0,
            false => results.iter().map(|row| row.score).sum::<f64>() / results.len() as f64,
        };
        Evaluation { results, score }
    }
}

/// A metric for the common case: every label field must match the prediction exactly.
pub fn exact_match(example: &Example, prediction: &Prediction) -> f64 {
    let labels = example.labels();
    if labels.is_empty() {
        return 0.0;
    }
    let matched = labels
        .fields()
        .filter(|(name, expected)| prediction.get(name) == Some(*expected))
        .count();
    match matched == labels.len() {
        true => 1.0,
        false => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use serde_json::json;

    fn devset() -> Vec<Example> {
        vec![
            example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
            example! { question: "capital of Germany?", answer: "Berlin" }.with_inputs(["question"]),
        ]
    }

    fn answering(answer: &'static str) -> impl Fn(Example) -> std::future::Ready<anyhow::Result<Prediction>> {
        move |_| {
            std::future::ready(Ok(Prediction::new(
                Example::new([("answer", json!(answer))]),
                "raw",
            )))
        }
    }

    #[tokio::test]
    async fn a_program_that_is_always_right_scores_one() {
        let program = |example: Example| {
            let answer = match example.get("question").and_then(|q| q.as_str()) {
                Some(q) if q.contains("France") => "Paris",
                _ => "Berlin",
            };
            std::future::ready(Ok(Prediction::new(
                Example::new([("answer", json!(answer))]),
                "raw",
            )))
        };
        let evaluation = Evaluate::new(devset(), program, exact_match).run().await;
        assert_eq!(evaluation.score, 1.0);
        assert_eq!(evaluation.failure_count(), 0);
    }

    #[tokio::test]
    async fn a_half_right_program_scores_one_half() {
        let evaluation = Evaluate::new(devset(), answering("Paris"), exact_match)
            .run()
            .await;
        assert_eq!(evaluation.score, 0.5);
        assert_eq!(evaluation.results.len(), 2);
    }

    #[tokio::test]
    async fn an_erroring_row_scores_the_failure_score_and_keeps_the_rest() {
        let program = |example: Example| {
            let france = example
                .get("question")
                .and_then(|q| q.as_str())
                .is_some_and(|q| q.contains("France"));
            std::future::ready(match france {
                true => Err(anyhow::anyhow!("provider timed out")),
                false => Ok(Prediction::new(
                    Example::new([("answer", json!("Berlin"))]),
                    "raw",
                )),
            })
        };
        let evaluation = Evaluate::new(devset(), program, exact_match).run().await;

        // One row errored, the other still scored: an outage must not discard the evidence.
        assert_eq!(evaluation.score, 0.5);
        assert_eq!(evaluation.failure_count(), 1);
        assert!(
            evaluation
                .failures()
                .next()
                .unwrap()
                .prediction
                .as_ref()
                .unwrap_err()
                .contains("provider timed out")
        );
    }

    #[tokio::test]
    async fn the_program_only_sees_the_declared_inputs() {
        // Handing the labels to the program would let it score perfectly by reading the answer.
        let program = |example: Example| {
            assert!(example.get("answer").is_none(), "the label must not reach the program");
            std::future::ready(Ok(Prediction::new(
                Example::new([("answer", json!("Paris"))]),
                "raw",
            )))
        };
        let evaluation = Evaluate::new(devset(), program, exact_match).run().await;
        assert_eq!(evaluation.score, 0.5);
    }

    #[tokio::test]
    async fn an_empty_devset_scores_zero_rather_than_dividing_by_nothing() {
        let evaluation = Evaluate::new(Vec::new(), answering("Paris"), exact_match)
            .run()
            .await;
        assert_eq!(evaluation.score, 0.0);
        assert!(evaluation.results.is_empty());
    }
}
