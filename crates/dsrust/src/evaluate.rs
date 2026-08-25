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

use futures_util::StreamExt;

use crate::example::{Example, Prediction};

pub mod dpr;
pub mod metrics;

/// dspy `settings.max_errors`: how many rows may fail before a run gives up.
pub const DEFAULT_MAX_ERRORS: usize = 10;

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
    /// How many rows are in flight at once — dspy's `Evaluate(num_threads=…)`, and `None` for one
    /// at a time as upstream's default is. See [`num_threads`](Self::num_threads).
    pub num_threads: Option<usize>,
    /// How many rows may fail before the run gives up — dspy's `Evaluate(max_errors=…)`, whose
    /// default comes from `dspy.settings.max_errors = 10`. See [`max_errors`](Self::max_errors).
    pub max_errors: usize,
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
            num_threads: None,
            max_errors: DEFAULT_MAX_ERRORS,
        }
    }

    /// Run this many rows at once — dspy's `Evaluate(num_threads=…)`.
    ///
    /// One at a time otherwise, which is upstream's default (`num_threads=None`). A devset of
    /// hundreds against a hosted model is dominated by waiting, so this is most of the wall clock an
    /// optimizer spends.
    ///
    /// Safe to raise now in a way it was not before: a rate limit is retried with dspy's own backoff
    /// (see [`retry`](crate::lm::retry)), so a run that asks too fast slows down rather than failing.
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads.max(1));
        self
    }

    /// How many rows may fail before the run gives up — dspy's `Evaluate(max_errors=…)`.
    ///
    /// A scored failure is still a failure. Without a bound, a devset run against a provider that is
    /// simply down scores `failure_score` five hundred times and reports the mean as a result, which
    /// is the shape of a number that gets believed. `usize::MAX` restores the old behaviour of
    /// scoring everything.
    pub fn max_errors(mut self, max_errors: usize) -> Self {
        self.max_errors = max_errors;
        self
    }

    /// Score every example, and report the rows in devset order however many ran at once.
    ///
    /// Order is `buffered` rather than `buffer_unordered`: a caller reads `results[i]` against
    /// `devset[i]`, and dspy's own results are aligned the same way.
    pub async fn run(&self) -> Evaluation {
        let threads = self.num_threads.unwrap_or(1);
        let watch = crate::observe::evaluating(self.devset.len(), threads);
        let scoring = futures_util::stream::iter(self.devset.clone())
            .map(|example| self.score_row(example))
            .buffered(threads);
        // dspy stops the run at `max_errors`, so the rows after the cap are never asked. `take_while`
        // is that: it ends the stream on the row that reaches the cap, and `buffered` stops pulling.
        // The failed rows up to and including it are kept, which is what makes the report readable.
        let failures = std::sync::atomic::AtomicUsize::new(0);
        let cap = self.max_errors;
        let bounded = scoring.take_while(move |row| {
            let seen = match row.failed() {
                true => failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1,
                false => failures.load(std::sync::atomic::Ordering::Relaxed),
            };
            std::future::ready(seen <= cap)
        });
        let results: Vec<Scored> =
            crate::observe::evaluated_within(&watch, bounded.collect()).await;
        let score = match results.is_empty() {
            true => 0.0,
            false => results.iter().map(|row| row.score).sum::<f64>() / results.len() as f64,
        };
        let evaluation = Evaluation { results, score };
        // No error arm: a run scores every row and a failing one scores `failure_score`, which is
        // dspy's choice too. So the span records what it found and never an exception.
        crate::observe::scored(&watch, &evaluation);
        evaluation
    }

    /// One example, run and scored.
    async fn score_row(&self, example: Example) -> Scored {
        // An example whose split was never declared is a devset mistake, not a program failure. It
        // scores like a failure but says so, rather than handing the program an empty input set and
        // reporting the resulting zero as a model problem.
        let inputs = match example.inputs() {
            Ok(inputs) => inputs,
            Err(error) => {
                return Scored {
                    example,
                    prediction: Err(format!("{error:#}")),
                    score: self.failure_score,
                };
            }
        };
        let (prediction, score) = match (self.program)(inputs).await {
            Ok(prediction) => {
                let score = (self.metric)(&example, &prediction);
                (Ok(prediction), score)
            }
            Err(error) => (Err(format!("{error:#}")), self.failure_score),
        };
        Scored {
            example,
            prediction,
            score,
        }
    }
}

/// A metric for the common case: every label field must match the prediction exactly.
/// dspy `Evaluate.score`: the metric's mean as a percentage, rounded to two places.
///
/// Not the 0..1 mean [`Evaluation::score`] carries — upstream reports a percentage, and it reaches
/// a model in COPRO's attempts block, so the rounding is part of the bytes.
pub fn percent(mean: f64) -> f64 {
    (10_000.0 * mean).round_ties_even() / 100.0
}

pub fn exact_match(example: &Example, prediction: &Prediction) -> f64 {
    // An undeclared example cannot be scored; the runner reports that as a row failure, so
    // reaching here with one means scoring nothing rather than everything.
    let Ok(labels) = example.labels() else {
        return 0.0;
    };
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn devset() -> Vec<Example> {
        vec![
            example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
            example! { question: "capital of Germany?", answer: "Berlin" }
                .with_inputs(["question"]),
        ]
    }

    fn answering(
        answer: &'static str,
    ) -> impl Fn(Example) -> std::future::Ready<anyhow::Result<Prediction>> {
        move |_| {
            std::future::ready(Ok(Prediction::new(
                Example::new([("answer", json!(answer))]),
                "raw",
            )))
        }
    }

    /// dspy `num_threads`: rows really do overlap, and a stored-but-unused knob would look the same
    /// from the outside. The program records how many calls are in flight and the high-water mark is
    /// what is asserted — a sequential runner never gets past one.
    #[tokio::test]
    async fn rows_run_concurrently_up_to_num_threads() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let (counting, highest) = (Arc::clone(&in_flight), Arc::clone(&peak));
        let program = move |_: Example| {
            let counting = Arc::clone(&counting);
            let highest = Arc::clone(&highest);
            async move {
                let now = counting.fetch_add(1, Ordering::SeqCst) + 1;
                highest.fetch_max(now, Ordering::SeqCst);
                tokio::task::yield_now().await;
                counting.fetch_sub(1, Ordering::SeqCst);
                Ok(Prediction::new(
                    Example::new([("answer", json!("Paris"))]),
                    "raw",
                ))
            }
        };

        let devset: Vec<Example> = (0..8)
            .map(|n| {
                example! { question: format!("q{n}"), answer: "Paris" }.with_inputs(["question"])
            })
            .collect();
        Evaluate::new(devset, program, |_: &Example, _: &Prediction| 1.0)
            .num_threads(4)
            .run()
            .await;

        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "num_threads(4) ran one row at a time"
        );
        assert!(
            peak.load(Ordering::SeqCst) <= 4,
            "more rows in flight than asked for: {}",
            peak.load(Ordering::SeqCst)
        );
    }

    /// The rows come back in devset order however many ran at once, because a caller reads
    /// `results[i]` against `devset[i]`.
    #[tokio::test]
    async fn concurrent_rows_are_still_reported_in_order() {
        let devset: Vec<Example> = (0..6)
            .map(|n| {
                example! { question: format!("q{n}"), answer: "Paris" }.with_inputs(["question"])
            })
            .collect();
        let evaluation = Evaluate::new(
            devset.clone(),
            |example: Example| async move {
                // The later rows finish first if nothing preserves order.
                for _ in 0..8 {
                    tokio::task::yield_now().await;
                }
                Ok(Prediction::new(
                    Example::new([("answer", json!(example.get("question").cloned()))]),
                    "raw",
                ))
            },
            |_: &Example, _: &Prediction| 1.0,
        )
        .num_threads(6)
        .run()
        .await;

        let asked: Vec<_> = evaluation
            .results
            .iter()
            .map(|row| row.example.get("question").cloned())
            .collect();
        let expected: Vec<_> = devset
            .iter()
            .map(|row| row.get("question").cloned())
            .collect();
        assert_eq!(asked, expected);
    }

    /// dspy `max_errors`, default 10: a run against a provider that is simply down stops rather than
    /// scoring five hundred zeroes and reporting the mean as a result.
    #[tokio::test]
    async fn a_run_gives_up_once_too_many_rows_fail() {
        let asked = Arc::new(AtomicUsize::new(0));
        let counting = Arc::clone(&asked);
        let devset: Vec<Example> = (0..100)
            .map(|n| example! { question: format!("q{n}"), answer: "x" }.with_inputs(["question"]))
            .collect();

        let evaluation = Evaluate::new(
            devset,
            move |_: Example| {
                counting.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Err(anyhow::anyhow!("the provider is down")))
            },
            |_: &Example, _: &Prediction| 1.0,
        )
        .max_errors(3)
        .run()
        .await;

        assert_eq!(
            evaluation.failure_count(),
            3,
            "the failures up to the cap are kept"
        );
        assert!(
            asked.load(Ordering::SeqCst) < 100,
            "every row was still asked: {}",
            asked.load(Ordering::SeqCst)
        );
    }

    /// The default is dspy's own, from `settings.max_errors`.
    #[test]
    fn the_default_error_budget_is_dspys() {
        assert_eq!(DEFAULT_MAX_ERRORS, 10);
        let evaluation =
            Evaluate::new(Vec::new(), answering("x"), |_: &Example, _: &Prediction| {
                1.0
            });
        assert_eq!(evaluation.max_errors, 10);
    }

    /// A run whose rows merely score badly is not a run that failed, so the budget never fires.
    #[tokio::test]
    async fn a_low_scoring_run_is_not_a_failing_one() {
        let devset: Vec<Example> = (0..40)
            .map(|n| {
                example! { question: format!("q{n}"), answer: "right" }.with_inputs(["question"])
            })
            .collect();
        let evaluation = Evaluate::new(devset, answering("wrong"), exact_match)
            .max_errors(1)
            .run()
            .await;
        assert_eq!(evaluation.results.len(), 40, "every row ran");
        assert_eq!(evaluation.score, 0.0);
        assert_eq!(evaluation.failure_count(), 0);
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
            assert!(
                example.get("answer").is_none(),
                "the label must not reach the program"
            );
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
