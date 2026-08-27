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

use anyhow::{Result, bail};
use futures_util::StreamExt;

use crate::example::{Example, Prediction};

pub mod auto;
pub mod dpr;
pub mod metrics;

/// dspy `settings.max_errors`: how many rows may fail before a run gives up.
pub const DEFAULT_MAX_ERRORS: usize = 10;

/// What scores one row — dspy's metric, which is any callable.
///
/// A plain closure is one through the blanket impl below, so `|example, prediction| …` keeps
/// working and costs a `ready` future per row, which is nothing beside a program call.
///
/// It is a trait rather than an `Fn` bound because upstream's metrics include **LM judges**:
/// `SemanticF1` and `CompleteAndGrounded` are `dspy.Module`s that call a model to score, and a
/// port whose metric could not await would have had them as values a caller cannot pass to
/// `Evaluate` or to any optimizer. dspy needs no such seam — every callable is a metric there —
/// so this is the shape that keeps the *capability* the same rather than the spelling.
pub trait Metric: Send + Sync {
    /// This prediction's score against that example's labels.
    fn score<'a>(
        &'a self,
        example: &'a Example,
        prediction: &'a Prediction,
    ) -> std::pin::Pin<Box<dyn Future<Output = f64> + Send + 'a>>;
}

/// A borrowed metric, which is how an optimizer lends its own to a scoring pass without giving it
/// up.
///
/// A newtype rather than `impl Metric for &M`, which cannot be written: `&F` is itself an `Fn`, so
/// that impl overlaps the closure one below and the compiler refuses both.
pub struct MetricRef<'a, M: ?Sized>(pub &'a M);

impl<M: Metric + ?Sized> Metric for MetricRef<'_, M> {
    fn score<'a>(
        &'a self,
        example: &'a Example,
        prediction: &'a Prediction,
    ) -> std::pin::Pin<Box<dyn Future<Output = f64> + Send + 'a>> {
        self.0.score(example, prediction)
    }
}

impl<F> Metric for F
where
    F: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    fn score<'a>(
        &'a self,
        example: &'a Example,
        prediction: &'a Prediction,
    ) -> std::pin::Pin<Box<dyn Future<Output = f64> + Send + 'a>> {
        Box::pin(std::future::ready(self(example, prediction)))
    }
}

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

/// dspy's `callback_metadata` on `Evaluate.__call__`: which pass of a search this scoring is.
///
/// Upstream hands handlers a free dict and puts one of three things in it — `{"metric_key":
/// "eval_full"}` from MIPROv2's whole-valset pass, `{"metric_key": "eval_minibatch"}` from its
/// subsample, and `{"disable_logging": True}` from GEPA's reflection minibatch. Two of the three
/// reach a handler here, because GEPA scores through its own path rather than through [`Evaluate`];
/// a closed set says which two, where a map would hand a reader keys to guess at.
///
/// Absent for a caller scoring a program directly, which is upstream's `callback_metadata=None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// dspy's `eval_full`: every row of the valset.
    Full,
    /// dspy's `eval_minibatch`: one subsample of it, whose score moves no winner.
    Minibatch,
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
    /// Which pass of a search this is, for a watcher to read — dspy's `callback_metadata`. `None`
    /// for a caller scoring directly. See [`pass`](Self::pass).
    pub pass: Option<Pass>,
}

impl<P, M, F> Evaluate<P, M>
where
    P: Fn(Example) -> F,
    F: Future<Output = anyhow::Result<Prediction>>,
    M: Metric,
{
    pub fn new(devset: Vec<Example>, program: P, metric: M) -> Self {
        Self {
            devset,
            program,
            metric,
            failure_score: 0.0,
            num_threads: None,
            max_errors: DEFAULT_MAX_ERRORS,
            pass: None,
        }
    }

    /// Say which pass of a search this is — dspy's `Evaluate(callback_metadata=…)`.
    ///
    /// Only a watcher reads it; nothing about the scoring changes. It is what lets a handler tell
    /// MIPROv2's whole-valset pass from its subsample, which upstream announces and a run here
    /// makes either way.
    pub fn pass(mut self, pass: Pass) -> Self {
        self.pass = Some(pass);
        self
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

    /// Order is `buffered` rather than `buffer_unordered`: a caller reads `results[i]` against
    /// `devset[i]`, and dspy's own results are aligned the same way.
    /// Score every example, and report the rows in devset order however many ran at once.
    ///
    /// `Err` when the failures reach [`max_errors`](Self::max_errors), which is upstream's
    /// behaviour and not a nicety: `ParallelExecutor` cancels at `error_count >= max_errors` and
    /// the exception propagates out of `Evaluate.__call__`. Returning what was scored instead
    /// hands back a number that reads as a result — the shape of a number that gets believed.
    ///
    /// The boundary is `>=`, so `max_errors` of 3 tolerates two failing rows and gives up on the
    /// third. A tolerated failure still scores `failure_score` and still appears in the results.
    pub async fn run(&self) -> Result<Evaluation> {
        let threads = self.num_threads.unwrap_or(1);
        let watch = crate::observe::evaluating(&self.devset, threads, self.pass);
        let scoring = futures_util::stream::iter(self.devset.clone())
            .map(|example| self.score_row(example))
            .buffered(threads);
        // dspy stops the run at `max_errors`, so the rows after the cap are never asked. `take_while`
        // is that: it ends the stream on the row that reaches the cap, and `buffered` stops pulling.
        // The failed rows up to and including it are kept, which is what makes the report readable.
        // Upstream cancels at `error_count >= max_errors`, so a cap of three tolerates two failing
        // rows and gives up on the third — `< cap`, not `<= cap`. Measured against dspy in
        // `tests/conformance/evaluate/max_errors.json` rather than read off the comparison.
        let failures = std::sync::atomic::AtomicUsize::new(0);
        let cap = self.max_errors;
        let bounded = scoring.take_while(move |row| {
            let seen = match row.failed() {
                true => failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1,
                false => failures.load(std::sync::atomic::Ordering::Relaxed),
            };
            std::future::ready(seen < cap)
        });
        let results: Vec<Scored> =
            crate::observe::evaluated_within(&watch, bounded.collect()).await;
        let evaluated = self.finish(results);
        // Both arms close the point, because upstream's decorator does: a run that gives up reaches
        // `on_evaluate_end` with the exception rather than leaving the handler's start unanswered.
        crate::observe::scored(&watch, evaluated.as_ref());
        evaluated
    }

    /// The devset's score, or the run that abandoned it.
    ///
    /// Short of the devset means the cap stopped the stream, since every row that runs is scored —
    /// a failing one takes `failure_score`.
    fn finish(&self, results: Vec<Scored>) -> Result<Evaluation> {
        if results.len() < self.devset.len() {
            // dspy's own words, so a caller who has read its traceback recognises this one.
            bail!("Execution cancelled due to errors or interruption.");
        }
        // dspy's headline number is `round(100 * ncorrect / ntotal, 2)`, so the aggregate a caller
        // reads is a *percentage* — `50.0`, never `0.5`. Scaling at each call site instead left
        // `Evaluation::score` a hundred times smaller than the field it is mapped to, and every
        // optimizer converting it back by hand through two identical functions.
        let mean = match results.is_empty() {
            true => 0.0,
            false => results.iter().map(|row| row.score).sum::<f64>() / results.len() as f64,
        };
        Ok(Evaluation {
            score: percent(mean),
            results,
        })
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
                let score = self.metric.score(&example, &prediction).await;
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

/// A mean as the percentage dspy reports — `round(mean * 100, 2)`, and the rounding matters.
///
/// The number [`Evaluation::score`] already carries, here for a caller folding [`Scored`] rows into
/// an aggregate of their own. Python rounds half to *even*, so a score landing exactly on a half
/// goes to the nearest even hundredth rather than always up, and a comparison against a number dspy
/// printed will drift on those without it. This one reaches a model in COPRO's attempts block, so
/// the rounding is part of the prompt bytes.
///
/// ```
/// use dsrust::evaluate::percent;
///
/// assert_eq!(percent(2.0 / 3.0), 66.67);
/// // Half-to-even: `.125` of a percent goes down to `.12`, where half-up would give `.13`.
/// assert_eq!(percent(0.00125), 0.12);
/// assert_eq!(percent(0.00375), 0.38);
/// ```
pub fn percent(mean: f64) -> f64 {
    (10_000.0 * mean).round_ties_even() / 100.0
}

/// A metric for the common case: every label field must match the prediction exactly.
///
/// dspy's `answer_exact_match` over all of them at once — one wrong field scores the row zero, and
/// an example with nothing labelled scores zero rather than vacuously matching.
///
/// ```
/// use dsrust::{Prediction, example, evaluate::exact_match};
///
/// let row = example! { question: "capital?", answer: "Paris" }.with_inputs(["question"]);
/// let answered = |answer| Prediction::new(example! { answer: answer }, "raw");
/// assert_eq!(exact_match(&row, &answered("Paris")), 1.0);
/// assert_eq!(exact_match(&row, &answered("Lyon")), 0.0);
/// ```
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
            .await
            .expect("the run stays inside its error budget");

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
        .await
        .expect("the run stays inside its error budget");

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

        // dspy raises rather than reporting what it managed to score, and says so in these words.
        let refused = evaluation.expect_err("a run past its budget is not a score");
        assert_eq!(
            refused.to_string(),
            "Execution cancelled due to errors or interruption."
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
            .await
            .expect("the run stays inside its error budget");
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
        let evaluation = Evaluate::new(devset(), program, exact_match)
            .run()
            .await
            .expect("the run stays inside its error budget");
        assert_eq!(evaluation.score, 100.0);
        assert_eq!(evaluation.failure_count(), 0);
    }

    #[tokio::test]
    async fn a_half_right_program_scores_one_half() {
        let evaluation = Evaluate::new(devset(), answering("Paris"), exact_match)
            .run()
            .await
            .expect("the run stays inside its error budget");
        assert_eq!(evaluation.score, 50.0);
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
        let evaluation = Evaluate::new(devset(), program, exact_match)
            .run()
            .await
            .expect("the run stays inside its error budget");

        // One row errored, the other still scored: an outage must not discard the evidence.
        assert_eq!(evaluation.score, 50.0);
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
        let evaluation = Evaluate::new(devset(), program, exact_match)
            .run()
            .await
            .expect("the run stays inside its error budget");
        assert_eq!(evaluation.score, 50.0);
    }

    #[tokio::test]
    async fn an_empty_devset_scores_zero_rather_than_dividing_by_nothing() {
        let evaluation = Evaluate::new(Vec::new(), answering("Paris"), exact_match)
            .run()
            .await
            .expect("the run stays inside its error budget");
        assert_eq!(evaluation.score, 0.0);
        assert!(evaluation.results.is_empty());
    }
}
