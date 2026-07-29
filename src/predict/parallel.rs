//! dspy `Parallel` (`predict/parallel.py`): several branches asked at once.
//!
//! Upstream uses a thread pool because a Python model call blocks. Nothing blocks here, so the
//! same shape is reached by polling many calls on one task: bounded the way `num_threads` bounds
//! that pool, and without spawning, because this crate takes no runtime and forcing one on a
//! caller is a promise it has declined to make.
//!
//! Three behaviours are the ones worth matching, and each is a decision rather than an accident:
//! results come back in the order they were asked for however they finish, a branch that fails
//! leaves a hole beside its siblings rather than sinking them, and enough failures abandon the
//! whole call rather than returning a mostly-empty answer.

use anyhow::{Result, anyhow};
use futures_util::StreamExt;
use futures_util::stream;

use crate::example::{Example, Prediction};
use crate::module::Module;

mod answered;
pub use answered::Answered;

/// dspy `Parallel`'s `timeout`, 120 seconds. See [`Parallel::timeout`].
const DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Ask several branches at once.
///
/// The fields are dspy's `Parallel.__init__` arguments, under the names that read in Rust. The two
/// that steer a latency hedge upstream — [`timeout`](Self::timeout) and
/// [`straggler_limit`](Self::straggler_limit) — are carried but not acted on, for the reason each
/// documents; every other field changes what the call does or returns.
pub struct Parallel {
    /// How many calls may be in flight, dspy's `num_threads`. Its default is 8.
    pub max_in_flight: usize,
    /// How many branches may fail before the call is abandoned. dspy reads `settings.max_errors`,
    /// whose default is 10.
    pub max_errors: usize,
    /// Whether a branch is asked with an example's input fields alone, dspy's `access_examples`.
    ///
    /// True — the default — strips each example to the fields it marked as inputs, upstream's
    /// `module(**example.inputs())`. False asks with the whole example, non-input fields included,
    /// upstream's `module(example)`.
    pub access_examples: bool,
    /// Whether [`Answered`] carries the branches that failed, dspy's `return_failed_examples`.
    ///
    /// False — the default — leaves [`Answered::failed`] empty, upstream returning results alone.
    /// The failures are still counted against [`max_errors`](Self::max_errors) either way; this
    /// governs only whether the caller is handed them.
    pub return_failed_examples: bool,
    /// Whether a failed branch logs its whole error chain rather than one line, dspy's
    /// `provide_traceback`.
    pub provide_traceback: bool,
    /// Whether the per-branch progress line is silenced, dspy's `disable_progress_bar`.
    ///
    /// Upstream draws a `tqdm` bar; a library draws nothing to a terminal it does not own, so the
    /// progress is a `tracing` line instead, and this silences it.
    pub disable_progress_bar: bool,
    /// How long a straggler runs before upstream resubmits it, dspy's `timeout` (120s).
    ///
    /// Carried, not acted on. dspy hedges a slow branch by resubmitting it once when few remain
    /// and taking whichever copy finishes first — a latency optimisation that returns the *same*
    /// result, since both copies compute it. Acting on it needs a timer, and a timer needs a
    /// runtime this crate declines to force on a caller. Kept so the constructor matches upstream
    /// and a runtime-aware layer above could honour it; its absence changes latency, never output.
    pub timeout: std::time::Duration,
    /// How few branches must remain before upstream watches for a straggler, dspy's
    /// `straggler_limit` (3). Carried for the same reason as [`timeout`](Self::timeout).
    pub straggler_limit: usize,
}

impl Default for Parallel {
    fn default() -> Self {
        Self {
            max_in_flight: 8,
            max_errors: 10,
            access_examples: true,
            return_failed_examples: false,
            provide_traceback: false,
            disable_progress_bar: false,
            timeout: DEFAULT_TIMEOUT,
            straggler_limit: 3,
        }
    }
}

impl Parallel {
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            max_in_flight,
            ..Default::default()
        }
    }

    /// Ask each module its own question, all at once.
    ///
    /// Answers sit where their question sat, so a caller reads them against the list it passed
    /// rather than against the order they arrived. A branch that failed is `None` there: dspy
    /// catches an exception per branch and leaves that slot empty rather than letting one failure
    /// take its siblings with it.
    ///
    /// The whole call fails once [`max_errors`](Self::max_errors) branches have — upstream
    /// cancelling its pool and raising. It stops pulling new work at that point rather than asking
    /// every remaining branch first, so the failure costs no more calls than it must. Partial
    /// results are discarded on that path, deliberately: a caller reading half an answer as a
    /// whole one is the failure this prevents.
    pub async fn run<'a>(
        &self,
        work: impl IntoIterator<Item = (&'a dyn Module, Example)>,
    ) -> Result<Answered> {
        let asked: Vec<(&dyn Module, Example)> = work.into_iter().collect();
        let total = asked.len();
        let access = self.access_examples;

        let mut answers = stream::iter(asked.into_iter().enumerate())
            .map(|(at, (module, example))| async move {
                // `access_examples`: the whole example, or only the fields it marked as inputs.
                let prepared = if access {
                    example.inputs()
                } else {
                    Ok(example)
                };
                let answer = match prepared {
                    Ok(inputs) => module.forward(inputs).await,
                    Err(error) => Err(error),
                };
                (at, answer)
            })
            // dspy bounds its pool rather than starting everything; a hundred branches against a
            // paid provider is a bill, not a speedup.
            .buffer_unordered(self.max_in_flight.max(1));

        let mut ordered: Vec<Option<Prediction>> = (0..total).map(|_| None).collect();
        let mut failed = Vec::new();
        let mut failures = 0;
        let mut done = 0;
        while let Some((at, answer)) = answers.next().await {
            match answer {
                Ok(prediction) => ordered[at] = Some(prediction),
                Err(error) => {
                    failures += 1;
                    self.log_failure(at, &error);
                    if self.return_failed_examples {
                        failed.push((at, error));
                    }
                }
            }
            done += 1;
            if !self.disable_progress_bar {
                tracing::debug!(done, total, "parallel progress");
            }
            // Abandon as upstream does, the moment the budget is spent, rather than asking the
            // branches still queued behind this one.
            if failures >= self.max_errors {
                return Err(anyhow!(
                    "parallel call abandoned: {failures} of {total} branches failed"
                ));
            }
        }

        Ok(Answered {
            results: ordered,
            failed,
        })
    }

    /// One failed branch, logged with as much of its cause as `provide_traceback` asks for.
    fn log_failure(&self, branch: usize, error: &anyhow::Error) {
        if self.provide_traceback {
            tracing::error!(cause = ?error, branch, "a branch of a parallel call failed");
        } else {
            tracing::error!(%error, branch, "a branch of a parallel call failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use crate::optimize::scripted::{Answers, Solver};
    use serde_json::json;

    fn asked(question: &str) -> Example {
        example! { question: question }.with_inputs(["question"])
    }

    /// Failures beside the results, which is the shape most tests want to read.
    fn reporting(max_in_flight: usize) -> Parallel {
        Parallel {
            return_failed_examples: true,
            ..Parallel::new(max_in_flight)
        }
    }

    /// Answers sit where their questions sat, whatever order they finished in.
    #[tokio::test]
    async fn answers_come_back_where_their_questions_were() {
        let solvers: Vec<Solver> = (0..4).map(|_| Solver::new(Answers::Correctly)).collect();
        let work: Vec<(&dyn Module, Example)> = solvers
            .iter()
            .map(|s| s as &dyn Module)
            .zip([
                asked("capital of France?"),
                asked("capital of Germany?"),
                asked("capital of France?"),
                asked("capital of Germany?"),
            ])
            .collect();

        let answers = Parallel::new(4).run(work).await.expect("runs");
        let read: Vec<&str> = answers
            .results
            .iter()
            .map(|a| a.as_ref().unwrap().get("answer").unwrap().as_str().unwrap())
            .collect();
        assert_eq!(read, ["Paris", "Berlin", "Paris", "Berlin"]);
    }

    /// One branch failing leaves a hole rather than taking its siblings with it.
    #[tokio::test]
    async fn a_failed_branch_leaves_a_hole_beside_the_others() {
        let good = Solver::new(Answers::Correctly);
        let bad = Solver::new(Answers::Failing);
        let work: Vec<(&dyn Module, Example)> = vec![
            (&good, asked("capital of France?")),
            (&bad, asked("capital of Germany?")),
            (&good, asked("capital of Germany?")),
        ];

        let answers = reporting(8).run(work).await.expect("runs");
        assert!(answers.results[0].is_some());
        assert!(answers.results[1].is_none(), "the failed branch is a hole");
        assert!(answers.results[2].is_some(), "a sibling is unaffected");

        // A hole says a branch failed and nothing else; `return_failed_examples` hands back the
        // reason, and reading it out of the log is not the same thing.
        assert_eq!(answers.failed.len(), 1);
        assert_eq!(answers.failed[0].0, 1, "named by the branch that failed");
        assert!(
            answers
                .failure(1)
                .expect("the reason survived")
                .to_string()
                .contains("provider is down")
        );
        assert!(
            answers.failure(0).is_none(),
            "a branch that worked has none"
        );
    }

    /// The failures are withheld by default, which is upstream returning results alone, and the
    /// count still drives `max_errors` regardless.
    #[tokio::test]
    async fn failures_are_withheld_unless_asked_for() {
        let good = Solver::new(Answers::Correctly);
        let bad = Solver::new(Answers::Failing);
        let work: Vec<(&dyn Module, Example)> = vec![(&good, asked("q")), (&bad, asked("q"))];

        let answers = Parallel::default().run(work).await.expect("runs");
        assert!(answers.results[1].is_none(), "the hole is still there");
        assert!(
            answers.failed.is_empty(),
            "but the reason is withheld until return_failed_examples asks for it"
        );
    }

    /// `access_examples` strips a branch to its input fields, so a non-input field a caller
    /// carried for its own bookkeeping does not reach the module.
    #[tokio::test]
    async fn access_examples_asks_with_the_input_fields_alone() {
        let seen = FieldsSeen::default();
        let carrying =
            example! { question: "capital of France?", gold: "Paris" }.with_inputs(["question"]);
        let work: Vec<(&dyn Module, Example)> = vec![(&seen, carrying.clone())];

        // Default (true): only `question` reaches the module.
        Parallel::default().run(work).await.expect("runs");
        assert_eq!(seen.last(), vec!["question".to_owned()]);

        // False: the whole example, `gold` included.
        let seen = FieldsSeen::default();
        let work: Vec<(&dyn Module, Example)> = vec![(&seen, carrying)];
        Parallel {
            access_examples: false,
            ..Parallel::default()
        }
        .run(work)
        .await
        .expect("runs");
        assert_eq!(
            seen.last(),
            vec!["question".to_owned(), "gold".to_owned()],
            "the whole example, in the order its fields were declared"
        );
    }

    /// Enough failures abandon the call, rather than handing back a mostly-empty answer.
    #[tokio::test]
    async fn enough_failures_abandon_the_whole_call() {
        let bad = Solver::new(Answers::Failing);
        let work: Vec<(&dyn Module, Example)> =
            (0..3).map(|_| (&bad as &dyn Module, asked("q"))).collect();

        let refused = Parallel {
            max_errors: 2,
            ..Default::default()
        }
        .run(work)
        .await;
        assert!(
            refused.is_err(),
            "two failures should have abandoned the call"
        );
    }

    /// One in flight is still every branch, just one at a time — the bound is a bill, not a plan.
    #[tokio::test]
    async fn a_bound_of_one_still_answers_everything() {
        let solver = Solver::new(Answers::Correctly);
        let work: Vec<(&dyn Module, Example)> = (0..3)
            .map(|_| (&solver as &dyn Module, asked("capital of France?")))
            .collect();

        let answers = Parallel::new(1).run(work).await.expect("runs");
        assert_eq!(answers.results.iter().filter(|a| a.is_some()).count(), 3);
    }

    /// A module that records the field names each call handed it, so `access_examples` is
    /// observable rather than inferred.
    #[derive(Default)]
    struct FieldsSeen(std::sync::Mutex<Vec<Vec<String>>>);

    impl FieldsSeen {
        fn last(&self) -> Vec<String> {
            self.0
                .lock()
                .expect("not poisoned")
                .last()
                .cloned()
                .unwrap_or_default()
        }
    }

    impl Module for FieldsSeen {
        fn forward<'a>(
            &'a self,
            inputs: Example,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
            let names: Vec<String> = inputs.fields().map(|(name, _)| name.to_owned()).collect();
            self.0.lock().expect("not poisoned").push(names);
            Box::pin(async move {
                Ok(Prediction::new(
                    Example::new([("answer", json!("ok"))]),
                    "raw",
                ))
            })
        }
    }
}
