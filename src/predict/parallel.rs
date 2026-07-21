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

/// Ask several branches at once.
pub struct Parallel {
    /// How many calls may be in flight, which is dspy's `num_threads`. Its default is 8.
    pub max_in_flight: usize,
    /// How many branches may fail before the call is abandoned. dspy reads
    /// `settings.max_errors`, whose default is 10.
    pub max_errors: usize,
}

impl Default for Parallel {
    fn default() -> Self {
        Self {
            max_in_flight: 8,
            max_errors: 10,
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
    /// catches an exception per branch and leaves that slot empty rather than letting one
    /// failure take its siblings with it.
    ///
    /// The whole call fails once `max_errors` branches have, which is upstream cancelling its
    /// pool and raising. Partial results are discarded on that path, deliberately: a caller
    /// reading half an answer as a whole one is the failure this prevents.
    pub async fn run<'a>(
        &self,
        work: impl IntoIterator<Item = (&'a dyn Module, Example)>,
    ) -> Result<Answered> {
        let asked: Vec<(&dyn Module, Example)> = work.into_iter().collect();
        let total = asked.len();

        let answers = stream::iter(asked.into_iter().enumerate())
            .map(|(at, (module, inputs))| async move { (at, module.forward(inputs).await) })
            // dspy bounds its pool rather than starting everything; a hundred branches against a
            // paid provider is a bill, not a speedup.
            .buffer_unordered(self.max_in_flight.max(1))
            .collect::<Vec<(usize, Result<Prediction>)>>()
            .await;

        let mut ordered: Vec<Option<Prediction>> = (0..total).map(|_| None).collect();
        let mut failures = 0;
        let mut failed = Vec::new();
        for (at, answer) in answers {
            match answer {
                Ok(prediction) => ordered[at] = Some(prediction),
                Err(error) => {
                    failures += 1;
                    tracing::error!(%error, branch = at, "a branch of a parallel call failed");
                    failed.push((at, error));
                }
            }
        }
        if failures >= self.max_errors {
            return Err(anyhow!(
                "parallel call abandoned: {failures} of {total} branches failed"
            ));
        }
        Ok(Answered {
            results: ordered,
            failed,
        })
    }
}

/// What a parallel call answered with.
///
/// The failures travel beside the results rather than only into the log, because a hole in
/// `results` says a branch failed and nothing else — and "why" is the question a caller has next.
/// dspy hands back the same pairing through `batch(return_failed_examples=True)`, which answers
/// with the results, the examples that failed, and the exceptions they raised.
pub struct Answered {
    /// Each branch's answer where its question sat, and `None` where the branch failed.
    pub results: Vec<Option<Prediction>>,
    /// Which branches failed and why, in the order they were asked.
    pub failed: Vec<(usize, anyhow::Error)>,
}

impl Answered {
    /// Just the answers, for a caller that has no use for the failures.
    pub fn into_results(self) -> Vec<Option<Prediction>> {
        self.results
    }

    /// The error a given branch failed with, if it did.
    pub fn failure(&self, branch: usize) -> Option<&anyhow::Error> {
        self.failed
            .iter()
            .find(|(at, _)| *at == branch)
            .map(|(_, error)| error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use crate::optimize::scripted::{Answers, Solver};

    fn asked(question: &str) -> Example {
        example! { question: question }.with_inputs(["question"])
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

        let answers = Parallel::default().run(work).await.expect("runs");
        assert!(answers.results[0].is_some());
        assert!(answers.results[1].is_none(), "the failed branch is a hole");
        assert!(answers.results[2].is_some(), "a sibling is unaffected");

        // A hole says a branch failed and nothing else; upstream's `return_failed_examples`
        // hands back the reason, and reading it out of the log is not the same thing.
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
}
