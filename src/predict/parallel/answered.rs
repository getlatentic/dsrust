//! What a [`Parallel`](super::Parallel) call answered with.

use crate::example::Prediction;

/// The results, and — when asked for — the branches that failed.
///
/// The failures travel beside the results rather than only into the log, because a hole in
/// `results` says a branch failed and nothing else — and "why" is the question a caller has next.
/// dspy hands back the same pairing through `return_failed_examples=True`, which answers with the
/// results, the examples that failed, and the exceptions they raised.
pub struct Answered {
    /// Each branch's answer where its question sat, and `None` where the branch failed.
    pub results: Vec<Option<Prediction>>,
    /// Which branches failed and why, in the order they were asked — empty unless
    /// [`return_failed_examples`](super::Parallel::return_failed_examples) asked for them.
    pub failed: Vec<(usize, anyhow::Error)>,
}

impl Answered {
    /// Just the answers, for a caller that has no use for the failures.
    pub fn into_results(self) -> Vec<Option<Prediction>> {
        self.results
    }

    /// The error a given branch failed with, if it did and it was kept.
    pub fn failure(&self, branch: usize) -> Option<&anyhow::Error> {
        self.failed
            .iter()
            .find(|(at, _)| *at == branch)
            .map(|(_, error)| error)
    }
}
