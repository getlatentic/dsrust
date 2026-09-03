//! gepa 0.1.4's proposal controls on the builder: what a proposal must do to be kept, which of an
//! iteration's kept proposals enter the pool, and how many the iteration makes.
//!
//! Their own file because they are one decision a caller makes together and none of them is the
//! optimizer's own configuration — dspy passes all three through `gepa_kwargs`.

use gepa::{Acceptance, Sampling, Selection};

use super::GEPA;
use crate::optimize::gepa::{Feedback, MetricContext};

impl<M> GEPA<M>
where
    M: Fn(&crate::Example, &crate::Prediction, &MetricContext<'_>) -> Feedback + Send + Sync,
{
    /// gepa 0.1.4's `acceptance_criterion`, reached through dspy 3.3.1's `gepa_kwargs`: what a
    /// proposal must do on its minibatch to be kept. Strict improvement by default.
    pub fn acceptance_criterion(mut self, acceptance: Acceptance) -> Self {
        self.acceptance = acceptance;
        self
    }

    /// gepa 0.1.4's `selection_strategy`: which of an iteration's accepted proposals enter the
    /// candidate pool. All of them by default.
    pub fn selection_strategy(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// gepa 0.1.4's `sampling_strategy`: how many (parent, minibatch) tasks each iteration
    /// proposes from. One by default, which is the run gepa always made.
    pub fn sampling_strategy(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }
}
