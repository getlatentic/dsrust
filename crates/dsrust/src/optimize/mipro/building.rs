//! Putting a MIPROv2 together: dspy's constructor arguments, each under its own name.
//!
//! Lifted out of `mod.rs` when that file crossed 400 lines. The split is the one dspy's own class
//! has — `__init__` takes these and `compile` runs the three steps — and nothing here reaches the
//! search.

use std::future::Future;
use std::sync::Arc;

use super::MIPROv2;
use crate::example::{Example, Prediction};
use crate::lm::DynChatModel;

impl<M> MIPROv2<M>
where
    M: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    /// A MIPROv2 proposing with this model. dspy's defaults for the counts are set per auto mode;
    /// here they are explicit — ten instruction candidates and twenty trials is a common medium run.
    pub fn new(metric: M, prompt_model: Arc<dyn DynChatModel>) -> Self {
        Self {
            metric,
            prompt_model,
            num_candidates: 10,
            num_trials: 20,
            seed: 9,
            init_temperature: 1.0,
            metric_threshold: None,
            scoring: crate::optimize::Scoring::default(),
            program_code: None,
            tip_aware: true,
            task_model: None,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 4,
        }
    }

    /// dspy `task_model`: run the program on this model while `prompt_model` writes the proposals.
    ///
    /// Upstream scopes it with `with dspy.context(lm=self.task_model)` around Step 1 and Step 3, so
    /// the bootstrap and the search ask it and the proposer does not — which is the point: a strong
    /// model writes a handful of instructions while a cheap one absorbs the many evaluation calls.
    ///
    /// Unset means the configured model, as upstream's `task_model or settings.lm` does. A
    /// predictor carrying its own model still asks that one, in both: dspy's `Predict.forward`
    /// resolves `self.lm or settings.lm`, so a scope reaches only the predictors that defer.
    pub fn task_model(mut self, task_model: Arc<dyn DynChatModel>) -> Self {
        self.task_model = Some(task_model);
        self
    }

    /// Run `work` on the task model, or as-is when none was named — dspy's
    /// `with dspy.context(lm=self.task_model)`.
    pub(super) async fn on_task_model<T>(&self, work: impl Future<Output = T>) -> T {
        match &self.task_model {
            Some(task_model) => {
                crate::lm::global::context_model(crate::lm::global::client(), task_model.clone())
                    .run(work)
                    .await
            }
            None => work.await,
        }
    }

    /// dspy `max_bootstrapped_demos`: how many bootstrapped demos a candidate set may hold, default
    /// 4 as upstream's is. With `max_labeled_demos` also zero the run is upstream's zero-shot one:
    /// the sets are still built, because they ground the proposer, but they are built at dspy's
    /// in-context constants and never searched.
    pub fn max_bootstrapped_demos(mut self, max_bootstrapped_demos: usize) -> Self {
        self.max_bootstrapped_demos = max_bootstrapped_demos;
        self
    }

    /// dspy `max_labeled_demos`: how many labelled trainset examples a candidate set may hold
    /// without bootstrapping, default 4 as upstream's is.
    pub fn max_labeled_demos(mut self, max_labeled_demos: usize) -> Self {
        self.max_labeled_demos = max_labeled_demos;
        self
    }

    /// Whether this run searches demos at all — dspy's `zeroshot_opt`, and the same test.
    pub(super) fn zeroshot(&self) -> bool {
        self.max_bootstrapped_demos == 0 && self.max_labeled_demos == 0
    }

    /// How many instructions to propose per predictor.
    pub fn num_candidates(mut self, num_candidates: usize) -> Self {
        self.num_candidates = num_candidates;
        self
    }

    /// How many instruction combinations the search evaluates.
    pub fn num_trials(mut self, num_trials: usize) -> Self {
        self.num_trials = num_trials;
        self
    }

    /// dspy `init_temperature`: the temperature instructions are proposed at, default 1.0. Lower it
    /// for proposals that stay close to the current instruction.
    pub fn init_temperature(mut self, temperature: f64) -> Self {
        self.init_temperature = temperature;
        self
    }

    /// dspy `metric_threshold`: the score a bootstrapped trace must beat to be kept in Step 1.
    /// Unset keeps every trace whose metric was truthy, which is upstream's default.
    pub fn metric_threshold(mut self, threshold: f64) -> Self {
        self.metric_threshold = Some(threshold);
        self
    }

    /// What each scoring pass is bounded by — dspy's `num_threads` and `max_errors`, which its
    /// teleprompters take and hand to the `Evaluate` they build.
    pub fn scoring(mut self, scoring: crate::optimize::Scoring) -> Self {
        self.scoring = scoring;
        self
    }

    /// The seed for the proposer's RNG and the TPE sampler — dspy seeds both from one number.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Turn on the program-aware proposer with this pseudo-code description of the program. dspy
    /// reads it from source; a Rust caller supplies it, which is dspy's own `program_code_string` seam.
    pub fn program_code(mut self, program_code: impl Into<String>) -> Self {
        self.program_code = Some(program_code.into());
        self
    }
}
