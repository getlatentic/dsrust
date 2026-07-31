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
    /// A MIPROv2 proposing with this model, at upstream's own defaults: `auto="light"`, which
    /// decides the counts, the valset and whether trials run on minibatches.
    ///
    /// Naming [`num_candidates`](Self::num_candidates) or [`num_trials`](Self::num_trials) clears
    /// the preset and runs to the explicit counts instead — ten and twenty here, where dspy has no
    /// default at all and raises for the one it is not given.
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
            auto: Some(super::Auto::Light),
            minibatch: true,
            minibatch_size: 35,
            minibatch_full_eval_steps: 5,
        }
    }

    /// dspy `auto="light" | "medium" | "heavy"`: run to a budget preset rather than to explicit
    /// counts.
    ///
    /// A preset does five things, not two. It picks a candidate count and a validation-set size;
    /// subsamples the valset to that size — off the same generator the bootstrap and the proposer
    /// read, and *before* either, so every later draw moves; turns minibatching on when the
    /// subsample exceeds 50; splits the candidate count in two, halving the instruction budget when
    /// demos are also searched; and derives the trial count from the search space.
    ///
    /// Mutually exclusive with [`num_candidates`](Self::num_candidates) and
    /// [`num_trials`](Self::num_trials), which upstream raises on. Here the last call wins, so the
    /// pair dspy rejects cannot be built.
    pub fn auto(mut self, auto: super::Auto) -> Self {
        self.auto = Some(auto);
        self
    }

    /// dspy `minibatch`: score each trial on a subsample of the valset rather than all of it,
    /// stopping every [`minibatch_full_eval_steps`](Self::minibatch_full_eval_steps) trials to score
    /// the best-averaging combination on the whole thing.
    ///
    /// On by default, as upstream's `compile` argument is. Read only when [`auto`](Self::auto) is
    /// unset: a preset recomputes it from the subsampled valset's size and discards what was asked
    /// for, which is upstream's precedence rather than this crate's.
    ///
    /// Note that the interleaved full evaluations are themselves fed back to the sampler, so this
    /// changes which combinations get tried and not only what a trial costs. On a valset no larger
    /// than [`minibatch_size`](Self::minibatch_size) every trial is still a full pass, but the
    /// winner then moves only on those full evaluations — so a run of fewer than
    /// `minibatch_full_eval_steps + 1` trials compiles the default program.
    pub fn minibatch(mut self, minibatch: bool) -> Self {
        self.minibatch = minibatch;
        self
    }

    /// dspy `minibatch_size`: how many examples one minibatch trial scores on, default 35. A size
    /// at or above the valset makes every trial a full pass — and takes no draw, as upstream's does.
    pub fn minibatch_size(mut self, minibatch_size: usize) -> Self {
        self.minibatch_size = minibatch_size;
        self
    }

    /// dspy `minibatch_full_eval_steps`: how often a full evaluation interrupts the minibatch
    /// trials, default 5.
    pub fn minibatch_full_eval_steps(mut self, minibatch_full_eval_steps: usize) -> Self {
        self.minibatch_full_eval_steps = minibatch_full_eval_steps;
        self
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

    /// How many instructions to propose per predictor. Clears [`auto`](Self::auto), which would
    /// otherwise override it — upstream refuses the pair rather than picking one.
    pub fn num_candidates(mut self, num_candidates: usize) -> Self {
        self.num_candidates = num_candidates;
        self.auto = None;
        self
    }

    /// How many instruction combinations the search evaluates. Clears [`auto`](Self::auto), for the
    /// reason [`num_candidates`](Self::num_candidates) gives.
    pub fn num_trials(mut self, num_trials: usize) -> Self {
        self.num_trials = num_trials;
        self.auto = None;
        self
    }

    /// dspy `tip_aware_proposer`: draw a writing tip per proposal and put it in the proposer's
    /// prompt, on by default as upstream's is.
    ///
    /// Turning it off removes the field *and* the draw, so it changes the whole generator sequence
    /// after it rather than only the prompt.
    pub fn tip_aware_proposer(mut self, tip_aware_proposer: bool) -> Self {
        self.tip_aware = tip_aware_proposer;
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
