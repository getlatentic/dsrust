//! GEPA in dsrs: instruction optimization by reflective evolution, wrapping the [`gepa`] crate the
//! way dspy's `GEPA` teleprompter wraps the gepa package.
//!
//! The [`gepa`] engine runs the loop (Pareto candidate selection, minibatch sampling, the accept
//! test, the budget); this supplies the LLM work through [`adapter::Adapter`] — running the student
//! program, scoring it with a feedback metric, and rewriting an instruction with a reflection model.
//!
//! This is the reflective-mutation path under dspy's defaults (`reflection_minibatch_size=3`, Pareto
//! selection, `skip_perfect_score`, round-robin components). Merge is deferred, so dspy's `use_merge`
//! default is turned off — the same shape of config boundary MIPROv2's zero-shot path draws.

mod adapter;
mod metric;

#[cfg(test)]
mod conformance;

use std::sync::Arc;

use anyhow::{Result, bail};
use gepa::{Candidate, GepaEngine};

pub use gepa::GepaOutcome;
pub use metric::Feedback;

use adapter::{Adapter, set_instructions};

use super::Optimizer;
use crate::example::{Example, Prediction};
use crate::lm::{DynChatModel, global};
use crate::module::Module;

/// dspy `GEPA`: evolve each predictor's instruction by reflecting on how the program did, keeping the
/// candidates that improve a validation metric.
///
/// The metric returns a [`Feedback`] — a score and the text GEPA reflects on. Reflection runs on its
/// own model (dspy requires a `reflection_lm`), independent of the model the program itself uses.
pub struct GEPA<M> {
    metric: M,
    reflection_model: Arc<dyn DynChatModel>,
    max_metric_calls: usize,
    reflection_minibatch_size: usize,
    perfect_score: f64,
    skip_perfect_score: bool,
    failure_score: f64,
    seed: u64,
}

impl<M> GEPA<M>
where
    M: Fn(&Example, &Prediction) -> Feedback + Send + Sync,
{
    /// A GEPA optimizer with dspy's defaults, save the budget — set it with
    /// [`with_max_metric_calls`](Self::with_max_metric_calls) before compiling.
    pub fn new(metric: M, reflection_model: Arc<dyn DynChatModel>) -> Self {
        Self {
            metric,
            reflection_model,
            max_metric_calls: 0,
            reflection_minibatch_size: 3,
            perfect_score: 1.0,
            skip_perfect_score: true,
            failure_score: 0.0,
            seed: 0,
        }
    }

    /// The rollout budget: GEPA stops once this many metric calls have been spent (dspy's
    /// `max_metric_calls`). Required — GEPA has no other stopping condition here.
    pub fn with_max_metric_calls(mut self, calls: usize) -> Self {
        self.max_metric_calls = calls;
        self
    }

    /// The minibatch size reflection evaluates on each iteration (dspy default 3).
    pub fn with_reflection_minibatch_size(mut self, size: usize) -> Self {
        self.reflection_minibatch_size = size;
        self
    }

    /// The RNG seed shared by candidate selection and minibatch sampling (dspy default 0).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Evolve `student`'s instructions, tracking Pareto scores on `valset` and reflecting on
    /// minibatches from `trainset`, then apply the best candidate found. Returns the run's outcome —
    /// the candidate pool, per-candidate valset scores, and eval totals (dspy's `detailed_results`).
    /// dspy reuses the trainset as the valset when none is given; call with `trainset` for both to
    /// match that.
    pub async fn compile<S: Module + ?Sized>(
        &self,
        student: &mut S,
        trainset: &[Example],
        valset: &[Example],
    ) -> Result<GepaOutcome> {
        assert!(self.max_metric_calls > 0, "GEPA needs a metric-call budget; set it with with_max_metric_calls");
        let seed_candidate: Candidate = student
            .named_predictors()
            .into_iter()
            .map(|predictor| (predictor.name.clone(), predictor.signature.instructions.clone()))
            .collect();

        let engine = GepaEngine {
            adapter: Adapter::new(
                student,
                &self.metric,
                self.reflection_model.clone(),
                global::client(),
                trainset,
                valset,
                self.failure_score,
            ),
            trainset_size: trainset.len(),
            valset_size: valset.len(),
            minibatch_size: self.reflection_minibatch_size,
            max_metric_calls: self.max_metric_calls,
            perfect_score: self.perfect_score,
            skip_perfect_score: self.skip_perfect_score,
            seed: self.seed,
        };
        let outcome = engine.optimize(seed_candidate).await;
        set_instructions(student, &outcome.best);
        Ok(outcome)
    }
}

impl<M> Optimizer for GEPA<M>
where
    M: Fn(&Example, &Prediction) -> Feedback + Send + Sync,
{
    /// GEPA has no teacher — it optimizes instructions from a metric — and reuses the trainset as the
    /// valset (dspy's default when none is given).
    fn compile<'a>(
        &'a self,
        student: &'a mut dyn Module,
        teacher: Option<&'a mut dyn Module>,
        trainset: &'a [Example],
    ) -> impl Future<Output = Result<()>> + Send + 'a {
        async move {
            if teacher.is_some() {
                bail!("GEPA optimizes instructions from a metric and has no teacher to learn from");
            }
            self.compile(student, trainset, trainset).await?;
            Ok(())
        }
    }
}
