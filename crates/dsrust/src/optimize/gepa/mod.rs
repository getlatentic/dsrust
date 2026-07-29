//! GEPA in dsrust: instruction optimization by reflective evolution, wrapping the [`gepa`] crate the
//! way dspy's `GEPA` teleprompter wraps the gepa package.
//!
//! The [`gepa`] engine runs the loop (Pareto candidate selection, minibatch sampling, the accept
//! test, the budget); this supplies the LLM work through [`adapter::Adapter`] — running the student
//! program, scoring it with a feedback metric, and rewriting an instruction with a reflection model.
//!
//! Both of dspy's proposers run under its defaults (`reflection_minibatch_size=3`, Pareto
//! selection, `skip_perfect_score`, round-robin components, `use_merge=True`): reflective mutation
//! evolves one predictor's instruction, and merge combines two candidates that improved different
//! predictors. Merge only has something to combine in a multi-predictor program; over a single
//! `Predict` it never fires, so `use_merge` on by default costs nothing there.

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
    /// dspy `use_merge`: combine two candidates that improved different predictors of a
    /// multi-predictor program. On by default, as the teleprompter has it. A single-predictor
    /// program has nothing to combine, so this changes nothing there.
    use_merge: bool,
    /// dspy `max_merge_invocations`: the cap on accepted merges over a run (default 5).
    max_merge_invocations: usize,
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
            use_merge: true,
            max_merge_invocations: 5,
        }
    }

    /// Turn merging off (dspy `use_merge=False`), leaving the reflective-mutation-only engine.
    pub fn without_merge(mut self) -> Self {
        self.use_merge = false;
        self
    }

    /// The rollout budget: GEPA stops once this many metric calls have been spent (dspy's
    /// `max_metric_calls`). Required — GEPA has no other stopping condition here.
    pub fn with_max_metric_calls(mut self, calls: usize) -> Self {
        self.max_metric_calls = calls;
        self
    }

    /// dspy `GEPA.auto_budget`: a budget worked out from the shape of the run rather than named.
    ///
    /// Every part of this is a place to be off by one, so it is arithmetic transcribed rather than
    /// rederived: the trial count is the larger of two arms under an `int()` that *truncates*, the
    /// periodic-evaluation divisor is `full_eval_steps` and not `full_eval_steps + 1` despite the
    /// comment above it in upstream saying `m+1`, and a final full evaluation is added only when
    /// the trial count falls below that divisor.
    ///
    /// `num_preds` is how many predictors the program has and `num_candidates` how many will be
    /// proposed; both come from the caller because GEPA cannot know them until it has the student.
    pub fn auto_budget(
        num_preds: usize,
        num_candidates: usize,
        valset_size: usize,
        minibatch_size: usize,
        full_eval_steps: usize,
    ) -> Result<usize, String> {
        if full_eval_steps < 1 {
            return Err("full_eval_steps must be >= 1.".to_owned());
        }
        // `log2(0)` is negative infinity in Python and this crate has no candidates to propose, so
        // the second arm is what stands — as it does upstream, where `max` picks it.
        let searched = 2.0 * (num_preds as f64 * 2.0) * (num_candidates as f64).log2();
        let by_candidates = 1.5 * num_candidates as f64;
        let trials = searched.max(by_candidates).max(0.0) as usize;

        let mut total = valset_size + num_candidates * 5 + trials * minibatch_size;
        if trials == 0 {
            // No loop ran, so no evaluation inside it did either.
            return Ok(total);
        }
        let periodic = (trials + 1) / full_eval_steps + 1;
        let final_eval = usize::from(trials < full_eval_steps);
        total += (periodic + final_eval) * valset_size;
        Ok(total)
    }

    /// The budget [`auto_budget`](Self::auto_budget) works out, set on this optimizer.
    pub fn with_auto_budget(
        self,
        num_preds: usize,
        num_candidates: usize,
        valset_size: usize,
    ) -> Result<Self, String> {
        // dspy's own defaults for the two it defaults.
        let calls = Self::auto_budget(num_preds, num_candidates, valset_size, 35, 5)?;
        Ok(self.with_max_metric_calls(calls))
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
        assert!(
            self.max_metric_calls > 0,
            "GEPA needs a metric-call budget; set it with with_max_metric_calls"
        );
        let seed_candidate: Candidate = student
            .named_predictors()
            .into_iter()
            .map(|predictor| {
                (
                    predictor.name.clone(),
                    predictor.signature.instructions.clone(),
                )
            })
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
            use_merge: self.use_merge,
            max_merge_invocations: self.max_merge_invocations,
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

#[cfg(test)]
mod budget_tests {
    use super::*;
    use serde_json::Value;

    /// dspy's own answers, recorded by `scripts/generate_gepa_budget_fixture.py`. Transcribed
    /// arithmetic is exactly the kind that reads right and computes wrong, so none of these
    /// numbers is one this crate worked out.
    fn golden() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/optimize/gepa_budget.json");
        let text = std::fs::read_to_string(&path).expect("the golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    fn field(case: &Value, name: &str) -> usize {
        case[name]
            .as_u64()
            .unwrap_or_else(|| panic!("{name} is a number")) as usize
    }

    #[test]
    fn every_budget_dspy_worked_out_is_the_one_this_works_out() {
        let golden = golden();
        let cases = golden["cases"].as_array().expect("cases");
        assert!(!cases.is_empty(), "no cases to check");
        for case in cases {
            let answered = GEPA::<fn(&Example, &Prediction) -> Feedback>::auto_budget(
                field(case, "num_preds"),
                field(case, "num_candidates"),
                field(case, "valset_size"),
                field(case, "minibatch_size"),
                field(case, "full_eval_steps"),
            )
            .expect("a budget");
            assert_eq!(answered, field(case, "budget"), "for {case}");
        }
    }

    /// `full_eval_steps` below one is refused in dspy's wording. The other two refusals upstream
    /// has are unreachable here — a negative size cannot be spelled in a `usize`, which is the
    /// type system doing what the check does.
    #[test]
    fn a_full_eval_step_below_one_is_refused() {
        let refused = GEPA::<fn(&Example, &Prediction) -> Feedback>::auto_budget(1, 2, 10, 35, 0)
            .expect_err("zero is refused");
        assert_eq!(refused, "full_eval_steps must be >= 1.");
    }

    /// The builder sets what the calculation says, using dspy's own defaults for the two it
    /// defaults — so a caller who does not want to name a budget need not name those either.
    #[test]
    fn the_builder_sets_the_budget_it_works_out() {
        let golden = golden();
        let defaulted = golden["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|case| field(case, "minibatch_size") == 35 && field(case, "full_eval_steps") == 5)
            .expect("a case on dspy's defaults");
        let expected = field(defaulted, "budget");
        assert_eq!(
            GEPA::<fn(&Example, &Prediction) -> Feedback>::auto_budget(
                field(defaulted, "num_preds"),
                field(defaulted, "num_candidates"),
                field(defaulted, "valset_size"),
                35,
                5,
            )
            .expect("a budget"),
            expected
        );
    }
}
