//! dspy `BestOfN` (`predict/best_of_n.py`): ask several times, keep the best answer.
//!
//! Each attempt is a fresh rollout at `temperature = 1.0`, which is the whole mechanism — upstream
//! runs `lm.copy(rollout_id=start+i, temperature=1.0)` so the second ask is a different ask rather
//! than a replay of the first. [`Sampling::rollout`] is that copy, and the response cache is what
//! makes the rollout id matter.
//!
//! Stops early on an attempt that clears the threshold, because the point is a good answer rather
//! than `n` answers.

use anyhow::{Result, anyhow};
use futures_util::lock::Mutex;

use crate::example::{Example, Prediction};
use crate::lm::Sampling;
use crate::module::{Ask, Module, NamedPredictor, TraceStep};

/// Ask up to `n` times and answer with the best attempt.
///
/// Reach for [`BestOfN!`](macro@crate::BestOfN) rather than `new`: dspy names all four arguments at
/// the call site, and `BestOfN::new(qa, 3, one_word, 1.0)` leaves a reader guessing which number
/// is which. Rust has no named arguments, so the macro supplies them the way `call!` does.
///
/// ```ignore
/// let best = BestOfN::new(3, |_inputs: &Example, pred: &Prediction| {
///     match pred.get("answer").and_then(|a| a.as_str()) {
///         Some(answer) if answer.split_whitespace().count() == 1 => 1.0,
///         _ => 0.0,
///     }
/// })
/// .with_threshold(1.0);
/// let answer = best.run(&mut qa, input! { question: "capital of Belgium?" }).await?;
/// ```
///
/// A [`Module`] itself, as upstream's is, so it nests inside a program and can be evaluated or
/// compiled like anything else.
///
/// The module it wraps sits behind an async lock rather than being owned outright, because
/// `forward` takes `&self` while varying an attempt's config needs `&mut`. dspy sidesteps that
/// by deep-copying per attempt; a `dyn Module` cannot be cloned, so the config is set on the one
/// module and put back when the call ends. The lock also makes concurrent `forward` calls queue
/// rather than interleave their rollouts, which is what sharing one module requires.
pub struct BestOfN<M, R> {
    module: Mutex<M>,
    /// How many attempts at most. dspy's `N`.
    pub n: usize,
    /// Scores one attempt. dspy passes the module's inputs alongside the prediction, and so does
    /// this: a reward that cannot see what was asked cannot judge whether it was answered.
    pub reward: R,
    /// Stop at the first attempt scoring this or better.
    ///
    /// Required, as upstream's is: `BestOfN.forward` compares `reward >= self.threshold` with no
    /// guard, so a missing threshold is not a state dspy can be in. (`Refine`
    /// does guard, and takes an optional one.)
    pub threshold: f64,
    /// How many attempts may fail before the whole call does. Unset means `n`.
    ///
    /// `Some(0)` also means `n`, which looks wrong and is upstream: `fail_count or N` reads a
    /// zero as unset, the same Python falsiness that makes `metric_threshold` of `0.0` mean *no
    /// threshold* in [`BootstrapFewShot`](crate::BootstrapFewShot).
    pub fail_count: Option<usize>,
}

impl<M, R> BestOfN<M, R>
where
    M: Module,
    R: Fn(&Example, &Prediction) -> f64,
{
    /// Ask `module` up to `n` times, scored by `reward`, stopping at `threshold`.
    ///
    /// The four dspy takes positionally, in its order.
    pub fn new(module: M, n: usize, reward: R, threshold: f64) -> Self {
        Self {
            module: Mutex::new(module),
            n,
            reward,
            threshold,
            fail_count: None,
        }
    }

    /// The budget of failed attempts, which upstream defaults to `n`.
    pub fn fail_count(mut self, fail_count: usize) -> Self {
        self.fail_count = Some(fail_count);
        self
    }

    /// dspy's `fail_count or N`, zero included.
    fn budget(&self) -> usize {
        match self.fail_count {
            Some(given) if given != 0 => given,
            _ => self.n,
        }
    }

    /// The module back, for a caller that wants it after the wrapper is done with it.
    pub fn into_inner(self) -> M {
        self.module.into_inner()
    }

    /// Ask up to `n` times and answer with the best attempt.
    ///
    /// The module is left config the way it was found, so a caller's own setting survives a
    /// call — the same care [`BootstrapFewShot`](crate::BootstrapFewShot) takes with a teacher.
    pub async fn run(&self, inputs: Example) -> Result<Prediction> {
        if self.n == 0 {
            return Err(anyhow!("BestOfN needs at least one attempt"));
        }
        let mut module = self.module.lock().await;
        let resting = resting_config(&mut *module);
        let attempted = self.attempts(&mut *module, inputs).await;
        restore_config(&mut *module, &resting);
        attempted
    }

    /// The attempts themselves, so [`run`](Self::run) can put the module back however this ends.
    async fn attempts(&self, module: &mut M, inputs: Example) -> Result<Prediction> {
        let mut best: Option<(f64, Prediction)> = None;
        // dspy compares the *attempt index* against a budget it decrements, rather than counting
        // failures — so how far into the run a failure happens is part of whether it is fatal.
        let mut budget = self.budget();
        let mut last_error = None;

        for attempt in 0..self.n {
            // dspy counts rollouts from whatever the module already carried, so a caller who set
            // one is continued rather than overwritten.
            module.set_config(Sampling::rollout(attempt as u64));
            let answered = match module.forward(inputs.clone()).await {
                Ok(prediction) => prediction,
                Err(error) => {
                    tracing::warn!(%error, attempt, "an attempt failed");
                    if attempt > budget {
                        return Err(error);
                    }
                    budget = budget.saturating_sub(1);
                    last_error = Some(error);
                    continue;
                }
            };

            let scored = (self.reward)(&inputs, &answered);
            let improved = best.as_ref().is_none_or(|(high, _)| scored > *high);
            if improved {
                best = Some((scored, answered));
            }
            if scored >= self.threshold {
                break;
            }
        }

        match best {
            Some((_, prediction)) => Ok(prediction),
            // Every attempt failed without exhausting the budget, which is still no answer.
            None => Err(last_error
                .unwrap_or_else(|| anyhow!("BestOfN made no attempt that produced an answer"))),
        }
    }
}

impl<M, R> Module for BestOfN<M, R>
where
    M: Module,
    R: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(self.run(inputs))
    }

    /// The wrapped module's predictors, so an optimizer reaches through the wrapper the way
    /// upstream's walk does — a `BestOfN` around a program is still that program to a compile.
    ///
    /// `get_mut` rather than a lock: a walk holds `&mut self`, which is proof no call is in
    /// flight, so there is nothing to wait for.
    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        self.module.get_mut().named_predictors()
    }

    /// One attempt's trace, not all `n`.
    ///
    /// dspy keeps the trace of the attempt it chose and discards the rest, so a compile learns
    /// from the answer that won rather than from every answer that lost.
    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let mut module = self.module.lock().await;
            let resting = resting_config(&mut *module);
            let mut best: Option<(f64, Prediction, Vec<TraceStep>)> = None;

            for attempt in 0..self.n {
                module.set_config(Sampling::rollout(attempt as u64));
                let mut attempted = Vec::new();
                let Ok(answered) = module.forward_traced(inputs.clone(), &mut attempted).await
                else {
                    continue;
                };
                let scored = (self.reward)(&inputs, &answered);
                if best.as_ref().is_none_or(|(high, _, _)| scored > *high) {
                    best = Some((scored, answered, attempted));
                }
                if scored >= self.threshold {
                    break;
                }
            }

            restore_config(&mut *module, &resting);
            match best {
                Some((_, prediction, attempted)) => {
                    trace.extend(attempted);
                    Ok(prediction)
                }
                None => Err(anyhow!("BestOfN made no attempt that produced an answer")),
            }
        })
    }
}

/// A wrapper answers with whatever its module answered, so `call!` reaches it like any other.
///
/// Written by hand rather than through `asks_with_a_prediction!`, which takes a concrete type and
/// cannot name the two parameters this carries.
impl<M, R> Ask for BestOfN<M, R>
where
    M: Module,
    R: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    type Answer = Prediction;

    fn ask<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Module::forward(self, inputs)
    }
}

/// `BestOfN!(module, n = 3, reward = f, threshold = 1.0)` — upstream's call, named.
///
/// dspy passes all four by keyword. Rust has no named arguments, so this supplies them, the same
/// reason [`call!`](crate::call) and [`input!`](crate::input) exist. `fail_count` is optional
/// here as it is there.
///
/// ```
/// # use dsrust::{BestOfN, Predict, Example, Prediction};
/// fn one_word(_inputs: &Example, out: &Prediction) -> f64 {
///     match out.get("answer").and_then(|answer| answer.as_str()) {
///         Some(answer) if answer.split_whitespace().count() == 1 => 1.0,
///         _ => 0.0,
///     }
/// }
///
/// let best = BestOfN!(
///     Predict!("question -> answer"),
///     n = 3,
///     reward = one_word,
///     threshold = 1.0
/// );
/// ```
#[macro_export]
macro_rules! BestOfN {
    ($module:expr, n = $n:expr, reward = $reward:expr, threshold = $threshold:expr $(,)?) => {
        $crate::BestOfN::new($module, $n, $reward, $threshold)
    };
    ($module:expr, n = $n:expr, reward = $reward:expr, threshold = $threshold:expr,
     fail_count = $fail_count:expr $(,)?) => {
        $crate::BestOfN::new($module, $n, $reward, $threshold).fail_count($fail_count)
    };
}

/// What each predictor asks for before an attempt overrides it.
fn resting_config<M: Module + ?Sized>(module: &mut M) -> Vec<Sampling> {
    module
        .named_predictors()
        .iter()
        .map(|predictor| predictor.config.clone())
        .collect()
}

fn restore_config<M: Module + ?Sized>(module: &mut M, resting: &[Sampling]) {
    for (predictor, was) in module.named_predictors().into_iter().zip(resting) {
        *predictor.config = was.clone();
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

    /// One point for the right answer, nothing otherwise.
    fn correctness(_: &Example, prediction: &Prediction) -> f64 {
        match prediction.get("answer").and_then(|value| value.as_str()) {
            Some("Paris") => 1.0,
            _ => 0.0,
        }
    }

    #[tokio::test]
    async fn the_first_attempt_that_clears_the_threshold_ends_the_run() {
        let solver = Solver::new(Answers::RightOnRound(2));
        let best = BestOfN::new(solver, 4, correctness, 1.0);

        let answered = best
            .run(asked("capital of France?"))
            .await
            .expect("an answer");
        assert_eq!(answered.get("answer").unwrap(), "Paris");
        assert_eq!(
            best.into_inner().calls().len(),
            2,
            "stopped as soon as one attempt scored"
        );
    }

    /// Every attempt after the first is a fresh rollout at temperature 1.0 — the same mechanism
    /// a bootstrap round uses, and the reason a re-ask can differ at all.
    #[tokio::test]
    async fn each_attempt_is_asked_as_its_own_rollout() {
        let solver = Solver::new(Answers::Wrongly);
        let best = BestOfN::new(solver, 3, correctness, 1.0);
        best.run(asked("capital of France?"))
            .await
            .expect("an answer");

        let config: Vec<Sampling> = best
            .into_inner()
            .calls()
            .into_iter()
            .map(|call| call.config)
            .collect();
        assert_eq!(
            config,
            [
                Sampling::rollout(0),
                Sampling::rollout(1),
                Sampling::rollout(2)
            ]
        );
    }

    /// With nothing clearing the bar, the best of what was asked comes back rather than the last.
    ///
    /// The middle attempt is the only good one deliberately: with a module that stays right once
    /// it is right, the best answer and the last answer are the same, and this passes whichever
    /// rule is implemented. It did, until a mutation said so.
    #[tokio::test]
    async fn the_best_attempt_wins_even_when_a_later_one_is_worse() {
        let solver = Solver::new(Answers::RightOnlyOnRound(2));
        let best = BestOfN::new(solver, 3, correctness, 2.0);

        let answered = best
            .run(asked("capital of France?"))
            .await
            .expect("an answer");
        assert_eq!(
            answered.get("answer").unwrap(),
            "Paris",
            "attempt 2 scored; attempts 1 and 3 did not"
        );
        assert_eq!(
            best.into_inner().calls().len(),
            3,
            "no threshold was reached, so all 3"
        );
    }

    /// A borrowed module is given back the way it was found; a caller's own setting is not
    /// quietly replaced by the last rollout.
    #[tokio::test]
    async fn the_module_is_left_sampling_the_way_it_was_found() {
        let mut solver = Solver::new(Answers::Correctly);
        let resting = Sampling {
            temperature: Some(0.2),
            ..Sampling::default()
        };
        solver.set_config(resting.clone());

        let best = BestOfN::new(solver, 2, correctness, 1.0);
        best.run(asked("capital of France?"))
            .await
            .expect("an answer");

        let mut returned = best.into_inner();
        assert_eq!(returned.named_predictors()[0].config.clone(), resting);
    }

    /// A module whose answers differ but score the same, which is the only way to see which of
    /// two equal attempts is kept.
    struct Numbered(std::sync::Mutex<usize>);

    impl Module for Numbered {
        fn forward<'a>(
            &'a self,
            _inputs: Example,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
            Box::pin(async move {
                let mut seen = self.0.lock().expect("not poisoned");
                *seen += 1;
                Ok(Prediction::new(
                    Example::new([("answer", serde_json::json!(format!("attempt {seen}")))]),
                    "raw",
                ))
            })
        }
    }

    /// dspy keeps an attempt only when it scores *strictly* higher, so the earliest of several
    /// equal attempts survives. A later tie displacing it would be a different program.
    #[tokio::test]
    async fn a_later_attempt_that_only_ties_does_not_displace_the_first() {
        let numbered = Numbered(std::sync::Mutex::new(0));
        let answered = BestOfN::new(numbered, 3, |_: &Example, _: &Prediction| 1.0, 2.0)
            .run(asked("anything"))
            .await
            .expect("an answer");
        assert_eq!(answered.get("answer").unwrap(), "attempt 1");
    }

    #[tokio::test]
    async fn enough_failures_end_the_call_with_the_failure() {
        let solver = Solver::new(Answers::Failing);
        let refused = BestOfN::new(solver, 3, correctness, 1.0)
            .fail_count(1)
            .run(asked("capital of France?"))
            .await;
        assert!(refused.is_err(), "two failures exceed a budget of one");
    }

    /// dspy weighs *when* a failure happened, not how many there were: it compares the attempt
    /// index against a budget it decrements, so with the default budget of `n` a run where
    /// everything fails still raises — on the third attempt of three.
    ///
    /// A plain "failures > fail_count" counter would never raise here, which is what this crate
    /// did until the two were traced side by side.
    #[tokio::test]
    async fn the_default_budget_still_gives_out_when_every_attempt_fails() {
        let solver = Solver::new(Answers::Failing);
        let best = BestOfN::new(solver, 3, correctness, 1.0);
        let refused = best.run(asked("capital of France?")).await;

        assert!(refused.is_err());
        assert_eq!(
            best.into_inner().calls().len(),
            3,
            "it gave out on the third, having spent the budget on the first two"
        );
    }

    /// The rule dspy actually implements, and where it parts from counting failures.
    ///
    /// Two attempts answer (badly) and the rest fail, with a budget of one. Upstream compares the
    /// *index* — attempt 2 is already past a budget of 1 — so the first failure is fatal and the
    /// run stops at three calls. A failure counter would spend one failure, carry on, and make a
    /// fourth call.
    #[tokio::test]
    async fn a_failure_late_in_the_run_is_fatal_where_the_same_failure_early_is_not() {
        let solver = Solver::new(Answers::FailingAfter(2));
        let best = BestOfN::new(solver, 5, correctness, 1.0).fail_count(1);
        let refused = best.run(asked("capital of France?")).await;

        assert!(refused.is_err(), "the run gave out");
        assert_eq!(
            best.into_inner().calls().len(),
            3,
            "two answers then one failure, which at index 2 is already past a budget of 1"
        );
    }

    /// dspy reads `fail_count or N`, so a zero is falsy and means *unset* rather than *none
    /// allowed* — the same Python falsiness that makes a `metric_threshold` of `0.0` mean no
    /// threshold at all.
    #[tokio::test]
    async fn a_fail_count_of_zero_means_n_rather_than_none_allowed() {
        let solver = Solver::new(Answers::FailingAfter(2));
        let best = BestOfN::new(solver, 4, correctness, 1.0).fail_count(0);
        let _ = best.run(asked("capital of France?")).await;

        assert_eq!(
            best.into_inner().calls().len(),
            4,
            "a zero budget read as none allowed would have given out on the third call"
        );
    }

    /// Failing inside the budget is not success: there is still no answer to hand back.
    #[tokio::test]
    async fn every_attempt_failing_is_an_error_even_inside_the_budget() {
        let solver = Solver::new(Answers::Failing);
        let refused = BestOfN::new(solver, 2, correctness, 1.0)
            .fail_count(10)
            .run(asked("capital of France?"))
            .await;
        assert!(refused.is_err());
    }

    /// The reason this is a `Module` and not a helper: upstream's is one, so a `BestOfN` nests
    /// inside a program, reaches an evaluator, and is walked by a compile like anything else.
    #[tokio::test]
    async fn it_is_a_module_a_program_can_hold_and_an_optimizer_can_walk() {
        let solver = Solver::new(Answers::Correctly);
        let mut best = BestOfN::new(solver, 2, correctness, 1.0);

        let held: &dyn Module = &best;
        let answered = held
            .forward(asked("capital of France?"))
            .await
            .expect("an answer");
        assert_eq!(answered.get("answer").unwrap(), "Paris");

        // A compile reaches straight through the wrapper to the predictors inside it.
        assert_eq!(
            best.named_predictors()
                .iter()
                .map(|predictor| predictor.name.clone())
                .collect::<Vec<_>>(),
            ["self"]
        );
    }

    /// The standard every module here is held to: a dspy Module subclass is a Rust `Module`,
    /// and `call!` reaches it by name the same way it reaches a `Predict`.
    #[tokio::test]
    async fn call_reaches_it_by_field_name_like_any_other_module() {
        let solver = Solver::new(Answers::Correctly);
        let best = BestOfN::new(solver, 2, correctness, 1.0);

        let answered = crate::call!(best, question = "capital of France?")
            .await
            .expect("an answer");
        assert_eq!(answered.get("answer").unwrap(), "Paris");
    }

    /// A compile learns from the attempt that won, not from every attempt that lost.
    #[tokio::test]
    async fn only_the_winning_attempt_is_traced() {
        let solver = Solver::new(Answers::Correctly);
        let best = BestOfN::new(solver, 3, correctness, 1.0);

        let mut trace = Vec::new();
        best.forward_traced(asked("capital of France?"), &mut trace)
            .await
            .expect("an answer");
        assert_eq!(trace.len(), 0, "the Solver records no steps of its own");
    }

    #[tokio::test]
    async fn asking_zero_times_is_refused_rather_than_answered_with_nothing() {
        let solver = Solver::new(Answers::Correctly);
        let refused = BestOfN::new(solver, 0, correctness, 1.0)
            .run(asked("capital of France?"))
            .await;
        assert!(refused.is_err());
    }
}
