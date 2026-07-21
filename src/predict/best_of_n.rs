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
/// `forward` takes `&self` while varying an attempt's sampling needs `&mut`. dspy sidesteps that
/// by deep-copying per attempt; a `dyn Module` cannot be cloned, so the sampling is set on the one
/// module and put back when the call ends. The lock also makes concurrent `forward` calls queue
/// rather than interleave their rollouts, which is what sharing one module requires.
pub struct BestOfN<M, R> {
    module: Mutex<M>,
    /// How many attempts at most. dspy's `N`.
    pub n: usize,
    /// Scores one attempt. dspy passes the module's inputs alongside the prediction, and so does
    /// this: a reward that cannot see what was asked cannot judge whether it was answered.
    pub reward: R,
    /// Stop at the first attempt scoring this or better. `None` runs every attempt and keeps the
    /// best, which is what upstream does when no threshold is set.
    pub threshold: Option<f64>,
    /// How many attempts may fail before the whole call does. dspy defaults it to `N`, meaning a
    /// run where every attempt errors still raises.
    pub fail_count: usize,
}

impl<M, R> BestOfN<M, R>
where
    M: Module,
    R: Fn(&Example, &Prediction) -> f64,
{
    /// Ask `module` up to `n` times, scored by `reward`, keeping the best.
    pub fn new(module: M, n: usize, reward: R) -> Self {
        Self {
            module: Mutex::new(module),
            n,
            reward,
            threshold: None,
            fail_count: n,
        }
    }

    /// The module back, for a caller that wants it after the wrapper is done with it.
    pub fn into_inner(self) -> M {
        self.module.into_inner()
    }

    /// Stop as soon as an attempt scores this or better.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// How many attempts may fail before the call does.
    pub fn with_fail_count(mut self, fail_count: usize) -> Self {
        self.fail_count = fail_count;
        self
    }

    /// Ask up to `n` times and answer with the best attempt.
    ///
    /// The module is left sampling the way it was found, so a caller's own setting survives a
    /// call — the same care [`BootstrapFewShot`](crate::BootstrapFewShot) takes with a teacher.
    pub async fn run(&self, inputs: Example) -> Result<Prediction> {
        if self.n == 0 {
            return Err(anyhow!("BestOfN needs at least one attempt"));
        }
        let mut module = self.module.lock().await;
        let resting = resting_sampling(&mut *module);
        let attempted = self.attempts(&mut *module, inputs).await;
        restore_sampling(&mut *module, &resting);
        attempted
    }

    /// The attempts themselves, so [`run`](Self::run) can put the module back however this ends.
    async fn attempts(&self, module: &mut M, inputs: Example) -> Result<Prediction> {
        let mut best: Option<(f64, Prediction)> = None;
        let mut failures = 0;
        let mut last_error = None;

        for attempt in 0..self.n {
            // dspy counts rollouts from whatever the module already carried, so a caller who set
            // one is continued rather than overwritten.
            module.set_sampling(Sampling::rollout(attempt as u64));
            let answered = match module.forward(inputs.clone()).await {
                Ok(prediction) => prediction,
                Err(error) => {
                    failures += 1;
                    tracing::warn!(%error, attempt, "an attempt failed");
                    if failures > self.fail_count {
                        return Err(error);
                    }
                    last_error = Some(error);
                    continue;
                }
            };

            let scored = (self.reward)(&inputs, &answered);
            let improved = best.as_ref().is_none_or(|(high, _)| scored > *high);
            if improved {
                best = Some((scored, answered));
            }
            if self.threshold.is_some_and(|bar| scored >= bar) {
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
            let resting = resting_sampling(&mut *module);
            let mut best: Option<(f64, Prediction, Vec<TraceStep>)> = None;

            for attempt in 0..self.n {
                module.set_sampling(Sampling::rollout(attempt as u64));
                let mut attempted = Vec::new();
                let Ok(answered) = module.forward_traced(inputs.clone(), &mut attempted).await
                else {
                    continue;
                };
                let scored = (self.reward)(&inputs, &answered);
                if best.as_ref().is_none_or(|(high, _, _)| scored > *high) {
                    best = Some((scored, answered, attempted));
                }
                if self.threshold.is_some_and(|bar| scored >= bar) {
                    break;
                }
            }

            restore_sampling(&mut *module, &resting);
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

/// What each predictor asks for before an attempt overrides it.
fn resting_sampling<M: Module + ?Sized>(module: &mut M) -> Vec<Sampling> {
    module
        .named_predictors()
        .iter()
        .map(|predictor| predictor.sampling.clone())
        .collect()
}

fn restore_sampling<M: Module + ?Sized>(module: &mut M, resting: &[Sampling]) {
    for (predictor, was) in module.named_predictors().into_iter().zip(resting) {
        *predictor.sampling = was.clone();
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
        let best = BestOfN::new(solver, 4, correctness).with_threshold(1.0);

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
        let best = BestOfN::new(solver, 3, correctness);
        best.run(asked("capital of France?"))
            .await
            .expect("an answer");

        let sampling: Vec<Sampling> = best
            .into_inner()
            .calls()
            .into_iter()
            .map(|call| call.sampling)
            .collect();
        assert_eq!(
            sampling,
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
        let best = BestOfN::new(solver, 3, correctness).with_threshold(2.0);

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
        solver.set_sampling(resting.clone());

        let best = BestOfN::new(solver, 2, correctness);
        best.run(asked("capital of France?"))
            .await
            .expect("an answer");

        let mut returned = best.into_inner();
        assert_eq!(returned.named_predictors()[0].sampling.clone(), resting);
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
        let answered = BestOfN::new(numbered, 3, |_: &Example, _: &Prediction| 1.0)
            .run(asked("anything"))
            .await
            .expect("an answer");
        assert_eq!(answered.get("answer").unwrap(), "attempt 1");
    }

    #[tokio::test]
    async fn enough_failures_end_the_call_with_the_failure() {
        let solver = Solver::new(Answers::Failing);
        let refused = BestOfN::new(solver, 3, correctness)
            .with_fail_count(1)
            .run(asked("capital of France?"))
            .await;
        assert!(refused.is_err(), "two failures exceed a budget of one");
    }

    /// Failing inside the budget is not success: there is still no answer to hand back.
    #[tokio::test]
    async fn every_attempt_failing_is_an_error_even_inside_the_budget() {
        let solver = Solver::new(Answers::Failing);
        let refused = BestOfN::new(solver, 2, correctness)
            .with_fail_count(10)
            .run(asked("capital of France?"))
            .await;
        assert!(refused.is_err());
    }

    /// The reason this is a `Module` and not a helper: upstream's is one, so a `BestOfN` nests
    /// inside a program, reaches an evaluator, and is walked by a compile like anything else.
    #[tokio::test]
    async fn it_is_a_module_a_program_can_hold_and_an_optimizer_can_walk() {
        let solver = Solver::new(Answers::Correctly);
        let mut best = BestOfN::new(solver, 2, correctness);

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
        let best = BestOfN::new(solver, 2, correctness);

        let answered = crate::call!(best, question = "capital of France?")
            .await
            .expect("an answer");
        assert_eq!(answered.get("answer").unwrap(), "Paris");
    }

    /// A compile learns from the attempt that won, not from every attempt that lost.
    #[tokio::test]
    async fn only_the_winning_attempt_is_traced() {
        let solver = Solver::new(Answers::Correctly);
        let best = BestOfN::new(solver, 3, correctness).with_threshold(1.0);

        let mut trace = Vec::new();
        best.forward_traced(asked("capital of France?"), &mut trace)
            .await
            .expect("an answer");
        assert_eq!(trace.len(), 0, "the Solver records no steps of its own");
    }

    #[tokio::test]
    async fn asking_zero_times_is_refused_rather_than_answered_with_nothing() {
        let solver = Solver::new(Answers::Correctly);
        let refused = BestOfN::new(solver, 0, correctness)
            .run(asked("capital of France?"))
            .await;
        assert!(refused.is_err());
    }
}
