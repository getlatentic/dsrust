//! dspy `Refine` (`predict/refine.py`): ask again, having been told what went wrong.
//!
//! `BestOfN` with a feedback step. Each attempt is a fresh rollout at `temperature = 1.0`, and
//! between attempts a model is asked — through [`OfferFeedback`](feedback) — what each predictor
//! should do differently. That advice reaches the next attempt as the predictor's `hint_` input,
//! so the module that went wrong is the one told about it.
//!
//! What separates it from `BestOfN`: the threshold is nullable (a run with none never stops
//! early on reward), and the losing attempts are what the feedback is built from rather than
//! discarded outright.

mod describe;
pub mod feedback;
mod trajectory;

use std::pin::Pin;

use anyhow::{Result, anyhow};
use futures_util::lock::Mutex;
use serde_json::Value;

use crate::example::{Example, Prediction};
use crate::lm::{DynChatModel, Sampling};
use crate::module::{Ask, Module, NamedPredictor, TraceStep};
use crate::predict::Predict;

/// Ask up to `n` times, advising each attempt from the last, and answer with the best.
///
/// A [`Module`], as upstream's is, so it nests inside a program and is walked by a compile like
/// anything else — the advisor it asks for feedback is deliberately not part of that walk, since
/// dspy builds one per call rather than holding it as a parameter.
///
/// Reach for [`Refine!`](macro@crate::Refine) rather than `new`: dspy names all its arguments at the
/// call site, and `Refine::new(qa, 3, one_word, 1.0)` leaves a reader guessing which number is
/// which.
///
/// The wrapped module sits behind an async lock for the same reason [`BestOfN`](struct@crate::BestOfN)'s
/// does: `forward` takes `&self` while varying an attempt's config and hint needs `&mut`, and a
/// `dyn Module` cannot be deep-copied the way dspy copies per attempt.
pub struct Refine<M, R> {
    module: Mutex<M>,
    /// Asked what each predictor should do differently. dspy builds a fresh
    /// `Predict(OfferFeedback)` per call; this holds one and reuses it, which is the same program.
    advisor: Predict,
    /// How many attempts at most. dspy's `N`.
    pub n: usize,
    /// Scores one attempt, the module's inputs alongside its prediction.
    pub reward: R,
    /// Stop at the first attempt scoring this or better, or run all `n` when unset.
    ///
    /// Nullable where [`BestOfN`](struct@crate::BestOfN)'s is required: `Refine.forward` guards with
    /// `if self.threshold is not None and reward >= self.threshold`, so a missing threshold is a
    /// state it can be in and `BestOfN` cannot.
    pub threshold: Option<f64>,
    /// How many attempts may fail before the whole call does. Unset means `n`, and so does
    /// `Some(0)` — dspy's `fail_count or N` reads a zero as unset.
    pub fail_count: Option<usize>,
    /// The wrapped program's source, for the advisor to read. dspy fills it with
    /// `inspect.getsource`; Rust has none at run time, so it is supplied at the call site or by
    /// the derive. Empty renders as an empty field rather than failing.
    pub program_code: String,
    /// The reward function's source, likewise. A Rust closure has no recoverable source, so this
    /// is always the caller's to give.
    pub reward_code: String,
}

impl<M, R> Refine<M, R>
where
    M: Module,
    R: Fn(&Example, &Prediction) -> f64,
{
    /// Ask `module` up to `n` times, scored by `reward`, stopping at `threshold`.
    ///
    /// The four dspy takes positionally, in its order. `threshold` is `impl Into<Option<f64>>`,
    /// so `1.0` reads as the common case and `None::<f64>` as the run-all-`n` one.
    pub fn new(module: M, n: usize, reward: R, threshold: impl Into<Option<f64>>) -> Self {
        Self {
            module: Mutex::new(module),
            advisor: Predict::from_signature(feedback::signature()),
            n,
            reward,
            threshold: threshold.into(),
            fail_count: None,
            program_code: String::new(),
            reward_code: String::new(),
        }
    }

    /// The budget of failed attempts, which upstream defaults to `n`.
    pub fn fail_count(mut self, fail_count: usize) -> Self {
        self.fail_count = Some(fail_count);
        self
    }

    /// The source the advisor reads: the program's, and the reward's. dspy's two
    /// `inspect.getsource` results, supplied rather than reflected.
    pub fn with_code(
        mut self,
        program_code: impl Into<String>,
        reward_code: impl Into<String>,
    ) -> Self {
        self.program_code = program_code.into();
        self.reward_code = reward_code.into();
        self
    }

    /// The model the advisor asks. Unset, it asks the configured one — dspy's `dspy.settings.lm`.
    pub fn with_advisor_lm(mut self, lm: std::sync::Arc<dyn DynChatModel>) -> Self {
        self.advisor = self.advisor.with_lm(lm);
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
    pub async fn run(&self, inputs: Example) -> Result<Prediction> {
        self.run_capturing(inputs).await.map(|(pred, _)| pred)
    }

    /// The same, keeping the winning attempt's trace so [`Module::forward_traced`] can hand it up.
    async fn run_capturing(&self, inputs: Example) -> Result<(Prediction, Vec<TraceStep>)> {
        if self.n == 0 {
            return Err(anyhow!("Refine needs at least one attempt"));
        }
        let mut module = self.module.lock().await;
        let resting = resting_state(&mut *module);
        let attempted = self.attempts(&mut *module, inputs).await;
        restore_state(&mut *module, &resting);
        attempted
    }

    /// The attempts, so [`run_capturing`](Self::run_capturing) can put the module back however
    /// this ends.
    async fn attempts(
        &self,
        module: &mut M,
        inputs: Example,
    ) -> Result<(Prediction, Vec<TraceStep>)> {
        let mut best: Option<(f64, Prediction, Vec<TraceStep>)> = None;
        let mut advice: Option<serde_json::Map<String, Value>> = None;
        // As `BestOfN` does, dspy compares the *attempt index* against a budget it decrements,
        // so how far into the run a failure lands is part of whether it is fatal.
        let mut budget = self.budget();
        let mut last_error = None;

        for attempt in 0..self.n {
            match self
                .attempt(module, &inputs, attempt, &mut best, advice.as_ref())
                .await
            {
                Ok(Outcome::Stop) => break,
                Ok(Outcome::Advised(next)) => advice = Some(next),
                Err(error) => {
                    tracing::warn!(%error, attempt, "an attempt failed");
                    if attempt > budget {
                        return Err(error);
                    }
                    budget = budget.saturating_sub(1);
                    last_error = Some(error);
                }
            }
        }

        match best {
            Some((_, prediction, trace)) => Ok((prediction, trace)),
            None => Err(last_error
                .unwrap_or_else(|| anyhow!("Refine made no attempt that produced an answer"))),
        }
    }

    /// One attempt: sample it as its own rollout, hinted by any advice, then either stop or ask
    /// what to advise the next one.
    ///
    /// Everything here is inside dspy's single `try`, the module call and the advice call alike —
    /// so a failed advisor call is an attempt failure the budget counts, exactly as a failed
    /// module call is.
    async fn attempt(
        &self,
        module: &mut M,
        inputs: &Example,
        attempt: usize,
        best: &mut Option<(f64, Prediction, Vec<TraceStep>)>,
        advice: Option<&serde_json::Map<String, Value>>,
    ) -> Result<Outcome> {
        apply(module, &Sampling::rollout(attempt as u64), advice);

        let mut trace = Vec::new();
        let answered = module.forward_traced(inputs.clone(), &mut trace).await?;
        let reward = (self.reward)(inputs, &answered);
        if best.as_ref().is_none_or(|(high, _, _)| reward > *high) {
            *best = Some((reward, answered.clone(), trace.clone()));
        }

        if self.threshold.is_some_and(|threshold| reward >= threshold) {
            return Ok(Outcome::Stop);
        }
        if attempt == self.n - 1 {
            return Ok(Outcome::Stop);
        }

        let (modules_defn, module_names) = describe_program(module);
        let advise = trajectory::advise_inputs(
            &self.program_code,
            &modules_defn,
            inputs,
            &trace,
            &answered,
            &self.reward_code,
            self.threshold,
            reward,
            &module_names,
        );
        let advised = Module::forward(&self.advisor, advise).await?;
        Ok(Outcome::Advised(feedback::advice_of(advised.get("advice"))))
    }
}

/// What one attempt decides for the loop around it.
enum Outcome {
    /// A cleared threshold or the last attempt: hand back the best so far.
    Stop,
    /// Advice for the next attempt, keyed by module name.
    Advised(serde_json::Map<String, Value>),
}

/// Set every predictor's config to this rollout, and its hint to its own advice.
///
/// With no advice — the first attempt — no predictor is hinted, matching dspy running the bare
/// module before any `WrapperAdapter` exists. With advice, *every* predictor is hinted, a named
/// one from the advice and the rest with `N/A`, which is upstream's `advice.get(name, "N/A")`.
fn apply<M: Module + ?Sized>(
    module: &mut M,
    config: &Sampling,
    advice: Option<&serde_json::Map<String, Value>>,
) {
    for predictor in module.named_predictors() {
        *predictor.config = config.clone();
        *predictor.hint = advice.map(|advice| hint_for(advice, &predictor.name));
    }
}

/// One module's hint: its own advice, or `N/A` when the advice did not name it.
fn hint_for(advice: &serde_json::Map<String, Value>, name: &str) -> String {
    advice
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or(feedback::NO_ADVICE)
        .to_owned()
}

/// The advisor's view of the program: each module described, and the names it seeks advice for.
fn describe_program<M: Module + ?Sized>(module: &mut M) -> (String, Vec<String>) {
    let predictors = module.named_predictors();
    let names = predictors.iter().map(|p| p.name.clone()).collect();
    (describe::modules(&predictors), names)
}

/// Each predictor's config and hint before an attempt overrides them, so a caller's module is
/// handed back the way it was found.
fn resting_state<M: Module + ?Sized>(module: &mut M) -> Vec<(Sampling, Option<String>)> {
    module
        .named_predictors()
        .iter()
        .map(|predictor| (predictor.config.clone(), predictor.hint.clone()))
        .collect()
}

fn restore_state<M: Module + ?Sized>(module: &mut M, resting: &[(Sampling, Option<String>)]) {
    for (predictor, (config, hint)) in module.named_predictors().into_iter().zip(resting) {
        *predictor.config = config.clone();
        *predictor.hint = hint.clone();
    }
}

impl<M, R> Module for Refine<M, R>
where
    M: Module,
    R: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(self.run(inputs))
    }

    /// The wrapped module's predictors, so a compile reaches through the wrapper — the advisor is
    /// not among them, matching dspy, whose advisor is never a parameter.
    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        self.module.get_mut().named_predictors()
    }

    /// The winning attempt's trace, not all `n` — a compile learns from the answer that won.
    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let (prediction, best_trace) = self.run_capturing(inputs).await?;
            trace.extend(best_trace);
            Ok(prediction)
        })
    }
}

/// A wrapper answers with whatever its module answered, so `call!` reaches it like any other.
impl<M, R> Ask for Refine<M, R>
where
    M: Module,
    R: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    type Answer = Prediction;

    fn ask<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Module::forward(self, inputs)
    }
}

/// `Refine!(module, n = 3, reward = f, threshold = 1.0)` — upstream's call, named.
///
/// dspy passes its arguments by keyword; Rust has none, so this supplies them, the same reason
/// [`call!`](crate::call) and [`BestOfN!`](macro@crate::BestOfN) exist. `fail_count` is optional
/// here as it is there, and `threshold` accepts `1.0` or an `Option<f64>`.
///
/// ```
/// # use dsrust::{Refine, Predict, Example, Prediction};
/// fn one_word(_inputs: &Example, out: &Prediction) -> f64 {
///     match out.get("answer").and_then(|answer| answer.as_str()) {
///         Some(answer) if answer.split_whitespace().count() == 1 => 1.0,
///         _ => 0.0,
///     }
/// }
///
/// let refined = Refine!(
///     Predict!("question -> answer"),
///     n = 3,
///     reward = one_word,
///     threshold = 1.0
/// );
/// ```
#[macro_export]
macro_rules! Refine {
    ($module:expr, n = $n:expr, reward = $reward:expr, threshold = $threshold:expr $(,)?) => {
        $crate::Refine::new($module, $n, $reward, $threshold)
    };
    ($module:expr, n = $n:expr, reward = $reward:expr, threshold = $threshold:expr,
     fail_count = $fail_count:expr $(,)?) => {
        $crate::Refine::new($module, $n, $reward, $threshold).fail_count($fail_count)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use crate::optimize::scripted::{Answers, Solver};
    use crate::predict::scripted::Scripted;

    fn asked(question: &str) -> Example {
        example! { question: question }.with_inputs(["question"])
    }

    fn correctness(_: &Example, prediction: &Prediction) -> f64 {
        match prediction.get("answer").and_then(Value::as_str) {
            Some("Paris") => 1.0,
            _ => 0.0,
        }
    }

    /// A completion the ChatAdapter parses into `OfferFeedback`'s two outputs, advising `self`.
    fn advice_reply(advice: &str) -> &'static str {
        Box::leak(
            format!(
                "[[ ## discussion ## ]]\nself went wrong\n\n\
                 [[ ## advice ## ]]\n{{\"self\": \"{advice}\"}}\n\n\
                 [[ ## completed ## ]]"
            )
            .into_boxed_str(),
        )
    }

    /// `n - 1` copies of one advice reply, for a run that asks the advisor after every attempt
    /// but the last.
    fn advice_replies(advice: &str, times: usize) -> Vec<&'static str> {
        (0..times).map(|_| advice_reply(advice)).collect()
    }

    /// The advisor scripted to answer, so a run that never clears its threshold still completes.
    fn advised<M, R>(refine: Refine<M, R>, replies: &[&'static str]) -> Refine<M, R>
    where
        M: Module,
        R: Fn(&Example, &Prediction) -> f64,
    {
        refine.with_advisor_lm(std::sync::Arc::new(Scripted::new(replies)))
    }

    #[tokio::test]
    async fn the_first_attempt_that_clears_the_threshold_ends_the_run() {
        // The threshold is cleared on attempt 2, so the advisor is asked exactly once — after
        // attempt 1, never after the winning one.
        let solver = Solver::new(Answers::RightOnRound(2));
        let refine = advised(
            Refine::new(solver, 4, correctness, 1.0),
            &advice_replies("try Paris", 1),
        );

        let answered = refine
            .run(asked("capital of France?"))
            .await
            .expect("an answer");
        assert_eq!(answered.get("answer").unwrap(), "Paris");
        assert_eq!(
            refine.into_inner().calls().len(),
            2,
            "stopped as soon as one attempt cleared the bar"
        );
    }

    /// Advice from one attempt is the next attempt's hint. dspy keys `advice[module_name]` onto
    /// each predictor, and here the one predictor is `self`.
    #[tokio::test]
    async fn advice_from_one_attempt_reaches_the_next_as_its_hint() {
        let solver = Solver::new(Answers::Wrongly);
        let refine = advised(
            Refine::new(solver, 2, correctness, 1.0),
            &advice_replies("say Paris", 1),
        );

        refine
            .run(asked("capital of France?"))
            .await
            .expect("an answer");

        let calls = refine.into_inner().calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].hint, None, "the first attempt is unhinted");
        assert_eq!(
            calls[1].hint.as_deref(),
            Some("say Paris"),
            "the second carries the first's advice"
        );
    }

    /// With no threshold a run never stops early on reward — it spends all `n` attempts. The
    /// advisor is asked after every attempt but the last, which is `n - 1` times.
    #[tokio::test]
    async fn a_run_with_no_threshold_uses_every_attempt() {
        let solver = Solver::new(Answers::Correctly);
        let refine = advised(
            Refine::new(solver, 3, correctness, None::<f64>),
            &advice_replies("keep going", 2),
        );

        refine
            .run(asked("capital of France?"))
            .await
            .expect("an answer");
        assert_eq!(
            refine.into_inner().calls().len(),
            3,
            "a perfect answer on attempt 1 does not stop a thresholdless run"
        );
    }

    /// Every attempt is a fresh rollout at temperature 1.0 — the same mechanism as `BestOfN`.
    #[tokio::test]
    async fn each_attempt_is_asked_as_its_own_rollout() {
        let solver = Solver::new(Answers::Wrongly);
        let refine = advised(
            Refine::new(solver, 3, correctness, 1.0),
            &advice_replies("hint", 2),
        );

        refine
            .run(asked("capital of France?"))
            .await
            .expect("an answer");
        let config: Vec<Sampling> = refine
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

    /// The best attempt wins, not the last — the middle one is the only good one.
    #[tokio::test]
    async fn the_best_attempt_wins_even_when_a_later_one_is_worse() {
        let solver = Solver::new(Answers::RightOnlyOnRound(2));
        // A threshold of 2.0 nothing reaches, so all three attempts run and the best is chosen.
        let refine = advised(
            Refine::new(solver, 3, correctness, 2.0),
            &advice_replies("hint", 2),
        );

        let answered = refine
            .run(asked("capital of France?"))
            .await
            .expect("an answer");
        assert_eq!(
            answered.get("answer").unwrap(),
            "Paris",
            "attempt 2 alone scored"
        );
    }

    /// A borrowed module is handed back with its config and hint the way they were found.
    #[tokio::test]
    async fn the_module_is_left_the_way_it_was_found() {
        let mut solver = Solver::new(Answers::Correctly);
        let resting = Sampling {
            temperature: Some(0.2),
            ..Sampling::default()
        };
        solver.set_config(resting.clone());

        let refine = advised(
            Refine::new(solver, 2, correctness, 1.0),
            &advice_replies("x", 1),
        );
        refine
            .run(asked("capital of France?"))
            .await
            .expect("an answer");

        let mut returned = refine.into_inner();
        let predictors = returned.named_predictors();
        assert_eq!(predictors[0].config.clone(), resting);
        assert_eq!(
            *predictors[0].hint, None,
            "the hint is cleared, not left set"
        );
    }

    /// A failed advisor call is an attempt failure the budget counts — it is inside dspy's one
    /// `try`. With a budget of one and an advisor that never answers, the second failure is fatal.
    #[tokio::test]
    async fn a_failing_advisor_spends_the_failure_budget() {
        let solver = Solver::new(Answers::Wrongly);
        // No replies: every advisor call fails. Attempt 0 answers then fails to advise (budget
        // 1 → 0); attempt 1 answers then fails to advise (index 1 > 0) → fatal.
        let refine = advised(Refine::new(solver, 4, correctness, 1.0).fail_count(1), &[]);

        let refused = refine.run(asked("capital of France?")).await;
        assert!(
            refused.is_err(),
            "the advisor's failures exhausted the budget"
        );
    }

    /// dspy weighs *when* a failure lands, comparing the attempt index against a decrementing
    /// budget rather than counting failures. With a budget of two and an advisor that never
    /// answers, the run survives attempts 0 and 1 and gives out on attempt 2 — three module calls
    /// in, since each attempt's module answers before its advisor fails. A `>=` boundary would
    /// give out one attempt sooner, which is what this pins.
    #[tokio::test]
    async fn the_failure_budget_counts_the_index_not_the_tally() {
        let solver = Solver::new(Answers::Wrongly);
        let refine = advised(Refine::new(solver, 5, correctness, 1.0).fail_count(2), &[]);

        let refused = refine.run(asked("capital of France?")).await;
        assert!(refused.is_err());
        assert_eq!(
            refine.into_inner().calls().len(),
            3,
            "survived the failures at 0 and 1, gave out on 2"
        );
    }

    #[tokio::test]
    async fn it_is_a_module_a_program_can_hold_and_an_optimizer_can_walk() {
        let solver = Solver::new(Answers::Correctly);
        let mut refine = advised(Refine::new(solver, 1, correctness, 1.0), &[]);

        let held: &dyn Module = &refine;
        let answered = held
            .forward(asked("capital of France?"))
            .await
            .expect("an answer");
        assert_eq!(answered.get("answer").unwrap(), "Paris");

        // The walk reaches the wrapped predictor, and not the advisor.
        assert_eq!(
            refine
                .named_predictors()
                .iter()
                .map(|predictor| predictor.name.clone())
                .collect::<Vec<_>>(),
            ["self"]
        );
    }

    #[tokio::test]
    async fn asking_zero_times_is_refused_rather_than_answered_with_nothing() {
        let refine = Refine::new(Solver::new(Answers::Correctly), 0, correctness, 1.0);
        assert!(refine.run(asked("capital of France?")).await.is_err());
    }
}
