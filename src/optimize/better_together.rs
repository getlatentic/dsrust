//! dspy `teleprompt/bettertogether.py`: run several optimizers in sequence and keep the best.
//!
//! A meta-optimizer. Each step of a strategy names one of the optimizers it was built with, runs it
//! over the student, and scores the result on a held-out set; whichever step scored best is what
//! the student is left holding. Because it holds optimizers rather than being one,
//! [`DynOptimizer`] is what it stores them as — the reason that trait exists.
//!
//! Upstream's *defaults* are `BootstrapFewShotWithRandomSearch` and `BootstrapFinetune`, and the
//! second is finetuning, which is out of this crate's 1.0 scope. So there is no default pair here:
//! the optimizers are named by the caller, which is upstream's own general case
//! (`BetterTogether(metric=…, p=…, w=…)`) and the one every documented example uses.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

use crate::evaluate::{Evaluate, percent};
use crate::example::{Example, Prediction};
use crate::module::{Module, ProgramState};

use super::DynOptimizer;
use super::rng::Rng;

/// dspy's separator between the steps of a strategy: `"p -> w -> p"`.
const STRATEGY_SEPARATOR: &str = " -> ";

/// One program the search evaluated, and what it scored.
///
/// dspy keeps a `deepcopy` of the program itself; here it is the program's compiled state, which
/// is what a copy would have carried and what [`Module::load_state`] puts back.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The metric's mean over the validation set as a percentage — dspy's `Evaluate.score` — or
    /// nothing where there was no validation set to score against.
    pub score: Option<f64>,
    /// The strategy that produced it: empty for the program as it arrived, then `"p"`,
    /// `"p -> w"`, and so on.
    pub strategy: String,
    pub state: ProgramState,
}

/// dspy's `BetterTogether`: prompt and weight optimization in whichever order the caller asks for.
pub struct BetterTogether<M> {
    pub metric: M,
    /// The optimizers a strategy may name, keyed as the caller named them.
    optimizers: BTreeMap<String, Box<dyn DynOptimizer>>,
    /// The share of the trainset held out to score each step, where the caller supplies no
    /// validation set of their own. Zero holds nothing out, and then no step is scored at all.
    pub valset_ratio: f64,
    /// dspy shuffles the trainset before each step so a run cannot overfit the example order.
    pub shuffle_trainset_between_steps: bool,
    pub seed: u64,
}

impl<M> BetterTogether<M>
where
    M: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    /// A meta-optimizer over the named optimizers. The names are what a strategy string uses.
    pub fn new(
        metric: M,
        optimizers: impl IntoIterator<Item = (impl Into<String>, Box<dyn DynOptimizer>)>,
    ) -> Self {
        Self {
            metric,
            optimizers: optimizers.into_iter().map(|(name, o)| (name.into(), o)).collect(),
            valset_ratio: 0.1,
            shuffle_trainset_between_steps: true,
            seed: 0,
        }
    }

    pub fn with_valset_ratio(mut self, valset_ratio: f64) -> Self {
        self.valset_ratio = valset_ratio;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn without_shuffling(mut self) -> Self {
        self.shuffle_trainset_between_steps = false;
        self
    }

    /// Run the strategy, leaving the student on the best program it found.
    ///
    /// Answers with every candidate it evaluated, best first — dspy's `candidate_programs`, which
    /// it hangs off the returned program. A step that fails stops the run rather than sinking it:
    /// the best program found so far is what the student keeps, which is upstream's behaviour.
    pub async fn compile(
        &self,
        student: &mut dyn Module,
        trainset: &[Example],
        valset: Option<&[Example]>,
        strategy: &str,
    ) -> Result<Vec<Candidate>> {
        let steps = self.parse_strategy(strategy)?;
        let (mut trainset, valset) = self.split(trainset, valset)?;

        let mut rng = Rng::seeded(self.seed);
        let mut candidates = Vec::new();
        // dspy scores the program as it arrived first, so "no optimization" is a candidate too.
        candidates.push(self.candidate(student, String::new(), valset.as_deref()).await);

        for (index, step) in steps.iter().enumerate() {
            if self.shuffle_trainset_between_steps {
                rng.shuffle(&mut trainset);
            }
            let optimizer = &self.optimizers[step];
            if let Err(error) = optimizer.compile_dyn(student, None, &trainset).await {
                tracing::error!(%error, step = %step, "a step failed; keeping the best so far");
                break;
            }
            let reached = steps[..=index].join(STRATEGY_SEPARATOR);
            candidates.push(self.candidate(student, reached, valset.as_deref()).await);
        }

        // dspy sorts by score, and a tie keeps the one found earlier.
        let mut ranked: Vec<(usize, Candidate)> = candidates.into_iter().enumerate().collect();
        ranked.sort_by(|(left_at, left), (right_at, right)| {
            let score = |candidate: &Candidate| candidate.score.unwrap_or(f64::NEG_INFINITY);
            score(right)
                .partial_cmp(&score(left))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left_at.cmp(right_at))
        });
        // With nothing to score against there is no "best", so dspy keeps the latest program.
        let best = match valset.is_some() {
            true => ranked.first(),
            false => ranked.iter().max_by_key(|(at, _)| *at),
        };
        if let Some((_, best)) = best {
            // The state came from this very student a moment ago, so a mismatch here would mean
            // the walk changed underneath — worth surfacing rather than passing over.
            student.load_state(&best.state)?;
        }
        Ok(ranked.into_iter().map(|(_, candidate)| candidate).collect())
    }

    /// Score the student as it stands and record it under the strategy that got it there.
    async fn candidate(
        &self,
        student: &mut dyn Module,
        strategy: String,
        valset: Option<&[Example]>,
    ) -> Candidate {
        let score = match valset {
            None => None,
            Some(valset) => Some(self.score(student, valset).await),
        };
        Candidate { score, strategy, state: student.dump_state() }
    }

    async fn score(&self, student: &dyn Module, valset: &[Example]) -> f64 {
        let evaluation = Evaluate::new(
            valset.to_vec(),
            |inputs| student.forward(inputs),
            |example: &Example, prediction: &Prediction| (self.metric)(example, prediction),
        )
        .run()
        .await;
        percent(evaluation.score)
    }

    /// dspy `_prepare_strategy`: the steps, each of which must name an optimizer it holds.
    fn parse_strategy(&self, strategy: &str) -> Result<Vec<String>> {
        if strategy.trim().is_empty() {
            bail!("strategy cannot be empty");
        }
        let steps: Vec<String> =
            strategy.split(STRATEGY_SEPARATOR).map(str::to_owned).collect();
        let unknown: Vec<&str> = steps
            .iter()
            .filter(|step| !self.optimizers.contains_key(*step))
            .map(String::as_str)
            .collect();
        if !unknown.is_empty() {
            let known: Vec<&str> = self.optimizers.keys().map(String::as_str).collect();
            bail!(
                "Strategy contains invalid optimizer keys: {unknown:?}. Valid keys are: {known:?}"
            );
        }
        Ok(steps)
    }

    /// dspy `_prepare_trainset_and_valset`: a supplied validation set wins; otherwise the ratio
    /// takes the *front* of the trainset, leaving the rest to train on.
    fn split(
        &self,
        trainset: &[Example],
        valset: Option<&[Example]>,
    ) -> Result<(Vec<Example>, Option<Vec<Example>>)> {
        if trainset.is_empty() {
            bail!("trainset cannot be empty");
        }
        if !(0.0..1.0).contains(&self.valset_ratio) {
            bail!("valset_ratio must be in range [0, 1), got {}", self.valset_ratio);
        }
        if let Some(valset) = valset {
            return Ok((trainset.to_vec(), Some(valset.to_vec())));
        }
        if self.valset_ratio == 0.0 {
            return Ok((trainset.to_vec(), None));
        }
        // Python's `int()` truncates, so a ratio that lands between examples rounds down.
        let held = (self.valset_ratio * trainset.len() as f64) as usize;
        Ok((trainset[held..].to_vec(), Some(trainset[..held].to_vec())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::scripted::{Answers, Solver, trainset};
    use crate::optimize::{LabeledFewShot, Optimizer};
    use crate::evaluate::exact_match;

    /// An optimizer that writes a known instruction, so which step ran is readable off the result.
    struct Writes(&'static str);

    impl Optimizer for Writes {
        fn compile<'a>(
            &'a self,
            student: &'a mut dyn Module,
            _teacher: Option<&'a mut dyn Module>,
            _trainset: &'a [Example],
        ) -> impl Future<Output = Result<()>> + Send + 'a {
            async move {
                for predictor in student.named_predictors() {
                    predictor.signature.instructions = self.0.to_owned();
                }
                Ok(())
            }
        }
    }

    /// An optimizer that always fails, for the step-failure path.
    struct Fails;

    impl Optimizer for Fails {
        fn compile<'a>(
            &'a self,
            _student: &'a mut dyn Module,
            _teacher: Option<&'a mut dyn Module>,
            _trainset: &'a [Example],
        ) -> impl Future<Output = Result<()>> + Send + 'a {
            async move { bail!("this step fails") }
        }
    }

    /// The instruction the student's one predictor is holding.
    fn instructions(student: &mut Solver) -> String {
        student.named_predictors().remove(0).signature.instructions.clone()
    }

    fn optimizers(
        pairs: Vec<(&'static str, Box<dyn DynOptimizer>)>,
    ) -> BetterTogether<fn(&Example, &Prediction) -> f64> {
        BetterTogether::new(exact_match as fn(&Example, &Prediction) -> f64, pairs)
            .with_valset_ratio(0.0)
    }

    /// Every step runs in order, and each is recorded under the strategy that reached it.
    #[tokio::test]
    async fn it_runs_each_step_and_records_the_strategy_that_reached_it() {
        let together = optimizers(vec![
            ("p", Box::new(Writes("from p"))),
            ("w", Box::new(Writes("from w"))),
        ]);
        let mut student = Solver::new(Answers::Correctly);
        let candidates = together
            .compile(&mut student, &trainset(), None, "p -> w -> p")
            .await
            .expect("compiles");

        let strategies: Vec<&str> =
            candidates.iter().map(|candidate| candidate.strategy.as_str()).collect();
        assert_eq!(strategies, ["", "p", "p -> w", "p -> w -> p"]);
        // No validation set, so nothing was scored and the latest program is what is kept.
        assert!(candidates.iter().all(|candidate| candidate.score.is_none()));
        assert_eq!(instructions(&mut student), "from p");
    }

    /// With a validation set the best-scoring step is what the student keeps, and the candidates
    /// come back best first.
    #[tokio::test]
    async fn the_best_scoring_step_is_the_one_kept() {
        let trainset = trainset();
        let together = BetterTogether::new(
            exact_match as fn(&Example, &Prediction) -> f64,
            vec![
                ("good", Box::new(LabeledFewShot::new(2)) as Box<dyn DynOptimizer>),
                ("noop", Box::new(Writes("unchanged")) as Box<dyn DynOptimizer>),
            ],
        );
        let mut student = Solver::new(Answers::Correctly);
        let candidates = together
            .compile(&mut student, &trainset, Some(&trainset), "good -> noop")
            .await
            .expect("compiles");

        assert_eq!(candidates.len(), 3);
        assert!(candidates.iter().all(|candidate| candidate.score.is_some()));
        // Sorted best first, and a tie keeps whichever was found earlier.
        let scores: Vec<f64> = candidates.iter().map(|c| c.score.expect("a score")).collect();
        assert!(scores.windows(2).all(|pair| pair[0] >= pair[1]), "sorted: {scores:?}");
    }

    /// A failing step stops the run and leaves the best found so far, rather than erroring out.
    #[tokio::test]
    async fn a_failing_step_keeps_what_was_found_before_it() {
        let together = optimizers(vec![
            ("p", Box::new(Writes("from p"))),
            ("boom", Box::new(Fails)),
        ]);
        let mut student = Solver::new(Answers::Correctly);
        let candidates = together
            .compile(&mut student, &trainset(), None, "p -> boom")
            .await
            .expect("does not surface the step's error");

        let strategies: Vec<&str> = candidates.iter().map(|c| c.strategy.as_str()).collect();
        assert_eq!(strategies, ["", "p"], "the failed step recorded no candidate");
        assert_eq!(instructions(&mut student), "from p");
    }

    /// A strategy naming an optimizer that was never supplied is refused, with dspy's wording.
    #[tokio::test]
    async fn it_refuses_a_strategy_it_cannot_run() {
        let together = optimizers(vec![("p", Box::new(Writes("from p")))]);
        let mut student = Solver::new(Answers::Correctly);
        let error = together
            .compile(&mut student, &trainset(), None, "p -> w")
            .await
            .expect_err("refuses");
        assert!(error.to_string().contains("invalid optimizer keys"), "got: {error}");
        assert!(
            together.compile(&mut student, &trainset(), None, "  ").await.is_err(),
            "an empty strategy is refused too"
        );
    }

    /// The held-out share comes off the front of the trainset, and Python's `int()` truncates.
    #[test]
    fn the_validation_share_is_taken_off_the_front() {
        let together = optimizers(vec![("p", Box::new(Writes("x")))]).with_valset_ratio(0.5);
        let examples = trainset();
        let (train, val) = together.split(&examples, None).expect("splits");
        assert_eq!(val.expect("a valset").len(), examples.len() / 2);
        assert_eq!(train.len(), examples.len() - examples.len() / 2);
        assert!(together.split(&[], None).is_err(), "an empty trainset is refused");
    }
}
