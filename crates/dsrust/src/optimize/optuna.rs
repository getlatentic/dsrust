//! dspy `teleprompt/teleprompt_optuna.py`: bootstrap once, then let optuna choose which single demo
//! each predictor keeps.
//!
//! The search space is one integer per predictor — an index into the demos the bootstrap earned it
//! — and each trial builds a program from those indices, scores it on the validation set, and
//! reports the score back. What optuna does with those reports is [`IntTpeSampler`], reproduced in
//! the `tpe` crate because the search *is* its decisions.
//!
//! **Upstream's runs are not reproducible and this crate's are.** dspy calls
//! `optuna.create_study()` with no sampler, so its study is seeded from entropy and no two of its
//! own runs agree with each other. A Rust caller has to be given something, and being handed a seed
//! is the only shape that is not worse: see [`seed`](BootstrapFewShotWithOptuna::seed).

use anyhow::Result;
use tpe::IntTpeSampler;

use super::BootstrapFewShot;
use crate::evaluate::Metric;
use crate::example::Example;
use crate::module::{Module, ProgramState};

/// One trial: the demo each predictor kept, and what the resulting program scored.
///
/// `compile` answers with every trial in the order they ran, which upstream leaves in
/// `study.trials` and throws away — so a caller can see what the search tried rather than only
/// what it settled on.
///
/// ```
/// # use dsrust::optimize::OptunaTrial;
/// # fn read(trials: Vec<OptunaTrial>) {
/// let best = trials.iter().max_by(|a, b| a.score.total_cmp(&b.score));
/// // The first ten trials are the sampler's random start; what it *learned* is after them.
/// let searched = trials.iter().skip(10).count();
/// if let Some(best) = best {
///     println!("demos {:?} scored {:.1}%, after {searched} guided trial(s)", best.indices, best.score);
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct OptunaTrial {
    /// The index optuna suggested per predictor, in `named_predictors` order — dspy's
    /// `demo_index_for_{name}`.
    pub indices: Vec<usize>,
    /// What the program built from those indices scored, on `Evaluate`'s percentage scale.
    pub score: f64,
}

/// dspy `BootstrapFewShotWithOptuna`.
pub struct BootstrapFewShotWithOptuna<M> {
    metric: M,
    max_labeled_demos: usize,
    max_rounds: usize,
    num_candidate_programs: usize,
    seed: u32,
    scoring: super::Scoring,
}

impl<M> BootstrapFewShotWithOptuna<M>
where
    M: Metric,
{
    /// dspy's defaults: sixteen labelled demos, one round, sixteen candidate sets.
    ///
    /// `max_bootstrapped_demos` is not here on purpose. Upstream takes it, stores it as
    /// `max_num_samples`, prints it, and never reads it again — what actually bounds the bootstrap
    /// is `compile`'s own `max_demos`, which has no default.
    pub fn new(metric: M) -> Self {
        Self {
            metric,
            max_labeled_demos: 16,
            max_rounds: 1,
            num_candidate_programs: 16,
            seed: 0,
            scoring: super::Scoring::default(),
        }
    }

    pub fn max_labeled_demos(mut self, demos: usize) -> Self {
        self.max_labeled_demos = demos;
        self
    }

    pub fn max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }

    /// How many trials the study runs — dspy's `num_candidate_programs`, stored as
    /// `num_candidate_sets`.
    pub fn num_candidate_programs(mut self, programs: usize) -> Self {
        self.num_candidate_programs = programs;
        self
    }

    /// The sampler's seed, which upstream does not have.
    ///
    /// dspy's `optuna.create_study()` takes no sampler, so optuna builds a `TPESampler` seeded from
    /// entropy: two dspy runs over the same trainset propose different trials and can return
    /// different programs. There is no sequence here to match, only an algorithm — so this crate
    /// takes the seed and defaults it to zero, which makes a compile repeatable and a regression
    /// visible. Everything the seed feeds is upstream's.
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn scoring(mut self, scoring: super::Scoring) -> Self {
        self.scoring = scoring;
        self
    }

    /// Bootstrap `student`, then search for the one demo per predictor that scores best.
    ///
    /// `max_demos` is a required argument upstream too, and it is the bootstrap's budget rather
    /// than the search's: every trial keeps exactly one demo per predictor whatever it is.
    ///
    /// Answers with every trial in the order they ran; `student` is left holding the best. dspy
    /// defaults `valset` to the trainset, so pass the trainset for both to match that.
    pub async fn compile<S: Module + ?Sized>(
        &self,
        student: &mut S,
        max_demos: usize,
        trainset: &[Example],
        valset: &[Example],
    ) -> Result<Vec<OptunaTrial>> {
        let start = student.dump_state();
        let mut bootstrap = BootstrapFewShot::new(crate::evaluate::MetricRef(&self.metric));
        bootstrap.max_bootstrapped_demos = max_demos;
        bootstrap.max_labeled_demos = self.max_labeled_demos;
        bootstrap.max_rounds = self.max_rounds;
        bootstrap.compile(student, trainset).await?;

        // The demos each predictor may choose from, and the range optuna searches per predictor.
        let offered: Vec<Vec<Example>> = student
            .named_predictors()
            .into_iter()
            .map(|predictor| predictor.demos.clone())
            .collect();
        let ranges: Vec<(i64, i64)> = offered
            .iter()
            .map(|demos| (0, demos.len() as i64 - 1))
            .collect();
        // Upstream reaches `suggest_int(name, 0, -1)` for a predictor the bootstrap taught nothing,
        // and optuna refuses a range whose high is below its low. Refusing here says which
        // predictor, which upstream's `ValueError` does not.
        if let Some(position) = ranges.iter().position(|&(_, high)| high < 0) {
            let names = student.named_predictors();
            anyhow::bail!(
                "predictor {:?} earned no demos to choose between; optuna cannot search an empty \
                 range. Raise `max_demos`, or give the bootstrap a trainset it can solve.",
                names[position].name
            );
        }

        let compiled = student.dump_state();
        let mut sampler = IntTpeSampler::new(self.seed, ranges);
        let mut trials: Vec<OptunaTrial> = Vec::new();
        let mut best: Option<(f64, ProgramState)> = None;
        for _ in 0..self.num_candidate_programs {
            let indices: Vec<usize> = sampler.ask().iter().map(|&i| i as usize).collect();
            // Each trial starts from the *pre-bootstrap* program, as `reset_copy` does, and is
            // taught the one demo it drew — not the whole earned set narrowed.
            student.load_state(&start)?;
            for (position, (predictor, &index)) in student
                .named_predictors()
                .into_iter()
                .zip(&indices)
                .enumerate()
            {
                *predictor.demos = vec![offered[position][index].clone()];
            }
            let score = self.score(student, valset).await?;
            sampler.tell(indices.iter().map(|&i| i as i64).collect(), score);
            if best.as_ref().is_none_or(|(seen, _)| score > *seen) {
                best = Some((score, student.dump_state()));
            }
            trials.push(OptunaTrial { indices, score });
        }

        match best {
            Some((_, state)) => student.load_state(&state)?,
            None => student.load_state(&compiled)?,
        }
        Ok(trials)
    }

    async fn score<S: Module + ?Sized>(&self, student: &S, valset: &[Example]) -> Result<f64> {
        Ok(self
            .scoring
            .apply(crate::evaluate::Evaluate::new(
                valset.to_vec(),
                |inputs| student.forward(inputs),
                crate::evaluate::MetricRef(&self.metric),
            ))
            .run()
            .await?
            .score)
    }
}
