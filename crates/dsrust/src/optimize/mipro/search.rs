//! dspy's Step 3: `_optimize_prompt_parameters`, the trial loop the TPE sampler drives.
//!
//! Split out of `mod.rs` when minibatching landed, which is the point at which the loop stopped
//! being "ask, score, tell" — a minibatch run scores on a subsample, keeps a running mean per
//! combination, and every few trials stops to score the best-averaging one on the whole valset and
//! feed *that* back to the sampler.

use anyhow::{Result, bail};

use super::minibatch::{self, Averages};
use super::{MIPROv2, RunMode, Slot, Trial, apply, search_space};
use crate::evaluate::{Evaluate, Pass};
use crate::example::Example;
use crate::module::Module;

use super::super::rng::Rng;

impl<M> MIPROv2<M>
where
    M: crate::evaluate::Metric,
{
    /// Seed the sampler with the default program as a baseline trial, run `num_trials` more, and
    /// leave the student on the best combination. Candidate zero of every predictor is its original
    /// instruction, so the all-zeros baseline is the default program.
    pub(super) async fn search<S: Module + ?Sized>(
        &self,
        student: &mut S,
        candidates: &[Vec<String>],
        demo_sets: Option<&[Vec<Vec<Example>>]>,
        mode: &RunMode,
        rng: &mut Rng,
    ) -> Result<Vec<Trial>> {
        if mode.minibatch && self.minibatch_full_eval_steps == 0 {
            bail!("minibatch_full_eval_steps must be greater than zero");
        }
        let named = search_space(candidates, demo_sets);
        let space: Vec<Slot> = named.iter().map(|(_, slot)| *slot).collect();
        let baseline = vec![0usize; space.len()];
        // The default program's own pass is always a full one, so it takes no draw.
        let default_score = self.score(student, &mode.valset, mode).await?;

        let mut sampler = tpe::TpeSampler::new(
            self.seed as u32,
            named
                .iter()
                .map(|(name, slot)| (name.clone(), slot.cardinality()))
                .collect(),
        );
        sampler.tell(baseline.clone(), default_score);
        let mut best = (default_score, baseline.clone());
        let mut trials = vec![Trial {
            params: baseline,
            score: default_score,
        }];
        let mut averages = Averages::default();
        let adjusted = minibatch::adjusted_trials(mode.num_trials, self.minibatch_full_eval_steps);

        // What the schedule below counts is not loop iterations but *trials the study has created* —
        // upstream reads `trial.number + 1`, and a full evaluation is added to the study as a trial
        // of its own, so it takes a number and shifts every trial after it. Counting iterations
        // instead agrees until the first full evaluation and then drifts by one per full evaluation,
        // which is enough to change which candidates the sampler is asked for.
        let mut created = 1;
        for _ in 0..mode.num_trials {
            let trial = created + 1;
            created += 1;
            let params = sampler.ask();
            apply(student, candidates, demo_sets, &space, &params);
            let batch = self.batch(mode, rng);
            let score = self.score(student, &batch, mode).await?;
            sampler.tell(params.clone(), score);
            trials.push(Trial {
                params: params.clone(),
                score,
            });
            // A minibatch score says nothing about the whole valset, so upstream lets only a full
            // evaluation move the winner.
            if !mode.minibatch {
                if score > best.0 {
                    best = (score, params);
                }
                continue;
            }
            averages.record(&params, score);
            if !minibatch::full_evaluation_due(trial, adjusted, self.minibatch_full_eval_steps) {
                continue;
            }
            let Some((params, score)) = self
                .evaluate_fully(student, candidates, demo_sets, &space, &mut averages, mode)
                .await?
            else {
                continue;
            };
            created += 1;
            // Told after the trial that triggered it, matching the numbers optuna hands out: the
            // running trial was created first and keeps the lower one however much later its value
            // arrives, and the sampler reads them in number order. Upstream's call order is the
            // other way round, which is what makes this worth saying.
            sampler.tell(params.clone(), score);
            trials.push(Trial {
                params: params.clone(),
                score,
            });
            if score > best.0 {
                best = (score, params);
            }
        }

        apply(student, candidates, demo_sets, &space, &best.1);
        Ok(trials)
    }

    /// dspy `_perform_full_evaluation`: score the best-averaging combination not yet fully
    /// evaluated on the whole valset.
    ///
    /// The score comes back to be told to the sampler as its own trial, which is the part that
    /// makes minibatching change the *search* and not only what each trial costs.
    async fn evaluate_fully<S: Module + ?Sized>(
        &self,
        student: &mut S,
        candidates: &[Vec<String>],
        demo_sets: Option<&[Vec<Vec<Example>>]>,
        space: &[Slot],
        averages: &mut Averages,
        mode: &RunMode,
    ) -> Result<Option<(Vec<usize>, f64)>> {
        let Some((params, _mean)) = averages.highest_average() else {
            return Ok(None);
        };
        let params = params.to_vec();
        averages.mark_evaluated(&params);
        apply(student, candidates, demo_sets, space, &params);
        let score = self.score(student, &mode.valset, mode).await?;
        Ok(Some((params, score)))
    }

    /// What one trial scores on — the whole valset, or a fresh subsample of it.
    fn batch(&self, mode: &RunMode, rng: &mut Rng) -> Vec<Example> {
        let size = if mode.minibatch {
            self.minibatch_size
        } else {
            mode.valset.len()
        };
        minibatch::batch(&mode.valset, size, rng)
    }

    /// dspy Evaluate's headline for one candidate: the metric's mean over the examples, as a
    /// percentage.
    ///
    /// Which pass this is comes from the same comparison upstream makes — `eval_candidate_program`
    /// reads `batch_size >= len(trainset)` — and reaches a watcher as dspy's `callback_metadata`.
    async fn score<S: Module + ?Sized>(
        &self,
        student: &S,
        examples: &[Example],
        mode: &RunMode,
    ) -> Result<f64> {
        let pass = match examples.len() >= mode.valset.len() {
            true => Pass::Full,
            false => Pass::Minibatch,
        };
        let evaluation = self
            .scoring
            .apply(Evaluate::new(
                examples.to_vec(),
                |inputs| student.forward(inputs),
                crate::evaluate::MetricRef(&self.metric),
            ))
            .pass(pass)
            .run()
            .await;
        Ok(evaluation?.score)
    }
}
