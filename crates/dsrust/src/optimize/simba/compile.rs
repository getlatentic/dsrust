//! The loop that drives [`search`](super::search)'s pieces.
//!
//! One `compile` is `max_steps` of: draw a mini-batch, sample trajectories, contrast them, build
//! candidates, score them on the same batch, and register them. Then a final slate is scored on the
//! whole trainset and the best of it is loaded into the student.
//!
//! Held to `optimize/simba.json`, which is every decision dspy's own SIMBA made on a scripted model.

use anyhow::{Result, bail};
use pyrng::cpython::Random;
use pyrng::pcg64::Pcg64;

use super::search::{Candidate, Compiled, Run, Simba, Step};
use super::{search, strategies};
use crate::evaluate::Metric;
use crate::example::Example;
use crate::module::{Module, ProgramState};

impl<M: Metric> Simba<M> {
    /// dspy `SIMBA.compile`, answering with the trace of what it decided.
    ///
    /// Returned rather than attached, for the reason `MIPROv2::compile_traced` returns its trials:
    /// a Rust program is the caller's value, and an optimizer writing `candidate_programs` and
    /// `trial_logs` onto it would have to invent somewhere to put them.
    pub async fn compile_traced(
        &self,
        student: &mut dyn Module,
        trainset: &[Example],
    ) -> Result<Compiled> {
        if trainset.len() < self.bsize {
            bail!("Trainset too small: {} < {}", trainset.len(), self.bsize);
        }
        // Taken once, as dspy takes `predictor2name` once: the search loads and reloads candidate
        // programs into the same predictors, which changes what they say and not which they are.
        let names = student.predictor_names();
        // Both generators take the same seed, and they are different generators: `Random` is
        // CPython's Mersenne Twister and `Pcg64` is numpy's `default_rng`.
        let mut rng = Random::seeded(self.seed);
        let mut numpy = Pcg64::seeded(self.seed);

        // dspy's `prompt_model or dspy.settings.lm`: the rule ask goes to the configured model.
        let advisor = crate::predict::Predict::from_signature(super::feedback::offer_feedback());
        let baseline = student.dump_state();
        let mut pool = search::Pool::seeded(baseline.clone());
        let mut winners: Vec<ProgramState> = vec![baseline.clone()];

        let mut indices: Vec<usize> = (0..trainset.len()).collect();
        rng.shuffle(&mut indices);
        let mut cursor = 0usize;

        let mut steps = Vec::new();
        for _ in 0..self.max_steps {
            let mut step = Step {
                batch: search::next_batch(
                    &mut rng,
                    &mut indices,
                    &mut cursor,
                    self.bsize,
                    trainset.len(),
                ),
                ..Step::default()
            };
            let batch: Vec<Example> = step.batch.iter().map(|at| trainset[*at].clone()).collect();

            let base = student.named_predictors().first().map(|p| p.config.clone());
            let configs =
                search::resampling_configs(&base.unwrap_or_default(), self.num_candidates);
            step.rollout_ids = configs.iter().filter_map(|c| c.rollout_id).collect();
            let top = pool.top_k_plus_baseline(self.num_candidates);

            // Model-major, which is the order the bucket stride below depends on.
            let mut runs = Vec::new();
            for config in &configs {
                for example in &batch {
                    let chosen =
                        pool.softmax_sample(&mut rng, &top, self.temperature_for_sampling)?;
                    pool.load(student, chosen)?;
                    for predictor in student.named_predictors() {
                        *predictor.config = config.clone();
                    }
                    runs.push(self.run_one(student, &names, example).await);
                }
            }
            step.percentiles = search::gates(&runs);
            let buckets = search::buckets_of(runs, self.bsize);

            let mut candidates: Vec<ProgramState> = Vec::new();
            for bucket in &buckets {
                let source =
                    pool.softmax_sample(&mut rng, &top, self.temperature_for_candidates)?;
                pool.load(student, source)?;

                let held = student
                    .named_predictors()
                    .iter()
                    .map(|predictor| predictor.demos.len())
                    .max()
                    .unwrap_or(0);
                let dropped = search::demos_to_drop(&mut rng, &mut numpy, held, self.max_demos);
                for predictor in student.named_predictors() {
                    let mut at = 0;
                    predictor.demos.retain(|_| {
                        let keep = !dropped.contains(&at);
                        at += 1;
                        keep
                    });
                }
                step.demos_dropped.push(dropped.len());

                let strategy = self.strategies()[rng.choice_index(self.strategies().len())].clone();
                let applied = strategies::apply(
                    &strategy,
                    bucket,
                    student,
                    step.percentiles,
                    self.demo_input_field_maxlen,
                    &advisor,
                )
                .await?;
                step.strategies.push((strategy, applied));

                candidates.push(student.dump_state());
                if candidates.len() > self.num_candidates {
                    break;
                }
            }

            for candidate in &candidates {
                student.load_state(candidate)?;
                let scores = self.score_over(student, &names, &batch).await;
                let average = mean(&scores);
                step.candidate_scores.push(average);
                step.candidates.push(candidate.clone());
                pool.register(candidate.clone(), scores);
            }
            if let Some(best) = best_index(&step.candidate_scores) {
                winners.push(candidates[best].clone());
            }
            steps.push(step);
        }

        // The final slate: `M` winners past the baseline over `N = num_candidates + 1` slots.
        let slate = search::slate(winners.len() - 1, self.num_candidates);
        let mut scored = Vec::new();
        for index in &slate {
            student.load_state(&winners[*index])?;
            scored.push(mean(&self.score_over(student, &names, trainset).await));
        }
        let best = best_index(&scored).unwrap_or(0);
        student.load_state(&winners[slate[best]])?;

        let mut candidates: Vec<Candidate> = slate
            .iter()
            .zip(&scored)
            .map(|(index, score)| Candidate {
                score: *score,
                program: winners[*index].clone(),
            })
            .collect();
        // dspy sorts `candidate_data` by score descending before attaching it. A stable sort keeps
        // the slate's own order among ties, as Python's does.
        candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
        Ok(Compiled { steps, candidates })
    }

    /// dspy `SIMBA.compile`, keeping only the compiled program.
    pub async fn compile(&self, student: &mut dyn Module, trainset: &[Example]) -> Result<()> {
        self.compile_traced(student, trainset).await.map(|_| ())
    }

    /// One example under the loaded program — dspy's `wrap_program`, whose whole job is that a
    /// program that raises scores zero rather than ending the run.
    async fn run_one(
        &self,
        student: &dyn Module,
        names: &crate::module::PredictorNames,
        example: &Example,
    ) -> Run {
        // dspy runs the program inside a `try` and logs a failure rather than raising: an example
        // the program cannot answer scores zero and the search carries on with the rest.
        let (prediction, trace) = match example.inputs() {
            Ok(inputs) => {
                let (answered, trace) = student.traced_with(names, inputs).await;
                (answered.ok(), trace)
            }
            Err(_) => (None, Vec::new()),
        };
        let score = match &prediction {
            Some(prediction) => self.metric.score(example, prediction).await,
            None => 0.0,
        };
        Run {
            example: example.clone(),
            prediction,
            trace,
            score,
        }
    }

    async fn score_over(
        &self,
        student: &dyn Module,
        names: &crate::module::PredictorNames,
        examples: &[Example],
    ) -> Vec<f64> {
        let mut scores = Vec::new();
        for example in examples {
            scores.push(self.run_one(student, names, example).await.score);
        }
        scores
    }
}

fn mean(scores: &[f64]) -> f64 {
    match scores.is_empty() {
        true => 0.0,
        false => scores.iter().sum::<f64>() / scores.len() as f64,
    }
}

/// dspy's `candidate_scores.index(max(...))`: the *first* maximum, which is what makes a tie keep
/// the earlier candidate.
fn best_index(scores: &[f64]) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .fold(None, |best: Option<(usize, f64)>, (at, score)| match best {
            Some((_, high)) if high >= *score => best,
            _ => Some((at, *score)),
        })
        .map(|(at, _)| at)
}
