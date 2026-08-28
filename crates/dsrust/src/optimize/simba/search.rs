//! SIMBA's compile loop: sample, contrast, and either show an example or write a rule.
//!
//! One step is: take the next mini-batch; run every example under several copies of the program at
//! different rollout ids; group the runs by example and sort those *buckets* by how much the score
//! varied; then for each bucket build a new candidate — dropping a random number of demos first —
//! by appending the best run as a demo, or by asking a model what the better run did that the worse
//! one did not. Score the candidates on the same batch, keep the best, and register them all.
//!
//! Everything stochastic here is Python's or numpy's, reproduced rather than reimplemented:
//! [`Random`] is CPython's Mersenne Twister and [`Pcg64`] is numpy's `default_rng`, which is a
//! *different* generator. The decisions the two produce are the whole of what
//! `optimize/simba.json` records.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use pyrng::cpython::Random;
use pyrng::pcg64::Pcg64;

use super::arithmetic::{final_slate, percentile};
use crate::evaluate::Metric;
use crate::example::{Example, Prediction};
use crate::lm::Sampling;
use crate::module::{Module, ProgramState, TraceStep};

/// What one program's run on one example produced — dspy's `wrap_program` output.
pub struct Run {
    pub example: Example,
    pub prediction: Option<Prediction>,
    pub trace: Vec<TraceStep>,
    pub score: f64,
}

/// One example's runs across the sampled programs, and the three numbers they are ordered by.
///
/// dspy sorts buckets by `(max_to_min_gap, max_score, max_to_avg_gap)` descending, so the example
/// whose outcome varied most is worked on first. The tuple order matters: a tie on the gap is
/// broken by the best score and only then by the gap to the average.
pub struct Bucket {
    pub runs: Vec<Run>,
    pub max_to_min_gap: f64,
    pub max_score: f64,
    pub max_to_avg_gap: f64,
}

impl Bucket {
    /// Runs sorted best first, and the three keys read off them.
    pub(super) fn of(mut runs: Vec<Run>) -> Self {
        runs.sort_by(|left, right| right.score.total_cmp(&left.score));
        let max_score = runs.first().map_or(0.0, |run| run.score);
        let min_score = runs.last().map_or(0.0, |run| run.score);
        let average = match runs.is_empty() {
            true => 0.0,
            false => runs.iter().map(|run| run.score).sum::<f64>() / runs.len() as f64,
        };
        Self {
            max_to_min_gap: max_score - min_score,
            max_score,
            max_to_avg_gap: max_score - average,
            runs,
        }
    }

    /// dspy's sort key, as one comparable value.
    pub(super) fn key(&self) -> (f64, f64, f64) {
        (self.max_to_min_gap, self.max_score, self.max_to_avg_gap)
    }
}

/// Which strategy a bucket invoked, and whether it changed the candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Strategy {
    /// dspy `append_a_demo`: the best run becomes a demo on every predictor it touched.
    AppendADemo,
    /// dspy `append_a_rule`: a model is asked what the better run did that the worse did not.
    AppendARule,
}

impl Strategy {
    /// dspy's `strategy.__name__`, which its log line prints and the golden records.
    pub fn name(&self) -> &'static str {
        match self {
            Strategy::AppendADemo => "append_a_demo_",
            Strategy::AppendARule => "append_a_rule",
        }
    }
}

/// What one step of the search decided — the trace `compile_traced` answers with.
#[derive(Debug, Clone, Default)]
pub struct Step {
    /// The trainset indices this step's mini-batch was drawn from.
    pub batch: Vec<usize>,
    /// The rollout ids `prepare_models_for_resampling` produced, all at temperature 1.0.
    pub rollout_ids: Vec<u64>,
    /// The 10th and 90th percentile of every score in the step, which gate both strategies.
    pub percentiles: (f64, f64),
    /// Per bucket, in the order they were worked: which strategy ran and whether it applied.
    pub strategies: Vec<(Strategy, bool)>,
    /// Per candidate, how many demos the poisson draw dropped.
    pub demos_dropped: Vec<usize>,
    /// Each new candidate's average score on the same mini-batch.
    pub candidate_scores: Vec<f64>,
    /// The candidates themselves, in the order they were built and registered.
    ///
    /// dspy registers all of them into its pool and keeps only the best in `winning_programs`, so
    /// a candidate a strategy changed can be scored, kept in the pool, and never reach the final
    /// slate. Held here because that is where the effect of a strategy is visible at all.
    pub candidates: Vec<ProgramState>,
}

/// One of the final slate, scored on the whole trainset — dspy's `candidate_programs` entry.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub score: f64,
    pub program: ProgramState,
}

/// What a run answers with: every step's decisions, and the scored slate.
///
/// dspy attaches `candidate_programs` and `trial_logs` to the compiled program. A Rust program is
/// the caller's value and an optimizer writing attributes onto it would have to invent somewhere to
/// put them, so they are returned — and the slate is *sorted by score, descending*, as upstream
/// sorts `candidate_data`.
#[derive(Debug, Clone, Default)]
pub struct Compiled {
    pub steps: Vec<Step>,
    pub candidates: Vec<Candidate>,
}

/// dspy `SIMBA`: stochastic introspective mini-batch ascent.
pub struct Simba<M> {
    pub metric: M,
    /// dspy `bsize=32`.
    pub bsize: usize,
    /// dspy `num_candidates=6`.
    pub num_candidates: usize,
    /// dspy `max_steps=8`.
    pub max_steps: usize,
    /// dspy `max_demos=4`. Zero drops `append_a_demo` from the strategy list entirely.
    pub max_demos: usize,
    /// dspy `demo_input_field_maxlen=100_000`.
    pub demo_input_field_maxlen: usize,
    /// dspy `temperature_for_sampling=0.2`.
    pub temperature_for_sampling: f64,
    /// dspy `temperature_for_candidates=0.2`.
    pub temperature_for_candidates: f64,
    /// dspy's `compile(seed=0)`, which seeds *both* generators.
    pub seed: u64,
}

impl<M: Metric> Simba<M> {
    pub fn new(metric: M) -> Self {
        Self {
            metric,
            bsize: 32,
            num_candidates: 6,
            max_steps: 8,
            max_demos: 4,
            demo_input_field_maxlen: 100_000,
            temperature_for_sampling: 0.2,
            temperature_for_candidates: 0.2,
            seed: 0,
        }
    }

    /// The strategies in dspy's order — a demo first, unless `max_demos` is zero.
    pub(super) fn strategies(&self) -> Vec<Strategy> {
        match self.max_demos > 0 {
            true => vec![Strategy::AppendADemo, Strategy::AppendARule],
            false => vec![Strategy::AppendARule],
        }
    }
}

/// The program pool: each entry is a saved state and the scores it has been given.
///
/// dspy keeps deep copies of `dspy.Module`; a Rust program is the caller's one value, so the pool
/// holds [`ProgramState`] and a candidate is *loaded* into the student to be run. The index is
/// upstream's `simba_idx`, and index 0 is always the baseline.
pub(super) struct Pool {
    states: Vec<ProgramState>,
    scores: Vec<Vec<f64>>,
}

impl Pool {
    pub(super) fn seeded(baseline: ProgramState) -> Self {
        Self {
            states: vec![baseline],
            scores: vec![Vec::new()],
        }
    }

    /// dspy's `calc_average_score`: zero for a program nothing has scored yet.
    pub(super) fn average(&self, index: usize) -> f64 {
        let scores = &self.scores[index];
        match scores.is_empty() {
            true => 0.0,
            false => scores.iter().sum::<f64>() / scores.len() as f64,
        }
    }

    /// dspy's `top_k_plus_baseline`: the k best by average, with the baseline forced into the last
    /// slot, deduped with order kept.
    ///
    /// The forcing overwrites rather than appends — `top_k[-1] = 0` — so a pool whose best k
    /// already holds the baseline keeps k entries and one that does not *loses* its k-th.
    pub(super) fn top_k_plus_baseline(&self, k: usize) -> Vec<usize> {
        let mut ranked: Vec<usize> = (0..self.states.len()).collect();
        // dspy sorts the program list, which is in registration order, by descending average. A
        // stable sort keeps registration order among ties, as Python's does.
        ranked.sort_by(|left, right| self.average(*right).total_cmp(&self.average(*left)));
        let mut top: Vec<usize> = ranked.into_iter().take(k).collect();
        if !top.contains(&0)
            && let Some(last) = top.last_mut()
        {
            *last = 0;
        }
        let mut seen = Vec::new();
        for index in top {
            if !seen.contains(&index) {
                seen.push(index);
            }
        }
        seen
    }

    /// dspy's `softmax_sample`: weights are `exp(average / temperature)`, normalised.
    ///
    /// A pool whose weights sum to zero falls back to a uniform `rng.choice`, which spends the
    /// generator differently from `rng.choices` — so the fallback is part of the stream and not
    /// only of the answer.
    pub(super) fn softmax_sample(
        &self,
        rng: &mut Random,
        among: &[usize],
        temperature: f64,
    ) -> Result<usize> {
        if among.is_empty() {
            bail!("No programs available for softmax sampling.");
        }
        let exponentials: Vec<f64> = among
            .iter()
            .map(|index| (self.average(*index) / temperature).exp())
            .collect();
        let total: f64 = exponentials.iter().sum();
        if total <= 0.0 {
            return Ok(among[rng.choice_index(among.len())]);
        }
        let weights: Vec<f64> = exponentials.iter().map(|value| value / total).collect();
        let picked = rng.choices(&weights, 1);
        Ok(among[picked[0]])
    }

    pub(super) fn register(&mut self, state: ProgramState, scores: Vec<f64>) -> usize {
        self.states.push(state);
        self.scores.push(scores);
        self.states.len() - 1
    }

    /// Put one pooled program into the student, which is how a Rust port holds a pool at all: dspy
    /// keeps deep copies and runs them, and here there is one program the caller owns.
    pub(super) fn load(&self, student: &mut dyn Module, index: usize) -> Result<()> {
        student.load_state(&self.states[index])
    }
}

/// dspy `prepare_models_for_resampling`: `n` copies of the model at rollout ids `0..n`,
/// temperature 1.0.
///
/// The rollout id is what makes two otherwise identical calls two calls rather than one cache hit,
/// so it is the whole mechanism behind sampling several trajectories for one example.
pub fn resampling_configs(base: &Sampling, count: usize) -> Vec<Sampling> {
    let start = base.rollout_id.unwrap_or(0);
    (0..count as u64)
        .map(|offset| Sampling {
            rollout_id: Some(start + offset),
            temperature: Some(1.0),
            ..base.clone()
        })
        .collect()
}

/// dspy's demo-drop: a poisson draw over `num_demos / max_demos`, floored at one once a predictor
/// is at or past the cap, then that many indices drawn *with replacement*.
///
/// With replacement is upstream's, and it matters: `[rng.randrange(n) for _ in range(k)]` can draw
/// the same index twice, so `k` draws drop *at most* `k` demos and often fewer.
pub(super) fn demos_to_drop(
    rng: &mut Random,
    numpy: &mut Pcg64,
    num_demos: usize,
    max_demos: usize,
) -> Vec<usize> {
    let cap = match max_demos > 0 {
        true => max_demos,
        false => 3,
    };
    let drawn = numpy.poisson(num_demos as f64 / cap as f64) as usize;
    let floor = usize::from(num_demos >= cap);
    let count = drawn.max(floor).min(num_demos);
    if num_demos == 0 {
        return Vec::new();
    }
    (0..count).map(|_| rng.below(num_demos)).collect()
}

/// The buckets one step produces, in dspy's worked order.
///
/// The chunking is by *stride*: `outputs[idx::bsize]` gathers every model's run of example `idx`,
/// which is only the same as grouping by example because the runs were appended model-major.
pub fn buckets_of(runs: Vec<Run>, bsize: usize) -> Vec<Bucket> {
    let mut grouped: BTreeMap<usize, Vec<Run>> = BTreeMap::new();
    for (at, run) in runs.into_iter().enumerate() {
        grouped.entry(at % bsize).or_default().push(run);
    }
    let mut buckets: Vec<Bucket> = grouped.into_values().map(Bucket::of).collect();
    buckets.sort_by(|left, right| {
        let (left, right) = (right.key(), left.key());
        left.0
            .total_cmp(&right.0)
            .then(left.1.total_cmp(&right.1))
            .then(left.2.total_cmp(&right.2))
    });
    buckets
}

/// The 10th and 90th percentile of every score in a step, which gate both strategies.
pub(super) fn gates(runs: &[Run]) -> (f64, f64) {
    let scores: Vec<f64> = runs.iter().map(|run| run.score).collect();
    (
        percentile(&scores, 10.0).unwrap_or(0.0),
        percentile(&scores, 90.0).unwrap_or(0.0),
    )
}

/// dspy's final slate over the winning programs — `[round(i * M / (N - 1)) for i in range(N)]`,
/// deduped.
pub(super) fn slate(winners: usize, num_candidates: usize) -> Vec<usize> {
    final_slate(winners, num_candidates)
}

/// The mini-batch this step draws, reshuffling when the cursor would run past the end.
///
/// dspy reshuffles the *whole* index list and restarts at zero rather than wrapping, so an example
/// can appear in two consecutive batches and the order after a reshuffle is not a rotation of the
/// order before it.
pub(super) fn next_batch(
    rng: &mut Random,
    indices: &mut Vec<usize>,
    cursor: &mut usize,
    bsize: usize,
    trainset: usize,
) -> Vec<usize> {
    if *cursor + bsize > trainset {
        rng.shuffle(indices);
        *cursor = 0;
    }
    let batch = indices[*cursor..(*cursor + bsize).min(indices.len())].to_vec();
    *cursor += bsize;
    batch
}
