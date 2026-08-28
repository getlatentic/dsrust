//! Scoring a trial on a subsample, and the full evaluations that interrupt a run of them.
//!
//! A minibatch run does not merely score cheaply. Upstream's `_perform_full_evaluation` picks the
//! best-*averaging* combination seen so far, scores it on the whole valset, and feeds that score
//! back into the sampler as an extra trial — so a full evaluation moves every draw after it. A port
//! that ran full evaluations and only recorded the number would agree on every score and diverge on
//! the search.

use crate::Example;

use super::super::rng::Rng;

/// dspy `MIN_MINIBATCH_SIZE`: the subsampled valset size above which `auto` turns minibatching on.
pub const MIN_MINIBATCH_SIZE: usize = 50;

/// dspy `auto="light" | "medium" | "heavy"`: a budget preset, which is a candidate count and a
/// valset size.
///
/// The valset size is the part with reach. `auto` subsamples the valset to it — off the same
/// generator the bootstrap and the proposer read, and before either of them — so choosing a preset
/// moves every later draw, not just how many trials run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Auto {
    /// dspy `"light"`: 6 candidates over 100 validation examples.
    Light,
    /// dspy `"medium"`: 12 candidates over 300.
    Medium,
    /// dspy `"heavy"`: 18 candidates over 1000.
    Heavy,
}

impl Auto {
    /// `AUTO_RUN_SETTINGS[mode]["n"]`: how many few-shot sets are built, and — halved when demos are
    /// searched — how many instructions are proposed.
    pub fn candidates(self) -> usize {
        match self {
            Self::Light => 6,
            Self::Medium => 12,
            Self::Heavy => 18,
        }
    }

    /// `AUTO_RUN_SETTINGS[mode]["val_size"]`: what the valset is subsampled to.
    pub fn val_size(self) -> usize {
        match self {
            Self::Light => 100,
            Self::Medium => 300,
            Self::Heavy => 1000,
        }
    }

    /// dspy's split of one count into two: instructions get half the budget when demos are also
    /// searched, since upstream found the budget better spent on the demos.
    pub fn instruction_candidates(self, zeroshot: bool) -> usize {
        if zeroshot {
            self.candidates()
        } else {
            // `int(n * 0.5)`, which truncates — 6/12/18 all halve exactly, but the rounding is
            // upstream's and a preset added later need not.
            (self.candidates() as f64 * 0.5) as usize
        }
    }
}

/// dspy `create_minibatch`: `batch_size` examples drawn without replacement, in draw order.
///
/// Kept separate from the decision to draw at all, because `eval_candidate_program` takes no draw
/// when the batch covers the set — so how far the generator advances turns on the size, and a port
/// that always drew would diverge on every trial after the first.
pub fn subsample(examples: &[Example], batch_size: usize, rng: &mut Rng) -> Vec<Example> {
    let batch_size = batch_size.min(examples.len());
    let indices: Vec<usize> = (0..examples.len()).collect();
    rng.sample(&indices, batch_size)
        .into_iter()
        .map(|index| examples[index].clone())
        .collect()
}

/// dspy `eval_candidate_program`: the examples one trial scores on — the whole set, or a fresh
/// subsample of it.
pub fn batch(examples: &[Example], batch_size: usize, rng: &mut Rng) -> Vec<Example> {
    if batch_size >= examples.len() {
        examples.to_vec()
    } else {
        subsample(examples, batch_size, rng)
    }
}

/// dspy's `adjusted_num_trials`: the trial count a minibatch run *displays*, counting the full
/// evaluations interleaved with the minibatch trials.
///
/// Only the display and [`full_evaluation_due`]'s tail condition read it; the study still runs
/// `num_trials` trials. Note it divides by `full_eval_steps` where the interval below divides by
/// `full_eval_steps + 1`, so the two disagree and the tail condition usually cannot fire. That is
/// upstream's arithmetic, reproduced rather than corrected.
pub fn adjusted_trials(num_trials: usize, full_eval_steps: usize) -> usize {
    let trailing = usize::from(!num_trials.is_multiple_of(full_eval_steps));
    num_trials + num_trials / full_eval_steps + 1 + trailing
}

/// Whether trial `trial` — counted from one, as upstream counts — is followed by a full evaluation.
pub fn full_evaluation_due(trial: usize, adjusted: usize, full_eval_steps: usize) -> bool {
    trial.is_multiple_of(full_eval_steps + 1) || trial + 1 == adjusted
}

/// dspy's `param_score_dict` and `fully_evaled_param_combos`: every score each parameter
/// combination has earned, and which combinations have already had a full pass.
///
/// Insertion-ordered rather than keyed, because `get_program_with_highest_avg_score` sorts by mean
/// with a stable sort — so two combinations averaging the same are separated by which was *scored*
/// first, not by how their parameters compare. A `BTreeMap` would agree until the first tie.
#[derive(Default)]
pub struct Averages {
    scored: Vec<(Vec<usize>, Vec<f64>)>,
    fully_evaluated: Vec<Vec<usize>>,
}

impl Averages {
    /// Record what one trial's combination scored on its minibatch.
    pub fn record(&mut self, params: &[usize], score: f64) {
        match self.scored.iter_mut().find(|(seen, _)| seen == params) {
            Some((_, scores)) => scores.push(score),
            None => self.scored.push((params.to_vec(), vec![score])),
        }
    }

    /// dspy `get_program_with_highest_avg_score`: the highest-averaging combination that has not
    /// already had a full evaluation, and its mean.
    ///
    /// Falls through to the *lowest*-averaging combination when every one has been fully evaluated,
    /// which is the pin leaving its loop variables bound rather than a considered choice. dspy's
    /// main branch raises there instead; the pin is the oracle.
    pub fn highest_average(&self) -> Option<(&[usize], f64)> {
        let mut ranked: Vec<(&[usize], f64)> = self
            .scored
            .iter()
            .map(|(params, scores)| {
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                (params.as_slice(), mean)
            })
            .collect();
        // Stable, and descending by the same comparison Python's `reverse=True` makes: ties keep the
        // order they were first scored in.
        ranked.sort_by(|(_, left), (_, right)| {
            right.partial_cmp(left).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
            .iter()
            .find(|(params, _)| !self.fully_evaluated.iter().any(|done| done == params))
            .or_else(|| ranked.last())
            .copied()
    }

    /// Note that a combination has had its full pass, so a later full evaluation skips it.
    pub fn mark_evaluated(&mut self, params: &[usize]) {
        if !self.fully_evaluated.iter().any(|done| done == params) {
            self.fully_evaluated.push(params.to_vec());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tie is the whole point: two combinations averaging the same are separated by which was
    /// scored first. A keyed map would return `[0, 1]` here whatever the order.
    #[test]
    fn a_tie_goes_to_whichever_was_scored_first() {
        let mut averages = Averages::default();
        averages.record(&[1, 0], 0.5);
        averages.record(&[0, 1], 0.5);
        assert_eq!(averages.highest_average().expect("scored").0, [1, 0]);
    }

    /// A mean over every batch, not the most recent score. The two disagree here: `[0]` averages
    /// 0.5 and last scored 0.0, so reading only the latest would pick `[1]`.
    #[test]
    fn the_pick_averages_every_batch_a_combination_scored() {
        let mut averages = Averages::default();
        averages.record(&[0], 1.0);
        averages.record(&[0], 0.0);
        averages.record(&[1], 0.4);
        let (params, mean) = averages.highest_average().expect("scored");
        assert_eq!(params, [0]);
        assert!((mean - 0.5).abs() < f64::EPSILON);
    }

    /// The pin's fall-through: every combination fully evaluated leaves the loop variables on the
    /// last one it looked at, which is the lowest-averaging.
    #[test]
    fn falling_through_lands_on_the_lowest_average() {
        let mut averages = Averages::default();
        averages.record(&[0], 0.9);
        averages.record(&[1], 0.1);
        averages.mark_evaluated(&[0]);
        averages.mark_evaluated(&[1]);
        assert_eq!(averages.highest_average().expect("scored").0, [1]);
    }

    /// dspy's two divisors disagree, so with the defaults the tail condition never fires and a
    /// 10-trial run takes one mid-run full evaluation rather than the three the count implies.
    #[test]
    fn the_displayed_trial_count_overcounts_the_full_evaluations_that_fire() {
        let adjusted = adjusted_trials(10, 5);
        assert_eq!(adjusted, 13);
        let fired: Vec<usize> = (1..=10)
            .filter(|trial| full_evaluation_due(*trial, adjusted, 5))
            .collect();
        assert_eq!(fired, [6]);
    }
}
