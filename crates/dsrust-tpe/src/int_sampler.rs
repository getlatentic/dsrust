//! optuna's `TPESampler` over `IntDistribution`s, at the defaults `optuna.create_study()` gives.
//!
//! Two things separate this from the categorical sampler beside it. The parameters are drawn
//! **independently** — `multivariate` is false by default, so each goes through
//! `sample_independent` with a search space of one rather than being drawn jointly — and each is
//! modelled as a point on a line rather than a choice among options, which is [`Numerical`].
//!
//! `BootstrapFewShotWithOptuna` is the caller: one `suggest_int` per predictor, naming the demo it
//! keeps.

use pyrng::RandomState;

use crate::numerical::{IntRange, Numerical, half_to_even};

/// optuna `TPESampler(n_startup_trials=10)`.
const N_STARTUP_TRIALS: usize = 10;
/// optuna `TPESampler(n_ei_candidates=24)`.
const N_EI_CANDIDATES: usize = 24;
/// optuna `_ParzenEstimatorParameters(prior_weight=1.0)`.
const PRIOR_WEIGHT: f64 = 1.0;

struct Trial {
    params: Vec<i64>,
    value: f64,
}

/// A sampler over integer parameters, seeded as `TPESampler(seed=...)`.
pub struct IntTpeSampler {
    ranges: Vec<IntRange>,
    /// Two generators from one seed, as optuna builds them: `TPESampler` keeps its own and hands
    /// the same seed to the `RandomSampler` it falls back on for the startup trials.
    startup_rng: RandomState,
    tpe_rng: RandomState,
    trials: Vec<Trial>,
}

impl IntTpeSampler {
    /// The parameters, in the order the objective suggests them. Each is `(low, high)` inclusive.
    pub fn new(seed: u32, parameters: Vec<(i64, i64)>) -> Self {
        Self {
            ranges: parameters
                .into_iter()
                .map(|(low, high)| IntRange::new(low, high))
                .collect(),
            startup_rng: RandomState::new(seed),
            tpe_rng: RandomState::new(seed),
            trials: Vec::new(),
        }
    }

    /// The next trial's value per parameter.
    ///
    /// Each parameter is drawn on its own, in suggest order, and every draw advances the same
    /// generator — so a port that batched them would agree on the first parameter and diverge on
    /// the rest.
    pub fn ask(&mut self) -> Vec<i64> {
        let startup = self.trials.len() < N_STARTUP_TRIALS;
        (0..self.ranges.len())
            .map(|parameter| match startup {
                true => self.sample_startup(parameter),
                false => self.sample_tpe(parameter),
            })
            .collect()
    }

    /// Record a trial and what it scored.
    pub fn tell(&mut self, params: Vec<i64>, value: f64) {
        self.trials.push(Trial { params, value });
    }

    /// optuna's `RandomSampler.sample_independent`: one uniform over the widened bounds, snapped
    /// back onto the grid.
    fn sample_startup(&mut self, parameter: usize) -> i64 {
        let range = self.ranges[parameter];
        let (low, high) = (range.low - 0.5, range.high + 0.5);
        let drawn = low + (high - low) * self.startup_rng.random_sample();
        untransform(drawn, range)
    }

    /// optuna's TPE step for one parameter: fit an estimator to the better trials and one to the
    /// rest, draw candidates from the better, and keep the candidate the density ratio likes most.
    fn sample_tpe(&mut self, parameter: usize) -> i64 {
        let range = self.ranges[parameter];
        let (below, above) = self.split(default_gamma(self.trials.len()), parameter);
        let below = Numerical::build(&below, range, PRIOR_WEIGHT);
        let above = Numerical::build(&above, range, PRIOR_WEIGHT);
        let candidates = below.sample(&mut self.tpe_rng, N_EI_CANDIDATES);
        let (below_pdf, above_pdf) = (below.log_pdf(&candidates), above.log_pdf(&candidates));
        let acquisitions: Vec<f64> = below_pdf
            .iter()
            .zip(&above_pdf)
            .map(|(below, above)| below - above)
            .collect();
        candidates[argmax(&acquisitions)] as i64
    }

    /// optuna `_split_trials`: the best `gamma` trials by score go below, the rest above, and each
    /// group is put back into trial order.
    fn split(&self, gamma: usize, parameter: usize) -> (Vec<f64>, Vec<f64>) {
        let n_below = gamma.min(self.trials.len());
        let mut order: Vec<usize> = (0..self.trials.len()).collect();
        order.sort_by(|&a, &b| self.trials[b].value.total_cmp(&self.trials[a].value));
        let mut below: Vec<usize> = order[..n_below].to_vec();
        let mut above: Vec<usize> = order[n_below..].to_vec();
        below.sort_unstable();
        above.sort_unstable();
        let values = |indices: Vec<usize>| {
            indices
                .iter()
                .map(|&i| self.trials[i].params[parameter] as f64)
                .collect()
        };
        (values(below), values(above))
    }
}

/// optuna `_untransform_numerical_param` for a non-log `IntDistribution` at step one.
fn untransform(value: f64, range: IntRange) -> i64 {
    (range.low + half_to_even(value - range.low)).clamp(range.low, range.high) as i64
}

/// optuna `default_gamma`: a tenth of the trials, rounded up, capped at 25.
fn default_gamma(n: usize) -> usize {
    ((0.1 * n as f64).ceil() as usize).min(25)
}

/// numpy `argmax`: the index of the largest value, the first on a tie.
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The untransform is relative to the range's own low, which every dspy caller leaves at zero —
    /// `suggest_int(name, 0, len - 1)` — so a range starting elsewhere is the only thing that can
    /// tell the offset from an absent one.
    #[test]
    fn the_untransform_is_relative_to_the_ranges_low() {
        let range = IntRange::new(3, 8);
        // 5.5 is 2.5 above the low, and half-to-even takes that to 2.
        assert_eq!(untransform(5.5, range), 5);
        assert_eq!(
            untransform(6.5, range),
            7,
            "the next half rounds the other way"
        );
        assert_eq!(untransform(2.6, range), 3, "below the low, clamped up");
        assert_eq!(untransform(99.0, range), 8, "above the high, clamped down");
        assert_eq!(
            untransform(3.0, IntRange::new(0, 5)),
            3,
            "and a zero low is the plain case"
        );
    }

    /// `default_gamma` is a tenth, rounded up, capped at 25 — the split every TPE trial is built on.
    #[test]
    fn the_split_size_is_a_capped_tenth() {
        assert_eq!(default_gamma(0), 0);
        assert_eq!(default_gamma(1), 1);
        assert_eq!(default_gamma(10), 1);
        assert_eq!(default_gamma(11), 2, "rounded up, not down");
        assert_eq!(default_gamma(250), 25);
        assert_eq!(default_gamma(1000), 25, "capped");
    }
}
