//! optuna's Parzen estimator over a *discrete* numerical parameter — an `IntDistribution`.
//!
//! The categorical estimator beside this one models a choice among unrelated options. This one
//! models a point on a line: each observed trial contributes a truncated normal centred on it, with
//! a width taken from how far its neighbours are, and the mixture is completed by a flat prior
//! kernel spanning the range.
//!
//! Held to `tests/conformance/int_tpe.json`, which records whole draw sequences from optuna rather
//! than any one of these numbers — the shape and the generator have to be right together.

use pyrng::RandomState;

use crate::parzen::pairwise_sum;
use crate::truncnorm;

/// One `IntDistribution`. `step` is one for every parameter dspy asks for, so the bounds the kernels
/// live between are the integer range widened by half a step on each side.
#[derive(Clone, Copy)]
pub(crate) struct IntRange {
    pub(crate) low: f64,
    pub(crate) high: f64,
}

impl IntRange {
    pub(crate) fn new(low: i64, high: i64) -> Self {
        Self {
            low: low as f64,
            high: high as f64,
        }
    }

    /// The continuous interval the kernels are truncated to: `[low - step/2, high + step/2]`.
    fn widened(&self) -> (f64, f64) {
        (self.low - 0.5, self.high + 0.5)
    }
}

/// optuna's `_ParzenEstimator` for one discrete numerical parameter.
pub(crate) struct Numerical {
    /// Mixture weight per kernel, normalised. One per observation plus the trailing prior.
    weights: Vec<f64>,
    mus: Vec<f64>,
    sigmas: Vec<f64>,
    range: IntRange,
}

impl Numerical {
    /// Build over the observed values of this parameter.
    pub(crate) fn build(observations: &[f64], range: IntRange, prior_weight: f64) -> Self {
        let (low, high) = range.widened();
        let mut weights = match observations.is_empty() {
            true => vec![1.0],
            false => {
                let mut w = default_weights(observations.len());
                w.push(prior_weight);
                w
            }
        };
        let total: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= total;
        }

        let mut sigmas = neighbour_widths(observations, low, high);
        // The "magic clip": a kernel may be no wider than the range and no narrower than a slice of
        // it that shrinks as kernels accumulate, so a run of identical observations cannot collapse
        // into a spike.
        let n_kernels = observations.len() + 1;
        let min_sigma = (high - low) / 100.0_f64.min(1.0 + n_kernels as f64);
        let max_sigma = high - low;
        for sigma in &mut sigmas {
            *sigma = sigma.clamp(min_sigma, max_sigma);
        }

        let mut mus = observations.to_vec();
        // The prior kernel: centred on the range, as wide as it is.
        mus.push(0.5 * (low + high));
        sigmas.push(high - low);
        Self {
            weights,
            mus,
            sigmas,
            range,
        }
    }

    /// optuna's `_MixtureOfProductDistribution.sample` for a single discrete truncated normal.
    ///
    /// One kernel is drawn per candidate from the mixture weights, then a truncated normal from
    /// that kernel, then the value is snapped back onto the integer grid.
    pub(crate) fn sample(&self, rng: &mut RandomState, count: usize) -> Vec<f64> {
        let active = rng.choice(&self.weights, count);
        let (low, high) = self.range.widened();
        let mus: Vec<f64> = active.iter().map(|&k| self.mus[k]).collect();
        let sigmas: Vec<f64> = active.iter().map(|&k| self.sigmas[k]).collect();
        let a: Vec<f64> = mus
            .iter()
            .zip(&sigmas)
            .map(|(mu, sigma)| (low - mu) / sigma)
            .collect();
        let b: Vec<f64> = mus
            .iter()
            .zip(&sigmas)
            .map(|(mu, sigma)| (high - mu) / sigma)
            .collect();
        let quantiles = rng.random_sample_n(count);
        let drawn = truncnorm::ppf(&quantiles, &a, &b);
        drawn
            .iter()
            .zip(&mus)
            .zip(&sigmas)
            .map(|((&x, &mu), &sigma)| {
                let value = x * sigma + mu;
                // `clip(low + round((v - low) / step) * step, low, high)` at `step == 1`,
                // against the *unwidened* bounds — and `np.round` is banker's rounding, so a value
                // landing exactly halfway goes to the even neighbour rather than away from zero.
                (self.range.low + half_to_even(value - self.range.low))
                    .clamp(self.range.low, self.range.high)
            })
            .collect()
    }

    /// optuna's `log_pdf` for the same: the mass this mixture puts on each candidate's own integer
    /// cell, normalised by the mass inside the bounds.
    pub(crate) fn log_pdf(&self, values: &[f64]) -> Vec<f64> {
        let (low, high) = self.range.widened();
        values
            .iter()
            .map(|&x| {
                let terms: Vec<f64> = self
                    .mus
                    .iter()
                    .zip(&self.sigmas)
                    .zip(&self.weights)
                    .map(|((&mu, &sigma), &weight)| {
                        let cell = truncnorm::log_gauss_mass(
                            (x - 0.5 - mu) / sigma,
                            (x + 0.5 - mu) / sigma,
                        );
                        let inside =
                            truncnorm::log_gauss_mass((low - mu) / sigma, (high - mu) / sigma);
                        cell - inside + weight.ln()
                    })
                    .collect();
                logsumexp(&terms)
            })
            .collect()
    }
}

/// optuna's non-multivariate sigma rule: each kernel is as wide as the larger gap to its
/// neighbours, with the range's ends standing in for the missing outer ones.
///
/// `consider_endpoints` is false at optuna's defaults, which rewrites the first and last widths to
/// ignore the endpoints they were just given — but only once there are enough points for the
/// endpoints to have been counted, which is the `>= 4` guard on the padded array.
fn neighbour_widths(observations: &[f64], low: f64, high: f64) -> Vec<f64> {
    if observations.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..observations.len()).collect();
    order.sort_by(|&a, &b| observations[a].total_cmp(&observations[b]));
    let mut padded = Vec::with_capacity(observations.len() + 2);
    padded.push(low);
    padded.extend(order.iter().map(|&i| observations[i]));
    padded.push(high);

    let mut sorted: Vec<f64> = (1..padded.len() - 1)
        .map(|i| (padded[i] - padded[i - 1]).max(padded[i + 1] - padded[i]))
        .collect();
    if padded.len() >= 4 {
        let last = sorted.len() - 1;
        sorted[0] = padded[2] - padded[1];
        sorted[last] = padded[padded.len() - 2] - padded[padded.len() - 3];
    }

    // Back into observation order — `sorted_sigmas[np.argsort(sorted_indices)]`.
    let mut widths = vec![0.0; observations.len()];
    for (position, &index) in order.iter().enumerate() {
        widths[index] = sorted[position];
    }
    widths
}

/// optuna's `default_weights` for the numerical path: the oldest trials fade linearly once there
/// are more than 25, and the newest 25 all count fully.
fn default_weights(n: usize) -> Vec<f64> {
    if n < 25 {
        return vec![1.0; n];
    }
    let ramp: Vec<f64> = (0..n - 25)
        .map(|i| (i as f64 + 1.0) / (n - 25) as f64)
        .collect();
    ramp.into_iter()
        .chain(std::iter::repeat_n(1.0, 25))
        .collect()
}

/// `np.round`: half to even, which Rust's `f64::round` is not.
pub(crate) fn half_to_even(value: f64) -> f64 {
    let nearest = value.round();
    match (value - value.trunc()).abs() == 0.5 && nearest % 2.0 != 0.0 {
        true => nearest - value.signum(),
        false => nearest,
    }
}

/// optuna's mixture reduction, which is numpy arithmetic and not the textbook one: the shift is
/// zeroed rather than kept when the maximum is `-inf`, and the sum is numpy's pairwise one.
///
/// The summation order is not a detail. Two candidates either side of a symmetric estimator earn
/// the same density, and which of them `argmax` keeps is decided by the last bits — a sequential
/// sum picked the wrong one of a tied pair at one trial in twelve recorded runs.
fn logsumexp(values: &[f64]) -> f64 {
    let mut largest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if largest == f64::NEG_INFINITY {
        largest = 0.0;
    }
    let shifted: Vec<f64> = values.iter().map(|value| (value - largest).exp()).collect();
    pairwise_sum(&shifted).ln() + largest
}
