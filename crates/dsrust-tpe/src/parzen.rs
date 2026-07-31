//! optuna's categorical Parzen estimator: the mixture of one kernel per observed trial plus a
//! uniform prior, over the categorical parameters MIPROv2 searches.
//!
//! Each kernel is a categorical distribution per parameter — an observation adds one to its own
//! category over a flat `prior_weight / n_kernels` floor, the prior kernel stays flat — and the
//! mixture weights come from optuna's `default_weights`. The estimator both draws candidates
//! ([`Parzen::sample`]) and scores them ([`Parzen::pdf`]); the sampler above ranks candidates by the
//! ratio of the "below" density to the "above" one.

use pyrng::RandomState;

pub(crate) struct Parzen {
    /// Mixture weight per kernel, normalised. Length is `observations + 1` (the trailing prior).
    mixture: Vec<f64>,
    /// Per parameter, per kernel, the normalised category distribution.
    categorical: Vec<Vec<Vec<f64>>>,
}

impl Parzen {
    /// Build the estimator over these observations (each a category per parameter). optuna's
    /// `_ParzenEstimator` for an all-categorical, multivariate search space.
    pub fn build(observations: &[Vec<usize>], cardinalities: &[usize], prior_weight: f64) -> Self {
        let mixture = mixture_weights(observations.len(), prior_weight);
        let categorical = cardinalities
            .iter()
            .enumerate()
            .map(|(param, &cardinality)| {
                categorical_kernel(observations, param, cardinality, prior_weight)
            })
            .collect();
        Self {
            mixture,
            categorical,
        }
    }

    /// Draw `n` candidates — optuna `_MixtureOfProductDistribution.sample`. First one kernel is
    /// chosen per candidate (`choice`), then each parameter's category is drawn from that kernel by
    /// placing a uniform into the kernel's cumulative weights. The draw order — the `choice`, then a
    /// `rand` per parameter — is the order optuna consumes the generator in.
    pub fn sample(&self, rng: &mut RandomState, n: usize, parameters: usize) -> Vec<Vec<usize>> {
        let active = rng.choice(&self.mixture, n);
        let mut candidates = vec![vec![0usize; parameters]; n];
        for (param, kernels) in self.categorical.iter().enumerate() {
            let quantiles = rng.random_sample_n(n);
            for (candidate, &kernel) in active.iter().enumerate() {
                candidates[candidate][param] =
                    pick_category(&kernels[kernel], quantiles[candidate]);
            }
        }
        candidates
    }

    /// One candidate's log density under the mixture — optuna
    /// `_MixtureOfProductDistribution.log_pdf`.
    ///
    /// The expression is reproduced rather than simplified. Summing the kernels in linear space and
    /// ranking by the ratio is algebraically the same and numerically is not: it breaks the exact
    /// ties optuna's acquisition is full of, and `np.argmax` keeps the first of a tie, so a tie
    /// broken by last-bit noise picks a different trial.
    pub fn log_pdf(&self, candidate: &[usize]) -> f64 {
        let weighted: Vec<f64> = self
            .mixture
            .iter()
            .enumerate()
            .map(|(kernel, weight)| {
                let logs: Vec<f64> = candidate
                    .iter()
                    .enumerate()
                    .map(|(param, &category)| self.categorical[param][kernel][category].ln())
                    .collect();
                pairwise_sum(&logs) + weight.ln()
            })
            .collect();
        // optuna zeroes a `-inf` maximum so the shift below never subtracts infinity from itself.
        let mut largest = weighted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if largest == f64::NEG_INFINITY {
            largest = 0.0;
        }
        let shifted: Vec<f64> = weighted
            .iter()
            .map(|value| (value - largest).exp())
            .collect();
        pairwise_sum(&shifted).ln() + largest
    }
}

/// numpy's `add.reduce` over a contiguous double array — pairwise summation, which is not a left
/// fold and rounds differently past eight elements.
///
/// Every sum numpy takes goes through this, not only the log-sum-exp: normalising a kernel row sums
/// over the categories, so a program proposing twelve instructions rounds differently from one
/// proposing six. That difference reaches the acquisition, which ties constantly, and `np.argmax`
/// keeps the first of a tie — so it decides which trial runs next.
fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        return values.iter().sum();
    }
    if n <= BLOCK {
        let mut accumulators: [f64; 8] = values[..8].try_into().expect("eight values");
        let mut index = 8;
        while index + 8 <= n {
            for (slot, value) in accumulators.iter_mut().zip(&values[index..index + 8]) {
                *slot += value;
            }
            index += 8;
        }
        let mut total = ((accumulators[0] + accumulators[1]) + (accumulators[2] + accumulators[3]))
            + ((accumulators[4] + accumulators[5]) + (accumulators[6] + accumulators[7]));
        for value in &values[index..] {
            total += value;
        }
        return total;
    }
    let half = (n / 2) - (n / 2) % 8;
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

/// optuna `default_weights` appended with the prior, then normalised: the mixture weights over the
/// observed kernels and the prior. Every kernel weighs one until there are 25 observations, past
/// which the oldest ramp down.
fn mixture_weights(observations: usize, prior_weight: f64) -> Vec<f64> {
    if observations == 0 {
        return vec![1.0];
    }
    let mut weights = default_weights(observations);
    weights.push(prior_weight);
    let sum = pairwise_sum(&weights);
    weights.iter().map(|weight| weight / sum).collect()
}

/// optuna `default_weights`: ones until 25 observations, then a ramp from `1/n` to `1` over the
/// oldest `n - 25` and a flat one over the newest 25.
fn default_weights(n: usize) -> Vec<f64> {
    if n < 25 {
        return vec![1.0; n];
    }
    let ramp = n - 25;
    let mut weights = Vec::with_capacity(n);
    for i in 0..ramp {
        let fraction = if ramp == 1 {
            0.0
        } else {
            i as f64 / (ramp - 1) as f64
        };
        weights.push(1.0 / n as f64 + (1.0 - 1.0 / n as f64) * fraction);
    }
    weights.extend(std::iter::repeat_n(1.0, 25));
    weights
}

/// One parameter's kernels — optuna `_calculate_categorical_distributions`. With no observations,
/// one uniform prior kernel; otherwise a kernel per observation over a `prior_weight / n_kernels`
/// floor with the observed category incremented, plus a trailing flat prior kernel, each row
/// normalised.
fn categorical_kernel(
    observations: &[Vec<usize>],
    param: usize,
    cardinality: usize,
    prior_weight: f64,
) -> Vec<Vec<f64>> {
    if observations.is_empty() {
        return vec![vec![1.0 / cardinality as f64; cardinality]];
    }
    let n_kernels = observations.len() + 1;
    let floor = prior_weight / n_kernels as f64;
    let mut kernels = vec![vec![floor; cardinality]; n_kernels];
    for (kernel, observation) in observations.iter().enumerate() {
        kernels[kernel][observation[param]] += 1.0;
    }
    for kernel in &mut kernels {
        // Summed pairwise because numpy's `weights.sum(axis=1)` is, and this row is as long as the
        // parameter has categories — so it crosses numpy's eight-element threshold exactly when a
        // program has eight or more candidates. Below that the two agree and nothing notices.
        let sum = pairwise_sum(kernel);
        for weight in kernel {
            *weight /= sum;
        }
    }
    kernels
}

/// optuna's category draw: the number of cumulative weights below the quantile, with the last
/// cumulative weight pinned to one so a quantile just under one never falls off the end.
fn pick_category(weights: &[f64], quantile: f64) -> usize {
    let last = weights.len() - 1;
    let mut cumulative = 0.0;
    let mut category = 0;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += weight;
        let bound = if index == last { 1.0 } else { cumulative };
        if bound < quantile {
            category += 1;
        }
    }
    category
}
