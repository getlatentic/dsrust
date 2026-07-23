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
        Self { mixture, categorical }
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
                candidates[candidate][param] = pick_category(&kernels[kernel], quantiles[candidate]);
            }
        }
        candidates
    }

    /// The density of one candidate under the mixture: each kernel's mixture weight times the
    /// product of its category weights, summed.
    ///
    /// optuna computes `log_pdf` and ranks candidates by `log_pdf_below - log_pdf_above`, i.e. by
    /// `log(pdf_below / pdf_above)`. Since `log` is monotonic, ranking by the raw ratio
    /// `pdf_below / pdf_above` gives the same order — and it avoids the `log`/`exp` whose
    /// cross-language last-bit differences would otherwise flip the order of two categories whose
    /// acquisitions sit only a few ULPs apart. The categorical densities are small sums of rational
    /// weights over the few parameters a program has, so the product neither under- nor overflows.
    pub fn pdf(&self, candidate: &[usize]) -> f64 {
        self.mixture
            .iter()
            .enumerate()
            .map(|(kernel, weight)| {
                let product: f64 = candidate
                    .iter()
                    .enumerate()
                    .map(|(param, &category)| self.categorical[param][kernel][category])
                    .product();
                weight * product
            })
            .sum()
    }
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
    let sum: f64 = weights.iter().sum();
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
        let fraction = if ramp == 1 { 0.0 } else { i as f64 / (ramp - 1) as f64 };
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
        let sum: f64 = kernel.iter().sum();
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

