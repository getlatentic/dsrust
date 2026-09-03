//! optuna's categorical Parzen estimator: the mixture of one kernel per observed trial plus a
//! uniform prior, over the categorical parameters MIPROv2 searches.
//!
//! Each kernel is a categorical distribution per parameter — an observation adds one to its own
//! category over a flat `prior_weight / n_kernels` floor, the prior kernel stays flat — and the
//! mixture weights come from optuna's `default_weights`. The estimator both draws candidates
//! ([`Parzen::sample`]) and scores them ([`Parzen::log_pdf`]); the sampler above ranks candidates by the
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

/// numpy's `add.reduce` over a contiguous array — pairwise summation, which is not a left fold and
/// rounds differently past eight elements.
///
/// Every sum numpy takes goes through this, not only the log-sum-exp: normalising a kernel row sums
/// over the categories, so a program proposing twelve instructions rounds differently from one
/// proposing six. That difference reaches the acquisition, which ties constantly, and `np.argmax`
/// keeps the first of a tie — so it decides which trial runs next. The same reduction, in `f32`,
/// is what `np.mean` and `np.linalg.norm` take over a float32 row.
pub fn pairwise_sum<T: PairwiseSummand>(values: &[T]) -> T {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        return values
            .iter()
            .copied()
            .fold(T::ZERO, |sum, value| sum + value);
    }
    if n <= BLOCK {
        let mut accumulators: [T; 8] = values[..8].try_into().expect("eight values");
        // `as_chunks` walks the same eight-wide blocks the cursor loop did, in the same order, so
        // every float lands in the same accumulator and the sum is bit-identical — and the
        // iterator owns the progress, so there is no `index += 8` for a mutant to stall.
        let (blocks, tail) = values[8..].as_chunks::<8>();
        for block in blocks {
            for (slot, value) in accumulators.iter_mut().zip(block) {
                *slot = *slot + *value;
            }
        }
        let mut total = ((accumulators[0] + accumulators[1]) + (accumulators[2] + accumulators[3]))
            + ((accumulators[4] + accumulators[5]) + (accumulators[6] + accumulators[7]));
        for value in tail {
            total = total + *value;
        }
        return total;
    }
    let half = (n / 2) - (n / 2) % 8;
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

/// A float numpy reduces pairwise: `f64` as `add.reduce` over a double array, `f32` over a float32
/// one, each accumulating in its own width.
pub trait PairwiseSummand: Copy + std::ops::Add<Output = Self> {
    const ZERO: Self;
}

impl PairwiseSummand for f64 {
    const ZERO: Self = 0.0;
}

impl PairwiseSummand for f32 {
    const ZERO: Self = 0.0;
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
///
/// The ramp is `np.linspace(1/n, 1.0, num=n-25)` and is computed the way `linspace` computes it —
/// `i * step + start` with `step = (stop - start) / (num - 1)`, and the last element assigned
/// `stop` outright rather than arrived at. Writing it as `start + delta * (i / div)` is the same
/// algebra and rounds differently: it disagreed with optuna in the last bit at two indices out of
/// fifty. That would be pedantic if the weights were only weights, but they reach an acquisition
/// whose ties `np.argmax` breaks by first index, so a last-bit difference picks a different trial.
pub(crate) fn default_weights(n: usize) -> Vec<f64> {
    if n < 25 {
        return vec![1.0; n];
    }
    let ramp = n - 25;
    let start = 1.0 / n as f64;
    let mut weights = Vec::with_capacity(n);
    if ramp > 0 {
        let step = (1.0 - start) / (ramp - 1).max(1) as f64;
        for i in 0..ramp {
            weights.push(i as f64 * step + start);
        }
        // `linspace` assigns the endpoint rather than computing it, but only when it was asked for
        // more than one point — `np.linspace(a, b, num=1)` is `[a]`.
        if ramp > 1 {
            weights[ramp - 1] = 1.0;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/parzen_arithmetic.json");
        let text = std::fs::read_to_string(&path).expect("the arithmetic golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    fn floats(value: &serde_json::Value) -> Vec<f64> {
        value
            .as_array()
            .expect("an array")
            .iter()
            .map(|number| match number {
                serde_json::Value::String(text) => text.parse().expect("a float"),
                other => other.as_f64().expect("a float"),
            })
            .collect()
    }

    /// `pairwise_sum` against numpy's own `add.reduce`, at every length that straddles a branch:
    /// under eight, the eight-accumulator path, the 128-element block boundary, and past it where
    /// numpy splits and recurses.
    ///
    /// Every corpus is built so a *left fold* disagrees — one large leading value and many small
    /// ones — and the golden records which lengths that holds for. Without that the case would pass
    /// against a plain `iter().sum()`, which is what this function exists not to be.
    #[test]
    fn the_sum_is_numpys_at_every_length_that_changes_branch() {
        let golden = golden();
        let cases = golden["sums"].as_array().expect("sums");
        let mut discriminating = 0;
        for case in cases {
            let values = floats(&case["values"]);
            let expected: f64 = case["sum"]
                .as_str()
                .expect("a sum")
                .parse()
                .expect("a float");
            let n = case["n"].as_u64().expect("n");
            assert_eq!(pairwise_sum(&values), expected, "sum of {n} values");

            if case["order_matters"].as_bool().unwrap_or(false) {
                discriminating += 1;
                let left_fold: f64 = case["left_fold"]
                    .as_str()
                    .expect("a left fold")
                    .parse()
                    .expect("a float");
                assert_ne!(
                    expected, left_fold,
                    "the golden says {n} discriminates but the two agree"
                );
            }
        }
        assert!(
            discriminating >= 4,
            "the corpus no longer tells pairwise summation from a left fold"
        );
    }

    /// `default_weights` against optuna's, including the `n >= 25` ramp — which no case reached,
    /// because the *below* group never grows that large in the recorded studies.
    #[test]
    fn the_weights_are_optunas_past_the_ramp() {
        let golden = golden();
        for case in golden["default_weights"].as_array().expect("weights") {
            let n = case["n"].as_u64().expect("n") as usize;
            assert_eq!(
                default_weights(n),
                floats(&case["weights"]),
                "weights for {n}"
            );
        }
    }

    /// The last cumulative weight is pinned to one, so a quantile just under one cannot fall off the
    /// end — optuna's `cum_probs[:, -1] = 1`, guarding the rounding in the cumulative sum.
    #[test]
    fn a_quantile_just_under_one_lands_on_the_last_category() {
        // A row whose cumulative sum falls clearly short of one, and a quantile in the gap. Without
        // the pin the count runs past the last category and returns an index out of range; with it
        // the last bound is one and the quantile stays inside. A row that only just falls short
        // would leave both answers equal and prove nothing.
        let weights = [0.1, 0.2, 0.6];
        assert!(weights.iter().sum::<f64>() < 1.0);
        assert_eq!(pick_category(&weights, 0.95), weights.len() - 1);
        // And the strict `<` keeps a quantile exactly on a boundary in the lower category.
        assert_eq!(pick_category(&[0.5, 0.5], 0.5), 0);
        assert_eq!(pick_category(&[0.5, 0.5], 0.500_000_001), 1);
    }
}
