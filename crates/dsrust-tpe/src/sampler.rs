//! optuna's `TPESampler` over categorical distributions, seeded and multivariate — the slice dspy's
//! MIPROv2 drives its Step-3 search with.
//!
//! Two generators, both seeded with the same seed, mirror optuna's two: one for the random startup
//! trials, one for the TPE phase. A caller asks for a trial's categories, evaluates it, and tells
//! the sampler the score — optuna's ask-and-tell, which is what MIPROv2's objective loop is.
//!
//! The generator draws are reproduced bit for bit, so the sampler proposes optuna's candidates
//! exactly, and it scores them with optuna's own arithmetic — a log-sum-exp over the kernels, summed
//! in numpy's pairwise order (see [`crate::parzen`]).
//!
//! Scoring them *equivalently* is not enough. The acquisition ties constantly: any two categories
//! the two estimators treat alike score the same, and `np.argmax` then keeps the first. Ranking by
//! the algebraically identical `pdf_below / pdf_above` computed in linear space breaks those ties by
//! last-bit noise instead — measured at twelve categories, where optuna produced two distinct
//! acquisition values across twenty-four candidates and the ratio produced five. Reproducing the
//! expression, not merely its ordering, is what makes the ties land where optuna's do.

use crate::parzen::Parzen;
use pyrng::RandomState;

const N_STARTUP_TRIALS: usize = 10;
const N_EI_CANDIDATES: usize = 24;
const PRIOR_WEIGHT: f64 = 1.0;

struct Trial {
    params: Vec<usize>,
    value: f64,
}

/// A seeded TPE sampler over a fixed list of categorical parameters, given by their cardinalities.
/// Maximises the told scores, as MIPROv2's study does.
pub struct TpeSampler {
    /// Cardinalities in the order the objective suggests the parameters.
    cardinalities: Vec<usize>,
    /// Indices into `cardinalities`, ordered by parameter name.
    ///
    /// The two phases read the search space in different orders, which is invisible until a run is
    /// long enough to reach both. A startup trial is drawn one parameter at a time by
    /// `sample_independent`, called from `suggest_categorical` — so in suggest order. A TPE trial is
    /// drawn all at once by `sample_relative`, whose search space comes from `IntersectionSearchSpace`
    /// and is `dict(sorted(...))` — so in name order.
    by_name: Vec<usize>,
    startup_rng: RandomState,
    tpe_rng: RandomState,
    trials: Vec<Trial>,
}

impl TpeSampler {
    /// A sampler over these parameters, named and in the order the objective suggests them, seeded
    /// as optuna's `TPESampler(seed=...)`.
    ///
    /// The names are not decoration: they decide the order the TPE phase draws in, and a caller
    /// whose names sort differently from its suggest order gets a different search.
    pub fn new(seed: u32, parameters: Vec<(String, usize)>) -> Self {
        let mut by_name: Vec<usize> = (0..parameters.len()).collect();
        by_name.sort_by(|&left, &right| parameters[left].0.cmp(&parameters[right].0));
        Self {
            cardinalities: parameters.into_iter().map(|(_, count)| count).collect(),
            by_name,
            startup_rng: RandomState::new(seed),
            tpe_rng: RandomState::new(seed),
            trials: Vec::new(),
        }
    }

    /// The next trial's category per parameter. Random while under the startup count, then TPE.
    pub fn ask(&mut self) -> Vec<usize> {
        if self.trials.len() < N_STARTUP_TRIALS {
            self.sample_startup()
        } else {
            self.sample_tpe()
        }
    }

    /// Record a trial's categories and the score it earned, so later trials can learn from it.
    pub fn tell(&mut self, params: Vec<usize>, value: f64) {
        self.trials.push(Trial { params, value });
    }

    /// optuna's `RandomSampler` startup: draw a uniform per category and take the largest. The
    /// uniforms are the generator's bit-exact draws, so `np.argmax`'s strict first-max reproduces.
    fn sample_startup(&mut self) -> Vec<usize> {
        self.cardinalities
            .iter()
            .map(|&cardinality| argmax(&self.startup_rng.random_sample_n(cardinality)))
            .collect()
    }

    /// optuna's TPE step: split the trials into a better and worse group, fit a Parzen estimator to
    /// each, draw candidates from the better one, and keep the candidate the estimators most prefer.
    fn sample_tpe(&mut self) -> Vec<usize> {
        let ordered = |params: &[usize]| -> Vec<usize> {
            self.by_name.iter().map(|&index| params[index]).collect()
        };
        let cardinalities = ordered(&self.cardinalities);
        let (below, above) = self.split(default_gamma(self.trials.len()));
        let reorder = |group: Vec<Vec<usize>>| {
            group
                .iter()
                .map(|params| ordered(params))
                .collect::<Vec<_>>()
        };
        let below = Parzen::build(&reorder(below), &cardinalities, PRIOR_WEIGHT);
        let above = Parzen::build(&reorder(above), &cardinalities, PRIOR_WEIGHT);
        let candidates = below.sample(&mut self.tpe_rng, N_EI_CANDIDATES, cardinalities.len());

        let acquisitions: Vec<f64> = candidates
            .iter()
            .map(|candidate| below.log_pdf(candidate) - above.log_pdf(candidate))
            .collect();
        // Back into suggest order, which is what a caller indexes its own parameters by.
        let chosen = &candidates[argmax(&acquisitions)];
        let mut params = vec![0; chosen.len()];
        for (position, &parameter) in self.by_name.iter().enumerate() {
            params[parameter] = chosen[position];
        }
        params
    }

    /// optuna `_split_trials`: the best `gamma` trials by score go below, the rest above. Sorted by
    /// score descending — ties keeping trial order — then each group put back into trial order.
    fn split(&self, gamma: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let n_below = gamma.min(self.trials.len());
        let mut order: Vec<usize> = (0..self.trials.len()).collect();
        order.sort_by(|&a, &b| self.trials[b].value.total_cmp(&self.trials[a].value));
        let mut below: Vec<usize> = order[..n_below].to_vec();
        let mut above: Vec<usize> = order[n_below..].to_vec();
        below.sort_unstable();
        above.sort_unstable();
        let params = |indices: Vec<usize>| {
            indices
                .iter()
                .map(|&i| self.trials[i].params.clone())
                .collect()
        };
        (params(below), params(above))
    }
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
