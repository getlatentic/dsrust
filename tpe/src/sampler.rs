//! optuna's `TPESampler` over categorical distributions, seeded and multivariate — the slice dspy's
//! MIPROv2 drives its Step-3 search with.
//!
//! Two generators, both seeded with the same seed, mirror optuna's two: one for the random startup
//! trials, one for the TPE phase. A caller asks for a trial's categories, evaluates it, and tells
//! the sampler the score — optuna's ask-and-tell, which is what MIPROv2's objective loop is.
//!
//! The generator draws are reproduced bit for bit, so the sampler proposes optuna's candidates
//! exactly. It also picks among them exactly, save for one boundary: optuna scores candidates with
//! numpy's vectorized `log`/`exp`, whose last bit this crate's scalar `ln`/`exp` cannot always
//! match, so two candidates whose acquisitions sit within a few ULPs — which a symmetric objective
//! can produce — may rank in the other order. Ranking by the raw density ratio rather than the
//! log-difference (see [`crate::parzen`]) keeps that to genuine near-ties; with distinct objective
//! scores the whole trial sequence reproduces. It is the one place exact reproduction meets the
//! limit of cross-language floating point.

use crate::mt19937::Mt19937;
use crate::parzen::Parzen;

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
    cardinalities: Vec<usize>,
    startup_rng: Mt19937,
    tpe_rng: Mt19937,
    trials: Vec<Trial>,
}

impl TpeSampler {
    /// A sampler over parameters with these cardinalities, seeded as optuna's `TPESampler(seed=...)`.
    pub fn new(seed: u32, cardinalities: Vec<usize>) -> Self {
        Self {
            cardinalities,
            startup_rng: Mt19937::new(seed),
            tpe_rng: Mt19937::new(seed),
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
        let (below, above) = self.split(default_gamma(self.trials.len()));
        let below = Parzen::build(&below, &self.cardinalities, PRIOR_WEIGHT);
        let above = Parzen::build(&above, &self.cardinalities, PRIOR_WEIGHT);
        let candidates = below.sample(&mut self.tpe_rng, N_EI_CANDIDATES, self.cardinalities.len());

        // optuna ranks by log_pdf_below - log_pdf_above = log(pdf_below / pdf_above); the raw ratio
        // ranks the same way without the log/exp that would flip near-ULP ties across languages.
        let acquisitions: Vec<f64> = candidates
            .iter()
            .map(|candidate| below.pdf(candidate) / above.pdf(candidate))
            .collect();
        candidates[argmax(&acquisitions)].clone()
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
        let params = |indices: Vec<usize>| indices.iter().map(|&i| self.trials[i].params.clone()).collect();
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
