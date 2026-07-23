//! numpy's legacy `RandomState` — the MT19937 generator optuna's `TPESampler` draws through.
//!
//! optuna seeds a `numpy.random.RandomState` and consumes it through exactly two calls: `rand`
//! (`random_sample`) for the startup samples, and `choice` for the TPE candidate draws. Both are
//! reproduced here over the shared [`Mt19937`], held to numpy in `tests/`.

use crate::mt19937::Mt19937;

/// numpy `RandomState`: seed it as `RandomState(int)`, then draw.
pub struct RandomState(Mt19937);

impl RandomState {
    /// numpy `RandomState(seed)` for a scalar seed: `init_genrand(seed)`.
    pub fn new(seed: u32) -> Self {
        Self(Mt19937::from_word(seed))
    }

    /// One word of the raw seeded state, for holding the seeding to numpy's `get_state()` directly.
    pub fn state_word(&self, index: usize) -> u32 {
        self.0.state_word(index)
    }

    /// numpy `random_sample`: one double in `[0, 1)` with 53 bits of precision.
    pub fn random_sample(&mut self) -> f64 {
        self.0.random_double()
    }

    /// `size` doubles, in order — numpy `random_sample(size)`.
    pub fn random_sample_n(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.random_sample()).collect()
    }

    /// numpy `choice(len(weights), p=weights, size=size)`: `size` categories drawn with the given
    /// probabilities. numpy normalises the cumulative sum by its last entry and places each uniform
    /// draw into it from the right, so a draw of `u` lands on the first category whose running total
    /// exceeds `u`.
    pub fn choice(&mut self, weights: &[f64], size: usize) -> Vec<usize> {
        let mut cdf = Vec::with_capacity(weights.len());
        let mut running = 0.0;
        for weight in weights {
            running += weight;
            cdf.push(running);
        }
        let total = *cdf.last().expect("choice needs at least one weight");
        for value in &mut cdf {
            *value /= total;
        }
        self.random_sample_n(size)
            .into_iter()
            .map(|u| searchsorted_right(&cdf, u))
            .collect()
    }
}

/// numpy `searchsorted(cdf, u, side="right")`: the count of entries less than or equal to `u`,
/// which for a normalised CDF and a `u` in `[0, 1)` is the chosen category.
fn searchsorted_right(cdf: &[f64], u: f64) -> usize {
    let mut low = 0;
    let mut high = cdf.len();
    while low < high {
        let mid = (low + high) / 2;
        if u < cdf[mid] {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    low
}
