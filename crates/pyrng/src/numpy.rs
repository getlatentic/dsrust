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
        // Two mutations of this block cannot be told apart from it, and both for the same reason:
        // the normalisation divides by the last entry. Accumulating with `-=` negates every entry
        // *and* the divisor, so the normalised CDF is identical; and every caller normalises before
        // it gets here — numpy requires `p` to sum to one, and `parzen::mixture_weights` divides by
        // its own sum — so the divisor is 1.0 and `*=` is `/=`.
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
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn searchsorted_right(cdf: &[f64], u: f64) -> usize {
    // `partition_point`, for the reason `cpython::bisect_right` uses it: a halving loop terminates
    // only because `(low + high) / 2` stays strictly inside the bounds, and changing that
    // arithmetic hangs rather than fails. The predicate is `!(u < v)` and not `v <= u` so the
    // comparison stays numpy's, which tests `u < cdf[mid]` and takes the else branch otherwise.
    // `!(u < v)` against `!(u <= v)` needs a draw landing exactly on a cumulative boundary, which
    // a float from 53 random bits does not do outside a constructed case.
    cdf.partition_point(|&v| !(u < v))
}
