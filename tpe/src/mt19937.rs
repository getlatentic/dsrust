//! numpy's legacy `RandomState` — the MT19937 generator optuna's `TPESampler` draws from.
//!
//! optuna seeds a `numpy.random.RandomState` and the TPE sampler consumes it through exactly two
//! methods: [`random_sample`](Mt19937::random_sample) (uniform doubles) and
//! [`choice`](Mt19937::choice) (weighted categorical). Both are reproduced here bit for bit —
//! numpy's `init_genrand` seeding, the same tempering, the 53-bit double, and `choice`'s
//! cumulative-search — so a search seeded the same way makes the same draws in the same order.
//!
//! numpy's scalar-int seeding is `init_genrand`, *not* the `init_by_array` that CPython's `random`
//! uses, so this is a distinct generator from `dsrs`'s own CPython Mersenne Twister. Held to
//! `tests/conformance/numpy_mt19937.json`, captured by running numpy.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// A numpy-compatible MT19937. Seed it as numpy seeds a `RandomState(int)`, then draw.
pub struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Mt19937 {
    /// numpy `RandomState(seed)` for a scalar seed: `init_genrand(seed)`. Each word follows from
    /// the last by Knuth's multiplier, and the seed itself is word zero.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: N }
    }

    /// One word of the raw state, for holding the seeding to numpy's `get_state()` directly rather
    /// than only through the draws that follow it.
    pub fn state_word(&self, index: usize) -> u32 {
        self.state[index]
    }

    /// One tempered 32-bit word, twisting the state forward when the block is spent — numpy's
    /// `rk_random`.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    /// One double in `[0, 1)` with 53 bits of precision — numpy `random_sample`, its `rk_double`:
    /// the high 27 bits of one word and the high 26 of the next.
    pub fn random_sample(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
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
