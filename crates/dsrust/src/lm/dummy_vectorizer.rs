//! dspy `utils/dummies.py::DummyVectorizer`: the n-gram vectorizer its own tests embed with.
//!
//! Character n-grams hashed with a polynomial whose coefficients come from Python's `random`
//! seeded at 123, counted into a fixed-width row, then centred and normalised in float32. Every
//! step is reproduced — the Mersenne Twister, the modular hash, numpy's pairwise float32 mean and
//! norm — because the vectors decide which examples a `KNN` chooses, and dspy's tests assert on
//! that choice.

use pyrng::cpython::Random;

use crate::numpy::{l2_norm_f32, mean_f32};

/// The prime the hash is taken modulo, upstream's `10**9 + 7`.
const P: u64 = 1_000_000_007;

#[derive(Debug, Clone)]
pub struct DummyVectorizer {
    max_length: usize,
    n_gram: usize,
    coeffs: Vec<u64>,
}

impl Default for DummyVectorizer {
    fn default() -> Self {
        Self::new(100, 2)
    }
}

impl DummyVectorizer {
    /// `DummyVectorizer(max_length, n_gram)`: `n_gram` coefficients drawn as `random.randrange(1, P)`
    /// after `random.seed(123)`.
    pub fn new(max_length: usize, n_gram: usize) -> Self {
        let mut random = Random::seeded(123);
        let coeffs = (0..n_gram)
            .map(|_| 1 + random.below((P - 1) as usize) as u64)
            .collect();
        Self {
            max_length,
            n_gram,
            coeffs,
        }
    }

    /// The hash coefficients, for a test to hold against what Python drew.
    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }

    /// Upstream's `_hash`: a polynomial over the gram's code points, modulo `P`, then modulo the
    /// row width. `zip` stops at the shorter of the coefficients and the gram, as Python's does.
    fn hash(&self, gram: &[char]) -> usize {
        let mut h: u64 = 1;
        for (coeff, c) in self.coeffs.iter().zip(gram) {
            h = (h * coeff + u64::from(*c as u32)) % P;
        }
        (h % self.max_length as u64) as usize
    }

    /// `vectorizer(texts)`: one row per text, centred on its mean and divided by its norm plus
    /// `1e-10`, in float32 as numpy computes it.
    pub fn vectorize(&self, texts: &[impl AsRef<str>]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|text| {
                let chars: Vec<char> = text.as_ref().chars().collect();
                let mut row = vec![0.0f32; self.max_length];
                for start in 0..chars.len().saturating_sub(self.n_gram - 1) {
                    let gram = &chars[start..start + self.n_gram];
                    row[self.hash(gram)] += 1.0;
                }
                let mean = mean_f32(&row);
                for value in &mut row {
                    *value -= mean;
                }
                let scale = l2_norm_f32(&row) + 1e-10f32;
                for value in &mut row {
                    *value /= scale;
                }
                row
            })
            .collect()
    }
}
