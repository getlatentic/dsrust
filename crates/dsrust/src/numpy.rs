//! The float32 arithmetic numpy does over an embedding row, rounding for rounding.
//!
//! A ranking is decided by the last bits of a score, so a mean, a norm and a dot product taken
//! over `f32` rows have to round as numpy rounds them: `np.mean` and `np.linalg.norm` reduce
//! pairwise in the array's own width, and `np.argsort` breaks ties as its introsort does.

use tpe::{argsort, pairwise_sum};

/// `np.mean(row)` over a float32 row: the pairwise sum, divided in float32.
pub fn mean_f32(row: &[f32]) -> f32 {
    pairwise_sum(row) / row.len() as f32
}

/// `np.linalg.norm(rows, axis=1)` over a float32 row — the form the vectorizer and the retriever
/// take — which is the square root of the pairwise sum of squares. numpy's one-dimensional
/// `norm(row)` goes through BLAS's dot instead and can differ in its last bit.
pub fn l2_norm_f32(row: &[f32]) -> f32 {
    let squares: Vec<f32> = row.iter().map(|value| value * value).collect();
    pairwise_sum(&squares).sqrt()
}

/// A dot product of two float32 rows, accumulated in float32 left to right.
///
/// numpy hands `np.dot` to BLAS, whose kernel accumulates in whatever lanes the platform's build
/// chose; the order is not one a port can read. The fixture this crate is held to ranks the same
/// under a left-to-right sum, and a tie between scores that differ only past the seventh digit is
/// where the two could part.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |sum, (x, y)| sum + x * y)
}

/// `np.argsort(scores)` over float32 scores: numpy's own permutation, ties included.
pub fn argsort_f32(scores: &[f32]) -> Vec<usize> {
    let widened: Vec<f64> = scores.iter().map(|score| f64::from(*score)).collect();
    argsort(&widened)
}
