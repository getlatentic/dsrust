//! dspy COPRO's `track_stats` block: what each predictor's scores looked like at each depth.
//!
//! Upstream attaches these to the compiled program as `results_best`, `results_latest` and
//! `total_calls`, keyed by `id(predictor)`. A predictor here is its position among
//! [`Module::named_predictors`](crate::Module::named_predictors), which is the same information
//! under a key a caller can actually hold.

/// One depth's summary of a set of scores — dspy appends these five together, per predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthScores {
    /// dspy `depth`: which round these came from, counting from zero.
    pub depth: usize,
    pub max: f64,
    /// The arithmetic mean, dspy's `sum(scores) / len(scores)`.
    pub average: f64,
    pub min: f64,
    /// dspy's `statistics.pstdev` — the *population* deviation, dividing by `n` rather than
    /// `n - 1`. A one-element set therefore has a deviation of zero rather than being undefined,
    /// which is what a single surviving candidate at a depth produces.
    pub std: f64,
}

impl DepthScores {
    /// Summarise one depth's scores. Empty is `None`: upstream calls `max()` on the list, which
    /// raises on an empty one, so there is no depth for which it would record anything.
    pub(super) fn of(depth: usize, scores: &[f64]) -> Option<Self> {
        if scores.is_empty() {
            return None;
        }
        let count = scores.len() as f64;
        let average = scores.iter().sum::<f64>() / count;
        let variance = scores
            .iter()
            .map(|score| (score - average).powi(2))
            .sum::<f64>()
            / count;
        Some(Self {
            depth,
            max: scores.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            average,
            min: scores.iter().copied().fold(f64::INFINITY, f64::min),
            std: variance.sqrt(),
        })
    }
}

/// dspy's `track_stats` output, which upstream hangs off the compiled program.
///
/// Returned rather than attached, for the reason [`MIPROv2::compile_traced`] returns its trials:
/// a Rust program is the caller's value and an optimizer writing attributes onto it would have to
/// invent somewhere to put them.
///
/// [`MIPROv2::compile_traced`]: crate::optimize::MIPROv2::compile_traced
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CoproStats {
    /// dspy `results_best`: per predictor, one entry per depth, over the **top ten** scores that
    /// predictor had seen by the end of that depth.
    pub best: Vec<Vec<DepthScores>>,
    /// dspy `results_latest`: per predictor, one entry per depth, over the scores of the newest
    /// `breadth` candidates — the ones that depth's proposal round produced.
    pub latest: Vec<Vec<DepthScores>>,
    /// dspy `total_calls`: how many candidates were scored across the whole run.
    pub total_calls: usize,
}

impl CoproStats {
    pub(super) fn for_predictors(predictors: usize) -> Self {
        Self {
            best: vec![Vec::new(); predictors],
            latest: vec![Vec::new(); predictors],
            total_calls: 0,
        }
    }

    /// The ten highest scores, descending — dspy sorts every evaluated candidate and slices.
    pub(super) fn top_ten(scores: &mut [f64]) -> &[f64] {
        scores.sort_by(|left, right| right.total_cmp(left));
        &scores[..scores.len().min(10)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Population deviation, not sample: dividing by `n - 1` makes a one-element set undefined and
    /// a two-element set wider than dspy reports.
    #[test]
    fn the_deviation_is_pythons_pstdev() {
        let one = DepthScores::of(0, &[0.5]).expect("one score");
        assert_eq!(one.std, 0.0, "pstdev of a single score is zero");

        // pstdev([0.0, 1.0]) is 0.5; the sample deviation would be ~0.707.
        let two = DepthScores::of(0, &[0.0, 1.0]).expect("two scores");
        assert_eq!(two.std, 0.5);
        assert_eq!(two.average, 0.5);
        assert_eq!((two.min, two.max), (0.0, 1.0));
    }

    #[test]
    fn a_depth_with_no_scores_records_nothing() {
        assert_eq!(DepthScores::of(0, &[]), None);
    }

    /// Descending, and no more than ten however many were evaluated.
    #[test]
    fn the_best_ten_are_the_highest_ten() {
        let mut scores: Vec<f64> = (0..15).map(|at| at as f64 / 10.0).collect();
        let top = CoproStats::top_ten(&mut scores);
        assert_eq!(top.len(), 10);
        assert_eq!(top[0], 1.4);
        assert_eq!(top[9], 0.5);
    }
}
