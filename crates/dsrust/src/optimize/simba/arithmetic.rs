//! Two pieces of arithmetic SIMBA leans on that Rust does not spell the same way.
//!
//! Both are the shape a re-implementation gets wrong by writing the obvious thing.

/// numpy's `percentile` at its default `linear` interpolation.
///
/// The request lands at `q/100 * (n - 1)` in the **sorted** sample and is interpolated between its
/// neighbours, so the answer is usually a value no observation has. SIMBA takes the 10th and 90th
/// over a mini-batch's scores to decide what counts as a bad or a good trajectory, and those two
/// numbers are handed to the rule proposer as prompt text.
///
/// Empty is `None`: numpy raises, and there is no percentile of nothing to return.
pub fn percentile(sample: &[f64], q: f64) -> Option<f64> {
    if sample.is_empty() {
        return None;
    }
    let mut sorted = sample.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = q / 100.0 * (sorted.len() - 1) as f64;
    let below = at.floor();
    let above = at.ceil();
    if below == above {
        return Some(sorted[at as usize]);
    }
    let weight = at - below;
    Some(sorted[below as usize] * (1.0 - weight) + sorted[above as usize] * weight)
}

/// Python's `round` for a float with no digit count: **banker's rounding**, which breaks a tie
/// toward the even neighbour.
///
/// `round(0.5)` is 0 and `round(1.5)` is 2, where Rust's `f64::round` gives 1 and 2 — it breaks
/// away from zero. SIMBA picks its final candidate slate with `round(i * M / (N - 1))`, so a half
/// lands on a different winner and the compiled program is a different program.
pub fn round_half_to_even(x: f64) -> i64 {
    let nearest = x.round();
    match (x - x.trunc()).abs() == 0.5 && nearest % 2.0 != 0.0 {
        true => (nearest - x.signum()) as i64,
        false => nearest as i64,
    }
}

/// dspy's final slate: `[round(i * M / (N - 1)) for i in range(N)]` over the winners, deduped with
/// order kept — `M` winners past the baseline and `N = num_candidates + 1` slots.
///
/// With no winners at all every slot is the baseline, which dedupes to one.
pub fn final_slate(winners: usize, num_candidates: usize) -> Vec<usize> {
    let slots = num_candidates + 1;
    let past_baseline = winners as i64;
    let picked: Vec<usize> = match past_baseline < 1 {
        true => vec![0; slots],
        false => (0..slots)
            .map(|at| {
                round_half_to_even(at as f64 * past_baseline as f64 / (slots - 1) as f64) as usize
            })
            .collect(),
    };
    let mut seen = Vec::new();
    for index in picked {
        if !seen.contains(&index) {
            seen.push(index);
        }
    }
    seen
}
