//! numpy's `argsort`, because its tie-breaking is observable.
//!
//! optuna's numerical Parzen estimator sorts the observations to measure how far each one's
//! neighbours are, then maps the widths back through the permutation `np.argsort` returned. With
//! duplicate observations — the normal case for an integer parameter — *which* duplicate lands at a
//! run's boundary decides which kernel gets the wide sigma, so the permutation decides the search.
//!
//! `np.argsort`'s default kind is `quicksort`, which is introsort: insertion sort while a partition
//! is 15 elements or fewer, quicksort above. Insertion sort is stable and quicksort is not, so an
//! array of 16 sorts like a stable sort and an array of 17 does not. Every recorded TPE run with
//! fewer than seventeen observations agreed with a stable port, and the first one over it did not —
//! which is why this is a transcription of `npy_aquicksort` rather than a call to `sort_by`.
//!
//! Held to `tests/conformance/argsort.json`, where 39 of 74 cases would be wrong under a stable
//! sort.

/// numpy's `SMALL_QUICKSORT`: the partition width at or below which it stops recursing.
const SMALL_QUICKSORT: isize = 15;

/// numpy's `DOUBLE_LT`, which orders NaN last rather than leaving it wherever it fell.
fn lt(a: f64, b: f64) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

/// `np.argsort(values)` — the indices that would sort `values`, numpy's own permutation.
pub fn argsort(values: &[f64]) -> Vec<usize> {
    let num = values.len();
    let mut order: Vec<usize> = (0..num).collect();
    if num < 2 {
        return order;
    }
    let at = |order: &Vec<usize>, i: isize| values[order[i as usize]];
    // Pointer arithmetic as indices. `pl` and `pr` are inclusive bounds, as numpy's pointers are.
    let mut stack: Vec<(isize, isize)> = Vec::new();
    let (mut pl, mut pr) = (0isize, num as isize - 1);
    loop {
        while pr - pl > SMALL_QUICKSORT {
            // Median of three, written into the ends and the middle exactly as numpy writes it —
            // the swaps are part of the answer, not just of choosing a pivot.
            let pm = pl + ((pr - pl) >> 1);
            if lt(at(&order, pm), at(&order, pl)) {
                order.swap(pm as usize, pl as usize);
            }
            if lt(at(&order, pr), at(&order, pm)) {
                order.swap(pr as usize, pm as usize);
            }
            if lt(at(&order, pm), at(&order, pl)) {
                order.swap(pm as usize, pl as usize);
            }
            let pivot = at(&order, pm);
            let (mut pi, mut pj) = (pl, pr - 1);
            order.swap(pm as usize, pj as usize);
            loop {
                // numpy walks these with `do { ++pi } while (LT(v[*pi], vp))`, relying on the
                // values the median of three left at each end to stop them. Written as the length of
                // the run each pointer skips: the same index either way, and progress the loop
                // cannot fail to make, since both counts start past the element they were given.
                //
                // A walk that consumes its whole range lands one *past* the end, where numpy's
                // pointer would be — which is what lets the pair cross on a range the sentinels
                // cannot reach. Clamping to `pr` or `pl` instead makes a fixed point: on an inverted
                // range both walks are empty every pass and neither index moves again.
                pi += 1
                    + (pi + 1..=pr)
                        .take_while(|&i| lt(at(&order, i), pivot))
                        .count() as isize;
                pj -= 1
                    + (pl..pj)
                        .rev()
                        .take_while(|&j| lt(pivot, at(&order, j)))
                        .count() as isize;
                if pi >= pj {
                    break;
                }
                order.swap(pi as usize, pj as usize);
            }
            order.swap(pi as usize, (pr - 1) as usize);
            // Every entry goes on at a range at most half the size of the one below it, so the
            // depth cannot pass the number of times `num` halves — let alone `num` itself. The
            // bound is written as the length rather than the log of it because a bound wants no
            // arithmetic of its own: this is where the loop's real limit lives, the partition's
            // width test being one operator away from never firing, and a pass that pushes an entry
            // while the stack never drains is what spins.
            assert!(
                stack.len() < num,
                "sorting {num} values put {} ranges on the stack, past what halving can reach",
                stack.len()
            );
            // The larger side is pushed and the smaller walked, which is what bounds the stack.
            //
            // Surviving mutants live on this comparison and are equivalent to it: the two halves are
            // disjoint and both are sorted before the function returns, so which one goes first
            // cannot change the permutation — only how deep the stack gets. Measured by flipping it,
            // which leaves every test green.
            if pi - pl < pr - pi {
                stack.push((pi + 1, pr));
                pr = pi - 1;
            } else {
                stack.push((pl, pi - 1));
                pl = pi + 1;
            }
        }
        // A range rather than a cursor: the walk's progress belongs to the iterator, so there is
        // no increment for a mutant to delete and leave the loop standing still.
        //
        // The `+ 1` carries a seventh equivalent: starting at `pl` instead re-inserts the element
        // already there into a prefix of one, which the inner `while pj > pl` refuses immediately.
        for pi in (pl + 1)..=pr {
            let carried = order[pi as usize];
            let value = values[carried];
            // How far back the carried element travels: the run of preceding elements greater than
            // it. Counted first and shifted second, so neither step is a cursor the loop's own
            // arithmetic has to keep advancing.
            let steps = (pl..pi)
                .rev()
                .take_while(|&k| lt(value, at(&order, k)))
                .count() as isize;
            for slot in ((pi - steps + 1)..=pi).rev() {
                order[slot as usize] = order[(slot - 1) as usize];
            }
            order[(pi - steps) as usize] = carried;
        }
        match stack.pop() {
            Some((left, right)) => {
                pl = left;
                pr = right;
            }
            None => break,
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn every_recorded_permutation_is_numpys() {
        let golden: Value = serde_json::from_str(include_str!("../tests/conformance/argsort.json"))
            .expect("the argsort golden is valid JSON");
        let cases = golden["cases"].as_array().expect("cases");
        assert!(cases.len() >= 70, "the golden lost cases: {}", cases.len());
        let mut unstable = 0;
        for case in cases {
            let name = case["name"].as_str().expect("a name");
            // The golden tags a NaN, which JSON cannot spell. They are half of `DOUBLE_LT`.
            let values: Vec<f64> = case["values"]
                .as_array()
                .expect("values")
                .iter()
                .map(|v| match v.as_str() {
                    Some("nan") => f64::NAN,
                    _ => v.as_f64().expect("a float"),
                })
                .collect();
            let expected: Vec<usize> = case["argsort"]
                .as_array()
                .expect("argsort")
                .iter()
                .map(|i| i.as_u64().expect("an index") as usize)
                .collect();
            assert_eq!(argsort(&values), expected, "argsort for {name}");

            let mut stable: Vec<usize> = (0..values.len()).collect();
            // `total_cmp` orders NaN differently from numpy too, which is why the comparison is here
            // rather than in the generator.
            stable.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
            unstable += usize::from(stable != expected);
        }
        assert!(
            unstable >= 30,
            "only {unstable} case(s) distinguish numpy's sort from a stable one; the corpus has \
             stopped covering the thing it exists for"
        );
    }
}
