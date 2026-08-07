//! GEPA's merge proposer (`proposer/merge.py`): combine two descendants of a common ancestor that
//! improved different components into one candidate carrying both improvements.
//!
//! The whole flow turns on the shared generator, drawn in one exact order: the candidate pair from
//! `rng.sample(list(set(...)), 2)`, the common ancestor from `rng.choices(list(set_a & set_b),
//! weights)`, and the evaluation subsample from a bucketed `rng.sample`. `list(set(...))` is
//! CPython's set iteration order ([`crate::pyset`]), not sorted, so the same draw lands on a
//! different program unless that order is reproduced.
//!
//! **Two different orders, and only one of them is an edge.** gepa reads a candidate's components
//! twice and not the same way:
//!
//!   - the predicate (`merge.py`, `filter_program_candidates`) walks
//!     `list(program_candidates[ancestor].keys())` — dict order, so the program's declaration order.
//!     Exact here, since [`Candidate`] keeps it.
//!   - the construction (`merge_programs_by_common_predictors`) walks
//!     `set(program_candidates[ancestor].keys())` — a *set* of strings, whose iteration order
//!     CPython randomises per process under siphash.
//!
//! The second is the bounded edge. A single-component candidate — one predictor, which is what GEPA
//! over a `Predict` produces — has a one-element set, so the order is fixed and this is exact. With
//! several components it reaches the generator only in the branch where both descendants changed the
//! same component to different values, where a descendant is drawn at random; there the draw would
//! follow CPython's string-set order, which this walks in declaration order instead. Left to a later
//! siphash port rather than papered over.
//!
//! This note used to attribute the set to the predicate loop, which reads a list. Corrected against
//! the pinned gepa when `Candidate` became insertion-ordered.

use pyrng::Random;

use crate::adapter::Candidate;
use crate::pyset::PyIntSet;

/// dspy's `merges_performed`: the ancestor triples already tried, and the merge descriptions
/// already produced, each a dedup guard against repeating a merge.
#[derive(Default)]
pub struct MergesPerformed {
    triples: Vec<(usize, usize, usize)>,
    descriptions: Vec<(usize, usize, Vec<usize>)>,
}

/// A merge the proposer found: the new candidate and the two parents plus ancestor it came from.
pub struct MergeAttempt {
    pub candidate: Candidate,
    pub id1: usize,
    pub id2: usize,
    pub ancestor: usize,
}

/// dspy `does_triplet_have_desirable_predictors`: whether `id1` and `id2` improved *different*
/// components relative to `ancestor` — one kept a component the ancestor had while the other
/// changed it. Only such a triple has something to merge.
fn has_desirable_predictors(
    candidates: &[Candidate],
    ancestor: usize,
    id1: usize,
    id2: usize,
) -> bool {
    components(&candidates[ancestor]).any(|name| {
        let anc = &candidates[ancestor][name];
        let a = &candidates[id1][name];
        let b = &candidates[id2][name];
        (anc == a || anc == b) && a != b
    })
}

/// dspy `filter_ancestors`: keep the common ancestors worth merging through — not already tried for
/// this pair, no better than either descendant, and carrying a desirable predictor split.
fn filter_ancestors(
    i: usize,
    j: usize,
    common: &[usize],
    merges: &MergesPerformed,
    agg_scores: &[f64],
    candidates: &[Candidate],
) -> Vec<usize> {
    common
        .iter()
        .copied()
        .filter(|&ancestor| {
            !merges.triples.contains(&(i, j, ancestor))
                && agg_scores[ancestor] <= agg_scores[i]
                && agg_scores[ancestor] <= agg_scores[j]
                && has_desirable_predictors(candidates, ancestor, i, j)
        })
        .collect()
}

/// Every ancestor of `node`, by dspy's `get_ancestors` DFS — the parents, then their parents,
/// accumulated into a set in visit order.
fn ancestors_of(node: usize, parents: &[Vec<usize>]) -> PyIntSet {
    fn walk(node: usize, parents: &[Vec<usize>], found: &mut PyIntSet) {
        for &parent in &parents[node] {
            if !found.contains(parent) {
                found.add(parent);
                walk(parent, parents, found);
            }
        }
    }
    let mut found = PyIntSet::new();
    walk(node, parents, &mut found);
    found
}

/// dspy `find_common_ancestor_pair`: sample a pair of merge candidates, and — when they share a
/// mergeable ancestor neither descends from — pick one, weighted by aggregate score. Returns
/// `(i, j, ancestor)` with `i < j`, or `None` after `max_attempts` fruitless draws.
fn find_common_ancestor_pair(
    rng: &mut Random,
    parents: &[Vec<usize>],
    program_indexes: &[usize],
    merges: &MergesPerformed,
    agg_scores: &[f64],
    candidates: &[Candidate],
    max_attempts: usize,
) -> Option<(usize, usize, usize)> {
    for _ in 0..max_attempts {
        if program_indexes.len() < 2 {
            return None;
        }
        let pair = rng.sample(program_indexes, 2);
        let (mut i, mut j) = (pair[0], pair[1]);
        if i == j {
            continue;
        }
        // Equality is refused above, so this orders a pair that already differs.
        if j < i {
            std::mem::swap(&mut i, &mut j);
        }
        // dspy re-wraps each DFS result — `set(list(set(...)))` — before intersecting; the wrap is
        // not order-idempotent, so it is reproduced rather than skipped.
        let ancestors_i = PyIntSet::from_keys(ancestors_of(i, parents).iter());
        let ancestors_j = PyIntSet::from_keys(ancestors_of(j, parents).iter());
        if ancestors_i.contains(j) || ancestors_j.contains(i) {
            continue; // one descends from the other
        }
        let common = filter_ancestors(
            i,
            j,
            &ancestors_i.intersection(&ancestors_j).to_vec(),
            merges,
            agg_scores,
            candidates,
        );
        if !common.is_empty() {
            let weights: Vec<f64> = common.iter().map(|&a| agg_scores[a]).collect();
            let chosen = common[rng.choices(&weights, 1)[0]];
            return Some((i, j, chosen));
        }
    }
    None
}

/// dspy `sample_and_attempt_merge_programs_by_common_predictors`: find a mergeable triple, then
/// build the merged candidate component by component. Records the description so the same merge is
/// not produced twice, and consults `overlap` — whether two candidates share enough validation
/// support to compare — exactly where dspy does.
#[allow(clippy::too_many_arguments)]
pub fn sample_and_attempt_merge(
    rng: &mut Random,
    agg_scores: &[f64],
    merge_candidates: &[usize],
    merges: &mut MergesPerformed,
    candidates: &[Candidate],
    parents: &[Vec<usize>],
    overlap: impl Fn(usize, usize) -> bool,
    max_attempts: usize,
) -> Option<MergeAttempt> {
    if merge_candidates.len() < 2 || parents.len() < 3 {
        return None;
    }
    for _ in 0..max_attempts {
        let Some((id1, id2, ancestor)) = find_common_ancestor_pair(
            rng,
            parents,
            merge_candidates,
            merges,
            agg_scores,
            candidates,
            max_attempts,
        ) else {
            continue;
        };
        if merges.triples.contains(&(id1, id2, ancestor)) {
            continue;
        }
        let (candidate, description) =
            merged_candidate(candidates, ancestor, id1, id2, agg_scores, rng);
        if merges
            .descriptions
            .contains(&(id1, id2, description.clone()))
        {
            continue;
        }
        if !overlap(id1, id2) {
            continue;
        }
        merges.descriptions.push((id1, id2, description));
        return Some(MergeAttempt {
            candidate,
            id1,
            id2,
            ancestor,
        });
    }
    None
}

/// dspy's predicate merge: for each component, take the descendant that changed it when the other
/// kept the ancestor's, the higher-scoring descendant when both changed it (a coin flip on a tie),
/// and `id1`'s when they agree. The description records which descendant each component came from.
fn merged_candidate(
    candidates: &[Candidate],
    ancestor: usize,
    id1: usize,
    id2: usize,
    agg_scores: &[f64],
    rng: &mut Random,
) -> (Candidate, Vec<usize>) {
    let mut new_program = candidates[ancestor].clone();
    let mut description = Vec::new();
    for name in components(&candidates[ancestor]) {
        let anc = &candidates[ancestor][name];
        let a = &candidates[id1][name];
        let b = &candidates[id2][name];
        let source = if (anc == a || anc == b) && a != b {
            // One matches the ancestor; take the other's value — the one that changed it.
            if anc == a { id2 } else { id1 }
        // The `&&` is upstream's, and it reads as an `||` here: a candidate matching the ancestor
        // on exactly one side also differs from it on the other, which the arm above already took.
        } else if anc != a && anc != b {
            // Both changed it: the stronger descendant, or a coin flip when they tie.
            match agg_scores[id1].total_cmp(&agg_scores[id2]) {
                std::cmp::Ordering::Greater => id1,
                std::cmp::Ordering::Less => id2,
                std::cmp::Ordering::Equal => [id1, id2][rng.choice_index(2)],
            }
        } else {
            // They agree (both equal the ancestor, or both equal each other): id1 will do.
            id1
        };
        new_program.insert(name.clone(), candidates[source][name].clone());
        description.push(source);
    }
    (new_program, description)
}

/// dspy `select_eval_subsample_for_merged_program`: choose up to `num` validation ids to score the
/// merged candidate on, spread across where its two parents disagree and agree, then topped up.
///
/// `scores1`/`scores2` are the two parents' per-id subscores; the ids are their shared support.
pub fn select_eval_subsample(
    scores1: &[f64],
    scores2: &[f64],
    common_ids: &[usize],
    rng: &mut Random,
    num: usize,
) -> Vec<usize> {
    let bucket = |keep: &dyn Fn(f64, f64) -> bool| -> Vec<usize> {
        common_ids
            .iter()
            .copied()
            .filter(|&id| keep(scores1[id], scores2[id]))
            .collect()
    };
    let ahead = bucket(&|a, b| a > b);
    let behind = bucket(&|a, b| b > a);
    // gepa spells the third bucket as the complement of the other two rather than as `a == b`, and
    // the two disagree on nan: it compares false against everything, so a nan pair falls out of
    // both orderings and lands here, where `a == b` would leave it in no bucket at all.
    let level: Vec<usize> = common_ids
        .iter()
        .copied()
        .filter(|id| !ahead.contains(id) && !behind.contains(id))
        .collect();
    let buckets = [ahead, behind, level];
    let n_each = 1.max(num.div_ceil(3));

    let mut selected: Vec<usize> = Vec::new();
    for bucket in &buckets {
        if selected.len() >= num {
            break;
        }
        let available: Vec<usize> = bucket
            .iter()
            .copied()
            .filter(|id| !selected.contains(id))
            .collect();
        let take = available.len().min(n_each).min(num - selected.len());
        // gepa's guard, kept though it decides nothing: `sample` at `k = 0` draws nothing and
        // returns nothing, so the branch it skips is already a no-op. Same below.
        if take > 0 {
            selected.extend(rng.sample(&available, take));
        }
    }

    let remaining = num.saturating_sub(selected.len());
    if remaining > 0 {
        let unused: Vec<usize> = common_ids
            .iter()
            .copied()
            .filter(|id| !selected.contains(id))
            .collect();
        if unused.len() >= remaining {
            selected.extend(rng.sample(&unused, remaining));
        } else if !common_ids.is_empty() {
            let weights = vec![1.0; common_ids.len()];
            for _ in 0..remaining {
                selected.push(common_ids[rng.choices(&weights, 1)[0]]);
            }
        }
    }
    selected.truncate(num);
    selected
}

impl MergesPerformed {
    /// Record an accepted merge's ancestor triple, so it is not attempted again.
    pub fn record_triple(&mut self, id1: usize, id2: usize, ancestor: usize) {
        self.triples.push((id1, id2, ancestor));
    }
}

/// The candidate's component names in declaration order — gepa's `list(candidate.keys())`.
///
/// Exact for the predicate. The construction loop reads the same names out of a *set* upstream; see
/// the module note on why that one is still an approximation.
fn components(candidate: &Candidate) -> impl Iterator<Item = &String> {
    candidate.keys()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(instruction: &str) -> Candidate {
        Candidate::from([("self".to_owned(), instruction.to_owned())])
    }

    /// The ancestor kept `A`; id1 changed it, id2 did not — so the merge takes id1's value, with no
    /// RNG draw at all.
    #[test]
    fn a_component_one_descendant_changed_comes_from_that_descendant() {
        let candidates = [candidate("A"), candidate("B"), candidate("A")]; // ancestor 0, id1 1, id2 2
        let mut rng = Random::seeded(0);
        let (merged, description) =
            merged_candidate(&candidates, 0, 1, 2, &[0.5, 0.6, 0.5], &mut rng);
        assert_eq!(
            merged["self"], "B",
            "id1 changed it, id2 kept the ancestor's"
        );
        assert_eq!(description, vec![1]);
    }

    /// Both descendants changed the component: the higher-scoring one wins, deterministically.
    #[test]
    fn both_changed_takes_the_stronger_descendant() {
        let candidates = [candidate("A"), candidate("B"), candidate("C")];
        let mut rng = Random::seeded(0);
        let (merged, _) = merged_candidate(&candidates, 0, 1, 2, &[0.5, 0.9, 0.6], &mut rng);
        assert_eq!(merged["self"], "B", "id1 scores higher");
    }

    #[test]
    fn ancestors_walk_the_parent_graph() {
        // 0 seed; 1,2 children of 0; 3 child of 1; 4 merge of 2 and 3.
        let parents = vec![vec![], vec![0], vec![0], vec![1], vec![2, 3]];
        assert_eq!(ancestors_of(4, &parents).to_vec(), {
            let mut set = PyIntSet::new();
            for a in [2, 0, 3, 1] {
                set.add(a);
            }
            set.to_vec()
        });
    }
}
