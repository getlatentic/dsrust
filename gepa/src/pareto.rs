//! GEPA's Pareto-front candidate selection (`gepa_utils.py`): pick the next program to mutate from
//! the frontier, favouring programs that uniquely win validation testcases.
//!
//! `fronts[testcase]` is the set of program indices that achieve the best score on that testcase.
//! Dominated programs — whose every best-on testcase is also won by a survivor — are dropped, then
//! a program is drawn weighted by how many testcases it still wins.
//!
//! The frontier is a CPython `set` per testcase, and its *iteration order* is load-bearing: it
//! seeds the first-appearance order the domination sweep walks, and — through
//! [`find_dominator_programs`] — the candidate list merge's `rng.sample` draws against. That order
//! is not sorted (it agrees with sorted only for indices 0-7), so the fronts are [`PyIntSet`]s,
//! built by the same add sequence dspy's `_update_pareto_front_for_val_id` uses.

use std::collections::BTreeSet;

use pyrng::Random;

use crate::pyset::PyIntSet;

/// GEPA's per-testcase Pareto front (`GEPAState.program_at_pareto_front_valset`): for each validation
/// testcase, the set of programs achieving the best score on it. [`select_candidate`] reads it; this
/// maintains it. The seed program starts on every front.
pub struct ParetoFront {
    best: Vec<f64>,
    fronts: Vec<PyIntSet>,
}

impl ParetoFront {
    /// dspy's `GEPAState` initialisation: the seed program (index 0) on every testcase's front.
    pub fn seeded(seed_scores: &[f64]) -> Self {
        Self {
            best: seed_scores.to_vec(),
            fronts: seed_scores.iter().map(|_| PyIntSet::from_keys([0])).collect(),
        }
    }

    /// dspy `_update_pareto_front_for_val_id` over every testcase: a strictly higher score replaces a
    /// testcase's front with a fresh one-element set, an exact tie adds to the existing set (keeping
    /// its insertion order, and so its iteration order), a worse score is ignored.
    pub fn add_program(&mut self, program: usize, scores: &[f64]) {
        for (testcase, &score) in scores.iter().enumerate() {
            let previous = self.best[testcase];
            if score > previous {
                self.best[testcase] = score;
                self.fronts[testcase] = PyIntSet::from_keys([program]);
            } else if score == previous {
                self.fronts[testcase].add(program);
            }
        }
    }

    /// The front per testcase, as [`select_candidate`] reads it.
    pub fn fronts(&self) -> &[PyIntSet] {
        &self.fronts
    }
}

/// dspy `select_program_candidate_from_pareto_front`: the survivor set, then a frequency-weighted
/// draw. `scores` is the weighted aggregate score per program, used only to order the domination
/// sweep.
pub fn select_candidate(fronts: &[PyIntSet], scores: &[f64], rng: &mut Random) -> usize {
    let survivors = remove_dominated(fronts, scores);
    let list = sampling_list(&survivors);
    let index = rng.choice_index(list.len());
    list[index]
}

/// dspy `find_dominator_programs`: the distinct programs left on any front after dominated ones are
/// dropped, in the order `list(set(...))` yields them.
///
/// This is merge's candidate pool — `rng.sample` draws its pair from exactly this list — so the
/// final `set` is rebuilt with a [`PyIntSet`] over the survivors in the order they appear across
/// the fronts, rather than sorted.
pub fn find_dominator_programs(fronts: &[PyIntSet], scores: &[f64]) -> Vec<usize> {
    let survivors = remove_dominated(fronts, scores);
    let mut unique = PyIntSet::new();
    for front in &survivors {
        for program in front.iter() {
            unique.add(program);
        }
    }
    unique.to_vec()
}

/// dspy `remove_dominated_programs`: drop every program whose best-on testcases are all also won by
/// some surviving program, so only programs carrying a unique win remain. The sweep runs in
/// ascending-score order and restarts whenever it removes one, matching upstream's `while` loop.
/// Each surviving front keeps its own iteration order minus the dropped programs — the `difference`
/// dspy takes.
fn remove_dominated(fronts: &[PyIntSet], scores: &[f64]) -> Vec<PyIntSet> {
    let mut programs = first_appearance_order(fronts);
    // dspy `sorted(programs, key=scores)`, a stable ascending sort — ties keep first-appearance order.
    programs.sort_by(|a, b| scores[*a].total_cmp(&scores[*b]));

    let mut dominated = BTreeSet::new();
    let mut removing = true;
    while removing {
        removing = false;
        for &y in &programs {
            if dominated.contains(&y) {
                continue;
            }
            let others: BTreeSet<usize> =
                programs.iter().copied().filter(|p| *p != y && !dominated.contains(p)).collect();
            if is_dominated(y, &others, fronts) {
                dominated.insert(y);
                removing = true;
                break;
            }
        }
    }

    fronts
        .iter()
        .map(|front| PyIntSet::from_keys(front.iter().filter(|p| !dominated.contains(p))))
        .collect()
}

/// dspy `is_dominated`: `y` is dominated unless some testcase it wins is won by nobody else in
/// `others` — that testcase is `y`'s alone, so it survives.
fn is_dominated(y: usize, others: &BTreeSet<usize>, fronts: &[PyIntSet]) -> bool {
    for front in fronts {
        if !front.contains(y) {
            continue;
        }
        if !front.iter().any(|program| others.contains(&program)) {
            return false;
        }
    }
    true
}

/// The program indices in the order they first appear across the fronts — dspy's `freq.keys()`,
/// whose insertion order is each front's CPython iteration order in turn.
fn first_appearance_order(fronts: &[PyIntSet]) -> Vec<usize> {
    let mut seen = BTreeSet::new();
    let mut order = Vec::new();
    for front in fronts {
        for program in front.iter() {
            if seen.insert(program) {
                order.push(program);
            }
        }
    }
    order
}

/// dspy's `sampling_list`: each program repeated once per testcase it wins, in first-appearance
/// order — so a program on more of the frontier is proportionally likelier to be drawn.
fn sampling_list(fronts: &[PyIntSet]) -> Vec<usize> {
    let mut counts: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for front in fronts {
        for program in front.iter() {
            *counts.entry(program).or_insert(0) += 1;
        }
    }
    first_appearance_order(fronts)
        .into_iter()
        .flat_map(|program| std::iter::repeat_n(program, counts[&program]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fronts_of(case: &Value) -> Vec<PyIntSet> {
        case["fronts"]
            .as_array()
            .expect("fronts")
            .iter()
            .map(|front| {
                PyIntSet::from_keys(
                    front.as_array().expect("a front").iter().map(|p| p.as_u64().unwrap() as usize),
                )
            })
            .collect()
    }

    /// The candidate GEPA's own selection returns, across many fronts and seeds — from
    /// `tests/conformance/pareto.json`, generated by running the real gepa package. A match pins the
    /// domination sweep, the ascending set/dict ordering, and the CPython `choice` draw together.
    #[test]
    fn selects_the_candidate_gepa_selects() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/pareto.json");
        let text = std::fs::read_to_string(&path).expect("the pareto golden is committed");
        let fixture: Value = serde_json::from_str(&text).expect("the golden parses");
        let seeds: Vec<u64> = fixture["seeds"].as_array().expect("seeds").iter().map(|s| s.as_u64().unwrap()).collect();

        for case in fixture["cases"].as_array().expect("cases") {
            let fronts = fronts_of(case);
            let scores: Vec<f64> =
                case["scores"].as_array().expect("scores").iter().map(|s| s.as_f64().unwrap()).collect();
            let picks: Vec<usize> =
                case["picks"].as_array().expect("picks").iter().map(|p| p.as_u64().unwrap() as usize).collect();
            for (seed, &expected) in seeds.iter().zip(&picks) {
                let mut rng = Random::seeded(*seed);
                assert_eq!(
                    select_candidate(&fronts, &scores, &mut rng),
                    expected,
                    "front {fronts:?} seed {seed}"
                );
            }
        }
    }

    /// The per-testcase front GEPA's `GEPAState` builds as programs are added — from
    /// `tests/conformance/front.json`, generated by running the real gepa package. Verifies the
    /// strictly-better-replaces, ties-join, worse-ignored update over every testcase.
    #[test]
    fn maintains_the_front_gepa_maintains() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/front.json");
        let text = std::fs::read_to_string(&path).expect("the front golden is committed");
        let fixture: Value = serde_json::from_str(&text).expect("the golden parses");

        for case in fixture["cases"].as_array().expect("cases") {
            let scores = |value: &Value| -> Vec<f64> {
                value.as_array().unwrap().iter().map(|s| s.as_f64().unwrap()).collect()
            };
            let seed = scores(&case["seed"]);
            let programs: Vec<Vec<f64>> = case["programs"].as_array().unwrap().iter().map(scores).collect();
            let snapshots = case["fronts"].as_array().expect("fronts");

            let mut front = ParetoFront::seeded(&seed);
            assert_front(&front, &snapshots[0], "after seed");
            for (index, program) in programs.iter().enumerate() {
                front.add_program(index + 1, program);
                assert_front(&front, &snapshots[index + 1], &format!("after program {}", index + 1));
            }
        }
    }

    /// Compare the crate's front to a fixture snapshot: `{testcase -> program indices in CPython
    /// set order}`.
    fn assert_front(front: &ParetoFront, snapshot: &Value, at: &str) {
        for (testcase, set) in front.fronts().iter().enumerate() {
            let got: Vec<usize> = set.iter().collect();
            let want: Vec<usize> = snapshot[testcase.to_string()]
                .as_array()
                .unwrap_or_else(|| panic!("{at}: testcase {testcase} missing"))
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(got, want, "{at}: testcase {testcase}");
        }
    }
}
