//! What a bootstrap pass earned, and which predictor earned it.
//!
//! dspy keeps this as `name2traces`, a dict it fills while walking the trainset and reads back
//! in `_train`. Attribution is its own decision — which predictor a demo belongs to, what to do
//! when one produced several for a single example, and what a program that reported nothing is
//! owed — so it is answered here rather than inside the walk that produces it.

use std::collections::BTreeMap;

use pyrng::Random;

use crate::example::Example;
use crate::hasher::Hasher;

/// What one bootstrap pass produced.
pub(super) struct Bootstrapped {
    /// dspy's `name2traces`: demos keyed by the predictor whose own calls earned them, so each
    /// step of a pipeline is taught by its own successes rather than the program's.
    ///
    /// A predictor that never ran gets nothing, which is dspy initialising every name to an
    /// empty list rather than to the program's demos.
    pub(super) per_predictor: BTreeMap<String, Vec<Example>>,
    /// Demos for a program that recorded no trace at all, which every predictor then receives.
    ///
    /// [`Module::forward_traced`](crate::Module::forward_traced) may record nothing, and then
    /// there is no attribution to make. For a program with one predictor the two are the same
    /// list anyway.
    pub(super) program: Vec<Example>,
    /// dspy's `validation`: the trainset examples no round solved, shuffled.
    pub(super) validation: Vec<Example>,
}

/// What one solved example earned, before it is filed under a predictor.
pub(super) struct Solved {
    /// The whole turn, used when nothing was traced.
    pub(super) program: Example,
    /// dspy's `Example(augmented=True, **inputs, **outputs)` per traced call.
    pub(super) per_predictor: Vec<(String, Example)>,
}

impl Bootstrapped {
    pub(super) fn empty() -> Self {
        Self {
            per_predictor: BTreeMap::new(),
            program: Vec::new(),
            validation: Vec::new(),
        }
    }

    /// File one solved example's demos under the predictors that earned them.
    ///
    /// A predictor that traced more than once for a single example is collapsed to one demo by
    /// [`collapse`], which is dspy's coin rather than a stand-in for it.
    pub(super) fn file(&mut self, solved: Solved) {
        if solved.per_predictor.is_empty() {
            self.program.push(solved.program);
            return;
        }
        // First-seen order, not sorted: the demos of one predictor reach `collapse` in the order
        // the trace recorded them, and it is the *last* of them the coin weighs against the rest.
        let mut traced: Vec<(String, Vec<Example>)> = Vec::new();
        for (predictor, demo) in solved.per_predictor {
            match traced.iter_mut().find(|(name, _)| *name == predictor) {
                Some((_, demos)) => demos.push(demo),
                None => traced.push((predictor, vec![demo])),
            }
        }
        for (predictor, mut demos) in traced {
            if demos.len() > 1 {
                demos = vec![demos.swap_remove(collapse(&demos))];
            }
            self.per_predictor
                .entry(predictor)
                .or_default()
                .append(&mut demos);
        }
    }

    /// The demos one predictor is taught by, capped at the bootstrapped budget.
    ///
    /// A traced program answers per predictor. An untraced one files everything under `program`
    /// instead, and the two are exclusive: nothing is filed both ways, so a predictor missing
    /// from a traced program falls through to an empty `program` and is taught by nothing —
    /// which is dspy starting every name at an empty list.
    pub(super) fn earned(&self, predictor: &str, most: usize) -> &[Example] {
        let earned = match self.per_predictor.get(predictor) {
            Some(earned) => earned.as_slice(),
            None => self.program.as_slice(),
        };
        &earned[..most.min(earned.len())]
    }
}

/// Which of a predictor's demos survives when it answered more than once in one example.
///
/// dspy flips a coin seeded by the demos themselves:
///
/// ```text
/// rng = random.Random(Hasher.hash(tuple(demos)))
/// demos = [rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]]
/// ```
///
/// Half the time the last trace, half the time a uniform draw from the earlier ones — and which
/// half is decided by the sha256 of the pickled tuple, so it is a fixed function of the demos and
/// not a coin this crate is free to flip its own way. Reproducing it needs the pickle bytes
/// ([`Hasher`]) and CPython's string seeding (`Random::from_seed_bytes`); with both, the answer is
/// upstream's answer.
///
/// This used to return the last demo unconditionally, justified in a comment that had two facts
/// wrong: dspy replaced xxhash with `hashlib.sha256` on 2026-05-07, and the `_input_keys` whose
/// set iteration order was said to make the seed unreproducible is `None` on this path — the demo
/// is `Example(augmented=True, **inputs, **outputs)` with no `with_inputs` after it, so no set is
/// ever pickled. Measured against upstream, the old answer differed on 8 of 24 recorded cases.
pub(super) fn collapse(demos: &[Example]) -> usize {
    debug_assert!(
        demos.len() > 1,
        "dspy reaches its coin only past a single trace"
    );
    let mut rng = Random::from_seed_bytes(Hasher::hash(demos).as_bytes());
    if rng.random() < 0.5 {
        rng.choice_index(demos.len() - 1)
    } else {
        demos.len() - 1
    }
}

#[cfg(test)]
mod collapse_tests {
    use super::*;

    /// Every recorded case, against the demo dspy's own coin kept.
    #[test]
    fn the_kept_demo_is_the_one_upstream_kept() {
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/conformance/optimize/hasher.json"))
                .expect("the hasher golden is valid JSON");
        let mut checked = 0;
        let mut not_the_last = 0;
        for case in golden["cases"].as_array().expect("cases") {
            let Some(expected) = case["index"].as_u64() else {
                continue; // a single demo never reaches the coin
            };
            let demos = rebuild(case);
            let kept = collapse(&demos);
            assert_eq!(
                kept,
                expected as usize,
                "the demo kept for {}",
                case["name"].as_str().expect("name")
            );
            checked += 1;
            not_the_last += usize::from(kept + 1 != demos.len());
        }
        assert!(checked >= 12, "only {checked} cases reached the coin");
        // Without this the suite would pass just as well against the last-demo answer this
        // replaced, which is exactly how that answer survived.
        assert!(
            not_the_last >= 3,
            "only {not_the_last} case(s) kept a demo other than the last"
        );
    }

    fn rebuild(case: &serde_json::Value) -> Vec<Example> {
        let keys: Vec<String> = case["input_keys"]
            .as_array()
            .expect("input_keys")
            .iter()
            .map(|key| key.as_str().expect("a key").to_owned())
            .collect();
        case["demos"]
            .as_array()
            .expect("demos")
            .iter()
            .map(|fields| {
                Example::new(
                    fields
                        .as_object()
                        .expect("a demo is an object")
                        .iter()
                        .map(|(name, value)| (name.clone(), value.clone())),
                )
                .with_inputs(keys.clone())
            })
            .collect()
    }
}
