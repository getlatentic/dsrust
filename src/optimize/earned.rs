//! What a bootstrap pass earned, and which predictor earned it.
//!
//! dspy keeps this as `name2traces`, a dict it fills while walking the trainset and reads back
//! in `_train`. Attribution is its own decision — which predictor a demo belongs to, what to do
//! when one produced several for a single example, and what a program that reported nothing is
//! owed — so it is answered here rather than inside the walk that produces it.

use std::collections::BTreeMap;

use crate::example::Example;

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
    /// dspy collapses a predictor that traced more than once for a single example down to one
    /// demo, choosing evenly between its last trace and a random earlier one. This takes the last
    /// trace, which agrees with upstream whenever its coin lands there — and a predictor called
    /// once per example never reaches the branch at all.
    ///
    /// Not for want of the generator: [`rng`](super::rng) already reproduces the one upstream
    /// draws with. The choice is seeded with `xxhash64(pickle.dumps(demos))`, and those bytes
    /// embed the iteration order of `_input_keys`, a Python `set` — fixed within a process and
    /// different between them, since CPython seeds string hashing per interpreter. Upstream's own
    /// seed is therefore not reproducible, so there is no single answer to match.
    ///
    /// Taking the last trace is also the best available answer rather than a concession. Deriving
    /// a seed here would flip a coin independent of upstream's, and two independent coins agree
    /// less often than one fixed choice matching a coin: measured, 0.500 at any number of traces
    /// against 0.376 at three and 0.287 at eight.
    pub(super) fn file(&mut self, solved: Solved) {
        if solved.per_predictor.is_empty() {
            self.program.push(solved.program);
            return;
        }
        let mut last: BTreeMap<String, Example> = BTreeMap::new();
        for (predictor, demo) in solved.per_predictor {
            last.insert(predictor, demo);
        }
        for (predictor, demo) in last {
            self.per_predictor.entry(predictor).or_default().push(demo);
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
