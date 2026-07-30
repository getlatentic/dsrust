//! Every option dspy's optimizer constructors accept, reachable from here.
//!
//! Five of GEPA's were fields with dspy's default and no setter: the algorithm read them, and a
//! caller could not change them. A constructor parameter that exists internally is still a
//! parameter the API does not have, which is the gap `scripts/check_api_surface.py`'s
//! `[constructors]` table was added to make visible.

use dsrust::{DummyLM, Example, Feedback, GEPA, Prediction};

/// A metric that ignores its inputs; these tests assert on the options, not on a compile.
fn metric(_: &Example, _: &Prediction) -> Feedback {
    Feedback::new(1.0, "fine")
}

/// The five that had no setter, plus the ones that did, all settable in one chain.
#[test]
fn gepas_options_are_all_reachable() {
    let model = std::sync::Arc::new(DummyLM::new([]));
    let tuned = GEPA::new(metric, model)
        .with_max_metric_calls(200)
        .with_reflection_minibatch_size(5)
        .with_seed(7)
        .with_perfect_score(0.95)
        .skipping_perfect_scores(false)
        .with_failure_score(-1.0)
        .with_merge(false)
        .with_max_merge_invocations(9);

    // The chain compiles and returns GEPA, which is what a caller needs; the values themselves are
    // held privately and are exercised by the conformance goldens rather than read back here.
    let _ = tuned;
}
