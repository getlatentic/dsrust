//! Recording a predictor's call without its caller having threaded a trace to it.
//!
//! dspy needs no such seam: `Predict.__call__` appends to `dspy.settings.trace` on the way out, so
//! every predictor a program runs is recorded whatever the program's `forward` looks like, and the
//! name comes afterwards from `predictor2name[id(predictor)]`. A composed module written by hand
//! traces itself, and its author never hears about it.
//!
//! Rust has no ambient anything by default, so [`Module::forward_traced`](crate::Module::forward_traced)
//! is a seam an author must implement — and one they will not know to, because a module that does
//! not implement it still compiles, still runs, and is still accepted by every optimizer. What it
//! is not is attributable, and the demos then belong to nobody.
//!
//! This is the same mechanism upstream's, transposed: a buffer that follows the *task* rather than
//! the thread (a future may resume anywhere, and a trace spanning awaits would otherwise scatter),
//! and identity taken from the address of a predictor's signature, which is what `id(predictor)` is
//! standing in for.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::TraceStep;

tokio::task_local! {
    static AMBIENT: Option<Ambient>;
}

/// What a run in progress is recording into, and what to call each predictor that records.
struct Ambient {
    steps: Arc<Mutex<Vec<TraceStep>>>,
    /// Identity to the name [`named_predictors`](crate::Module::named_predictors) gives it. A
    /// predictor absent from the map is one the walk did not reach, and records under its own
    /// default name rather than not at all.
    names: Arc<HashMap<usize, String>>,
}

/// The identity of a predictor, for the run's lifetime.
///
/// The address of its signature: a `Predict` owns its signature inline, so the two addresses are
/// one identity, and it is stable while the predictor is. Compared, never dereferenced.
pub(crate) fn identity(signature: &crate::signature::Signature) -> usize {
    std::ptr::from_ref(signature) as usize
}

/// Record one call, if this task is recording. Nothing installed means nothing to do.
pub(crate) fn record(identity: usize, mut step: TraceStep) {
    let _ = AMBIENT.try_with(|ambient| {
        let Some(ambient) = ambient else { return };
        if let Some(name) = ambient.names.get(&identity) {
            step.predictor.clone_from(name);
        }
        if let Ok(mut steps) = ambient.steps.lock() {
            steps.push(step);
        }
    });
}

/// Run `work` recording every predictor it reaches, under the names given.
pub(crate) async fn recording<T>(
    names: &Arc<HashMap<usize, String>>,
    work: impl Future<Output = T>,
) -> (T, Vec<TraceStep>) {
    let steps = Arc::new(Mutex::new(Vec::new()));
    let ambient = Ambient {
        steps: Arc::clone(&steps),
        names: Arc::clone(names),
    };
    let answered = AMBIENT.scope(Some(ambient), work).await;
    let recorded = steps.lock().map(|held| held.clone()).unwrap_or_default();
    (answered, recorded)
}

/// Run `work` with no ambient recording, so a predictor that records its own step does not also
/// record one here.
///
/// [`Predict::forward_traced`](struct@crate::Predict) writes the step its caller asked for and then calls
/// its own `forward`, which would otherwise record a second copy into whatever buffer an outer
/// run had installed.
pub(crate) async fn detached<T>(work: impl Future<Output = T>) -> T {
    AMBIENT.scope(None, work).await
}
