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
//!
//! Following the task is what [`Recording`] is for, and it is why the thread-local below is not
//! the bug it looks like: the value is installed only across one `poll` and taken down after, so
//! two runs interleaved on one thread never see each other's buffer, and a run resumed on another
//! thread carries its own. That is how `tokio::task_local!` works from the inside, written out
//! here because tokio does not offer it on `wasm32` — and written once rather than behind a `cfg`,
//! so the arm a Worker runs is the arm the test suite already covers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use super::TraceStep;

thread_local! {
    static AMBIENT: RefCell<Option<Ambient>> = const { RefCell::new(None) };
}

/// What a run in progress is recording into, and what to call each predictor that records.
#[derive(Clone)]
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

/// Whether this task is recording — the run is under [`Module::traced`](crate::Module::traced).
pub(crate) fn listening() -> bool {
    AMBIENT.with(|slot| slot.borrow().is_some())
}

/// Record one call, if this task is recording. Nothing installed means nothing to do.
pub(crate) fn record(identity: usize, mut step: TraceStep) {
    AMBIENT.with(|slot| {
        let installed = slot.borrow();
        let Some(ambient) = installed.as_ref() else {
            return;
        };
        if let Some(name) = ambient.names.get(&identity) {
            step.predictor.clone_from(name);
        }
        if let Ok(mut steps) = ambient.steps.lock() {
            steps.push(step);
        }
    });
}

/// The buffer installed for exactly as long as one `poll`, and whatever it displaced.
///
/// A guard rather than a pair of calls so an unwind through `poll` puts the outer run's buffer
/// back: a panic in one predictor must not leave the next run recording into a dead trace.
struct Entered(Option<Ambient>);

impl Entered {
    fn holding(ambient: Ambient) -> Self {
        Self(AMBIENT.with(|slot| slot.borrow_mut().replace(ambient)))
    }
}

impl Drop for Entered {
    fn drop(&mut self) {
        AMBIENT.with(|slot| *slot.borrow_mut() = self.0.take());
    }
}

/// `work`, with `ambient` installed across each of its polls and nothing between them.
struct Recording<F> {
    ambient: Ambient,
    work: Pin<Box<F>>,
}

impl<F: Future> Future for Recording<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        // Both fields are `Unpin` — the future is held boxed — so this needs no projection.
        let this = self.get_mut();
        let _entered = Entered::holding(this.ambient.clone());
        this.work.as_mut().poll(cx)
    }
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
    let answered = Recording {
        ambient,
        work: Box::pin(work),
    }
    .await;
    let recorded = steps.lock().map(|held| held.clone()).unwrap_or_default();
    (answered, recorded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::Signature;

    fn step(named: &str) -> TraceStep {
        TraceStep {
            predictor: named.into(),
            inputs: crate::Example::default(),
            outputs: super::super::StepOutputs::Answered(crate::Example::default()),
            signature: "q -> a".parse::<Signature>().expect("parses"),
        }
    }

    /// The property the thread-local owes that a task-local gave for free.
    ///
    /// Two runs on one thread, each yielding between installing its buffer and recording into it,
    /// so the executor is guaranteed to interleave them. A buffer that outlived a `poll` would
    /// leak either way: the second run's step would land in the first's trace, or the first would
    /// finish holding both. Neither is a failure any existing test would show, because every other
    /// one traces a single run.
    #[tokio::test]
    async fn two_runs_interleaved_on_one_thread_keep_their_traces_apart() {
        let names = Arc::new(HashMap::new());
        let one = recording(&names, async {
            tokio::task::yield_now().await;
            record(1, step("first"));
            tokio::task::yield_now().await;
        });
        let two = recording(&names, async {
            tokio::task::yield_now().await;
            record(2, step("second"));
            tokio::task::yield_now().await;
        });
        let ((_, first), (_, second)) = tokio::join!(one, two);

        assert_eq!(
            first
                .iter()
                .map(|s| s.predictor.as_str())
                .collect::<Vec<_>>(),
            ["first"],
            "the first run recorded only its own call"
        );
        assert_eq!(
            second
                .iter()
                .map(|s| s.predictor.as_str())
                .collect::<Vec<_>>(),
            ["second"],
        );
    }

    /// Nothing installed records nothing, rather than recording into the run that just ended.
    #[tokio::test]
    async fn a_call_outside_any_run_is_not_recorded() {
        let names = Arc::new(HashMap::new());
        let (_, recorded) = recording(&names, async {}).await;
        assert!(recorded.is_empty());
        assert!(!listening(), "the buffer came down with the run");
        record(1, step("stray"));
    }
}
