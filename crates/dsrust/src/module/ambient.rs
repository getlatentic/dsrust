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
//! The thread-local below is not the bug it looks like. It is installed across one `poll` and
//! taken down after, so two runs interleaved on one thread never see each other's buffer and a run
//! resumed on another thread carries its own — the trace follows the *task*. That is
//! `tokio::task_local!` written out, because tokio does not offer it on `wasm32`, and written once
//! rather than behind a `cfg` so the arm a Worker runs is the arm the suite covers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::TraceStep;

thread_local! {
    static ACTIVE_TRACE: RefCell<Option<TraceContext>> = const { RefCell::new(None) };
}

/// What a run in progress is recording into, and what to call each predictor that records.
struct TraceContext {
    steps: Vec<TraceStep>,
    /// Identity to the name [`named_predictors`](crate::Module::named_predictors) gives it. A
    /// predictor absent from the map is one the walk did not reach, and records under its own
    /// default name rather than not at all.
    predictor_names: Arc<HashMap<usize, String>>,
}

/// The identity of a predictor, for the run's lifetime.
///
/// The address of its signature: a `Predict` owns its signature inline, so the two addresses are
/// one identity, and it is stable while the predictor is. Compared, never dereferenced. This is
/// what `id(predictor)` stands in for upstream.
pub(crate) fn identity(signature: &crate::signature::Signature) -> usize {
    std::ptr::from_ref(signature) as usize
}

/// Whether this task is recording — the run is under [`Module::traced`](crate::Module::traced).
pub(crate) fn listening() -> bool {
    ACTIVE_TRACE.with_borrow(|context| context.is_some())
}

/// Record one call, if this task is recording. Nothing installed means nothing to do.
pub(crate) fn record(identity: usize, mut step: TraceStep) {
    ACTIVE_TRACE.with_borrow_mut(|context| {
        let Some(context) = context else { return };
        if let Some(name) = context.predictor_names.get(&identity) {
            step.predictor.clone_from(name);
        }
        context.steps.push(step);
    });
}

/// The context installed for exactly as long as one `poll`, and whatever it displaced.
///
/// A guard rather than a pair of calls so an unwind puts the outer run's context back: a panic in
/// one predictor must not leave the next run recording into a dead trace.
struct TraceGuard<'a> {
    suspended_context: &'a mut Option<TraceContext>,
}

impl<'a> TraceGuard<'a> {
    fn enter(context: &'a mut Option<TraceContext>) -> Self {
        ACTIVE_TRACE.with_borrow_mut(|active| std::mem::swap(active, context));
        Self {
            suspended_context: context,
        }
    }
}

impl Drop for TraceGuard<'_> {
    fn drop(&mut self) {
        ACTIVE_TRACE.with_borrow_mut(|active| std::mem::swap(active, self.suspended_context));
    }
}

/// `work`, with its context installed across each poll — and across the destruction of the
/// future, which is why the future is held in an `Option` rather than inline.
struct Recording<F> {
    context: Option<TraceContext>,
    future: Option<Pin<Box<F>>>,
}

impl<F: Future> Future for Recording<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let recording = self.get_mut();
        let _guard = TraceGuard::enter(&mut recording.context);
        let result = recording
            .future
            .as_mut()
            .expect("recording polled after completion")
            .as_mut()
            .poll(cx);
        // Dropped here, inside the guard, rather than left for the caller: a future that records
        // as it unwinds — a predictor charging a cancelled call — would otherwise attribute that
        // step to whatever trace is ambient once this one has been taken down.
        if result.is_ready() {
            drop(recording.future.take());
        }
        result
    }
}

/// The same rule for a run that never completed. A cancelled future's cleanup belongs to the
/// trace it was cancelled inside, not to whichever run happens to be polling when it is dropped.
impl<F> Drop for Recording<F> {
    fn drop(&mut self) {
        let _guard = TraceGuard::enter(&mut self.context);
        drop(self.future.take());
    }
}

/// Run `work` recording every predictor it reaches, under the names given.
pub(crate) async fn recording<T>(
    names: &Arc<HashMap<usize, String>>,
    work: impl Future<Output = T>,
) -> (T, Vec<TraceStep>) {
    let mut recording = Recording {
        context: Some(TraceContext {
            steps: Vec::new(),
            predictor_names: Arc::clone(names),
        }),
        future: Some(Box::pin(work)),
    };
    let output = (&mut recording).await;
    let TraceContext { steps, .. } = recording
        .context
        .take()
        .expect("trace context restored after polling");
    (output, steps)
}

#[cfg(test)]
mod tests;
