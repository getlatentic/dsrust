//! Which call a handler is being told about, and which call enclosed it.
//!
//! Upstream keeps this in a `ContextVar`: `with_callbacks` sets `ACTIVE_CALL_ID` to the new id just
//! before running the wrapped function and restores it afterwards, so a handler reached during that
//! function reads its caller's id. `CallId` carries the parent it was born under instead, which
//! answers the same question without a second lookup — and means a handler is never asked to read
//! ambient state to make sense of what it was handed.

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};

thread_local! {
    /// The call currently running on this thread, which is the parent of the next one started.
    static ACTIVE: Cell<Option<u64>> = const { Cell::new(None) };
}

/// One call, and the call it happened inside — dspy's `call_id` plus its `ACTIVE_CALL_ID` parent.
///
/// A counter rather than upstream's random uuid: what the identifier is for is connecting a start
/// handler to its end handler and a child to its parent, and a counter does both without a
/// dependency. It is unique within the process, which is as far as the callbacks reach.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CallId {
    id: u64,
    parent: Option<u64>,
}

impl CallId {
    /// The next call, born under whichever one is running here.
    pub(crate) fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            parent: ACTIVE.with(Cell::get),
        }
    }

    /// This call's own identifier, the same value at its start and at its end.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The call this one happened inside, or nothing if it is the outermost.
    ///
    /// A program's shape is this: `ChainOfThought` starts, `Predict` starts under it, the model call
    /// starts under that. Upstream's `test_active_id` asserts exactly this chain.
    pub fn parent(&self) -> Option<u64> {
        self.parent
    }
}

impl std::fmt::Display for CallId {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "{:016x}", self.id)
    }
}

/// Restores the enclosing call when it is dropped.
pub(crate) struct Entered(Option<u64>);

impl Drop for Entered {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.set(self.0));
    }
}

/// Make `call` the running one until the guard is dropped — upstream's `ACTIVE_CALL_ID.set(call_id)`
/// and the `finally` that puts the parent back.
///
/// For the points that do not await. An asynchronous one uses [`Under`], because a guard held across
/// an await would claim whatever the runtime polled next.
///
/// Entered *after* the start handlers have run, never before — upstream is explicit about the same
/// ordering, and it is what a handler reading the tree depends on: while `on_module_start` runs, the
/// call still running is the parent. Reversing it would make every call its own parent.
pub(crate) fn entered(call: &CallId) -> Entered {
    Entered(ACTIVE.with(|active| active.replace(Some(call.id))))
}

/// A future that runs under `call`: the enclosing call is set for the duration of each poll and put
/// back when the poll returns.
///
/// Per poll rather than for the whole future, which is what makes this right where upstream's
/// `ContextVar` would not be. `Evaluate` runs its rows with `buffered`, so several rows interleave
/// inside one task: a value set once and left would be read by whichever row was polled next. It is
/// also what carries the parent across a runtime that moves a future between threads.
pub(crate) struct Under<F> {
    call: CallId,
    inner: Pin<Box<F>>,
}

impl<F: Future> Under<F> {
    /// Boxed so the projection needs no `unsafe`. One allocation per watched point, which is beside
    /// a model call.
    pub(crate) fn new(call: CallId, inner: F) -> Self {
        Self {
            call,
            inner: Box::pin(inner),
        }
    }
}

impl<F: Future> Future for Under<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<F::Output> {
        let under = self.get_mut();
        let _entered = entered(&under.call);
        under.inner.as_mut().poll(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A call started inside another names it, and the outermost names nothing.
    #[test]
    fn a_call_is_born_under_the_one_running() {
        let outer = CallId::next();
        assert_eq!(outer.parent(), None);

        let _entered = entered(&outer);
        let inner = CallId::next();
        assert_eq!(inner.parent(), Some(outer.id()));
        assert_ne!(inner.id(), outer.id());
    }

    /// Leaving a call puts its parent back, so a sibling started afterwards is a sibling and not a
    /// child — upstream's `finally: ACTIVE_CALL_ID.set(parent_call_id)`.
    #[test]
    fn leaving_a_call_restores_the_one_that_enclosed_it() {
        let parent = CallId::next();
        let entered_parent = entered(&parent);

        let first = {
            let child = CallId::next();
            let _entered = entered(&child);
            child
        };
        let second = CallId::next();

        assert_eq!(first.parent(), Some(parent.id()));
        assert_eq!(second.parent(), Some(parent.id()));
        drop(entered_parent);
        assert_eq!(CallId::next().parent(), None);
    }

    /// Two futures interleaved in one task each see their own call, which is the case a value set
    /// once and left would get wrong.
    #[tokio::test]
    async fn interleaved_futures_do_not_borrow_each_others_parent() {
        async fn child(parent: CallId) -> Option<u64> {
            // Two yields, so the runtime is guaranteed to poll the other future in between.
            tokio::task::yield_now().await;
            let seen = CallId::next().parent();
            tokio::task::yield_now().await;
            assert_eq!(seen, Some(parent.id()));
            seen
        }

        let left = CallId::next();
        let right = CallId::next();
        let (first, second) = futures_util::future::join(
            Under::new(left, child(left)),
            Under::new(right, child(right)),
        )
        .await;

        assert_eq!(first, Some(left.id()));
        assert_eq!(second, Some(right.id()));
    }
}
