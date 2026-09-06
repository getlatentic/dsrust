//! A value installed for exactly as long as one future is being polled — or destroyed.
//!
//! Four things in this crate follow a task rather than a thread: the trace a run records into, the
//! model a scope configures, the tracker usage is charged to, and the callbacks watching a point.
//! A thread-local left installed across an `await` is read by whatever the executor polls next, and
//! a Worker isolate interleaves requests on one thread, so each is installed per poll instead.
//!
//! Destruction counts as being polled. A future that touches the ambient value as it is dropped —
//! a predictor recording a cancelled call, an LM charging a partial spend — must see the scope it
//! ran under and not whichever one happens to be installed when the executor lets it go. That is
//! the half three of the four copies were missing, so there is one copy now.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Somewhere a value can be installed for the current thread, and taken back out.
pub(crate) trait Ambience: 'static {
    /// `Unpin` because [`Scoped`] moves it in and out of the thread-local between polls; every
    /// state in this crate is an owned struct or an `Arc`, so the bound costs nothing.
    type State: Unpin;

    /// Exchange what is installed with `held`, leaving each where the other was.
    ///
    /// One operation rather than an install and a restore, because the two must not be able to
    /// disagree: the enclosing scope is whatever this displaced, and it goes back untouched.
    fn exchange(held: &mut Option<Self::State>);
}

/// `future`, with `state` installed across every poll and across its destruction.
pub(crate) struct Scoped<A: Ambience, F> {
    held: Option<A::State>,
    future: Option<Pin<Box<F>>>,
}

impl<A: Ambience, F> Scoped<A, F> {
    pub(crate) fn new(state: A::State, future: F) -> Self {
        Self {
            held: Some(state),
            future: Some(Box::pin(future)),
        }
    }

    /// The state after the run, for a caller that put something in it.
    pub(crate) fn take(&mut self) -> Option<A::State> {
        self.held.take()
    }
}

/// Installed on entry, put back on exit — including when polling unwinds.
struct Entered<'a, A: Ambience>(&'a mut Option<A::State>);

impl<'a, A: Ambience> Entered<'a, A> {
    fn enter(held: &'a mut Option<A::State>) -> Self {
        A::exchange(held);
        Self(held)
    }
}

impl<A: Ambience> Drop for Entered<'_, A> {
    fn drop(&mut self) {
        A::exchange(self.0);
    }
}

impl<A: Ambience, F: Future> Future for Scoped<A, F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<F::Output> {
        // Both fields are `Unpin` — the future is boxed — so this needs no projection.
        let scoped = self.get_mut();
        let _entered = Entered::<A>::enter(&mut scoped.held);
        let polled = scoped
            .future
            .as_mut()
            .expect("a scoped future polled after it completed")
            .as_mut()
            .poll(context);
        if polled.is_ready() {
            scoped.future = None;
        }
        polled
    }
}

impl<A: Ambience, F> Drop for Scoped<A, F> {
    fn drop(&mut self) {
        if self.future.is_some() {
            let _entered = Entered::<A>::enter(&mut self.held);
            self.future = None;
        }
    }
}

#[cfg(test)]
mod tests;
