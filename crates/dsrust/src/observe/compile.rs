//! dspy 3.3.1's compile point, around one optimizer's `compile`.
//!
//! Upstream decorates every `Teleprompter` subclass through `__init_subclass__`, so a compile
//! written anywhere fires the point. Rust has no metaclass, so each optimizer goes through one of
//! these two functions instead, and `tests/callback.rs` is what says they all do.
use std::future::Future;

use anyhow::Result;
use tracing::{Instrument, field};

use super::{TARGET, Watch, opening};
use crate::callback::{self, Under};
use crate::example::Example;

thread_local! {
    /// The optimizer instances whose `compile` is running on this thread — dspy 3.3.1's
    /// `_ACTIVE_COMPILES`, keyed by the instance: a compile an optimizer runs on itself from inside
    /// its own compile is one call to the caller, and reports once.
    static ACTIVE_COMPILES: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// dspy 3.3.1 `on_compile_start`/`on_compile_end`, around one optimizer's `compile`. `instance`
/// identifies the optimizer — its address, which is stable for the call — so that a re-entry by
/// the same instance inside its own compile runs unreported, as upstream's does.
pub async fn compiling<T>(
    optimizer: &'static str,
    instance: usize,
    trainset: &[Example],
    valset: Option<&[Example]>,
    work: impl Future<Output = Result<T>>,
) -> Result<T> {
    if ACTIVE_COMPILES.with_borrow(|active| active.contains(&instance)) {
        return work.await;
    }
    let watch = compile_point(optimizer);
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_compile_start(&watch.call, optimizer, trainset, valset)
        });
    }
    let answered = ActiveCompile {
        instance,
        inner: Under::new(watch.call, work),
    }
    .instrument(watch.span.clone())
    .await;
    watch.finished(answered.as_ref(), |_| String::new());
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_compile_end(&watch.call, answered.as_ref().map(|_| ()))
        });
    }
    answered
}

/// [`compiling`] for an optimizer whose compile does not await and cannot fail: upstream's end
/// handler sees its program and no exception.
pub fn compiling_sync<T>(
    optimizer: &'static str,
    instance: usize,
    trainset: &[Example],
    valset: Option<&[Example]>,
    work: impl FnOnce() -> T,
) -> T {
    if ACTIVE_COMPILES.with_borrow(|active| active.contains(&instance)) {
        return work();
    }
    let watch = compile_point(optimizer);
    let _entered = watch.span.enter();
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_compile_start(&watch.call, optimizer, trainset, valset)
        });
    }
    let _under = callback::entered(&watch.call);
    ACTIVE_COMPILES.with_borrow_mut(|active| active.push(instance));
    let compiled = work();
    ACTIVE_COMPILES.with_borrow_mut(|active| active.retain(|held| *held != instance));
    watch.finished(Ok::<&(), &anyhow::Error>(&()), |()| String::new());
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_compile_end(&watch.call, Ok(()))
        });
    }
    compiled
}

fn compile_point(optimizer: &'static str) -> Watch {
    opening(tracing::info_span!(
        target: TARGET,
        "compile",
        optimizer = optimizer,
        inputs = field::Empty,
        outputs = field::Empty,
        error = field::Empty,
    ))
}

/// A future polled with its optimizer marked active on the polling thread, and unmarked between
/// polls — so two compiles interleaved on one thread each see only their own.
struct ActiveCompile<F> {
    instance: usize,
    inner: F,
}

impl<F: Future> Future for ActiveCompile<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // SAFETY: `inner` is never moved out of `self`, which is pinned; the projection keeps the
        // pin, and `instance` is `Copy`.
        let (instance, inner) = unsafe {
            let this = self.get_unchecked_mut();
            (this.instance, std::pin::Pin::new_unchecked(&mut this.inner))
        };
        ACTIVE_COMPILES.with_borrow_mut(|active| active.push(instance));
        let polled = inner.poll(context);
        ACTIVE_COMPILES.with_borrow_mut(|active| {
            if let Some(at) = active.iter().rposition(|held| *held == instance) {
                active.remove(at);
            }
        });
        polled
    }
}
