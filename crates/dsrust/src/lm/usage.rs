//! What a run of a program cost, totalled across every call it made.
//!
//! dspy answers this with `with dspy.track_usage() as tracker:` — a tracker scoped by a context
//! manager, which every `LM` call adds to as it returns. That shape is the reason upstream can
//! report usage from *anywhere*, including calls whose return value has no room for it.
//!
//! Which is the case that matters here. [`Prediction`](crate::Prediction) carries a
//! [`LmUsage`] and the value-level paths read it off there, but `Predict::call_typed` and the
//! derived-task paths answer with the caller's own struct — there is nowhere in a `T` for a token
//! count to live. A scoped tracker reaches those calls because it does not travel on the answer.
//!
//! Scoped rather than always-on, matching upstream: totalling every call a long-lived process ever
//! makes is a different measurement from totalling one run, and it is the second one a caller
//! asking this question wants.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
#[cfg(feature = "process-global")]
use std::sync::{MutexGuard, OnceLock};

use super::LmUsage;

/// Tokens spent, per model.
///
/// Kept per model because a program that uses a cheap model to draft and an expensive one to judge
/// has two numbers worth knowing apart, and dspy keys its tracker the same way.
#[derive(Debug, Default)]
pub struct UsageTracker {
    by_model: Mutex<BTreeMap<String, LmUsage>>,
}

impl UsageTracker {
    /// Charge one call to a model.
    pub fn add(&self, model: &str, usage: LmUsage) {
        let mut totals = self.by_model.lock().expect("not poisoned");
        let running = totals.entry(model.to_owned()).or_default();
        // Through `merge` rather than by hand, so every counter is carried — a run that spent
        // reasoning tokens totals them, where adding two fields would quietly drop eight.
        *running = LmUsage::merge(Some(running.clone()), Some(usage)).unwrap_or_default();
    }

    /// What each model was asked for, dspy's `get_total_tokens`.
    pub fn by_model(&self) -> BTreeMap<String, LmUsage> {
        self.by_model.lock().expect("not poisoned").clone()
    }

    /// What the whole run cost, across every model.
    pub fn total(&self) -> LmUsage {
        self.by_model
            .lock()
            .expect("not poisoned")
            .values()
            .fold(None, |running, usage| {
                LmUsage::merge(running, Some(usage.clone()))
            })
            .unwrap_or_default()
    }

    /// Whether anything has been charged yet.
    pub fn is_empty(&self) -> bool {
        self.by_model.lock().expect("not poisoned").is_empty()
    }
}

thread_local! {
    /// The tracker installed for the future currently being polled. A Worker isolate may
    /// interleave many requests on one thread, so this is restored after every poll rather than
    /// left installed across an await.
    static SCOPED: std::cell::RefCell<Option<Arc<UsageTracker>>> =
        const { std::cell::RefCell::new(None) };
}

/// Create a request-local usage scope that remains isolated when futures are interleaved.
pub fn scoped() -> UsageScope {
    UsageScope {
        tracker: Arc::new(UsageTracker::default()),
    }
}

/// Usage accounting owned by one asynchronous operation.
#[derive(Clone)]
pub struct UsageScope {
    tracker: Arc<UsageTracker>,
}

impl UsageScope {
    pub fn tracker(&self) -> &UsageTracker {
        &self.tracker
    }

    pub fn total(&self) -> LmUsage {
        self.tracker.total()
    }

    pub async fn run<T>(&self, work: impl Future<Output = T>) -> T {
        crate::scoped::Scoped::<Charging, _>::new(Arc::clone(&self.tracker), work).await
    }
}

/// The tracker a call is charged to, for as long as the future owning it is polled or destroyed.
pub(crate) struct Charging;

impl crate::scoped::Ambience for Charging {
    type State = Arc<UsageTracker>;

    fn exchange(held: &mut Option<Arc<UsageTracker>>) {
        SCOPED.with_borrow_mut(|installed| std::mem::swap(installed, held));
    }
}

/// The tracker calls are charged to while one is scoped, and the lock that keeps two overlapping
/// scopes from totalling into each other.
#[cfg(feature = "process-global")]
fn installed() -> &'static Mutex<Option<Arc<UsageTracker>>> {
    static INSTALLED: OnceLock<Mutex<Option<Arc<UsageTracker>>>> = OnceLock::new();
    INSTALLED.get_or_init(|| Mutex::new(None))
}

#[cfg(feature = "process-global")]
fn scope() -> &'static Mutex<()> {
    static SCOPE: OnceLock<Mutex<()>> = OnceLock::new();
    SCOPE.get_or_init(|| Mutex::new(()))
}

/// Count what every call costs until this is dropped. dspy's `with dspy.track_usage() as t:`.
///
/// Reached as `lm::track_usage` from outside, which is the name dspy uses:
///
/// ```
/// use dsrust::lm::{Tracking, track_usage};
///
/// # async fn wrapper(program: dsrust::Predict) -> anyhow::Result<()> {
/// let counting: Tracking = track_usage();
/// program.call("a question").await?;
/// println!("{:?} tokens", counting.total().total());
/// // Dropping it stops the counting, so the scope is the block rather than a call to end it.
/// # Ok(()) }
/// ```
#[cfg(feature = "process-global")]
pub fn track() -> Tracking {
    // Held for the whole scope, so a second `track()` waits rather than silently splitting one
    // run's calls between two totals.
    let held = scope()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let tracker = Arc::new(UsageTracker::default());
    *installed().lock().expect("not poisoned") = Some(Arc::clone(&tracker));
    Tracking {
        tracker,
        _held: held,
    }
}

/// A scope that is counting. Charges stop when it is dropped.
#[cfg(feature = "process-global")]
pub struct Tracking {
    tracker: Arc<UsageTracker>,
    _held: MutexGuard<'static, ()>,
}

#[cfg(feature = "process-global")]
impl Tracking {
    pub fn tracker(&self) -> &UsageTracker {
        &self.tracker
    }

    /// What has been spent so far in this scope.
    pub fn total(&self) -> LmUsage {
        self.tracker.total()
    }
}

#[cfg(feature = "process-global")]
impl Drop for Tracking {
    fn drop(&mut self) {
        *installed().lock().expect("not poisoned") = None;
    }
}

/// Charge a call to whichever scope is counting, if any.
///
/// A replay is not charged: the caller reads [`LmResponse::spend`](super::LmResponse::spend),
/// which is nothing on a cache hit, so a cached run totals what it actually bought.
pub(super) fn record(model: &str, spend: Option<LmUsage>) {
    let Some(usage) = spend else { return };
    if let Some(tracker) = SCOPED.with(|slot| slot.borrow().clone()) {
        tracker.add(model, usage);
        return;
    }
    #[cfg(feature = "process-global")]
    if let Some(tracker) = installed().lock().expect("not poisoned").as_ref() {
        tracker.add(model, usage);
    }
}

#[cfg(all(test, feature = "process-global"))]
mod tests {
    use super::*;

    fn usage(input_tokens: u32, output_tokens: u32) -> LmUsage {
        LmUsage::counted(input_tokens, output_tokens)
    }

    #[test]
    fn a_scope_totals_every_call_charged_to_it() {
        let counting = track();
        record("anthropic/claude", Some(usage(10, 4)));
        record("anthropic/claude", Some(usage(6, 2)));

        assert_eq!(counting.total(), usage(16, 6));
        assert_eq!(counting.total().total(), Some(22));
    }

    /// A drafting model and a judging model are two numbers worth knowing apart.
    #[test]
    fn each_model_is_totalled_on_its_own() {
        let counting = track();
        record("openai/gpt-4o-mini", Some(usage(10, 4)));
        record("anthropic/claude", Some(usage(100, 40)));

        let by_model = counting.tracker().by_model();
        assert_eq!(by_model["openai/gpt-4o-mini"], usage(10, 4));
        assert_eq!(by_model["anthropic/claude"], usage(100, 40));
        assert_eq!(counting.total(), usage(110, 44), "and summed across both");
    }

    /// Hold the tracking scope with nothing installed, so a test of the *out-of-scope* behaviour
    /// serialises against the tests that do install a tracker rather than racing one. Without
    /// this these tests' unscoped `record` calls leaked into a concurrent scope's tracker — a
    /// stray 70 tokens once landing in [`each_model_is_totalled_on_its_own`].
    fn exclusive() -> MutexGuard<'static, ()> {
        let held = scope()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *installed().lock().expect("not poisoned") = None;
        held
    }

    fn install(tracker: &Arc<UsageTracker>) {
        *installed().lock().expect("not poisoned") = Some(Arc::clone(tracker));
    }

    fn uninstall() {
        *installed().lock().expect("not poisoned") = None;
    }

    /// Nothing is counted outside a scope, so a long-lived process does not accumulate a total
    /// nobody asked for.
    #[test]
    fn a_call_outside_every_scope_is_charged_to_nothing() {
        let _held = exclusive();
        record("anthropic/claude", Some(usage(999, 999)));

        let tracker = Arc::new(UsageTracker::default());
        install(&tracker);
        assert!(tracker.is_empty(), "the out-of-scope call charged nothing");
        record("anthropic/claude", Some(usage(1, 1)));
        assert_eq!(tracker.total(), usage(1, 1));
        uninstall();
    }

    /// The scope ends where the guard does, which is what makes this one run's number rather
    /// than the process's.
    #[test]
    fn a_dropped_scope_stops_counting() {
        let _held = exclusive();
        let first = Arc::new(UsageTracker::default());
        install(&first);
        record("anthropic/claude", Some(usage(5, 5)));
        assert_eq!(first.total(), usage(5, 5));
        uninstall();

        record("anthropic/claude", Some(usage(70, 70)));

        let second = Arc::new(UsageTracker::default());
        install(&second);
        assert!(second.is_empty(), "a new scope starts at nothing");
        uninstall();
    }

    /// The guard's own `Drop`, which `a_dropped_scope_stops_counting` never reaches: that one
    /// drives `install`/`uninstall` and so proves the mechanism while leaving `Tracking::drop`
    /// unexercised.
    ///
    /// No `exclusive()` here, and that is not an oversight. `track()` takes the same scope lock
    /// that `exclusive()` holds, and a `std::sync::Mutex` is not reentrant — taking both on one
    /// thread deadlocks, which is why every other test in this module reaches for the helpers
    /// instead. `track()` serialises this test against them on its own.
    ///
    /// Asserted on the slot rather than on a later `track()`, because `track()` overwrites the
    /// slot unconditionally: a second scope reads empty whether or not the first cleared up.
    #[test]
    fn dropping_the_guard_clears_the_installed_tracker() {
        {
            let counting = track();
            record("anthropic/claude", Some(usage(5, 5)));
            assert_eq!(counting.total(), usage(5, 5));
        }

        assert!(
            installed().lock().expect("not poisoned").is_none(),
            "the guard took the tracker down with it"
        );
    }

    /// `is_empty` answering `true` unconditionally survived: every use of it asserted the empty
    /// direction, so the loaded one was never named.
    #[test]
    fn a_tracker_that_has_been_charged_is_not_empty() {
        let tracker = UsageTracker::default();
        assert!(tracker.is_empty(), "nothing charged yet");
        tracker.add("anthropic/claude", usage(1, 1));
        assert!(!tracker.is_empty(), "and not once something has been");
    }

    /// A replay was already paid for once, and `spend` is how that arrives here as nothing.
    #[test]
    fn a_call_that_spent_nothing_is_not_charged() {
        let counting = track();
        record("anthropic/claude", None);
        assert!(counting.tracker().is_empty());
    }
}

#[cfg(test)]
mod scoped_tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn interleaved_scopes_never_share_usage() {
        let first = scoped();
        let second = scoped();

        let first_run = first.run(async {
            record("first", Some(LmUsage::counted(2, 3)));
            tokio::task::yield_now().await;
            record("first", Some(LmUsage::counted(5, 7)));
        });
        let second_run = second.run(async {
            record("second", Some(LmUsage::counted(11, 13)));
            tokio::task::yield_now().await;
            record("second", Some(LmUsage::counted(17, 19)));
        });

        tokio::join!(first_run, second_run);

        assert_eq!(first.total(), LmUsage::counted(7, 10));
        assert_eq!(second.total(), LmUsage::counted(28, 32));
        assert_eq!(first.tracker().by_model().len(), 1);
        assert_eq!(second.tracker().by_model().len(), 1);
    }
}

#[cfg(test)]
mod destruction_scope {
    use super::*;

    struct ChargeOnDrop;
    impl Future for ChargeOnDrop {
        type Output = ();
        fn poll(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<()> {
            std::task::Poll::Ready(())
        }
    }
    impl Drop for ChargeOnDrop {
        fn drop(&mut self) {
            record("m", Some(LmUsage::counted(9, 9)));
        }
    }

    #[tokio::test]
    async fn a_future_charging_as_it_is_destroyed_bills_its_own_scope() {
        let outer = scoped();
        let inner = scoped();
        outer
            .run(async {
                inner.run(ChargeOnDrop).await;
            })
            .await;
        assert_eq!(
            inner.total().input_tokens,
            Some(9),
            "the inner scope was billed"
        );
        assert_eq!(outer.total().input_tokens, None, "the outer scope was not");
    }
}
