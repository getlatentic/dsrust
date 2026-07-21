//! What a run of a program cost, totalled across every call it made.
//!
//! dspy answers this with `with dspy.track_usage() as tracker:` — a tracker scoped by a context
//! manager, which every `LM` call adds to as it returns. That shape is the reason upstream can
//! report usage from *anywhere*, including calls whose return value has no room for it.
//!
//! Which is the case that matters here. [`Prediction`](crate::Prediction) carries a
//! [`Usage`] and the value-level paths read it off there, but `Predict::call_typed` and the
//! derived-task paths answer with the caller's own struct — there is nowhere in a `T` for a token
//! count to live. A scoped tracker reaches those calls because it does not travel on the answer.
//!
//! Scoped rather than always-on, matching upstream: totalling every call a long-lived process ever
//! makes is a different measurement from totalling one run, and it is the second one a caller
//! asking this question wants.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use super::Usage;

/// Tokens spent, per model.
///
/// Kept per model because a program that uses a cheap model to draft and an expensive one to judge
/// has two numbers worth knowing apart, and dspy keys its tracker the same way.
#[derive(Debug, Default)]
pub struct UsageTracker {
    by_model: Mutex<BTreeMap<String, Usage>>,
}

impl UsageTracker {
    /// Charge one call to a model.
    pub fn add(&self, model: &str, usage: Usage) {
        let mut totals = self.by_model.lock().expect("not poisoned");
        let running = totals.entry(model.to_owned()).or_default();
        running.input_tokens += usage.input_tokens;
        running.output_tokens += usage.output_tokens;
    }

    /// What each model was asked for, dspy's `get_total_tokens`.
    pub fn by_model(&self) -> BTreeMap<String, Usage> {
        self.by_model.lock().expect("not poisoned").clone()
    }

    /// What the whole run cost, across every model.
    pub fn total(&self) -> Usage {
        self.by_model.lock().expect("not poisoned").values().fold(
            Usage::default(),
            |running, usage| Usage {
                input_tokens: running.input_tokens + usage.input_tokens,
                output_tokens: running.output_tokens + usage.output_tokens,
            },
        )
    }

    /// Whether anything has been charged yet.
    pub fn is_empty(&self) -> bool {
        self.by_model.lock().expect("not poisoned").is_empty()
    }
}

/// The tracker calls are charged to while one is scoped, and the lock that keeps two overlapping
/// scopes from totalling into each other.
fn installed() -> &'static Mutex<Option<Arc<UsageTracker>>> {
    static INSTALLED: OnceLock<Mutex<Option<Arc<UsageTracker>>>> = OnceLock::new();
    INSTALLED.get_or_init(|| Mutex::new(None))
}

fn scope() -> &'static Mutex<()> {
    static SCOPE: OnceLock<Mutex<()>> = OnceLock::new();
    SCOPE.get_or_init(|| Mutex::new(()))
}

/// Count what every call costs until this is dropped. dspy's `with dspy.track_usage() as t:`.
///
/// ```
/// # async fn wrapper(program: dsrs::Predict) -> anyhow::Result<()> {
/// let counting = dsrs::lm::usage::track();
/// program.call("a question").await?;
/// println!("{} tokens", counting.tracker().total().total());
/// # Ok(()) }
/// ```
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
pub struct Tracking {
    tracker: Arc<UsageTracker>,
    _held: MutexGuard<'static, ()>,
}

impl Tracking {
    pub fn tracker(&self) -> &UsageTracker {
        &self.tracker
    }

    /// What has been spent so far in this scope.
    pub fn total(&self) -> Usage {
        self.tracker.total()
    }
}

impl Drop for Tracking {
    fn drop(&mut self) {
        *installed().lock().expect("not poisoned") = None;
    }
}

/// Charge a call to whichever scope is counting, if any.
///
/// A replay is not charged: the caller reads [`LmResponse::spend`](super::LmResponse::spend),
/// which is nothing on a cache hit, so a cached run totals what it actually bought.
pub(super) fn record(model: &str, spend: Option<Usage>) {
    let Some(usage) = spend else { return };
    if let Some(tracker) = installed().lock().expect("not poisoned").as_ref() {
        tracker.add(model, usage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input_tokens: u32, output_tokens: u32) -> Usage {
        Usage {
            input_tokens,
            output_tokens,
        }
    }

    #[test]
    fn a_scope_totals_every_call_charged_to_it() {
        let counting = track();
        record("anthropic/claude", Some(usage(10, 4)));
        record("anthropic/claude", Some(usage(6, 2)));

        assert_eq!(counting.total(), usage(16, 6));
        assert_eq!(counting.total().total(), 22);
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

    /// Nothing is counted outside a scope, so a long-lived process does not accumulate a total
    /// nobody asked for.
    #[test]
    fn a_call_outside_every_scope_is_charged_to_nothing() {
        record("anthropic/claude", Some(usage(999, 999)));
        let counting = track();
        assert!(counting.tracker().is_empty());
        record("anthropic/claude", Some(usage(1, 1)));
        assert_eq!(counting.total(), usage(1, 1));
    }

    /// The scope ends where the guard does, which is what makes this one run's number rather
    /// than the process's.
    #[test]
    fn a_dropped_scope_stops_counting() {
        let first = track();
        record("anthropic/claude", Some(usage(5, 5)));
        assert_eq!(first.total(), usage(5, 5));
        drop(first);

        record("anthropic/claude", Some(usage(70, 70)));

        let second = track();
        assert!(second.tracker().is_empty(), "a new scope starts at nothing");
    }

    /// A replay was already paid for once, and `spend` is how that arrives here as nothing.
    #[test]
    fn a_call_that_spent_nothing_is_not_charged() {
        let counting = track();
        record("anthropic/claude", None);
        assert!(counting.tracker().is_empty());
    }
}
