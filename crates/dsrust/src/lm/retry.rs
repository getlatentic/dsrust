//! dspy `LM(num_retries=3)`: asking a second time when the first ask failed transiently.
//!
//! dspy has no retry code of its own — it passes `num_retries=self.num_retries` and
//! `retry_strategy="exponential_backoff_retry"` into litellm, which hands both to `tenacity`. So the
//! oracle for this file is litellm 1.93.0's `completion_with_retries` and the tenacity policy it
//! builds, read rather than remembered:
//!
//! ```text
//! tenacity.AsyncRetrying(
//!     wait=tenacity.wait_exponential(multiplier=1, max=10),
//!     stop=tenacity.stop_after_attempt(num_retries),
//!     reraise=True,
//! )
//! ```
//!
//! Three details in there are easy to get wrong, and each is reproduced deliberately:
//!
//! * **`stop_after_attempt` counts attempts, not retries.** `num_retries=3` is three asks — two
//!   retries. dspy's own docstring calls it "the number of times to retry", which is off by one.
//! * **The wait is `min(multiplier * 2^(n-1), max)`,** with `n` the attempt that just failed and no
//!   jitter: 1s, then 2s, then 4s, capped at 10s.
//! * **Rate limits back off; other provider errors do not.** litellm overwrites the strategy dspy
//!   asked for — `RateLimitError` keeps exponential backoff, and any other `APIError` is downgraded
//!   to `constant_retry`, which is tenacity's default *no* wait.
//!
//! Two things here are this crate's own, and both are named where they happen: **which** failures
//! are retried comes from dspy 3.3's own `_RETRYABLE_LM_ERRORS` rather than from litellm retrying
//! every exception it sees, and a provider's `Retry-After` is honoured.

use std::future::Future;
use std::time::Duration;

use anyhow::Result;

use super::LmFailure;

/// dspy's `num_retries` default.
pub const DEFAULT_ATTEMPTS: usize = 3;

/// tenacity's `wait_exponential(max=10)`: the longest this crate waits of its own accord.
const MAX_WAIT_SECONDS: f64 = 10.0;

/// tenacity's `wait_exponential(multiplier=1, exp_base=2)`.
const MULTIPLIER: f64 = 1.0;
const EXP_BASE: f64 = 2.0;

/// How many times a failing provider call is asked again, and how long between asks.
///
/// dspy's `LM(num_retries=…)`. Set it with [`LmBuilder::num_retries`](super::LmBuilder::num_retries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Retry {
    /// Total asks, not retries — tenacity's `stop_after_attempt`, which counts the same way. One
    /// means never ask twice.
    pub attempts: usize,
}

impl Default for Retry {
    fn default() -> Self {
        Self {
            attempts: DEFAULT_ATTEMPTS,
        }
    }
}

impl Retry {
    /// A policy asking at most this many times. Zero reads as one: a call is always made.
    pub fn attempts(attempts: usize) -> Self {
        Self {
            attempts: attempts.max(1),
        }
    }

    /// Never ask twice — what `LM(num_retries=1)` gives, and what a test measuring one call wants.
    pub fn once() -> Self {
        Self::attempts(1)
    }

    /// How long to wait before ask number `next`, or `None` to give up and hand the failure back.
    ///
    /// `failed` is the ask that just failed, 1-based, so the first failure is `failed = 1`.
    pub fn wait_after(&self, failed: usize, failure: &LmFailure) -> Option<Duration> {
        if failed >= self.attempts || !failure.is_retryable() {
            return None;
        }
        Some(self.wait(failed, failure))
    }

    /// The delay itself, once it is settled that there will be another ask.
    ///
    /// A `retry-after` the provider sent wins, uncapped: waiting ten seconds when the server asked
    /// for sixty earns a second refusal. litellm ignores the header, which is the part of upstream
    /// not worth reproducing.
    ///
    /// Otherwise a rate limit backs off and nothing else does — litellm's `constant_retry`, whose
    /// tenacity default is no wait. A timeout has already spent its own delay, and a refused
    /// connection has nothing to wait for.
    fn wait(&self, failed: usize, failure: &LmFailure) -> Duration {
        if let Some(seconds) = failure.retry_after.filter(|seconds| *seconds > 0.0) {
            return Duration::from_secs_f64(seconds);
        }
        match failure.kind {
            super::LmErrorKind::RateLimit => Duration::from_secs_f64(exponential(failed)),
            _ => Duration::ZERO,
        }
    }
}

/// tenacity `wait_exponential.__call__`: `min(multiplier * exp_base ** (n - 1), max)`.
fn exponential(failed: usize) -> f64 {
    let exponent = u32::try_from(failed.saturating_sub(1)).unwrap_or(u32::MAX);
    let grown = MULTIPLIER * EXP_BASE.powi(exponent as i32);
    grown.min(MAX_WAIT_SECONDS)
}

/// Ask, and ask again while the failure is one dspy would have retried.
///
/// The failure handed back is the **last** one, which is what tenacity's `reraise=True` means. (In
/// litellm the retryer's own exception is swallowed by a bare `except Exception: pass` and the
/// *first* failure is raised instead; that is an accident of its control flow rather than a contract,
/// and reproducing it would hide the failure that actually ended the call.)
///
/// Anything that is not an [`LmFailure`] is handed straight back: a reply that would not parse is
/// not a transient provider failure, and it has its own recovery in
/// [`Predict`](crate::predict::Predict).
pub async fn asking<T, Attempt, Asked>(policy: Retry, mut attempt: Attempt) -> Result<T>
where
    Attempt: FnMut() -> Asked,
    Asked: Future<Output = Result<T>>,
{
    for number in 1.. {
        let error = match attempt().await {
            Ok(answered) => return Ok(answered),
            Err(error) => error,
        };
        let Some(failure) = error.downcast_ref::<LmFailure>() else {
            return Err(error);
        };
        let Some(wait) = policy.wait_after(number, failure) else {
            return Err(error);
        };
        tracing::warn!(
            attempt = number,
            of = policy.attempts,
            kind = failure.kind.code(),
            wait_ms = wait.as_millis(),
            "the provider failed transiently; asking again"
        );
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
    }
    unreachable!("the loop returns on the last attempt, which `attempts >= 1` guarantees exists")
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::lm::{LmErrorKind, api};

    fn failed(kind: LmErrorKind) -> LmFailure {
        LmFailure::new(kind, "no")
    }

    /// tenacity's curve, at each step and at the cap. Read off
    /// `wait_exponential(multiplier=1, max=10)`: 1, 2, 4, 8, then 10 forever.
    #[test]
    fn the_wait_is_tenacitys_exponential_capped_at_ten_seconds() {
        let waits: Vec<f64> = (1..=6).map(exponential).collect();
        assert_eq!(waits, vec![1.0, 2.0, 4.0, 8.0, 10.0, 10.0]);
    }

    /// `num_retries=3` is three asks, so there is a wait after the first and second and none after
    /// the third. Reading it as three *retries* would ask four times.
    #[test]
    fn three_attempts_means_two_retries() {
        let policy = Retry::default();
        assert_eq!(policy.attempts, 3);
        let limited = failed(LmErrorKind::RateLimit);
        assert!(policy.wait_after(1, &limited).is_some());
        assert!(policy.wait_after(2, &limited).is_some());
        assert!(
            policy.wait_after(3, &limited).is_none(),
            "the third ask is the last"
        );
    }

    /// Only dspy 3.3's `_RETRYABLE_LM_ERRORS` are asked again. A rejected key fails identically the
    /// second time, and litellm retrying it three times is the behaviour not worth reproducing.
    #[test]
    fn only_a_retryable_kind_is_asked_again() {
        let policy = Retry::default();
        for kind in [
            LmErrorKind::RateLimit,
            LmErrorKind::Timeout,
            LmErrorKind::Server,
            LmErrorKind::Transport,
        ] {
            assert!(policy.wait_after(1, &failed(kind)).is_some(), "{kind}");
        }
        for kind in [
            LmErrorKind::Auth,
            LmErrorKind::Billing,
            LmErrorKind::InvalidRequest,
            LmErrorKind::UnsupportedModel,
            LmErrorKind::UnsupportedFeature,
            LmErrorKind::Configuration,
            LmErrorKind::NotConfigured,
            LmErrorKind::Provider,
            LmErrorKind::Unexpected,
        ] {
            assert!(policy.wait_after(1, &failed(kind)).is_none(), "{kind}");
        }
    }

    /// litellm's downgrade to `constant_retry` for anything that is not a rate limit: those are
    /// asked again immediately.
    #[test]
    fn only_a_rate_limit_backs_off() {
        let policy = Retry::default();
        assert_eq!(
            policy.wait_after(1, &failed(LmErrorKind::RateLimit)),
            Some(Duration::from_secs(1))
        );
        for kind in [
            LmErrorKind::Timeout,
            LmErrorKind::Server,
            LmErrorKind::Transport,
        ] {
            assert_eq!(
                policy.wait_after(1, &failed(kind)),
                Some(Duration::ZERO),
                "{kind}"
            );
        }
    }

    /// A provider that named a delay is obeyed rather than second-guessed, and not clamped to the
    /// ten-second ceiling that bounds this crate's own curve.
    #[test]
    fn a_named_retry_after_wins_over_the_curve() {
        let policy = Retry::default();
        let asked = failed(LmErrorKind::RateLimit).with_retry_after(42.0);
        assert_eq!(
            policy.wait_after(1, &asked),
            Some(Duration::from_secs(42)),
            "the header, not min(2^0, 10)"
        );
    }

    /// The driver asks again and hands back the reply that finally worked.
    #[tokio::test]
    async fn a_transient_failure_is_asked_again_and_succeeds() {
        let asks = AtomicUsize::new(0);
        let answered: api::LmResponse = asking(Retry::default(), || {
            let number = asks.fetch_add(1, Ordering::SeqCst) + 1;
            async move {
                match number {
                    1 => Err(anyhow::Error::new(failed(LmErrorKind::Server))),
                    _ => Ok(api::LmResponse::default()),
                }
            }
        })
        .await
        .expect("the second ask answers");
        assert_eq!(asks.load(Ordering::SeqCst), 2);
        assert!(answered.outputs.is_empty());
    }

    /// Exhausting the budget hands back the last failure, not tenacity's own wrapper and not the
    /// first one. A caller reading `kind` has to see what ended the call.
    #[tokio::test]
    async fn the_last_failure_is_what_the_caller_sees() {
        let asks = AtomicUsize::new(0);
        let error = asking::<api::LmResponse, _, _>(Retry::default(), || {
            let number = asks.fetch_add(1, Ordering::SeqCst) + 1;
            async move {
                Err(anyhow::Error::new(match number {
                    1 => failed(LmErrorKind::Server),
                    _ => failed(LmErrorKind::Transport),
                }))
            }
        })
        .await
        .expect_err("every ask failed");
        assert_eq!(asks.load(Ordering::SeqCst), 3);
        assert_eq!(
            error
                .downcast_ref::<LmFailure>()
                .expect("an LmFailure")
                .kind,
            LmErrorKind::Transport
        );
    }

    /// A rejected key is asked once. Retrying it spends the caller's time to be told the same thing.
    #[tokio::test]
    async fn an_unretryable_failure_is_asked_once() {
        let asks = AtomicUsize::new(0);
        let error = asking::<api::LmResponse, _, _>(Retry::default(), || {
            asks.fetch_add(1, Ordering::SeqCst);
            async { Err(anyhow::Error::new(failed(LmErrorKind::Auth))) }
        })
        .await
        .expect_err("auth fails");
        assert_eq!(asks.load(Ordering::SeqCst), 1);
        assert_eq!(
            error.downcast_ref::<LmFailure>().map(|failed| failed.kind),
            Some(LmErrorKind::Auth)
        );
    }

    /// An error that is not an `LmFailure` at all is not a transient provider failure, so it goes
    /// straight back rather than being asked again three times.
    #[tokio::test]
    async fn something_that_is_not_a_provider_failure_is_handed_straight_back() {
        let asks = AtomicUsize::new(0);
        let error = asking::<api::LmResponse, _, _>(Retry::default(), || {
            asks.fetch_add(1, Ordering::SeqCst);
            async { Err(anyhow::anyhow!("the reply would not parse")) }
        })
        .await
        .expect_err("it fails");
        assert_eq!(asks.load(Ordering::SeqCst), 1);
        assert_eq!(error.to_string(), "the reply would not parse");
    }
}
