//! Replaying a reply instead of paying for it again.
//!
//! dspy caches every LM call and keys the entry on the whole request, `rollout_id` included. That
//! is what makes `lm.copy(rollout_id=n, temperature=1.0)` mean anything: two attempts that would
//! otherwise be the same request become two different keys, so the second is answered rather than
//! replayed. Without a cache the field would be inert, and every retry-shaped module upstream —
//! `BestOfN`, `Refine`, `BootstrapFewShot`'s later rounds — is built on it.
//!
//! In memory only. dspy also writes a disk layer, which buys a warm cache across processes and
//! costs a serialisation format to keep compatible; nothing here runs long enough to want it yet.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::Result;

use super::{ChatModel, LmRequest, LmResponse};

/// A model that answers a repeated request from memory.
///
/// Wraps any [`ChatModel`], so it composes with the scripted ones as readily as with a provider,
/// and a caller who does not want caching simply does not wrap.
pub struct Cached<M> {
    inner: M,
    entries: Mutex<HashMap<String, LmResponse>>,
}

impl<M> Cached<M> {
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// How many distinct requests have been answered and kept.
    pub fn len(&self) -> usize {
        self.entries.lock().expect("not poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget everything, so the next call of every shape reaches the model again.
    pub fn clear(&self) {
        self.entries.lock().expect("not poisoned").clear();
    }
}

impl<M: ChatModel + Send + Sync> ChatModel for Cached<M> {
    async fn chat(&self, http: &reqwest::Client, request: &LmRequest<'_>) -> Result<LmResponse> {
        let key = request.cache_key();
        if let Some(hit) = self.entries.lock().expect("not poisoned").get(&key) {
            // The usage stays as the original call reported it, so a caller can still see what
            // the answer was worth. `cache_hit` is how they tell that from a fresh charge.
            return Ok(LmResponse {
                cache_hit: true,
                ..hit.clone()
            });
        }
        let answered = self.inner.chat(http, request).await?;
        self.entries
            .lock()
            .expect("not poisoned")
            .insert(key, answered.clone());
        Ok(answered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use crate::lm::dummy::DummyLM;
    use crate::lm::{ChatTurn, OutputMode, Sampling};

    fn ask(lm: &Cached<DummyLM>, sampling: Sampling) -> LmResponse {
        let turns = [ChatTurn::user("what colour?")];
        let request = LmRequest::new("be helpful", &turns, OutputMode::Text).sampled(sampling);
        futures_lite_block_on(lm.chat(&reqwest::Client::new(), &request)).expect("an answer")
    }

    /// The dummy never awaits anything real, so a trivial executor keeps these synchronous.
    fn futures_lite_block_on<F: std::future::Future>(future: F) -> F::Output {
        use std::task::{Context, Poll, Wake, Waker};
        struct Noop;
        impl Wake for Noop {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = Waker::from(std::sync::Arc::new(Noop));
        let mut context = Context::from_waker(&waker);
        let mut future = Box::pin(future);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
        }
    }

    #[test]
    fn the_same_request_twice_reaches_the_model_once() {
        let lm = Cached::new(DummyLM::new([
            example! { answer: "red" },
            example! { answer: "blue" },
        ]));

        let first = ask(&lm, Sampling::default());
        assert!(!first.cache_hit);
        assert!(first.text_ref().contains("red"));

        let second = ask(&lm, Sampling::default());
        assert!(second.cache_hit, "the second ask was replayed");
        assert!(
            second.text_ref().contains("red"),
            "and replayed the first answer, not the next scripted one"
        );
    }

    /// The whole reason `rollout_id` exists: it is not sent to any provider, so if it did not
    /// change the key it would change nothing at all, and a re-ask could never differ.
    #[test]
    fn a_fresh_rollout_id_misses_the_cache_and_is_answered_again() {
        let lm = Cached::new(DummyLM::new([
            example! { answer: "red" },
            example! { answer: "blue" },
        ]));

        let first = ask(&lm, Sampling::rollout(0));
        let second = ask(&lm, Sampling::rollout(1));

        assert!(!first.cache_hit);
        assert!(!second.cache_hit, "a new rollout is a new key");
        assert!(first.text_ref().contains("red"));
        assert!(
            second.text_ref().contains("blue"),
            "the model was asked a second time and gave its next answer"
        );
        assert_eq!(lm.len(), 2, "two rollouts are two entries");
    }

    /// Temperature is sent, so it is part of what the reply depends on and must key separately.
    #[test]
    fn a_different_temperature_is_a_different_entry() {
        let lm = Cached::new(DummyLM::new([]).with_fallback(example! { answer: "any" }));
        ask(&lm, Sampling::default());
        ask(
            &lm,
            Sampling {
                temperature: Some(1.0),
                ..Sampling::default()
            },
        );
        assert_eq!(lm.len(), 2);
    }

    #[test]
    fn clearing_sends_the_next_ask_back_to_the_model() {
        let lm = Cached::new(DummyLM::new([
            example! { answer: "red" },
            example! { answer: "blue" },
        ]));
        ask(&lm, Sampling::default());
        lm.clear();
        assert!(lm.is_empty());

        let after = ask(&lm, Sampling::default());
        assert!(!after.cache_hit);
        assert!(after.text_ref().contains("blue"));
    }
}
