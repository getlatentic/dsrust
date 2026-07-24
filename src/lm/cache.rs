//! Replaying a reply instead of paying for it again.
//!
//! dspy caches every LM call and keys the entry on the whole request, `rollout_id` included. That
//! is what makes `lm.copy(rollout_id=n, temperature=1.0)` mean anything: two attempts that would
//! otherwise be the same request become two different keys, so the second is answered rather than
//! replayed. Every retry-shaped module upstream — `BestOfN`, `Refine`, `BootstrapFewShot`'s later
//! rounds — is built on that.
//!
//! Shaped after upstream in three ways that are easy to get wrong. The store is process-wide, so
//! two `LM` values naming the same model share replies rather than each paying separately — which
//! is why the model id is part of the key. It is on by default, because dspy's `LM.__init__` takes
//! `cache: bool = True`. And it is bounded, because an unbounded one is a leak in anything
//! long-lived; dspy's is an `LRUCache(maxsize=1_000_000)`.
//!
//! Only [`LM`](super::LM) caches by default, matching upstream exactly: dspy's `DummyLM` extends
//! `BaseLM` rather than `LM`, so the cache decoration never reaches it and a scripted model always
//! advances its script. [`Cached`] gives any other model a store of its own for callers who want
//! one anyway.

use std::num::NonZeroUsize;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use lru::LruCache;

pub mod disk;

pub use disk::DiskCache;

use super::ChatModel;
use super::api::{self, LmResponse};

/// dspy's `memory_max_entries`, which is effectively "grow until something is clearly wrong"
/// rather than a tuned figure.
const MAX_ENTRIES: usize = 1_000_000;

/// Replies kept against the requests that produced them.
///
/// Two tiers, as upstream's is: memory first, then disk, and a disk hit is promoted into memory
/// so the next read of it is quick. The disk half is what makes re-running a compile cheap.
pub struct ResponseCache {
    entries: Mutex<LruCache<String, LmResponse>>,
    disk: Option<DiskCache>,
}

impl ResponseCache {
    /// A store holding at most `max_entries` in memory, dropping the least recently read to stay
    /// under, and keeping nothing on disk.
    pub fn new(max_entries: NonZeroUsize) -> Self {
        Self {
            entries: Mutex::new(LruCache::new(max_entries)),
            disk: None,
        }
    }

    /// The same, also writing through to a directory that outlives the process.
    pub fn with_disk(mut self, disk: DiskCache) -> Self {
        self.disk = Some(disk);
        self
    }

    /// What one call answered with before, marked as the replay it is.
    ///
    /// Reading counts as a use, so a request asked throughout a long run is not the one evicted.
    pub fn replay(&self, key: &str) -> Option<LmResponse> {
        let kept = self.remembered(key).or_else(|| self.recovered(key))?;
        Some(LmResponse {
            cache_hit: true,
            ..kept
        })
    }

    fn remembered(&self, key: &str) -> Option<LmResponse> {
        self.entries.lock().expect("not poisoned").get(key).cloned()
    }

    /// A reply an earlier run paid for, brought back into memory so the next read is quick.
    fn recovered(&self, key: &str) -> Option<LmResponse> {
        let found = self.disk.as_ref()?.get(key)?;
        self.entries
            .lock()
            .expect("not poisoned")
            .put(key.to_owned(), found.clone());
        Some(found)
    }

    /// Keep this reply against the request that produced it.
    pub fn keep(&self, key: String, response: LmResponse) {
        if let Some(disk) = &self.disk {
            disk.put(&key, &response);
        }
        self.entries
            .lock()
            .expect("not poisoned")
            .put(key, response);
    }

    /// How many distinct requests are held in memory.
    pub fn len(&self) -> usize {
        self.entries.lock().expect("not poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The directory this writes through to, if it has one.
    pub fn disk(&self) -> Option<&DiskCache> {
        self.disk.as_ref()
    }

    /// Forget everything, in memory and on disk, so the next call of every shape reaches a model
    /// again.
    pub fn clear(&self) {
        self.entries.lock().expect("not poisoned").clear();
        if let Some(disk) = &self.disk {
            disk.clear();
        }
    }
}

impl Default for ResponseCache {
    /// Memory only. [`shared`] adds the disk half; a cache built by hand is usually a test's, and
    /// a test writing into someone's home directory would be a surprise.
    fn default() -> Self {
        Self::new(NonZeroUsize::new(MAX_ENTRIES).expect("a non-zero constant"))
    }
}

static SHARED: OnceLock<ResponseCache> = OnceLock::new();

/// The process-wide store every [`LM`](super::LM) reads, dspy's module-level `DSPY_CACHE`.
///
/// Shared rather than per-model so that two `LM` values built for the same model — which is what
/// a program that constructs one per call ends up with — answer each other's repeated requests.
/// Backed by disk as well as memory when there is a directory to use, which is upstream's
/// default and what makes a repeated compile cheap.
pub fn shared() -> &'static ResponseCache {
    SHARED.get_or_init(|| match DiskCache::from_env() {
        Some(disk) => ResponseCache::default().with_disk(disk),
        None => ResponseCache::default(),
    })
}

/// A model that answers a repeated request from memory, with a store of its own.
///
/// [`LM`](super::LM) already caches, so this is for giving some *other* model the same behaviour:
/// a scripted one in a test that wants to prove a caller re-asked, or a wrapper of a caller's own.
/// Its store is private to the wrapper rather than [`shared`], because a model that is not an `LM`
/// has no model id to keep its entries apart from anyone else's.
pub struct Cached<M> {
    inner: M,
    cache: ResponseCache,
}

impl<M> Cached<M> {
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            cache: ResponseCache::default(),
        }
    }

    /// A wrapper holding at most `max_entries` replies.
    pub fn bounded(inner: M, max_entries: NonZeroUsize) -> Self {
        Self {
            inner,
            cache: ResponseCache::new(max_entries),
        }
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn clear(&self) {
        self.cache.clear();
    }
}

impl<M: ChatModel + Send + Sync> ChatModel for Cached<M> {
    async fn forward(
        &self,
        http: &reqwest::Client,
        request: &api::LmRequest,
    ) -> Result<LmResponse> {
        // The store belongs to this wrapper and holds one model's replies, so there is no other
        // model's entries for a name to keep these apart from.
        let key = request.cache_key("");
        if let Some(replayed) = self.cache.replay(&key) {
            return Ok(replayed);
        }
        let answered = self.inner.forward(http, request).await?;
        self.cache.keep(key, answered.clone());
        Ok(answered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use crate::lm::api::interop::raise_request;
    use crate::lm::dummy::DummyLM;
    use crate::lm::{ChatTurn, LmConfig, LmUsage, OutputMode};

    fn ask(lm: &Cached<DummyLM>, config: LmConfig) -> LmResponse {
        let request = raise_request(
            "be helpful",
            &[ChatTurn::user("what colour?")],
            OutputMode::Text,
            &config,
        );
        futures_lite_block_on(lm.forward(&reqwest::Client::new(), &request)).expect("an answer")
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

        let first = ask(&lm, LmConfig::default());
        assert!(!first.cache_hit);
        assert!(first.first_text().contains("red"));

        let second = ask(&lm, LmConfig::default());
        assert!(second.cache_hit, "the second ask was replayed");
        assert!(
            second.first_text().contains("red"),
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

        let first = ask(&lm, LmConfig::rollout(0));
        let second = ask(&lm, LmConfig::rollout(1));

        assert!(!first.cache_hit);
        assert!(!second.cache_hit, "a new rollout is a new key");
        assert!(first.first_text().contains("red"));
        assert!(
            second.first_text().contains("blue"),
            "the model was asked a second time and gave its next answer"
        );
        assert_eq!(lm.len(), 2, "two rollouts are two entries");
    }

    /// Temperature is sent, so it is part of what the reply depends on and must key separately.
    #[test]
    fn a_different_temperature_is_a_different_entry() {
        let lm = Cached::new(DummyLM::new([]).with_fallback(example! { answer: "any" }));
        ask(&lm, LmConfig::default());
        ask(
            &lm,
            LmConfig {
                temperature: Some(1.0),
                ..LmConfig::default()
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
        ask(&lm, LmConfig::default());
        lm.clear();
        assert!(lm.is_empty());

        let after = ask(&lm, LmConfig::default());
        assert!(!after.cache_hit);
        assert!(after.first_text().contains("blue"));
    }

    /// An unbounded cache is a leak in anything long-lived, so the bound has to actually evict.
    #[test]
    fn the_oldest_entry_is_dropped_once_the_bound_is_reached() {
        let lm = Cached::bounded(
            DummyLM::new([]).with_fallback(example! { answer: "any" }),
            NonZeroUsize::new(2).expect("two"),
        );
        for id in 0..3 {
            ask(&lm, LmConfig::rollout(id));
        }
        assert_eq!(lm.len(), 2, "three distinct requests, room for two");
    }

    /// What the disk half is for: a second run pays nothing for what the first one bought. A
    /// fresh `ResponseCache` stands in for the fresh process, since the memory half of the first
    /// one is exactly what a restart loses.
    #[test]
    fn a_reply_kept_on_disk_outlives_the_cache_that_wrote_it() {
        let dir = std::env::temp_dir().join("dsrust-cache-across-runs");
        let _ = std::fs::remove_dir_all(&dir);

        let first = ResponseCache::default().with_disk(DiskCache::new(&dir, 1_000_000));
        first.keep("key".to_owned(), LmResponse::text("bought once"));

        let restarted = ResponseCache::default().with_disk(DiskCache::new(&dir, 1_000_000));
        assert!(
            restarted.is_empty(),
            "memory starts cold, as a new process would"
        );

        let replayed = restarted.replay("key").expect("recovered from disk");
        assert!(replayed.cache_hit);
        assert_eq!(replayed.first_text(), "bought once");
        assert_eq!(restarted.len(), 1, "and is promoted into memory once read");

        restarted.clear();
        assert_eq!(
            ResponseCache::default()
                .with_disk(DiskCache::new(&dir, 1_000_000))
                .replay("key"),
            None,
            "clearing reaches the disk half too"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A hit is a replay, not a purchase. dspy skips its usage tracker on one for the same
    /// reason, so summing `spend` over a run cannot bill the same tokens twice.
    #[test]
    fn a_replay_reports_what_it_was_worth_but_no_new_spend() {
        let cache = ResponseCache::default();
        let usage = LmUsage::counted(10, 4);
        cache.keep(
            "key".to_owned(),
            LmResponse::text("the reply").with_usage(Some(usage.clone())),
        );

        let replayed = cache.replay("key").expect("a hit");
        assert!(replayed.cache_hit);
        assert_eq!(replayed.usage, Some(usage), "what it was worth is readable");
        assert_eq!(replayed.spend(), None, "and it is not charged again");
    }
}
