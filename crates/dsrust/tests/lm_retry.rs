//! The retry, over a real socket, on the path a program actually takes.
//!
//! A retry policy that is stored but never reaches the wire looks identical to one that works, so
//! nothing here asserts on the field: a server that *refuses* a fixed number of times stands in for
//! a rate-limited provider, and each case counts the requests that arrived and how far apart they
//! were. The gaps are what separate the three behaviours upstream has — a rate limit backs off for a
//! second, a `Retry-After` header replaces that with what the server asked for, and any other
//! provider failure is asked again immediately.
//!
//! Both directions, always: a case that only proved "it retried" would pass for a policy that
//! retried everything, including a rejected key.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use dsrust::lm::{ChatModel, LM, LmErrorKind, LmFailure, api};
use dsrust::{Predict, call};
use serde_json::json;

/// tenacity's first exponential step, which is what the gaps here are measured against.
const FIRST_BACKOFF: Duration = Duration::from_secs(1);

/// A stub that refuses the first `refusals` requests with `status`, then answers.
///
/// Threaded rather than async because it has to observe *when* each request arrived, and a server
/// sharing the runtime under test would report the scheduler's timing as the provider's.
struct Refusing {
    address: String,
    arrivals: Arc<std::sync::Mutex<Vec<Instant>>>,
    stopping: Arc<AtomicBool>,
    served: JoinHandle<()>,
}

impl Refusing {
    fn new(refusals: usize, status: u16, retry_after: Option<&'static str>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let address = format!("http://{}", listener.local_addr().expect("a bound address"));
        let arrivals = Arc::new(std::sync::Mutex::new(Vec::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let recorded = Arc::clone(&arrivals);
        let asked_to_stop = Arc::clone(&stopping);
        let served = std::thread::spawn(move || {
            // Most cases leave refusals unspent on purpose, so the loop has to be interruptible:
            // `done` sets the flag and opens one connection to release the blocking accept.
            for number in 0..=refusals {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                if asked_to_stop.load(Ordering::SeqCst) {
                    return;
                }
                recorded.lock().expect("the arrivals").push(Instant::now());
                drain_request(&mut stream);
                let _ = match number < refusals {
                    true => write_refusal(&mut stream, status, retry_after),
                    false => write_reply(&mut stream),
                };
            }
        });
        Self {
            address,
            arrivals,
            stopping,
            served,
        }
    }

    fn base_url(&self) -> String {
        format!("{}/v1", self.address)
    }

    /// A model pointed at this stub, never cached — a replayed answer is not a call.
    fn model(&self, attempts: usize) -> LM {
        LM::builder("openai/refusing-model")
            .api_base(self.base_url())
            .api_key("stub")
            .cache(false)
            .num_retries(attempts)
            .build()
            .expect("a valid reference")
    }

    fn asks(&self) -> usize {
        self.arrivals.lock().expect("the arrivals").len()
    }

    /// How long the caller waited between one ask and the next.
    fn gap(&self, after: usize) -> Duration {
        let arrivals = self.arrivals.lock().expect("the arrivals");
        assert!(arrivals.len() > after, "there was no ask {}", after + 1);
        arrivals[after] - arrivals[after - 1]
    }

    /// Let the server stop before the listener drops, so a later case binds a clean port. The
    /// connect fails harmlessly when every refusal was spent and the thread has already returned.
    fn done(self) {
        self.stopping.store(true, Ordering::SeqCst);
        drop(TcpStream::connect(
            self.address.trim_start_matches("http://"),
        ));
        let _ = self.served.join();
    }
}

fn drain_request(stream: &mut TcpStream) {
    let mut reader = BufReader::new(stream);
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse().unwrap_or_default();
        }
    }
    let mut body = vec![0u8; length];
    let _ = reader.read_exact(&mut body);
}

fn write_refusal(
    stream: &mut TcpStream,
    status: u16,
    retry_after: Option<&str>,
) -> std::io::Result<()> {
    let body = json!({ "error": { "message": "slow down", "code": "over_limit" } }).to_string();
    let header = retry_after.map_or_else(String::new, |after| format!("retry-after: {after}\r\n"));
    write!(
        stream,
        "HTTP/1.1 {status} Refused\r\ncontent-type: application/json\r\n{header}content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

fn write_reply(stream: &mut TcpStream) -> std::io::Result<()> {
    let body = json!({
        "choices": [{ "message": { "content": "[[ ## answer ## ]]\nParis" } }]
    })
    .to_string();
    write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

fn asking() -> api::LmRequest {
    api::LmRequest {
        messages: vec![api::LmMessage::user(vec![api::LmPart::text("hello")])],
        ..api::LmRequest::default()
    }
}

/// Two refusals then an answer, with dspy's default budget of three asks: the caller never sees the
/// failure. This is the whole point of the story — a program moving from Python keeps its retry.
#[tokio::test]
async fn two_rate_limits_are_ridden_out_within_dspys_default_budget() {
    let stub = Refusing::new(2, 429, None);
    let answered = stub
        .model(3)
        .forward(&asking())
        .await
        .expect("the third ask answers");

    assert!(answered.first_text().contains("Paris"));
    assert_eq!(stub.asks(), 3, "three asks, which is two retries");
    assert!(
        stub.gap(1) >= FIRST_BACKOFF,
        "the first backoff is tenacity's 1s, not {:?}",
        stub.gap(1)
    );
    stub.done();
}

/// The budget is a budget: three refusals outlast three asks, and the caller gets the last failure
/// with the kind intact rather than a wrapper.
#[tokio::test]
async fn a_provider_that_never_recovers_hands_back_the_failure() {
    let stub = Refusing::new(9, 429, None);
    // Two asks, so the exhaustion case costs one backoff rather than three.
    let error = stub
        .model(2)
        .forward(&asking())
        .await
        .expect_err("every ask was refused");

    let failed = error.downcast_ref::<LmFailure>().expect("an LmFailure");
    assert_eq!(failed.kind, LmErrorKind::RateLimit);
    assert_eq!(failed.status, Some(429));
    assert_eq!(failed.provider_code.as_deref(), Some("over_limit"));
    assert_eq!(stub.asks(), 2, "the budget was two asks");
    stub.done();
}

/// A rejected key is asked exactly once. Without this the case above would pass for a policy that
/// retried everything — which is what litellm does, and the part of upstream dspy 3.3's own
/// `_RETRYABLE_LM_ERRORS` says not to reproduce.
#[tokio::test]
async fn a_rejected_key_is_never_asked_twice() {
    let stub = Refusing::new(9, 401, None);
    let error = stub
        .model(3)
        .forward(&asking())
        .await
        .expect_err("the key is rejected");

    assert_eq!(
        error.downcast_ref::<LmFailure>().map(|failed| failed.kind),
        Some(LmErrorKind::Auth)
    );
    assert_eq!(stub.asks(), 1, "auth fails the same way twice");
    stub.done();
}

/// A server error is asked again *immediately* — litellm downgrades anything that is not a rate
/// limit to `constant_retry`, whose tenacity default is no wait. A gap near the backoff would mean
/// the curve was applied to every kind alike.
#[tokio::test]
async fn a_server_error_is_asked_again_without_waiting() {
    let stub = Refusing::new(1, 503, None);
    let answered = stub
        .model(3)
        .forward(&asking())
        .await
        .expect("the second ask answers");

    assert!(answered.first_text().contains("Paris"));
    assert_eq!(stub.asks(), 2);
    assert!(
        stub.gap(1) < FIRST_BACKOFF,
        "a 5xx takes constant_retry's no wait, not {:?}",
        stub.gap(1)
    );
    stub.done();
}

/// A provider that named its own delay is obeyed, and the header replaces the curve rather than
/// being added to it. Asserted as an inequality against the 1s first step, so the case does not
/// depend on the scheduler being punctual.
#[tokio::test]
async fn a_retry_after_header_replaces_the_curve() {
    let stub = Refusing::new(1, 429, Some("0.25"));
    let answered = stub
        .model(3)
        .forward(&asking())
        .await
        .expect("the second ask answers");

    assert!(answered.first_text().contains("Paris"));
    let waited = stub.gap(1);
    assert!(
        waited >= Duration::from_millis(200),
        "the header asked for 0.25s, and it waited {waited:?}"
    );
    assert!(
        waited < FIRST_BACKOFF,
        "the header replaces the 1s curve rather than adding to it: {waited:?}"
    );
    stub.done();
}

/// `num_retries(1)` never asks twice — dspy's `LM(num_retries=1)`, and what a test that measures one
/// call needs.
#[tokio::test]
async fn one_attempt_never_asks_twice() {
    let stub = Refusing::new(9, 429, None);
    let error = stub
        .model(1)
        .forward(&asking())
        .await
        .expect_err("the one ask was refused");

    assert_eq!(
        error.downcast_ref::<LmFailure>().map(|failed| failed.kind),
        Some(LmErrorKind::RateLimit)
    );
    assert_eq!(stub.asks(), 1);
    stub.done();
}

/// The retry reaches a module and not only a bare `LM`, which is the only path a program uses. A
/// `Predict` over a refusing provider answers, and the refusal never surfaces as a parse failure.
#[tokio::test]
async fn a_module_rides_out_a_rate_limit_too() {
    let stub = Refusing::new(1, 429, Some("0.1"));
    let qa = Predict!("question -> answer").set_lm(Arc::new(stub.model(3)));

    let out = call!(qa, question = "capital of France?")
        .await
        .expect("the second ask answers");

    assert_eq!(
        out.get("answer").and_then(|answer| answer.as_str()),
        Some("Paris")
    );
    assert_eq!(stub.asks(), 2);
    stub.done();
}

/// The default is dspy's, so a caller who sets nothing gets what upstream's caller gets.
#[test]
fn the_default_budget_is_dspys_num_retries() {
    assert_eq!(dsrust::lm::retry::DEFAULT_ATTEMPTS, 3);
    assert_eq!(
        LM::new("openai/gpt-4o-mini")
            .expect("a valid reference")
            .retry
            .attempts,
        3
    );
}

/// A counted asker, so the retry can be driven without a socket: `num_retries` is on `LM`, and a
/// caller's own `ChatModel` is not retried by this crate at all — it owns its own transport.
#[tokio::test]
async fn a_callers_own_model_is_not_retried_behind_its_back() {
    struct Counting(AtomicUsize);
    impl ChatModel for Counting {
        async fn forward(&self, _request: &api::LmRequest) -> anyhow::Result<api::LmResponse> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::Error::new(LmFailure::from_status(429, "slow down")))
        }
    }

    let counting = Counting(AtomicUsize::new(0));
    let error = counting.forward(&asking()).await.expect_err("it refuses");
    assert_eq!(
        error.downcast_ref::<LmFailure>().map(|failed| failed.kind),
        Some(LmErrorKind::RateLimit)
    );
    assert_eq!(
        counting.0.load(Ordering::SeqCst),
        1,
        "the retry belongs to LM, where dspy's num_retries lives"
    );
}
