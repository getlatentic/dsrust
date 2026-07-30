//! The provider timeout, over a real socket, on every route that can be pointed at one.
//!
//! A knob that is stored but never reaches the wire looks identical to one that works, so nothing
//! here asserts the field: a server that answers *slowly* stands in for a loaded model, and each
//! case asks whether the call was abandoned or waited for. Both directions are checked — a bound
//! under the delay must give up, and a bound over it must get the answer — because a route that
//! quietly dropped its `.timeout()` would pass the first check on its own.
//!
//! Anthropic is not reachable here — it addresses a hard-coded host, so there is no way to put a
//! slow server in front of it — and its two call sites are the only ones these cases do not cover.
//! OpenRouter is covered despite the same fixed host, because it and the OpenAI-compatible route
//! are one `Endpoint` reading one field.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::JoinHandle;
use std::time::Duration;

use dsrust::DEFAULT_PROVIDER_TIMEOUT;
use dsrust::lm::{ChatModel, LM, api};
use futures_util::StreamExt;
use serde_json::json;

/// Long enough that a bound under it must fire, short enough to keep the suite quick.
const DELAY: Duration = Duration::from_millis(600);
const TOO_SHORT: Duration = Duration::from_millis(150);
const LONG_ENOUGH: Duration = Duration::from_secs(10);

/// A server that waits, then answers — one loaded model, in the only way that matters here.
struct Slow {
    address: String,
    served: JoinHandle<()>,
}

impl Slow {
    fn answering(body: String) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let address = format!("http://{}", listener.local_addr().expect("a bound address"));
        let served = std::thread::spawn(move || {
            // A client that gave up still opened the connection, so the accept succeeds either
            // way and the write is allowed to fail on a socket that is already gone.
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            drain_request(&mut stream);
            std::thread::sleep(DELAY);
            let _ = write_response(&mut stream, &body);
        });
        Self { address, served }
    }

    fn base_url(&self) -> String {
        format!("{}/v1", self.address)
    }

    /// Let the server finish before the listener drops, so a later case binds a clean port.
    fn done(self) {
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

fn write_response(stream: &mut TcpStream, body: &str) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

fn asking() -> api::LmRequest {
    api::LmRequest {
        messages: vec![api::LmMessage::user(vec![api::LmPart::text("hello")])],
        ..api::LmRequest::default()
    }
}

/// What each route answers with, so a call that beats the bound has something to parse.
fn ollama_chat_reply() -> String {
    json!({ "message": { "content": "the reply" }, "done": true }).to_string()
}

fn ollama_generate_reply() -> String {
    json!({ "response": "the reply", "done": true }).to_string()
}

fn openai_reply() -> String {
    json!({ "choices": [{ "message": { "content": "the reply" } }] }).to_string()
}

/// The Responses wire is its own envelope, so the raised-bound case has a reply it can parse
/// rather than a chat-completions body it would reject for the wrong reason.
fn openai_responses_reply() -> String {
    json!({
        "output": [
            { "type": "message", "content": [{ "type": "output_text", "text": "the reply" }] }
        ]
    })
    .to_string()
}

/// The four routes a slow server can be put in front of, each built against one.
fn routes(stub: &Slow) -> Vec<(&'static str, LM)> {
    vec![
        (
            "ollama_chat",
            LM::new("ollama_chat/slow-model")
                .expect("a valid reference")
                .with_ollama_host(&stub.address),
        ),
        (
            "ollama",
            LM::new("ollama/slow-model")
                .expect("a valid reference")
                .with_ollama_host(&stub.address),
        ),
        (
            "openai chat",
            LM::new("openai/slow-model")
                .expect("a valid reference")
                .with_openai_base_url(stub.base_url())
                .with_openai_key("stub"),
        ),
        (
            "openai responses",
            LM::new("openai/slow-model")
                .expect("a valid reference")
                .with_openai_base_url(stub.base_url())
                .with_openai_key("stub")
                .with_openai_responses_api(),
        ),
    ]
}

fn reply_for(route: &str) -> String {
    match route {
        "ollama_chat" => ollama_chat_reply(),
        "ollama" => ollama_generate_reply(),
        "openai responses" => openai_responses_reply(),
        _ => openai_reply(),
    }
}

/// A model slower than the bound is abandoned, on every route the bound has to reach.
#[tokio::test]
async fn a_short_timeout_abandons_every_slow_route() {
    let http = reqwest::Client::new();
    for route in ["ollama_chat", "ollama", "openai chat", "openai responses"] {
        let stub = Slow::answering(reply_for(route));
        let lm = routes(&stub)
            .into_iter()
            .find(|(name, _)| *name == route)
            .expect("the route")
            .1
            .with_cache(false)
            .with_timeout(TOO_SHORT);

        let error = lm
            .forward(&http, &asking())
            .await
            .expect_err("the bound fires");
        assert!(
            error
                .downcast_ref::<dsrust::lm::LmFailure>()
                .is_some_and(|failed| failed.kind == dsrust::lm::LmErrorKind::Timeout),
            "{route} failed for another reason: {error:#}"
        );
        stub.done();
    }
}

/// The same server, the same delay, a bound above it — so the failure above is the bound and not
/// the stub being unreachable.
#[tokio::test]
async fn a_raised_timeout_waits_for_every_slow_route() {
    let http = reqwest::Client::new();
    for route in ["ollama_chat", "ollama", "openai chat", "openai responses"] {
        let stub = Slow::answering(reply_for(route));
        let lm = routes(&stub)
            .into_iter()
            .find(|(name, _)| *name == route)
            .expect("the route")
            .1
            .with_cache(false)
            .with_timeout(LONG_ENOUGH);

        let answered = lm
            .forward(&http, &asking())
            .await
            .unwrap_or_else(|error| panic!("{route} should have waited: {error:#}"));
        assert_eq!(answered.first_text(), "the reply", "the reply for {route}");
        stub.done();
    }
}

/// The streaming path carries the bound too — it is a second set of call sites, and one that
/// dropped it would leave a stalled stream hanging for the whole request.
#[tokio::test]
async fn the_bound_reaches_the_streaming_routes() {
    let http = reqwest::Client::new();
    for route in ["ollama_chat", "ollama", "openai chat"] {
        let stub = Slow::answering(reply_for(route));
        let lm = routes(&stub)
            .into_iter()
            .find(|(name, _)| *name == route)
            .expect("the route")
            .1
            .with_timeout(TOO_SHORT);

        let request = asking();
        let mut events = lm.forward_stream(&http, &request);
        let first = events.next().await.expect("an event");
        assert!(first.is_err(), "{route} streamed past its bound: {first:?}");
        stub.done();
    }
}

/// The capability probe is its own request to the same host, so it carries the bound as well.
#[tokio::test]
async fn the_bound_reaches_the_capability_probe() {
    let stub = Slow::answering(json!({ "template": "{{ if .Tools }}{{ end }}" }).to_string());
    let lm = LM::new("ollama_chat/a-model-litellm-never-heard-of:timeout")
        .expect("a valid reference")
        .with_ollama_host(&stub.address)
        .with_timeout(TOO_SHORT);

    // A probe that gave up reports nothing rather than raising, which is how an unreachable
    // server has always been treated — what is asserted is that it gave up inside the bound.
    let started = std::time::Instant::now();
    let found = lm.capabilities(&reqwest::Client::new()).await;
    assert!(started.elapsed() < DELAY, "the probe waited past its bound");
    assert!(
        !found.function_calling,
        "a probe that timed out offers nothing"
    );
    stub.done();
}

/// A caller who sets none gets what dspy's caller gets: litellm's `request_timeout`, which dspy
/// never overrides. The old default of twenty seconds was this crate's own idea of responsive, and
/// it failed a local model's first call — a divergence a program hits before it hits anything else.
#[test]
fn the_default_is_the_one_litellm_applies() {
    assert_eq!(DEFAULT_PROVIDER_TIMEOUT, Duration::from_secs(6000));
    assert_eq!(
        LM::new("openai/gpt-4o-mini")
            .expect("a valid reference")
            .timeout,
        DEFAULT_PROVIDER_TIMEOUT
    );
}

/// The builder is a copy like every other `LM` builder, so raising it does not disturb the rest.
#[test]
fn raising_the_bound_leaves_the_rest_of_the_model_alone() {
    let lm = LM::new("ollama_chat/qwen2.5:7b-instruct")
        .expect("a valid reference")
        .with_ollama_host("http://elsewhere:11434")
        .with_timeout(Duration::from_secs(120));
    assert_eq!(lm.timeout, Duration::from_secs(120));
    assert_eq!(lm.ollama_host, "http://elsewhere:11434");
    assert!(lm.cache, "the cache setting is untouched");
}
