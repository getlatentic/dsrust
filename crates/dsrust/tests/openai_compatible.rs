//! The OpenAI-compatible provider over a real socket.
//!
//! A one-shot server on the loopback interface stands in for OpenAI, Groq, vLLM or LM Studio,
//! so the route, the credential header and the request body are asserted as they leave the
//! process rather than as the builder imagined them. The stub is std's `TcpListener`: no
//! network, and no mocking crate to add.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::JoinHandle;

use dsrust::lm::{ChatModel, JsonFormat, LM, OutputMode, TokenLimitRule, api};
use serde_json::{Value, json};

const REPLY: &str = r#"{"choices":[{"message":{"content":"the reply"}}]}"#;

/// What the provider actually received.
struct Request {
    path: String,
    headers: Vec<(String, String)>,
    body: Value,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        find_header(&self.headers, name)
    }
}

/// A server that answers exactly one request and hands it back for inspection.
struct Stub {
    base_url: String,
    served: JoinHandle<Request>,
}

impl Stub {
    fn answering(status: u16, body: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let base_url = format!(
            "http://{}/v1",
            listener.local_addr().expect("a bound address")
        );
        let body = body.to_owned();
        let served = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("one connection");
            let request = read_request(&mut stream);
            write_response(&mut stream, status, &body);
            request
        });
        Self { base_url, served }
    }

    /// Blocks until the exchange is over, which it is once the call under test has returned.
    fn received(self) -> Request {
        self.served.join().expect("the stub thread finished")
    }
}

fn read_request(stream: &mut TcpStream) -> Request {
    let mut reader = BufReader::new(stream);
    let head = read_head(&mut reader);
    let path = head[0]
        .split_whitespace()
        .nth(1)
        .expect("a request target")
        .to_owned();
    let headers: Vec<(String, String)> = head[1..]
        .iter()
        .filter_map(|line| split_header(line))
        .collect();
    let mut body = vec![0; content_length(&headers)];
    reader.read_exact(&mut body).expect("the whole body");
    Request {
        path,
        headers,
        body: serde_json::from_slice(&body).expect("a JSON body"),
    }
}

/// The request line and its headers, up to the blank line that ends the head.
fn read_head(reader: &mut impl BufRead) -> Vec<String> {
    let mut lines = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).expect("a header line");
        let line = line.trim_end().to_owned();
        if line.is_empty() {
            return lines;
        }
        lines.push(line);
    }
}

fn split_header(line: &str) -> Option<(String, String)> {
    let (name, value) = line.split_once(':')?;
    Some((name.to_ascii_lowercase(), value.trim().to_owned()))
}

fn find_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

fn content_length(headers: &[(String, String)]) -> usize {
    find_header(headers, "content-length")
        .and_then(|value| value.parse().ok())
        .unwrap_or_default()
}

fn write_response(stream: &mut TcpStream, status: u16, body: &str) {
    let reason = match status {
        200..=299 => "OK",
        _ => "Error",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .expect("the reply goes out");
    stream.flush().expect("the reply is flushed");
}

/// The probe every case here asks through: one ask, uncached, so what the stub answered is what is
/// asserted.
///
/// Caching is off because the store is process-wide and keyed on the request, and the base URL is
/// not part of a request — so two tests asking the same model the same thing collide however
/// different their stubs are. The second would be replayed, its stub would wait forever for a
/// connection, and the test would hang rather than fail.
///
/// Asking once rather than three times is what makes a refusal readable. Every stub here serves a
/// fixed number of connections, so a retried 429 would arrive at a closed listener and be reported
/// as the transport failure that followed it rather than the rate limit that caused it.
/// `tests/lm_retry.rs` is where the retry itself is held.
fn probe_lm_for(stub: &Stub, model: &str) -> LM {
    caching_lm_for(stub, model)
        .cache(false)
        .retry(dsrust::lm::Retry::once())
}

/// The same probe with the cache left on, for the one test that is about the cache.
fn caching_lm_for(stub: &Stub, model: &str) -> LM {
    LM::new(model)
        .expect("valid model ref")
        .openai_api_key("sk-test")
        .openai_base_url(stub.base_url.as_str())
}

fn probe_lm(stub: &Stub) -> LM {
    probe_lm_for(stub, "openai/gpt-4o-mini")
}

/// The typed request a `be helpful` / `hi` exchange builds — the same messages an adapter would
/// render, so what the provider serializes is what the wire has always carried.
fn probe_request(mode: &OutputMode<'_>) -> api::LmRequest {
    let mut request = api::LmRequest::new(
        "",
        vec![
            api::LmMessage::system(vec![api::LmPart::text("be helpful")]),
            api::LmMessage::user(vec![api::LmPart::text("hi")]),
        ],
    );
    if let OutputMode::Json { schema } = mode {
        request.config.response_format = Some((*schema).clone());
    }
    request
}

async fn ask(lm: &LM, mode: &OutputMode<'_>) -> anyhow::Result<api::LmResponse> {
    lm.forward(&probe_request(mode)).await
}

/// The same, naming a token cap — what the key-routing tests need, now that a bare call sends
/// none.
async fn ask_capped(lm: &LM, mode: &OutputMode<'_>, cap: u32) -> anyhow::Result<api::LmResponse> {
    let mut request = probe_request(mode);
    request.config.max_tokens = Some(cap);
    lm.forward(&request).await
}

fn probe_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false,
    })
}

#[tokio::test]
async fn a_call_reaches_chat_completions_under_the_configured_base_url() {
    let stub = Stub::answering(200, REPLY);
    let reply = ask(&probe_lm(&stub), &OutputMode::Text)
        .await
        .expect("the stub answers");
    assert_eq!(reply.first_text(), "the reply");

    let request = stub.received();
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(request.header("authorization"), Some("Bearer sk-test"));
    assert_eq!(request.body["model"], "gpt-4o-mini");
    assert_eq!(request.body["messages"][0]["role"], "system");
    assert_eq!(request.body["messages"][0]["content"], "be helpful");
    assert_eq!(request.body["messages"][1]["content"], "hi");
    assert_eq!(request.body.get("response_format"), None);
}

#[tokio::test]
async fn a_base_url_with_a_trailing_slash_reaches_the_same_route() {
    let stub = Stub::answering(200, REPLY);
    let lm = LM::new("openai/gpt-4o-mini")
        .expect("valid model ref")
        .openai_api_key("sk-test")
        .openai_base_url(format!("{}/", stub.base_url))
        .cache(false);
    ask(&lm, &OutputMode::Text).await.expect("the stub answers");
    assert_eq!(stub.received().path, "/v1/chat/completions");
}

#[tokio::test]
async fn json_mode_asks_for_an_object_by_default() {
    let stub = Stub::answering(200, REPLY);
    let schema = probe_schema();
    ask(&probe_lm(&stub), &OutputMode::Json { schema: &schema })
        .await
        .expect("the stub answers");
    assert_eq!(
        stub.received().body["response_format"],
        json!({ "type": "json_object" })
    );
}

#[tokio::test]
async fn the_schema_envelope_is_opt_in_and_carries_the_schema() {
    let stub = Stub::answering(200, REPLY);
    let lm = probe_lm(&stub).openai_json_format(JsonFormat::Schema);
    let schema = probe_schema();
    ask(&lm, &OutputMode::Json { schema: &schema })
        .await
        .expect("the stub answers");

    let request = stub.received();
    let format = &request.body["response_format"];
    assert_eq!(format["type"], "json_schema");
    assert_eq!(format["json_schema"]["strict"], true);
    assert_eq!(format["json_schema"]["schema"], schema);
}

/// Streaming: a `stream:true` request, the Server-Sent Events read back as the typed vocabulary.
/// The stub sends the whole SSE body at once, which is all a socket test can do; what this pins
/// is the request flag and the chunk→event→text mapping.
#[tokio::test]
async fn forward_stream_reads_sse_into_typed_events() {
    use dsrust::lm::api::{LmDelta, LmStreamEvent};
    use futures_util::StreamExt;

    let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Par\"}}]}\n\n\
               data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"is\"}}]}\n\n\
               data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
               data: [DONE]\n\n";
    let stub = Stub::answering(200, sse);
    let lm = probe_lm(&stub);
    let request = probe_request(&OutputMode::Text);

    let http = reqwest::Client::new();
    let events: Vec<LmStreamEvent> = lm
        .forward_stream_on(&http, &request)
        .map(|event| event.expect("a valid event"))
        .collect()
        .await;

    assert_eq!(
        stub.received().body["stream"],
        json!(true),
        "the request asked to stream"
    );
    assert!(
        matches!(events.first(), Some(LmStreamEvent::Start { .. })),
        "the stream opens with Start"
    );
    let streamed: String = events
        .iter()
        .filter_map(|event| match event {
            LmStreamEvent::Delta {
                delta: LmDelta::TextDelta { text },
                ..
            } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(streamed, "Paris", "the deltas reassemble into the reply");
    assert!(
        matches!(events.last(), Some(LmStreamEvent::End { .. })),
        "and closes with End"
    );
}

/// dspy 3.3 sends no `max_tokens` when the caller named none — a bare chat call carries a cap
/// under neither key, where this crate once defaulted 1024.
#[tokio::test]
async fn a_bare_call_sends_no_token_cap_on_the_wire() {
    let stub = Stub::answering(200, REPLY);
    ask(&probe_lm(&stub), &OutputMode::Text)
        .await
        .expect("the stub answers");

    let request = stub.received();
    assert_eq!(request.body.get("max_tokens"), None);
    assert_eq!(request.body.get("max_completion_tokens"), None);
}

/// OpenAI's reasoning models reject `max_tokens` outright, so a cap for one of them has to carry
/// under the other key and nothing under the old one.
#[tokio::test]
async fn a_reasoning_model_caps_completion_tokens_on_the_wire() {
    let stub = Stub::answering(200, REPLY);
    ask_capped(&probe_lm_for(&stub, "openai/o3"), &OutputMode::Text, 1024)
        .await
        .expect("the stub answers");

    let request = stub.received();
    assert_eq!(request.body["max_completion_tokens"], 1024);
    assert_eq!(request.body.get("max_tokens"), None);
}

#[tokio::test]
async fn a_chat_model_caps_max_tokens_on_the_wire() {
    let stub = Stub::answering(200, REPLY);
    ask_capped(&probe_lm(&stub), &OutputMode::Text, 1024)
        .await
        .expect("the stub answers");

    let request = stub.received();
    assert_eq!(request.body["max_tokens"], 1024);
    assert_eq!(request.body.get("max_completion_tokens"), None);
}

/// `gpt-5-chat` is the plain chat model of the family; only its reasoning siblings moved.
#[tokio::test]
async fn the_gpt_5_chat_line_keeps_max_tokens_on_the_wire() {
    let stub = Stub::answering(200, REPLY);
    ask_capped(
        &probe_lm_for(&stub, "openai/gpt-5-chat-latest"),
        &OutputMode::Text,
        1024,
    )
    .await
    .expect("the stub answers");

    let request = stub.received();
    assert_eq!(request.body["max_tokens"], 1024);
    assert_eq!(request.body.get("max_completion_tokens"), None);
}

/// A self-hosted server behind the same wire format need not know the newer field, and
/// saying so has to hold for a model whose name looks like one of OpenAI's.
#[tokio::test]
async fn a_host_pinned_to_max_tokens_sends_it_for_a_reasoning_model_too() {
    let stub = Stub::answering(200, REPLY);
    let lm =
        probe_lm_for(&stub, "openai/o3").openai_token_limit_rule(TokenLimitRule::AlwaysMaxTokens);
    ask_capped(&lm, &OutputMode::Text, 1024)
        .await
        .expect("the stub answers");

    let request = stub.received();
    assert_eq!(request.body["max_tokens"], 1024);
    assert_eq!(request.body.get("max_completion_tokens"), None);
}

/// dspy 3.3 normalizes an LM failure rather than handing back a provider string. The status, the
/// provider and the kind are fields a caller branches on; the rendered line is upstream's
/// `[model] message`.
#[tokio::test]
async fn a_refused_call_arrives_as_a_typed_failure() {
    let stub = Stub::answering(401, r#"{"error":{"message":"Incorrect API key provided"}}"#);
    let error = ask(&probe_lm(&stub), &OutputMode::Text)
        .await
        .expect_err("401 is a failure");

    let failed = error
        .downcast_ref::<dsrust::lm::LmFailure>()
        .unwrap_or_else(|| panic!("a typed LM failure, got: {error:#}"));
    assert_eq!(failed.kind, dsrust::lm::LmErrorKind::Auth);
    assert_eq!(failed.status, Some(401));
    assert_eq!(failed.provider.as_deref(), Some("openai"));
    assert_eq!(failed.message, "Incorrect API key provided");
    assert!(
        !failed.is_retryable(),
        "a rejected key fails the same way twice"
    );
}

/// And a 429 is the one a caller may act on, which is the point of the taxonomy.
#[tokio::test]
async fn a_rate_limit_is_retryable_where_a_rejected_key_is_not() {
    let stub = Stub::answering(
        429,
        r#"{"error":{"message":"Rate limit reached","code":"rate_limit_exceeded"}}"#,
    );
    let error = ask(&probe_lm(&stub), &OutputMode::Text)
        .await
        .expect_err("429 is a failure");

    let failed = error
        .downcast_ref::<dsrust::lm::LmFailure>()
        .unwrap_or_else(|| panic!("a typed LM failure, got: {error:#}"));
    assert_eq!(failed.kind, dsrust::lm::LmErrorKind::RateLimit);
    assert!(failed.is_retryable());
    assert_eq!(failed.provider_code.as_deref(), Some("rate_limit_exceeded"));
}

/// No stub: the call must fail on the missing credential before anything is sent.
#[tokio::test]
async fn a_missing_key_names_the_variable_the_endpoint_was_told_to_read() {
    // Not about the cache, and asking through it would initialise the process-global one before
    // the cache test can point it at a scratch directory — which is a race, since the two run
    // on different threads.
    let lm = LM::new("openai/llama-3.3-70b")
        .expect("valid model ref")
        .openai_key_var("DSRS_TEST_KEY_THAT_IS_NOT_SET")
        .cache(false);
    let error = ask(&lm, &OutputMode::Text)
        .await
        .expect_err("no credential is configured");
    assert!(
        error
            .to_string()
            .contains("DSRS_TEST_KEY_THAT_IS_NOT_SET is not set"),
        "got: {error}"
    );
}

/// dspy's `LM(cache=True)`: a repeated request is replayed rather than bought again. The stub
/// answers exactly one connection, so a second call that reached the wire would hang here.
#[tokio::test]
async fn an_identical_request_is_replayed_rather_than_sent_again() {
    // The shared cache is backed by a directory that outlives the process, so without this the
    // entry written by the *last* `cargo test` replays and the first call here is already a hit.
    // Pointing it at a scratch path also keeps a test run from writing into the developer's own
    // cache, which is a 30 GB directory nobody asked us to fill.
    //
    // Every other test in this binary asks with the cache off, so nothing has initialised
    // the shared cache yet and this is the value it takes.
    let scratch = std::env::temp_dir().join(format!(
        "dsrust-openai-compatible-cache-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&scratch);
    // SAFETY: set before this binary's first use of the cache, and read only through it.
    unsafe { std::env::set_var("DSRS_CACHEDIR", &scratch) };
    assert_eq!(
        dsrust::lm::cache::shared()
            .disk()
            .expect("a disk layer")
            .root(),
        scratch,
        "the shared cache was initialised before this test could redirect it"
    );

    let stub = Stub::answering(200, REPLY);
    // A model name no other test uses, so this owns its entry in the shared cache.
    let lm = caching_lm_for(&stub, "openai/cache-probe-model");

    let first = ask(&lm, &OutputMode::Text).await.expect("the stub answers");
    assert!(!first.cache_hit, "the first call is a real one");

    let second = ask(&lm, &OutputMode::Text).await.expect("replayed");
    assert!(second.cache_hit, "the second never reached the provider");
    assert_eq!(second.first_text(), "the reply");
    assert_eq!(stub.received().path, "/v1/chat/completions");
}

/// A request that never reaches a server is `transport`, and retryable — dspy classifies the same
/// failure as `LMTransportError` and `is_retryable_lm_error` says yes.
///
/// Compared live against the pinned dspy, both pointed at a closed port: dspy answered
/// `{"code": "transport", "retryable": true}` and this crate answered an untyped `anyhow` until
/// the send site was given a kind.
#[tokio::test]
async fn a_refused_connection_is_a_retryable_transport_failure() {
    let lm = dsrust::lm::LM::new("openai/gemma")
        .expect("a model id")
        // Port 9 is discard: nothing listens, so the connect fails before any HTTP happens.
        .openai_base_url("http://127.0.0.1:9/v1")
        .openai_api_key("x")
        .cache(false);

    let error = ask(&lm, &OutputMode::Text)
        .await
        .expect_err("nothing is listening");
    let failed = error
        .downcast_ref::<dsrust::lm::LmFailure>()
        .unwrap_or_else(|| panic!("a typed LM failure, got: {error:#}"));
    assert_eq!(failed.kind, dsrust::lm::LmErrorKind::Transport);
    assert!(
        failed.is_retryable(),
        "a refused connection is worth asking again"
    );
    assert_eq!(
        failed.status, None,
        "nothing answered, so there is no status to report"
    );
    assert_eq!(failed.provider.as_deref(), Some("openai"));
}

/// dspy's `dspy.LM(model, temperature=…, max_tokens=…)` keeps those on the instance and merges them
/// beneath every call — `kwargs = {**self.kwargs, **kwargs}`. So must these.
#[tokio::test]
async fn the_models_own_settings_reach_a_call_that_did_not_state_them() {
    let stub = Stub::answering(
        200,
        r#"{"choices":[{"message":{"content":"[[ ## answer ## ]]\nok"}}]}"#,
    );
    let lm = dsrust::lm::LM::builder("openai/gpt-4o-mini")
        .api_base(&stub.base_url)
        .api_key("x")
        .temperature(0.25)
        .max_tokens(321)
        .cache(false)
        .build()
        .expect("a model id");

    ask(&lm, &OutputMode::Text).await.expect("the stub answers");

    let sent = stub.received();
    assert_eq!(sent.body["temperature"], 0.25);
    assert_eq!(sent.body["max_tokens"], 321);
}

/// And a call stating its own wins, which is what makes an LM-wide default overridable.
#[tokio::test]
async fn a_calls_own_setting_overrides_the_models() {
    let stub = Stub::answering(
        200,
        r#"{"choices":[{"message":{"content":"[[ ## answer ## ]]\nok"}}]}"#,
    );
    let lm = dsrust::lm::LM::builder("openai/gpt-4o-mini")
        .api_base(&stub.base_url)
        .api_key("x")
        .temperature(0.25)
        .max_tokens(321)
        .cache(false)
        .build()
        .expect("a model id");

    let mut request = probe_request(&OutputMode::Text);
    request.config.temperature = Some(0.9);
    lm.forward(&request).await.expect("the stub answers");

    let sent = stub.received();
    assert_eq!(sent.body["temperature"], 0.9, "the call's");
    assert_eq!(
        sent.body["max_tokens"], 321,
        "and the model's, unstated by the call"
    );
}

/// The builder's `api_key` follows the model's own prefix.
///
/// Routing it to OpenAI regardless would leave an `anthropic/` model's real credential unset and
/// the call refused, with nothing in the error saying the key had gone to the wrong field.
#[test]
fn the_builders_key_goes_to_the_provider_the_model_names() {
    let anthropic = dsrust::lm::LM::builder("anthropic/claude-sonnet-4-5")
        .api_key("sk-ant-probe")
        .build()
        .expect("a model id");
    assert_eq!(anthropic.anthropic_api_key.as_deref(), Some("sk-ant-probe"));
    assert_ne!(anthropic.openai.api_key.as_deref(), Some("sk-ant-probe"));

    let openai = dsrust::lm::LM::builder("openai/gpt-4o-mini")
        .api_key("sk-openai-probe")
        .build()
        .expect("a model id");
    assert_eq!(openai.openai.api_key.as_deref(), Some("sk-openai-probe"));

    let routed = dsrust::lm::LM::builder("openrouter/openai/gpt-oss-120b")
        .api_key("sk-or-probe")
        .build()
        .expect("a model id");
    assert_eq!(routed.openrouter_api_key.as_deref(), Some("sk-or-probe"));
}

/// The whole audio path at once: samples through `Audio::from_samples`, into a signature that
/// declares the field, rendered by `Predict`, and read off the socket as the provider will.
///
/// Each stage below this has its own oracle — the WAV bytes are pinned to libsndfile's, the
/// adapter render to a golden generated from dspy, the block shape to the wire tests — but nothing
/// proved the stages compose. The strong assertions ride on that: the `input_audio` block's data
/// must be the *same string* the constructor produced (transport changed nothing) and must equal
/// the committed golden's bytes (the constructor produced what libsndfile does).
#[tokio::test]
async fn an_audio_value_rides_the_wire_as_the_block_dspy_sends() {
    use dsrust::adapter::types::Audio;
    use dsrust::{Predict, call};

    let reply = serde_json::to_string(&json!({
        "choices": [{ "message": {
            "content": "[[ ## transcript ## ]]\nwater sounds\n\n[[ ## completed ## ]]"
        } }]
    }))
    .expect("a reply body");
    let stub = Stub::answering(200, &reply);

    // The golden's `a_few_samples` case, so the encoded bytes have a measured answer.
    let audio = Audio::from_samples(&[0.0, 0.5, -0.5, 1.0], 8000);
    let encoded = audio.data.clone();
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/conformance/constants/wav_pcm16.json"),
        )
        .expect("the wav golden is committed"),
    )
    .expect("the golden parses");
    assert_eq!(
        encoded, golden["cases"]["a_few_samples"]["base64"],
        "the constructor writes libsndfile's bytes"
    );

    let qa = Predict!("clip: Audio -> transcript").set_lm(std::sync::Arc::new(probe_lm(&stub)));
    let out = call!(qa, clip = audio).await.expect("the stub answers");
    assert_eq!(
        out.get("transcript").and_then(Value::as_str),
        Some("water sounds")
    );

    let request = stub.received();
    let content = &request.body["messages"][1]["content"];
    let blocks = content.as_array().expect("a media turn renders as blocks");
    assert_eq!(
        blocks[0],
        json!({ "type": "text", "text": "[[ ## clip ## ]]\n" })
    );
    assert_eq!(
        blocks[1],
        json!({ "type": "input_audio", "input_audio": { "data": encoded, "format": "wav" } }),
        "the bytes the constructor made are the bytes on the wire"
    );
    assert_eq!(blocks.len(), 3, "prefix, audio, and the respond-with tail");
}
