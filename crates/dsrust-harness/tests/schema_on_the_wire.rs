//! What the schema looks like by the time it reaches a provider.
//!
//! dsrust's `JsonAdapter` puts a schema on the request; the bridge hands it to
//! agent-harness as `output_schema`; agent-harness wraps it the way the provider
//! wants. Three hands, and a shape mistake at any one of them — a schema wrapped
//! twice, say — is invisible to a live test, because a capable model answers in
//! JSON whether or not the constraint was well-formed. So this test is the
//! provider: it records the body and answers with a canned completion, and the
//! assertion is on the bytes.
#![cfg(feature = "openai-compatible")]

use std::sync::{Arc, Mutex};
use std::thread;

use dsrust::lm::DynChatModel;
use dsrust::{Example, JsonAdapter, Module, Predict};
use dsrust_harness::HarnessModel;
use dsrust_harness::harness::{ModelChoice, OpenHarness, OpenHarnessConfig};
use serde_json::{Value, json};

/// A `/v1/chat/completions` that remembers what it was asked and answers `reply`.
fn fake_openai(reply: &str) -> (String, Arc<Mutex<Vec<Value>>>) {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("bind");
    let base = format!("http://{}", server.server_addr());
    let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
    let log = Arc::clone(&seen);
    let reply = reply.to_owned();
    thread::spawn(move || {
        while let Ok(mut request) = server.recv() {
            let mut raw = String::new();
            let _ = std::io::Read::read_to_string(request.as_reader(), &mut raw);
            log.lock()
                .unwrap()
                .push(serde_json::from_str(&raw).unwrap_or(Value::Null));
            let frames = [
                json!({ "choices": [{ "delta": { "content": reply } }] }),
                json!({ "choices": [{ "delta": {}, "finish_reason": "stop" }], "usage": { "prompt_tokens": 1, "completion_tokens": 1 } }),
            ];
            let mut body = String::new();
            for frame in frames {
                body.push_str(&format!("data: {frame}\n\n"));
            }
            body.push_str("data: [DONE]\n\n");
            let _ = request.respond(
                tiny_http::Response::from_string(body).with_header(
                    tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..])
                        .unwrap(),
                ),
            );
        }
    });
    (base, seen)
}

#[tokio::test]
async fn the_json_adapters_schema_arrives_wrapped_exactly_once_as_response_format() {
    let (base, seen) = fake_openai(r#"{"category": "shipping", "urgency": "high"}"#);
    let harness = OpenHarness::custom(OpenHarnessConfig {
        id: "stand-in".into(),
        display_name: "Stand-in".into(),
        base_url: base,
        models: vec![ModelChoice {
            value: "test-model".into(),
            label: "Test".into(),
        }],
        ..Default::default()
    })
    // 127.0.0.1 reads as local, and the stand-in answers no `/props` probe.
    .with_context_tokens(32_000);
    let model: Arc<dyn DynChatModel> = Arc::new(
        HarnessModel::new(harness)
            .with_model("test-model")
            .with_cwd(std::env::temp_dir()),
    );

    let out = Predict::parse("ticket -> category, urgency")
        .expect("a signature")
        .adapter(JsonAdapter::default())
        .set_lm(model)
        .forward(Example::new([("ticket", json!("the mug arrived smashed"))]))
        .await
        .expect("the call completes");
    assert_eq!(
        out.get("category").and_then(Value::as_str),
        Some("shipping"),
        "the canned reply was read back"
    );

    let seen = seen.lock().unwrap();
    let body = &seen[0];
    let rf = &body["response_format"];
    assert_eq!(rf["type"], "json_schema", "OpenAI's wrapper: {rf}");
    let schema = &rf["json_schema"]["schema"];
    assert!(
        schema["properties"]["category"].is_object() && schema["properties"]["urgency"].is_object(),
        "the wrapper holds dsrust's schema — its output fields as properties — not another wrapper: {rf}"
    );
    assert!(
        schema.get("json_schema").is_none() && schema.get("type") != Some(&json!("json_schema")),
        "wrapped once, not twice: {rf}"
    );
}
