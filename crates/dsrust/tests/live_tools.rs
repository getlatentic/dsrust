//! Live, end-to-end checks against a real provider — ignored by default, since they need one running.
//!
//! One knob per axis, so the same tests reach every OpenAI-shaped or native provider this crate has:
//!
//! - `LIVE_LM` — the model ref (`ollama/…`, `openrouter/…`, `openai/…`). Defaults to a local ollama.
//! - `LIVE_BASE_URL` — an OpenAI-compatible base url, for a local `llama-server` (LlamaBarn), vLLM,
//!   LM Studio, etc.; pair it with `openai/<model>` and any `OPENAI_API_KEY` the server ignores.
//! - `LIVE_PROVIDER` — an OpenRouter upstream slug to pin, e.g. `amazon-bedrock`, forcing the request
//!   through that provider (this is OpenRouter's `provider: {only, allow_fallbacks:false}` routing).
//!
//! ```text
//! cargo test --test live_tools -- --ignored --nocapture
//! LIVE_LM=openrouter/amazon/nova-lite-v1 LIVE_PROVIDER=amazon-bedrock OPENROUTER_API_KEY=… \
//!   cargo test --test live_tools a_tool_conversation -- --ignored --nocapture
//! LIVE_LM=openai/<model> LIVE_BASE_URL=http://localhost:8080/v1 OPENAI_API_KEY=x \
//!   cargo test --test live_tools a_text_round_trip -- --ignored --nocapture
//! ```

use dsrust::lm::api::{self, LmConfig, LmMessage, LmPart, LmToolSpec};
use dsrust::{ChatModel, LM};

/// The model, provider config, and cache-off LM the env asks for.
fn live_setup() -> (LM, String, LmConfig) {
    let model =
        std::env::var("LIVE_LM").unwrap_or_else(|_| "ollama_chat/qwen2.5:7b-instruct".to_owned());
    let mut lm = LM::new(&model)
        .expect("a valid model ref")
        .with_cache(false);
    if let Ok(base_url) = std::env::var("LIVE_BASE_URL") {
        lm = lm.with_openai_base_url(base_url);
    }
    // Pinning an OpenRouter upstream rides through as the `provider` field, which the OpenAI-shaped
    // request builder passes straight onto the wire from config extensions.
    let config = match std::env::var("LIVE_PROVIDER") {
        Ok(provider) => LmConfig::from_kwargs([(
            "provider".to_owned(),
            serde_json::json!({ "only": [provider], "allow_fallbacks": false }),
        )])
        .expect("a valid provider routing config"),
        Err(_) => LmConfig::default(),
    };
    (lm, model, config)
}

/// The one tool the conversation offers.
fn weather_tool() -> LmToolSpec {
    let parameters = serde_json::from_value(serde_json::json!({
        "type": "object",
        "properties": { "city": { "type": "string", "description": "The city to look up" } },
        "required": ["city"],
    }))
    .expect("a valid schema");
    LmToolSpec::new("get_weather", parameters).described("Look up the current weather for a city")
}

/// One turn: the messages so far, the tool on offer, whatever the provider answers.
async fn turn(
    lm: &LM,
    http: &reqwest::Client,
    model: &str,
    messages: Vec<LmMessage>,
    tool: &LmToolSpec,
    config: &LmConfig,
) -> api::LmResponse {
    let request = api::LmRequest::new(model, messages)
        .with_tools(vec![tool.clone()])
        .configured(config.clone());
    lm.forward(http, &request)
        .await
        .expect("the provider answered")
}

/// The plainest round trip — a question in, some text out — so any OpenAI-compatible server (a local
/// `llama-server`, a small model that does not do tools) can prove it reaches the wire and back.
#[tokio::test]
#[ignore = "talks to a live provider"]
async fn a_text_round_trip_runs_end_to_end() {
    let (lm, model, config) = live_setup();
    let http = reqwest::Client::new();
    let request = api::LmRequest::new(
        &model,
        vec![LmMessage::user(vec![LmPart::text(
            "Reply with a one-sentence greeting.",
        )])],
    )
    .configured(config);
    let answer = lm
        .forward(&http, &request)
        .await
        .expect("the provider answered")
        .first_text();
    println!("[{model}] text round trip → {answer}");
    assert!(!answer.trim().is_empty(), "the provider returned some text");
}

/// The whole tool loop this crate owns, against a real model: send a tool, read the call out of the
/// response, replay the call and a result as a multi-turn conversation, read the final answer.
#[tokio::test]
#[ignore = "talks to a live provider; runs a local ollama model by default"]
async fn a_tool_conversation_runs_end_to_end() {
    let (lm, model, config) = live_setup();
    let http = reqwest::Client::new();
    let tool = weather_tool();

    // Turn 1 — ask, with the tool available. The model should answer with a call, which the
    // provider's response parsing surfaces as a `ToolCall` part.
    let ask = LmMessage::user(vec![LmPart::text(
        "What is the weather in Paris right now? Use the get_weather tool to find out.",
    )]);
    let first = turn(&lm, &http, &model, vec![ask.clone()], &tool, &config).await;
    let output = first.outputs.first().expect("one output");

    let call = output
        .parts
        .iter()
        .find(|part| matches!(part, LmPart::ToolCall { .. }))
        .unwrap_or_else(|| panic!("expected a tool call from {model}, got: {:?}", output.parts));
    let LmPart::ToolCall { id, name, args, .. } = call else {
        unreachable!()
    };
    assert_eq!(
        name, "get_weather",
        "the model called the tool it was offered"
    );
    assert!(
        args.contains_key("city"),
        "the call carried its arguments, got: {args:?}"
    );
    println!(
        "[{model}] turn 1 → tool call: {name}({})",
        serde_json::Value::Object(args.clone())
    );

    // Turn 2 — replay the assistant's call and a tool result, and let the model answer. This is the
    // multi-turn rendering under test: an assistant `ToolCall` part and a `ToolResult` message.
    let result = LmPart::ToolResult {
        call_id: id.clone(),
        name: Some("get_weather".to_owned()),
        content: vec![LmPart::text("It is sunny and 22°C in Paris.")],
        is_error: false,
        provider_data: api::Metadata::new(),
        metadata: api::Metadata::new(),
    };
    let conversation = vec![
        ask,
        LmMessage::assistant(vec![call.clone()]),
        LmMessage::new("tool", vec![result]),
    ];
    let answer = turn(&lm, &http, &model, conversation, &tool, &config)
        .await
        .first_text();
    println!("[{model}] turn 2 → answer: {answer}");
    assert!(
        !answer.trim().is_empty(),
        "the model gave a final answer once it had the tool result"
    );
}
