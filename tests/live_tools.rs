//! A live, end-to-end tool conversation against a real provider.
//!
//! Ignored by default — it needs a running model. ollama is the zero-config default (a local
//! tool-capable model); point it elsewhere with `LIVE_LM`, and for OpenRouter pass a key too:
//!
//! ```text
//! cargo test --test live_tools -- --ignored --nocapture
//! LIVE_LM=openrouter/openai/gpt-4o-mini OPENROUTER_API_KEY=… cargo test --test live_tools -- --ignored --nocapture
//! ```
//!
//! It drives the whole loop this crate is responsible for on a real model: send a tool, read the
//! model's tool call back out of the response, replay that call and a result as a multi-turn
//! conversation, and read the final answer. Passing means request tools, response tool-call
//! parsing, and the assistant/tool message rendering all agree with a provider that is not a fixture.

use dsrs::lm::api::{self, LmMessage, LmPart, LmToolSpec};
use dsrs::{ChatModel, LM};

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

/// One turn: the messages so far, the tool on offer, and whatever the provider answers.
async fn turn(
    lm: &LM,
    http: &reqwest::Client,
    model: &str,
    messages: Vec<LmMessage>,
    tool: &LmToolSpec,
) -> api::LmResponse {
    let request = api::LmRequest::new(model, messages).with_tools(vec![tool.clone()]);
    lm.forward(http, &request).await.expect("the provider answered")
}

#[tokio::test]
#[ignore = "talks to a live provider; runs a local ollama model by default"]
async fn a_tool_conversation_runs_end_to_end() {
    let model = std::env::var("LIVE_LM").unwrap_or_else(|_| "ollama/qwen2.5:7b-instruct".to_owned());
    let lm = LM::new(&model).expect("a valid model ref").without_cache();
    let http = reqwest::Client::new();
    let tool = weather_tool();

    // Turn 1 — ask, with the tool available. The model should answer with a call, which the
    // provider's response parsing surfaces as a `ToolCall` part.
    let ask = LmMessage::user(vec![LmPart::text(
        "What is the weather in Paris right now? Use the get_weather tool to find out.",
    )]);
    let first = turn(&lm, &http, &model, vec![ask.clone()], &tool).await;
    let output = first.outputs.first().expect("one output");

    let call = output
        .parts
        .iter()
        .find(|part| matches!(part, LmPart::ToolCall { .. }))
        .unwrap_or_else(|| panic!("expected a tool call from {model}, got: {:?}", output.parts));
    let LmPart::ToolCall { id, name, args, .. } = call else { unreachable!() };
    assert_eq!(name, "get_weather", "the model called the tool it was offered");
    assert!(args.contains_key("city"), "the call carried its arguments, got: {args:?}");
    println!("turn 1 → tool call: {name}({})", serde_json::Value::Object(args.clone()));

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
    let answer = turn(&lm, &http, &model, conversation, &tool).await.first_text();
    println!("turn 2 → answer: {answer}");
    assert!(
        !answer.trim().is_empty(),
        "the model gave a final answer once it had the tool result"
    );
}
