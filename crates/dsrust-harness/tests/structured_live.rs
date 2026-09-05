//! Structured output through the bridge, on real backends.
//!
//! dsrust's `JsonAdapter` puts its output schema on the request as
//! `response_format`; the bridge hands that to the run as `output_schema`. What
//! happens next differs by adapter, and the two tests here are the two cases:
//! Claude Code ignores the schema and answers in prose JSON that dsrust parses;
//! the OpenAI-compatible runtime sends it to the provider as a constraint.
//!
//! ```text
//! cargo test -p dsrust-harness --test structured_live -- --ignored
//! cargo test -p dsrust-harness --features openai-compatible --test structured_live -- --ignored
//! ```

use std::sync::Arc;

use dsrust::lm::DynChatModel;
use dsrust::{Example, JsonAdapter, Module, Predict};
use dsrust_harness::HarnessModel;
use serde_json::{Value, json};

const TICKET: &str = "My order arrived with the box crushed and the mug inside in three pieces. \
    I need a replacement before the 14th, it's a birthday present.";

fn workspace() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("dsrust-harness-structured");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

async fn triage(model: Arc<dyn DynChatModel>) -> dsrust::Prediction {
    Predict::parse("ticket -> category, urgency")
        .expect("a signature")
        .adapter(JsonAdapter::default())
        .set_lm(model)
        .forward(Example::new([("ticket", json!(TICKET))]))
        .await
        .expect("the call completes")
}

fn both_fields(prediction: &dsrust::Prediction) -> (String, String) {
    let field = |name: &str| {
        prediction
            .get(name)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("`{name}` is present and a string: {prediction:?}"))
            .to_owned()
    };
    (field("category"), field("urgency"))
}

/// Claude Code's adapter ignores `output_schema` (a preference it cannot express),
/// so this is dsrust's JSON adapter reading prose JSON out of an agent's reply.
#[tokio::test]
#[ignore = "live: needs the claude CLI installed and signed in; costs tokens"]
async fn the_json_adapter_over_claude_code_fills_every_typed_field() {
    use dsrust_harness::harness::Claude;
    let model = Arc::new(
        HarnessModel::new(Claude::new())
            .with_cwd(workspace())
            .with_max_turns(2),
    );
    let (category, urgency) = both_fields(&triage(model).await);
    assert!(
        !category.is_empty() && !urgency.is_empty(),
        "{category:?} / {urgency:?}"
    );
}

/// The OpenAI-compatible runtime sends the schema as `response_format`, so the
/// provider is constrained to it — the tighter of the two cases.
#[cfg(feature = "openai-compatible")]
#[tokio::test]
#[ignore = "live: needs a running Ollama with gpt-oss:20b pulled"]
async fn the_json_adapter_over_ollama_gets_a_schema_constrained_reply() {
    use dsrust_harness::harness::OpenHarness;
    let model = Arc::new(
        HarnessModel::new(OpenHarness::ollama().with_context_tokens(32_000))
            .with_model("gpt-oss:20b")
            .with_cwd(workspace())
            .with_max_turns(2),
    );
    let (category, urgency) = both_fields(&triage(model).await);
    assert!(
        !category.is_empty() && !urgency.is_empty(),
        "{category:?} / {urgency:?}"
    );
}

/// The distinction the `Predict` tests above cannot draw: a schema *enforced*
/// hands back exactly the JSON, while a schema *ignored* hands back prose that a
/// forgiving parser happens to rescue. This reads the reply's text directly.
#[tokio::test]
#[ignore = "live: needs the claude CLI installed and signed in; costs tokens"]
async fn claude_codes_reply_text_is_exactly_the_json_when_a_schema_is_asked_for() {
    use dsrust::lm::ChatModel;
    use dsrust::lm::api::{LmMessage, LmRequest};
    use dsrust_harness::harness::Claude;

    let model = HarnessModel::new(Claude::new())
        .with_cwd(workspace())
        .with_max_turns(2);
    let mut request = LmRequest::from_messages(
        "sonnet",
        vec![LmMessage::user([format!("Triage this ticket: {TICKET}")])],
    );
    request.config.response_format = Some(json!({
        "type": "object",
        "properties": { "category": { "type": "string" }, "urgency": { "type": "string" } },
        "required": ["category", "urgency"]
    }));
    let reply = model.forward(&request).await.expect("the call completes");
    let text = reply.outputs[0]
        .parts
        .iter()
        .find_map(|p| p.as_text())
        .expect("a text part");
    let value: Value = serde_json::from_str(text).unwrap_or_else(|e| {
        panic!("the reply is exactly the JSON, not prose around it ({e}): {text:?}")
    });
    assert!(
        value["category"].is_string() && value["urgency"].is_string(),
        "{value}"
    );
}
