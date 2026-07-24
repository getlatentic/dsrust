//! What a `Reasoning` output field changes about a request and its reply.
//!
//! dspy's `Reasoning.adapt_to_native_lm_feature`: a reasoning model with a `Reasoning` field reasons
//! on its own channel, so the field leaves the render and the request carries a `reasoning_effort`
//! instead. `parse_lm_response` then reads the reply's thinking back into that field. A model that
//! cannot reason renders the field as prose, unchanged.

use std::sync::{Arc, Mutex};

use dsrust::lm::api::{self};
use dsrust::lm::{Capabilities, ChatModel, DynChatModel};
use dsrust::module::Module;
use dsrust::predict::Predict;
use dsrust::signature::{FieldKind, OutField, Signature};
use serde_json::{Value, json};

/// A model of a stated ability that answers with a fixed reply and keeps the requests it saw.
struct Recorder {
    capabilities: Capabilities,
    reply: api::LmResponse,
    seen: Mutex<Vec<api::LmRequest>>,
}

impl Recorder {
    fn new(reasoning: bool, reply: api::LmResponse) -> Arc<Self> {
        Arc::new(Self {
            capabilities: Capabilities { reasoning, ..Default::default() },
            reply,
            seen: Mutex::new(Vec::new()),
        })
    }

    fn request(&self) -> api::LmRequest {
        self.seen.lock().expect("not poisoned").first().cloned().expect("a request went out")
    }
}

impl ChatModel for Recorder {
    async fn forward(
        &self,
        _http: &reqwest::Client,
        request: &api::LmRequest,
    ) -> anyhow::Result<api::LmResponse> {
        self.seen.lock().expect("not poisoned").push(request.clone());
        Ok(self.reply.clone())
    }

    fn capabilities<'a>(
        &'a self,
        _http: &'a reqwest::Client,
    ) -> impl std::future::Future<Output = Capabilities> + Send + 'a {
        std::future::ready(self.capabilities)
    }
}

/// `reasoning: Reasoning` then `answer`, dspy's `ChainOfThought` shape.
fn reasoning_signature() -> Signature {
    let mut signature = Signature::single_input(
        "Answer the question.",
        vec![OutField { name: "answer".into(), ..Default::default() }],
    );
    signature.outputs.insert(
        0,
        OutField { name: "reasoning".into(), kind: FieldKind::Reasoning, ..Default::default() },
    );
    signature
}

/// A reply carrying its reasoning on the thinking channel, and the answer as prose.
fn reply_with_thinking(thinking: &str) -> api::LmResponse {
    api::LmResponse {
        outputs: vec![api::LmOutput {
            parts: vec![
                api::LmPart::Thinking {
                    text: thinking.to_owned(),
                    redacted: false,
                    metadata: Default::default(),
                },
                api::LmPart::text("[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]"),
            ],
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn asked() -> dsrust::example::Example {
    dsrust::example::Example::new([("question", json!("what is six times seven"))])
}

/// A reasoning model: the request carries `reasoning_effort: "low"` and the reasoning field never
/// reaches the prompt — it is asked for on the model's own channel, not as a rendered block.
#[tokio::test]
async fn a_reasoning_model_carries_the_effort_and_drops_the_field_from_the_prompt() {
    let model = Recorder::new(true, reply_with_thinking("six sevens are forty-two"));
    let predict = Predict::from_signature(reasoning_signature())
        .with_lm(model.clone() as Arc<dyn DynChatModel>);
    let _ = predict.forward(asked()).await;

    let request = model.request();
    assert_eq!(
        request.config.reasoning.and_then(|r| r.effort).as_deref(),
        Some("low"),
        "the default effort rides on the request"
    );
    let prompt = serde_json::to_string(&request.messages).expect("messages serialize");
    assert!(!prompt.contains("[[ ## reasoning ## ]]"), "the field is not rendered: {prompt}");
}

/// dspy `parse_lm_response`: the reply's thinking fills the reasoning field, and the answer is read
/// from the prose beside it.
#[tokio::test]
async fn the_reply_thinking_fills_the_reasoning_field() {
    let model = Recorder::new(true, reply_with_thinking("six sevens are forty-two"));
    let predict = Predict::from_signature(reasoning_signature())
        .with_lm(model as Arc<dyn DynChatModel>);
    let prediction = predict.forward(asked()).await.expect("a reasoning reply parses");

    assert_eq!(
        prediction.get("reasoning").and_then(Value::as_str),
        Some("six sevens are forty-two")
    );
    assert_eq!(prediction.get("answer").and_then(Value::as_str), Some("42"));
}

/// A model that cannot reason renders the field as prose and carries no effort — the field stays in
/// the signature and the reply is read the ordinary way.
#[tokio::test]
async fn a_model_that_cannot_reason_renders_the_field() {
    let reply = api::LmResponse::text(
        "[[ ## reasoning ## ]]\nsix sevens are forty-two\n\n[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]",
    );
    let model = Recorder::new(false, reply);
    let predict = Predict::from_signature(reasoning_signature())
        .with_lm(model.clone() as Arc<dyn DynChatModel>);
    let prediction = predict.forward(asked()).await.expect("a prose reply parses");

    assert!(model.request().config.reasoning.is_none(), "no effort for a non-reasoning model");
    let prompt = serde_json::to_string(&model.request().messages).expect("serializes");
    assert!(prompt.contains("[[ ## reasoning ## ]]"), "the field is rendered as prose");
    assert_eq!(prediction.get("answer").and_then(Value::as_str), Some("42"));
    assert_eq!(
        prediction.get("reasoning").and_then(Value::as_str),
        Some("six sevens are forty-two")
    );
}
