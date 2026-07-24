//! What `use_native_function_calling` changes about the request that actually goes out.
//!
//! dspy's rule has four parts — the adapter asks for it, the signature declares tools coming in
//! and `ToolCalls` going out, and the model can take a tool list — and only when all four hold do
//! the tools move from the prompt onto the request. A flag that reached the adapter and stopped
//! there would leave every one of these assertions passing on the prompt side and failing here,
//! which is the point of asserting on the request rather than on the rendering.

use std::sync::{Arc, Mutex};

use dsrust::adapter::{ChatAdapter, Input};
use dsrust::example::Example;
use dsrust::module::Module;
use dsrust::lm::api::{self, LmToolSpec};
use dsrust::lm::{Capabilities, ChatModel, DynChatModel};
use dsrust::predict::Predict;
use dsrust::signature::{FieldKind, InField, JsonType, OutField, Signature};
use serde_json::{Value, json};

/// A model that answers nothing useful and keeps the request it was handed.
struct Recorder {
    capabilities: Capabilities,
    seen: Mutex<Vec<api::LmRequest>>,
}

impl Recorder {
    fn able(function_calling: bool) -> Arc<Self> {
        Arc::new(Self {
            capabilities: Capabilities { function_calling, ..Default::default() },
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
        Ok(api::LmResponse::text("[[ ## tool_calls ## ]]\n{\"tool_calls\": []}"))
    }

    fn capabilities<'a>(
        &'a self,
        _http: &'a reqwest::Client,
    ) -> impl std::future::Future<Output = Capabilities> + Send + 'a {
        std::future::ready(self.capabilities)
    }
}

fn tool_signature() -> Signature {
    Signature {
        instructions: "Answer the question.".into(),
        inputs: vec![
            InField { name: "question".into(), ..Default::default() },
            InField {
                name: "tools".into(),
                kind: FieldKind::Json(JsonType {
                    annotation: "list[Tool]".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
        ],
        outputs: vec![OutField {
            name: "tool_calls".into(),
            kind: FieldKind::Json(JsonType {
                annotation: "ToolCalls".into(),
                ..Default::default()
            }),
            ..Default::default()
        }],
    }
}

fn search_tool() -> Value {
    json!([{
        "name": "search",
        "desc": "look something up",
        "args": { "query": { "type": "string" } },
    }])
}

/// Run one call through a predictor built on `adapter`, against a model of the given ability.
async fn ask(adapter: ChatAdapter, model: Arc<Recorder>) -> api::LmRequest {
    let predict = Predict::from_signature(tool_signature())
        .with_adapter(adapter)
        .with_lm(model.clone() as Arc<dyn DynChatModel>);
    // The reply is deliberately thin, so a parse failure surfaces here rather than as a
    // confusing assertion later. What is under test is the request.
    let _ = predict.forward(asked_example()).await;
    model.request()
}

/// The inputs every call in this file is made with.
fn asked_example() -> Example {
    Example::new([("question", json!("what is dspy")), ("tools", search_tool())])
}

fn prompt_of(request: &api::LmRequest) -> String {
    serde_json::to_string(&request.messages).expect("messages serialize")
}

#[tokio::test]
async fn the_tools_ride_on_the_request_and_leave_the_prompt() {
    let adapter = ChatAdapter::default().with_native_function_calling(true);
    let request = ask(adapter, Recorder::able(true)).await;

    assert_eq!(request.tools.len(), 1, "the request carries the tool list");
    assert_eq!(request.tools[0].name, "search");
    assert_eq!(request.tools[0].description.as_deref(), Some("look something up"));
    assert_eq!(
        Value::Object(request.tools[0].parameters.clone()),
        json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"],
        })
    );
    // Both fields left the signature, so neither is announced or filled in the conversation.
    let prompt = prompt_of(&request);
    assert!(!prompt.contains("tool_calls"), "the output field is still rendered: {prompt}");
    assert!(!prompt.contains("look something up"), "the tools are still in the prompt: {prompt}");
}

#[tokio::test]
async fn a_model_that_cannot_call_tools_gets_them_in_the_prompt() {
    let adapter = ChatAdapter::default().with_native_function_calling(true);
    let request = ask(adapter, Recorder::able(false)).await;

    assert!(request.tools.is_empty(), "tools went to a model that cannot take them");
    let prompt = prompt_of(&request);
    assert!(prompt.contains("tool_calls"), "the output field should still be asked for");
    assert!(prompt.contains("look something up"), "the tools should still be rendered");
}

#[tokio::test]
async fn an_adapter_that_does_not_ask_natively_renders_them_however_able_the_model_is() {
    // ChatAdapter's own default, and the reason the flag has to be read from the adapter rather
    // than inferred from the model.
    let request = ask(ChatAdapter::default(), Recorder::able(true)).await;

    assert!(request.tools.is_empty());
    assert!(prompt_of(&request).contains("look something up"));
}

/// dspy sets `parallel_tool_calls` only where the adapter states one; the normalized config
/// spells it `tool_choice.parallel`.
#[tokio::test]
async fn parallel_tool_calls_reaches_the_request_only_when_stated() {
    let asked = ChatAdapter::default()
        .with_native_function_calling(true)
        .with_parallel_tool_calls(Some(false));
    let request = ask(asked, Recorder::able(true)).await;
    assert_eq!(request.config.tool_choice.expect("a tool choice").parallel, Some(false));

    // Unset is not the same as `Some(false)`: upstream leaves the provider option alone.
    let silent = ChatAdapter::default().with_native_function_calling(true);
    let request = ask(silent, Recorder::able(true)).await;
    assert!(request.config.tool_choice.is_none_or(|choice| choice.parallel.is_none()));
}

/// A signature asking for calls it offers no tools for is refused before any request goes out —
/// upstream raises rather than send something whose output field cannot be filled.
#[tokio::test]
async fn asking_for_calls_without_offering_tools_never_reaches_the_model() {
    let model = Recorder::able(true);
    let mut signature = tool_signature();
    signature.inputs.retain(|field| field.name != "tools");
    let predict = Predict::from_signature(signature)
        .with_adapter(ChatAdapter::default().with_native_function_calling(true))
        .with_lm(model.clone() as Arc<dyn DynChatModel>);

    let without_tools = Example::new([("question", json!("what is dspy"))]);
    let refused = predict.forward(without_tools).await.expect_err("refused");
    assert!(
        refused.to_string().contains("did not provide any tools as the input"),
        "{refused}"
    );
    assert!(model.seen.lock().expect("not poisoned").is_empty(), "a request went out anyway");
}

/// The spec is upstream's `format_as_litellm_function_call`, and every argument is required
/// because upstream says so rather than because the schema does.
#[test]
fn a_tool_with_no_arguments_still_states_an_object() {
    let signature = tool_signature();
    let inputs = [Input::new("tools", json!([{ "name": "finish", "desc": "stop" }]))];
    let planned = dsrust::adapter::native_tools::plan(
        &signature,
        &inputs,
        Capabilities { function_calling: true, ..Default::default() },
    )
    .expect("plans")
    .expect("native");
    let expected: LmToolSpec = serde_json::from_value(json!({
        "type": "function",
        "name": "finish",
        "description": "stop",
        "parameters": { "type": "object", "properties": {}, "required": [] },
    }))
    .expect("a spec");
    assert_eq!(planned.tools, vec![expected]);
}
