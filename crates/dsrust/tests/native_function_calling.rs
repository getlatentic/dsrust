//! What `use_native_function_calling` changes about the request that actually goes out.
//!
//! dspy's rule has four parts — the adapter asks for it, the signature declares tools coming in
//! and `ToolCalls` going out, and the model can take a tool list — and only when all four hold do
//! the tools move from the prompt onto the request. A flag that reached the adapter and stopped
//! there would leave every one of these assertions passing on the prompt side and failing here,
//! which is the point of asserting on the request rather than on the rendering.

use std::sync::{Arc, Mutex};

use dsrust::Adapter;
use dsrust::adapter::{ChatAdapter, Input, ToolCalls};
use dsrust::example::Example;
use dsrust::lm::api::{self, LmToolSpec};
use dsrust::lm::dummy::DummyLM;
use dsrust::lm::{Capabilities, ChatModel, DynChatModel};
use dsrust::module::Module;
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
            capabilities: Capabilities {
                function_calling,
                ..Default::default()
            },
            seen: Mutex::new(Vec::new()),
        })
    }

    fn request(&self) -> api::LmRequest {
        self.seen
            .lock()
            .expect("not poisoned")
            .first()
            .cloned()
            .expect("a request went out")
    }
}

impl ChatModel for Recorder {
    async fn forward(&self, request: &api::LmRequest) -> anyhow::Result<api::LmResponse> {
        self.seen
            .lock()
            .expect("not poisoned")
            .push(request.clone());
        Ok(api::LmResponse::text(
            "[[ ## tool_calls ## ]]\n{\"tool_calls\": []}",
        ))
    }

    fn capabilities(&self) -> impl std::future::Future<Output = Capabilities> + Send {
        std::future::ready(self.capabilities)
    }
}

fn tool_signature() -> Signature {
    Signature {
        instructions: "Answer the question.".into(),
        inputs: vec![
            InField {
                name: "question".into(),
                ..Default::default()
            },
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
        .adapter(adapter)
        .set_lm(model.clone() as Arc<dyn DynChatModel>);
    // The reply is deliberately thin, so a parse failure surfaces here rather than as a
    // confusing assertion later. What is under test is the request.
    let _ = predict.forward(asked_example()).await;
    model.request()
}

/// The inputs every call in this file is made with.
fn asked_example() -> Example {
    Example::new([
        ("question", json!("what is dspy")),
        ("tools", search_tool()),
    ])
}

fn prompt_of(request: &api::LmRequest) -> String {
    serde_json::to_string(&request.messages).expect("messages serialize")
}

#[tokio::test]
async fn the_tools_ride_on_the_request_and_leave_the_prompt() {
    let adapter = ChatAdapter::default().use_native_function_calling(true);
    let request = ask(adapter, Recorder::able(true)).await;

    assert_eq!(request.tools.len(), 1, "the request carries the tool list");
    assert_eq!(request.tools[0].name, "search");
    assert_eq!(
        request.tools[0].description.as_deref(),
        Some("look something up")
    );
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
    assert!(
        !prompt.contains("tool_calls"),
        "the output field is still rendered: {prompt}"
    );
    assert!(
        !prompt.contains("look something up"),
        "the tools are still in the prompt: {prompt}"
    );
}

#[tokio::test]
async fn a_model_that_cannot_call_tools_gets_them_in_the_prompt() {
    let adapter = ChatAdapter::default().use_native_function_calling(true);
    let request = ask(adapter, Recorder::able(false)).await;

    assert!(
        request.tools.is_empty(),
        "tools went to a model that cannot take them"
    );
    let prompt = prompt_of(&request);
    assert!(
        prompt.contains("tool_calls"),
        "the output field should still be asked for"
    );
    assert!(
        prompt.contains("look something up"),
        "the tools should still be rendered"
    );
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
        .use_native_function_calling(true)
        .parallel_tool_calls(Some(false));
    let request = ask(asked, Recorder::able(true)).await;
    assert_eq!(
        request.config.tool_choice.expect("a tool choice").parallel,
        Some(false)
    );

    // Unset is not the same as `Some(false)`: upstream leaves the provider option alone.
    let silent = ChatAdapter::default().use_native_function_calling(true);
    let request = ask(silent, Recorder::able(true)).await;
    assert!(
        request
            .config
            .tool_choice
            .is_none_or(|choice| choice.parallel.is_none())
    );
}

/// A signature asking for calls it offers no tools for is refused before any request goes out —
/// upstream raises rather than send something whose output field cannot be filled.
#[tokio::test]
async fn asking_for_calls_without_offering_tools_never_reaches_the_model() {
    let model = Recorder::able(true);
    let mut signature = tool_signature();
    signature.inputs.retain(|field| field.name != "tools");
    let predict = Predict::from_signature(signature)
        .adapter(ChatAdapter::default().use_native_function_calling(true))
        .set_lm(model.clone() as Arc<dyn DynChatModel>);

    let without_tools = Example::new([("question", json!("what is dspy"))]);
    let refused = predict.forward(without_tools).await.expect_err("refused");
    assert!(
        refused
            .to_string()
            .contains("did not provide any tools as the input"),
        "{refused}"
    );
    assert!(
        model.seen.lock().expect("not poisoned").is_empty(),
        "a request went out anyway"
    );
}

/// A model that answers with a native tool call of its own — content and calls beside each other,
/// the way a provider replies once it has been handed a tool list.
struct NativeReplier(api::LmResponse);

impl ChatModel for NativeReplier {
    async fn forward(&self, _request: &api::LmRequest) -> anyhow::Result<api::LmResponse> {
        Ok(self.0.clone())
    }

    fn capabilities(&self) -> impl std::future::Future<Output = Capabilities> + Send {
        std::future::ready(Capabilities {
            function_calling: true,
            ..Default::default()
        })
    }
}

fn call_part(id: &str, name: &str, args: Value) -> api::LmPart {
    api::LmPart::ToolCall {
        id: Some(id.to_owned()),
        name: name.to_owned(),
        args: args.as_object().expect("an object").clone(),
        provider_data: Default::default(),
        metadata: Default::default(),
    }
}

/// dspy `_call_postprocess`: a native reply's calls fill the tool-call output field, and the
/// provider's ids survive onto it — they are what later pairs each result to the call it answers.
#[tokio::test]
async fn a_native_reply_fills_the_output_field_with_the_providers_ids() {
    let reply = api::LmResponse {
        outputs: vec![api::LmOutput {
            parts: vec![call_part(
                "call_provider_1",
                "search",
                json!({ "query": "cats" }),
            )],
            finish_reason: Some("tool_calls".into()),
            ..Default::default()
        }],
        ..Default::default()
    };
    let predict = Predict::from_signature(tool_signature())
        .adapter(ChatAdapter::default().use_native_function_calling(true))
        .set_lm(Arc::new(NativeReplier(reply)) as Arc<dyn DynChatModel>);

    let prediction = predict
        .forward(asked_example())
        .await
        .expect("a native reply parses");
    let calls: ToolCalls = serde_json::from_value(
        prediction
            .get("tool_calls")
            .cloned()
            .expect("the output field was filled"),
    )
    .expect("a ToolCalls value");

    assert_eq!(calls.tool_calls.len(), 1);
    assert_eq!(calls.tool_calls[0].id.as_deref(), Some("call_provider_1"));
    assert_eq!(calls.tool_calls[0].name, "search");
    assert_eq!(
        calls.tool_calls[0].args,
        *json!({ "query": "cats" }).as_object().unwrap()
    );
}

/// Parallel native calls come back on the one reply, and both keep their ids and order.
#[tokio::test]
async fn parallel_native_calls_all_reach_the_output_field() {
    let reply = api::LmResponse {
        outputs: vec![api::LmOutput {
            parts: vec![
                call_part("call_provider_1", "search", json!({ "query": "cats" })),
                call_part("call_provider_2", "search", json!({ "query": "dogs" })),
            ],
            finish_reason: Some("tool_calls".into()),
            ..Default::default()
        }],
        ..Default::default()
    };
    let predict = Predict::from_signature(tool_signature())
        .adapter(ChatAdapter::default().use_native_function_calling(true))
        .set_lm(Arc::new(NativeReplier(reply)) as Arc<dyn DynChatModel>);

    let prediction = predict
        .forward(asked_example())
        .await
        .expect("a native reply parses");
    let calls: ToolCalls =
        serde_json::from_value(prediction.get("tool_calls").cloned().expect("filled"))
            .expect("a ToolCalls");
    let ids: Vec<&str> = calls
        .tool_calls
        .iter()
        .filter_map(|c| c.id.as_deref())
        .collect();
    assert_eq!(ids, ["call_provider_1", "call_provider_2"]);
}

/// The spec is upstream's `format_as_litellm_function_call`, and every argument is required
/// because upstream says so rather than because the schema does.
#[test]
fn a_tool_with_no_arguments_still_states_an_object() {
    let signature = tool_signature();
    let inputs = [Input::new(
        "tools",
        json!([{ "name": "finish", "desc": "stop" }]),
    )];
    let planned = dsrust::adapter::native_tools::plan(
        &signature,
        &inputs,
        Capabilities {
            function_calling: true,
            ..Default::default()
        },
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

/// A replayed tool result reaches the model as a `tool` message, and a scripted model records it
/// as one.
///
/// Both halves matter and neither is covered elsewhere. `adapter/history.rs` builds the tool turn
/// only on the native path, which nothing tested; and `DummyLM` used to re-derive its record from
/// the request with `match role { "assistant" => Assistant, _ => User }`, so this arrived as
/// something the user said. A test asserting on the render alone would still pass with the
/// recorder wrong, and one asserting on the recorder alone would pass with no tool turn rendered.
#[tokio::test]
async fn a_replayed_tool_result_is_a_tool_message_and_is_recorded_as_one() {
    let mut signature = tool_signature();
    signature.inputs.push(InField {
        name: "history".into(),
        kind: FieldKind::Json(JsonType {
            annotation: "History".into(),
            ..Default::default()
        }),
        ..Default::default()
    });

    let history = json!({
        "messages": [{
            "question": "weather in Paris?",
            "tool_calls": {
                "tool_calls": [{ "id": "call_1", "name": "get_weather", "args": { "city": "Paris" } }],
                "tool_call_results": {
                    "tool_call_results": [{
                        "call_id": "call_1",
                        "name": "get_weather",
                        "value": "17C and clear",
                        "is_error": false,
                    }],
                },
            },
        }],
    });

    let adapter = ChatAdapter {
        use_native_function_calling: true,
        ..ChatAdapter::default()
    };
    let rendered = adapter
        .format(
            &signature,
            &[],
            &[
                Input::new("question", json!("and tomorrow?")),
                Input::new("history", history),
            ],
        )
        .expect("the history renders");

    let roles: Vec<String> = rendered.iter().map(|m| m.role.clone()).collect();
    assert!(
        roles.iter().any(|role| role == "tool"),
        "a replayed tool result is its own message: {roles:?}"
    );

    let lm = DummyLM::new([dsrust::example! { tool_calls: "{\"tool_calls\": []}" }]);
    lm.forward(&api::LmRequest::from_messages("dummy", rendered))
        .await
        .expect("the scripted model answers");

    let seen = lm.asked();
    let recorded: Vec<String> = seen[0].messages.iter().map(|m| m.role.clone()).collect();
    assert_eq!(
        recorded, roles,
        "a scripted model records the roles it was sent"
    );
}
