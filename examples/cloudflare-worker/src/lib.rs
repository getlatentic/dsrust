use std::sync::Arc;

use dsrust::lm::api::{LmMessage, LmRequest};
use dsrust::lm::{Cached, ChatModel, DynChatModel};
use dsrust::{DummyLM, FnTool, Module, Predict, ReAct, example};
use futures_util::{StreamExt, pin_mut};
use serde::Deserialize;
use serde_json::{Value, json};
use worker::{Context, Env, Request, Response, ResponseBuilder, Result, Router, event};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[allow(dead_code)]
mod compatibility_compile_proof {
    use std::rc::Rc;

    use dsrust::{Example, Forward, Prediction, tool};

    #[derive(dsrust::Module)]
    pub struct WorkerLocalModule {
        #[not_a_step]
        pub local: Rc<()>,
    }

    impl Forward for WorkerLocalModule {
        async fn forward(&self, inputs: Example) -> dsrust::anyhow::Result<Prediction> {
            let local = Rc::clone(&self.local);
            std::future::ready(()).await;
            drop(local);
            Ok(Prediction::new(inputs, "fixture"))
        }
    }

    #[tool]
    /// Prove that an async tool future may remain local to a Worker isolate.
    pub async fn worker_local_tool(value: String) -> dsrust::anyhow::Result<String> {
        let local = Rc::new(value);
        std::future::ready(()).await;
        Ok((*local).clone())
    }
}

#[derive(Deserialize)]
struct Prompt {
    question: String,
}

fn worker_error(error: impl std::fmt::Display) -> worker::Error {
    worker::Error::RustError(error.to_string())
}

async fn predict(mut request: Request) -> Result<Response> {
    let prompt: Prompt = request.json().await?;
    let model = Arc::new(DummyLM::new([example! { answer: "Paris" }])) as Arc<dyn DynChatModel>;
    let program = Predict::parse("question -> answer")
        .map_err(worker_error)?
        .set_lm(model);
    let prediction = program
        .forward(example! { question: prompt.question }.with_inputs(["question"]))
        .await
        .map_err(worker_error)?;
    Response::from_json(&json!({ "answer": prediction.get("answer") }))
}

async fn stream_response(mut request: Request) -> Result<Response> {
    let prompt: Prompt = request.json().await?;
    let chunks = async_stream::stream! {
        let model = DummyLM::new([example! { answer: format!("echo: {}", prompt.question) }]);
        let asked = LmRequest::new("fixture", vec![LmMessage::user([prompt.question])]);
        let events = model.forward_stream(&asked);
        pin_mut!(events);
        while let Some(event) = events.next().await {
            let chunk: Result<Vec<u8>> = event
                .map_err(worker_error)
                .and_then(|event| serde_json::to_string(&event).map_err(worker_error))
                .map(|encoded| format!("data: {encoded}\n\n").into_bytes());
            yield chunk;
        }
    };
    ResponseBuilder::new()
        .with_header("content-type", "text/event-stream; charset=utf-8")?
        .with_header("cache-control", "no-cache")?
        .from_stream(chunks)
}

async fn react(mut request: Request) -> Result<Response> {
    let prompt: Prompt = request.json().await?;
    let tool = Box::new(FnTool::new(
        "lookup",
        "look up a deterministic fixture value",
        json!({ "key": { "type": "string" } }),
        |args: &Value| {
            let key = args.get("key").and_then(Value::as_str).unwrap_or_default();
            Ok(format!("value-for-{key}"))
        },
    ));
    let model = Arc::new(DummyLM::new([
        example! {
            next_thought: "look up the value",
            next_tool_name: "lookup",
            next_tool_args: json!({ "key": "worker" }),
        },
        example! {
            next_thought: "the value is known",
            next_tool_name: "finish",
            next_tool_args: json!({}),
        },
        example! { reasoning: "used lookup", answer: "value-for-worker" },
    ])) as Arc<dyn DynChatModel>;
    let signature = "question -> answer".parse().map_err(worker_error)?;
    let program = ReAct::new(signature, vec![tool]).set_lm(model);
    let prediction = program
        .forward(example! { question: prompt.question }.with_inputs(["question"]))
        .await
        .map_err(worker_error)?;
    Response::from_json(&json!({ "answer": prediction.get("answer") }))
}

async fn cache_check(mut request: Request) -> Result<Response> {
    let prompt: Prompt = request.json().await?;
    let model = Cached::new(DummyLM::new([example! { answer: "cached" }]));
    let asked = LmRequest::new("fixture", vec![LmMessage::user([prompt.question])]);
    let first = model.forward(&asked).await.map_err(worker_error)?;
    let second = model.forward(&asked).await.map_err(worker_error)?;
    Response::from_json(&json!({
        "first_cache_hit": first.cache_hit,
        "second_cache_hit": second.cache_hit,
        "entries": model.len(),
    }))
}

#[event(fetch)]
pub async fn fetch(request: Request, env: Env, _context: Context) -> Result<Response> {
    Router::new()
        .post_async(
            "/predict",
            |request, _| async move { predict(request).await },
        )
        .post_async("/stream", |request, _| async move {
            stream_response(request).await
        })
        .post_async("/react", |request, _| async move { react(request).await })
        .post_async("/cache-check", |request, _| async move {
            cache_check(request).await
        })
        .run(request, env)
        .await
}
