//! Every spelling opens the module point, and every tool opens the tool point.
//!
//! dspy decorates `Module.__call__` in `__init_subclass__`, so every module upstream is watched
//! whatever it is. Here each spelling reaches the point by a different route — a derive, a macro,
//! a wrapper around another module — and a route that missed it would be a module invisible to a
//! trace and to every callback a caller registered.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use dsrust::callback::{CallId, Callback, configure_callbacks};
use dsrust::lm::DynChatModel;
use dsrust::signature::SignatureSpec;
use dsrust::{
    ChainOfThought, DummyLM, Example, Forward, Module, Predict, Prediction, ReActV2, Signature,
    Tool, call, configure_model, example, tool,
};
use serde_json::{Value, json};

#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

#[tool]
/// Look one term up.
fn lookup(query: String) -> anyhow::Result<String> {
    Ok(format!("found {query}"))
}

#[derive(Module)]
#[task(QA)]
struct Mine {
    only: Predict,
}

impl Forward for Mine {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        self.only.forward(inputs).await
    }
}

#[derive(Default)]
struct Recorder {
    seen: Mutex<Vec<String>>,
}

impl Callback for Recorder {
    fn on_module_start(&self, _call: &CallId, module: &str, _inputs: &Example) {
        self.seen
            .lock()
            .expect("not poisoned")
            .push(format!("module:{module}"));
    }

    fn on_tool_start(&self, _call: &CallId, tool: &str, _args: &Value) {
        self.seen
            .lock()
            .expect("not poisoned")
            .push(format!("tool:{tool}"));
    }

    fn on_tool_end(&self, _call: &CallId, _answered: Result<&Value, &anyhow::Error>) {
        self.seen
            .lock()
            .expect("not poisoned")
            .push("tool:end".to_owned());
    }
}

fn turn(calls: Value) -> Example {
    Example::new([("next_thought", json!("go")), ("tool_calls", calls)])
}

#[tokio::test]
async fn every_spelling_and_every_tool_is_watched() {
    let recorder = Arc::new(Recorder::default());
    configure_model(
        reqwest::Client::new(),
        Arc::new(DummyLM::new([
            example! { answer: "Paris" },
            example! { reasoning: "because", answer: "Paris" },
            turn(json!({ "tool_calls": [{ "name": "lookup", "args": { "query": "France" } }] })),
            turn(json!({ "tool_calls": [{ "name": "submit", "args": { "answer": "Paris" } }] })),
            example! { answer: "Paris" },
            example! { answer: "Paris" },
        ])) as Arc<dyn DynChatModel>,
    );
    configure_callbacks([recorder.clone() as Arc<dyn Callback>]);

    call!(Predict!(QA), question = "q").await.expect("runs");
    call!(ChainOfThought!(QA), question = "q")
        .await
        .expect("runs");
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(Lookup)];
    call!(ReActV2!(QA, tools), question = "q")
        .await
        .expect("runs");
    call!(
        Mine {
            only: Predict::from_signature(QA::signature())
        },
        question = "q"
    )
    .await
    .expect("runs");
    call!(Predict!("question -> answer"), question = "q")
        .await
        .expect("runs");
    configure_callbacks([]);

    let seen = recorder.seen.lock().expect("not poisoned").clone();
    assert_eq!(
        seen,
        [
            // `Predict!(QA)` and `ChainOfThought!(QA)` answer through the typed path, which built
            // the caller's struct straight from the validated reply and threw the rest away — so
            // neither opened a point, and both were invisible to every callback.
            "module:Predict",
            "module:ChainOfThought",
            // The agent, and each tool it called, start and end.
            "module:ReActV2",
            "tool:lookup",
            "tool:end",
            "tool:submit",
            "tool:end",
            // A module of the caller's own, then the `Predict` inside it.
            "module:Mine",
            "module:Predict",
            // The one spelling with no task, which was watched all along.
            "module:Predict",
        ]
    );
}
