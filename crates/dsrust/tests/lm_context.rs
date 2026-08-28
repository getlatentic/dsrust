//! `dspy.context(lm=...)`: one piece of work asks another model, and nothing is rebuilt.
//!
//! Upstream's own example is the shape under test — a `Predict` built against the configured model,
//! then a block that points it at a second one:
//!
//! ```python
//! dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))
//! qa = dspy.Predict("question -> answer")
//! with dspy.context(lm=dspy.LM("anthropic/claude-sonnet-4-6")):
//!     result = qa(question="...")   # claude answers
//! # gpt-5-mini again here
//! ```
//!
//! The difference from `set_lm` is that nothing is rebuilt: the program may be one someone else
//! constructed and handed over, five modules deep.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use dsrust::lm::api::{self, LmResponse};
use dsrust::lm::{ChatModel, DynChatModel, global};
use dsrust::signature::Signature;
use dsrust::{Example, Forward, Module, Prediction, example, predict::Predict};
use serde_json::Value;

/// The configured model is process-wide, so these tests take turns.
static SERIAL: Mutex<()> = Mutex::new(());

/// A model that names itself in its answer, so a prediction says which one produced it.
struct Named(&'static str);

impl ChatModel for Named {
    async fn forward(&self, _request: &api::LmRequest) -> Result<LmResponse> {
        Ok(LmResponse::text(format!(
            "[[ ## answer ## ]]\n{}\n\n[[ ## completed ## ]]",
            self.0
        )))
    }
}

fn signature() -> Signature {
    "question -> answer".parse().expect("parses")
}

fn asked() -> Example {
    example! { question: "which model?" }.with_inputs(["question"])
}

fn answered(prediction: &Prediction) -> String {
    prediction
        .get("answer")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// A program built against the configured model asks the scoped one inside the block, and the
/// configured one again outside it — without being rebuilt, which is the whole point.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn a_scope_redirects_a_program_that_was_already_built() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    global::configure_model(
        reqwest::Client::new(),
        Arc::new(Named("configured")) as Arc<dyn DynChatModel>,
    );

    // Built once, against whatever was configured. Never touched again.
    let program = Predict::from_signature(signature());

    let before = program.forward(asked()).await.expect("parses");
    assert_eq!(answered(&before), "configured");

    // A scripted model stands in for `dspy.LM("anthropic/...")`; what is under test is which one a
    // module reaches, not which provider answers.
    let inside = global::context_model(
        reqwest::Client::new(),
        Arc::new(Named("scoped")) as Arc<dyn DynChatModel>,
    )
    .run(program.forward(asked()))
    .await
    .expect("parses");
    assert_eq!(
        answered(&inside),
        "scoped",
        "the scope should win inside it"
    );

    let after = program.forward(asked()).await.expect("parses");
    assert_eq!(
        answered(&after),
        "configured",
        "the scope should not outlive its own work"
    );
}

/// A nested program — a module inside a module — reaches the scoped model too, however deep.
/// `set_lm` would have to be threaded through every predictor to do the same.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn a_scope_reaches_a_nested_program() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    #[derive(Module)]
    struct Outer {
        inner: Predict,
    }

    impl Forward for Outer {
        async fn forward(&self, inputs: Example) -> Result<Prediction> {
            self.inner.forward(inputs).await
        }
    }

    global::configure_model(
        reqwest::Client::new(),
        Arc::new(Named("configured")) as Arc<dyn DynChatModel>,
    );
    let program = Outer {
        inner: Predict::from_signature(signature()),
    };

    let outside = Module::forward(&program, asked()).await.expect("parses");
    assert_eq!(answered(&outside), "configured");

    // The inner predictor was never told about the scope, and reaches it anyway.
    let inside = global::context_model(
        reqwest::Client::new(),
        Arc::new(Named("scoped")) as Arc<dyn DynChatModel>,
    )
    .run(Module::forward(&program, asked()))
    .await
    .expect("parses");
    assert_eq!(answered(&inside), "scoped");
}

/// Two scopes interleaved in one task each keep their own model.
///
/// This is why the scope is a future rather than a guard: a thread-local set once and held across
/// an `.await` would be read by whichever piece of work the runtime polled next, and `Evaluate`
/// runs its rows exactly that way.
#[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
#[tokio::test]
async fn interleaved_scopes_do_not_borrow_each_others_model() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    global::configure_model(
        reqwest::Client::new(),
        Arc::new(Named("configured")) as Arc<dyn DynChatModel>,
    );

    async fn asks_the_current_model() -> String {
        // Two yields, so the runtime is guaranteed to poll the other side in between — once before
        // the model is reached and once after.
        tokio::task::yield_now().await;
        let program = Predict::from_signature(signature());
        let answered = program.forward(asked()).await.expect("parses");
        tokio::task::yield_now().await;
        answered
            .get("answer")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    }

    fn scope(name: &'static str) -> global::Scope {
        global::context_model(
            reqwest::Client::new(),
            Arc::new(Named(name)) as Arc<dyn DynChatModel>,
        )
    }

    // Two *different* scopes, interleaved. A thread-local set once and held would give both sides
    // whichever was installed last.
    let (left, right) = futures_util::future::join(
        scope("left").run(asks_the_current_model()),
        scope("right").run(asks_the_current_model()),
    )
    .await;
    assert_eq!(left, "left");
    assert_eq!(right, "right");

    // And neither outlived its own work.
    assert_eq!(asks_the_current_model().await, "configured");
}
