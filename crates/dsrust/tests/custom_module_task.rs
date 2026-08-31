//! A module of your own answers the way the built-in ones do.
//!
//! `Predict!(QA)`, `ChainOfThought!(QA)` and `ReActV2!(QA, tools)` all hand back the task's own
//! outputs struct. A module written outside this crate reaches the same spelling through
//! `Module::task`, which is what the built-in macros call for you.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use dsrust::lm::DynChatModel;
use dsrust::signature::SignatureSpec;
use dsrust::{
    ChainOfThought, DummyLM, Example, Module, Predict, Prediction, Signature, call,
    configure_model, example,
};

#[derive(Signature)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

/// Two predictors, the second reading the first — the "module of your own" the guide describes.
struct AskTwice {
    first: Predict,
    second: Predict,
}

impl Module for AskTwice {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let draft = self.first.forward(inputs.clone()).await?;
            let mut second = inputs;
            second.set("question", draft.get("answer").cloned().unwrap_or_default());
            self.second.forward(second).await
        })
    }
}

/// One test, not two: `configure_model` installs process-wide and `cargo test` runs a file's tests
/// at once, so a second test would take replies meant for this one.
#[tokio::test]
async fn a_custom_module_answers_with_the_task_too() {
    configure_model(
        reqwest::Client::new(),
        Arc::new(DummyLM::new([
            example! { answer: "France" },
            example! { answer: "Paris" },
            example! { reasoning: "France's capital.", answer: "Paris" },
            example! { answer: "France" },
            example! { answer: "Paris" },
        ])) as Arc<dyn DynChatModel>,
    );
    let mine = || AskTwice {
        first: Predict::from_signature(QA::signature()),
        second: Predict::from_signature(QA::signature()),
    };

    // The line the built-in macros answer with, from a module written here.
    let out = call!(mine().task::<QA>(), question = "capital of France?")
        .await
        .expect("the program runs");
    assert_eq!(out.answer, "Paris");

    let built_in = call!(ChainOfThought!(QA), question = "capital of France?")
        .await
        .expect("the program runs");
    assert_eq!(built_in.answer, out.answer, "the same line, either way");

    // Untouched underneath: the module still answers with a `Prediction`, which is what an
    // optimizer and an evaluator take.
    let prediction = mine()
        .forward(Example::new([("question", serde_json::json!("q"))]))
        .await
        .expect("the program runs");
    assert_eq!(
        prediction.get("answer").and_then(serde_json::Value::as_str),
        Some("Paris")
    );
}

/// Naming a task changes how the answer is read, not what the thing is: an optimizer still takes
/// it. Wrapping the task arms of the module macros made this a real question — before, they gave
/// back the module itself.
#[tokio::test]
async fn a_task_named_module_is_still_optimizable() {
    use dsrust::{BootstrapFewShot, exact_match};

    // Its own model, not the process-wide one: the test above installs that, and both run at once.
    let mut program = Predict::from_signature(QA::signature())
        .set_lm(Arc::new(DummyLM::new([
            example! { answer: "Paris" },
            example! { answer: "Paris" },
            example! { answer: "Paris" },
            example! { answer: "Paris" },
        ])) as Arc<dyn DynChatModel>)
        .task::<QA>();
    let trainset = vec![
        example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
    ];
    BootstrapFewShot::new(exact_match)
        .compile(&mut program, &trainset)
        .await
        .expect("the optimizer takes a task-named module");
    // And it still answers the typed way afterwards.
    let out = call!(program, question = "capital of France?")
        .await
        .expect("the program runs");
    assert_eq!(out.answer, "Paris");
}

/// A module of your own, written the way the guide writes one — and naming its task, so there is
/// nothing to say at the call site. This is what the built-in macros' task arms do for you.
#[derive(dsrust::Module)]
#[task(QA)]
struct Twice {
    first: Predict,
    second: Predict,
}

impl dsrust::Forward for Twice {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let draft = self.first.forward(inputs.clone()).await?;
        let mut again = inputs;
        again.set("question", draft.get("answer").cloned().unwrap_or_default());
        self.second.forward(again).await
    }
}

#[tokio::test]
async fn a_derived_module_names_its_task() {
    // One scripted model each, because a `DummyLM` answers from its own list and the two steps
    // are separate predictors.
    let scripted = |answer: &str| {
        Arc::new(DummyLM::new([Example::new([(
            "answer",
            serde_json::json!(answer),
        )])])) as Arc<dyn DynChatModel>
    };
    let program = Twice {
        first: Predict::from_signature(QA::signature()).set_lm(scripted("France")),
        second: Predict::from_signature(QA::signature()).set_lm(scripted("Paris")),
    };
    // No `.task::<QA>()`: the derive carries it, so this is the built-in modules' own line.
    let out = call!(program, question = "capital of France?")
        .await
        .expect("the program runs");
    assert_eq!(out.answer, "Paris");
}
