//! One line, every module, against a real provider.
//!
//!     OPENAI_BASE_URL=… OPENAI_API_KEY=… LIVE_MODEL=… cargo run --example every_spelling_live
//!
//! Everything else about `Module::task` is checked with a scripted model, which says nothing about
//! whether a provider's own reply parses into the task's struct. This asks four ways and prints
//! what each answered.

use std::sync::Arc;

use anyhow::Result;
use dsrust::lm::api::LmConfig;
use dsrust::lm::{DynChatModel, LM};
use dsrust::signature::SignatureSpec;
use dsrust::{
    ChainOfThought, Example, FnTool, Module, Predict, Prediction, ReActV2, Signature, Tool, call,
    configure_model, tool,
};

#[derive(Signature)]
/// Answer the question in one word.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

#[tool]
/// Look one fact up in the reference.
fn lookup(query: String) -> anyhow::Result<String> {
    Ok(format!("The reference says: {query} is Paris."))
}

/// A module of the caller's own, naming the task it answers with — so its call site is the same
/// line as the built-in modules', with nothing added.
#[derive(Module)]
#[task(QA)]
struct AskTwice {
    first: Predict,
    second: Predict,
}

impl dsrust::Forward for AskTwice {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let draft = self.first.forward(inputs.clone()).await?;
        let mut again = inputs;
        again.set(
            "question",
            serde_json::json!(format!(
                "Repeat this single word exactly: {}",
                draft
                    .get("answer")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
            )),
        );
        self.second.forward(again).await
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var("LIVE_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".to_owned());
    let lm = LM::builder(format!("openai/{model}"))
        .config(LmConfig {
            max_tokens: Some(16384),
            ..Default::default()
        })
        .cache(false)
        .build()?;
    configure_model(
        reqwest::Client::new(),
        Arc::new(lm) as Arc<dyn DynChatModel>,
    );
    let question = "What is the capital of France?";

    let out = call!(Predict!(QA), question = question).await?;
    println!("Predict!(QA)          -> out.answer = {:?}", out.answer);

    let out = call!(ChainOfThought!(QA), question = question).await?;
    println!("ChainOfThought!(QA)   -> out.answer = {:?}", out.answer);

    let tools: Vec<Box<dyn Tool>> = vec![Box::new(Lookup)];
    let out = call!(ReActV2!(QA, tools), question = question).await?;
    println!("ReActV2!(QA, tools)   -> out.answer = {:?}", out.answer);

    let mine = AskTwice {
        first: Predict::from_signature(QA::signature()),
        second: Predict::from_signature(QA::signature()),
    };
    let out = call!(mine, question = question).await?;
    println!("custom #[task(QA)]    -> out.answer = {:?}", out.answer);

    // The one spelling with no task to answer with, for contrast.
    let out = call!(Predict!("question -> answer"), question = question).await?;
    println!(
        "Predict!(\"q -> a\")     -> out.get  = {:?}",
        out.get("answer")
    );
    let _ = FnTool::new(
        "unused",
        "",
        serde_json::json!({}),
        |_: &serde_json::Value| Ok(String::new()),
    );
    Ok(())
}
