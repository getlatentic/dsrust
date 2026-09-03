// Included at the top of every file scripts/docs_snippets.py generates.
//
// A guide's fragment names things the surrounding *prose* established rather than the code — a
// trainset, your own metric, a task called `Haiku`. This is the only place such a name may come
// from, and that is the point: what is declared here is what a guide may leave to prose, and
// everything else has to appear in the block a reader copies. Adding a fixture to green a failing
// block is how a guide starts lying again, so each one below is something the page really does
// introduce in words.
#![allow(unused, unused_mut, non_snake_case)]

use std::future::Future;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use dsrust::lm::{DynChatModel, LM, LmFailure, configure};
use dsrust::optimize::{Feedback, MetricContext};
use dsrust::signature::SignatureSpec;
use dsrust::{
    BestOfN, BootstrapFewShot, ChainOfThought, ChatModel, CodeAct, Evaluate, Example, Forward, GEPA,
    Module,
    MultiChainComparison, Parallel, Predict, Prediction, ProgramOfThought, RLM, ReAct, ReActV2,
    Assistant, Developer, LmMessage, LmPart, Refine, Rlm, Signature, System, Tool, User, call,
    exact_match, example, input, items, make_signature, tool, typed_tool,
};

/// The two-in, two-out task the guide writes in Python once and then keeps using.
#[derive(Signature)]
/// Write a haiku.
struct Haiku {
    #[input]
    subject: String,
    #[input]
    tone: String,
    #[output]
    haiku: String,
    #[output]
    mood: String,
}

/// The declared task the "either spelling" section hands to each module in turn.
#[derive(Signature)]
/// Investigate the question.
struct Investigate {
    #[input]
    question: String,
    #[output]
    answer: String,
}

/// What the prose around a fragment already established.
struct Prose {
    trainset: Vec<Example>,
    valset: Vec<Example>,
    metric: fn(&Example, &Prediction, &MetricContext<'_>) -> Feedback,
    reflection_lm: Arc<dyn DynChatModel>,
    tools: Vec<Box<dyn Tool>>,
    held: Arc<Held>,
    /// The module a fragment is calling — named for whatever the section is about.
    extractor: Predict,
    inputs: Example,
    program: Predict,
    qa: Predict,
    /// A reward function, named as dspy's `reward_fn=f` keyword is.
    f: fn(&Example, &Prediction) -> f64,
    /// The image a caller already has, in the multimodal examples.
    url: String,
    /// The examples a run is scored over — "a devset", which the page introduces in words.
    devset: Vec<Example>,
}

fn prose() -> Prose {
    Prose {
        trainset: Vec::new(),
        valset: Vec::new(),
        metric: |_, _, _| Feedback::new(0.0, String::new()),
        reflection_lm: Arc::new(dsrust::DummyLM::new(Vec::new())),
        tools: Vec::new(),
        held: Arc::new(Held),
        extractor: Predict!("question -> answer"),
        inputs: input! { question: "capital of France?" },
        program: Predict!("question -> answer"),
        qa: Predict!("question -> answer"),
        f: |_, _| 0.0,
        url: "https://example.com/a.jpg".to_owned(),
        devset: Vec::new(),
    }
}

/// The guide's stand-in for whatever a caller does about a retryable failure. `retry_after` is the
/// seconds a provider asked for, so this takes what `LmFailure` carries.
async fn back_off(_seconds: Option<f64>) {}

/// What the guide's tool page leaves to prose: the state a roster writes into.
pub struct Held;

impl Held {
    pub fn add_block(&self, block_type: String, text: String) -> Result<String> {
        Ok(format!("{block_type}: {text}"))
    }
}
