//! Every built-in, held in a composed module that traces nothing itself, still attributes.
//!
//! An optimizer learns which predictor earned a demo from the trace a run leaves, filed under the
//! names `named_predictors` yields. Two things used to empty or misfile that trace for a built-in
//! held inside a composed module, and nothing checked either across the roster:
//!
//! * A plain `forward` routed through the traced path with a scratch buffer — `ReAct`, `Refine`
//!   and `ProgramOfThought` all did — threw the explicit record away, and had suppressed the
//!   ambient one to avoid a double. Held in a module tracing nothing itself: nothing recorded.
//! * Every built-in's explicit trace named its steps by hand-written relabel chains, and all six
//!   disagreed with their own `named_predictors` — `self` where the walk says `predict`, `extract`
//!   where it says `extract.predict`. A demo filed under one name is not found under the other.
//!
//! Both are gone the same way: under a listening run a predictor records ambiently, once, under
//! the name the run resolved for it, whichever path reached it. The explicit push remains for a
//! module tracing for its own purposes, where nothing looks a name up. This test walks the roster
//! so a built-in added later is covered without anyone deciding to cover it.

use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use dsrust::lm::DynChatModel;
use dsrust::{
    BestOfN, ChainOfThought, DummyLM, Example, Module, MultiChainComparison, Prediction,
    ProgramOfThought, ReAct, Refine,
};
use serde_json::{Value, json};

/// A composed module that traces nothing of its own — the shape the guide writes, over any
/// built-in. Hand-written because the roster is dynamic and the derive takes concrete fields; it
/// does exactly what the derive would: plain `forward`, and predictors named under the field.
struct Holding {
    inner: Box<dyn Module>,
}

impl Module for Holding {
    fn forward<'a>(&'a self, inputs: Example) -> Boxed<'a> {
        Box::pin(async move { self.inner.forward(inputs).await })
    }

    fn named_predictors(&mut self) -> Vec<dsrust::NamedPredictor<'_>> {
        dsrust::module::under("inner", self.inner.named_predictors())
    }
}

/// One reply shape every built-in here can parse: each step reads the fields it declares.
fn scripted() -> Arc<dyn DynChatModel> {
    Arc::new(DummyLM::new((0..32).map(|i| {
        Example::new([
            ("reasoning", Value::String(format!("r{i}"))),
            ("b", Value::String(format!("b{i}"))),
            ("next_thought", Value::String("done".to_owned())),
            ("next_tool_name", Value::String("finish".to_owned())),
            ("next_tool_args", json!({})),
            ("generated_code", Value::String("b = 1".to_owned())),
            ("rationale", Value::String(format!("r{i}"))),
        ])
    }))) as Arc<dyn DynChatModel>
}

fn asked() -> Example {
    let mut inputs = Example::default();
    inputs.set("a", json!("x"));
    inputs
}

/// Every built-in that holds predictors, built over `a -> b`.
fn built_ins() -> Vec<(&'static str, Box<dyn Module>)> {
    let reward = |_: &Example, _: &Prediction| 1.0;
    vec![
        ("ChainOfThought", Box::new(ChainOfThought!("a -> b"))),
        ("ReAct", Box::new(ReAct!("a -> b", vec![]))),
        (
            "MultiChainComparison",
            Box::new(MultiChainComparison::parse("a -> b", 2).expect("parses")),
        ),
        (
            "BestOfN",
            Box::new(BestOfN::new(ChainOfThought!("a -> b"), 2, reward, 1.0)),
        ),
        (
            "Refine",
            Box::new(Refine::new(ChainOfThought!("a -> b"), 2, reward, 1.0)),
        ),
        ("ProgramOfThought", Box::new(ProgramOfThought!("a -> b"))),
    ]
}

fn names(program: &mut dyn Module) -> BTreeSet<String> {
    program
        .named_predictors()
        .into_iter()
        .map(|predictor| predictor.name)
        .collect()
}

/// Invariant 1: held in a module that traces nothing itself, every built-in still attributes.
#[tokio::test]
async fn every_built_in_attributes_when_held_in_a_composed_module() {
    let mut silent = Vec::new();
    for (what, inner) in built_ins() {
        let mut held = Holding { inner };
        for predictor in held.named_predictors() {
            *predictor.lm = Some(scripted());
        }
        let expected = names(&mut held);
        let (answered, trace) = held.traced(asked()).await;
        let answered = answered.map(|_| ()).map_err(|error| error.to_string());
        let recorded: BTreeSet<String> = trace.iter().map(|step| step.predictor.clone()).collect();
        if recorded.is_empty() || !recorded.is_subset(&expected) {
            silent.push(format!(
                "  {what}: named {expected:?}, trace recorded {recorded:?} (run: {answered:?})"
            ));
        }
    }
    assert!(
        silent.is_empty(),
        "held in a composed module, these built-ins lose attribution:\n{}",
        silent.join("\n")
    );
}

type Boxed<'a> = Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>>;

/// A built-in that threads `forward_traced` all the way up records each call exactly once.
///
/// The other half of the same rule. Under a listening run, a predictor reached through its own
/// `forward_traced` records ambiently under its resolved name; if it *also* pushed into the
/// caller's buffer, a bare `ChainOfThought` handed straight to an optimizer would file one call
/// twice — once as `predict`, once as `self` — and be taught its own demo twice over.
#[tokio::test]
async fn a_threaded_built_in_records_each_call_once() {
    let mut cot = ChainOfThought!("a -> b");
    for predictor in cot.named_predictors() {
        *predictor.lm = Some(scripted());
    }
    let (answered, trace) = cot.traced(asked()).await;
    answered.expect("it runs");
    let recorded: Vec<&str> = trace.iter().map(|step| step.predictor.as_str()).collect();
    assert_eq!(
        recorded,
        ["predict"],
        "one call, one step, under the walk's name"
    );
}
