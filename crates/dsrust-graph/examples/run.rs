//! Load a graph document, run it, and optimize it — the whole loop against a real provider.
//!
//!     export OPENAI_BASE_URL=https://api.deepseek.com
//!     export OPENAI_API_KEY=sk-...
//!
//!     cargo run --example run -- --document               # print the built-in document
//!     cargo run --example run -- --subject "winter mornings"
//!     cargo run --example run -- --optimize
//!
//! `--document FILE` takes a document of your own, in the shape `--document` prints.

use std::sync::Arc;

use dsrust::anyhow::{Context, Result};
use dsrust::lm::{DynChatModel, LM};
use dsrust::serde_json::{Value, json};
use dsrust::{BootstrapFewShot, Example, Module, exact_match, example};
use dsrust_graph::{Graph, GraphDocument};

/// A document of either shape: this crate's own, or Calibrate's — told apart by whether it has
/// `edges`, since only Calibrate's carries them.
fn main_document(path: Option<&str>) -> Result<GraphDocument> {
    let Some(path) = path else {
        return Ok(GraphDocument::example());
    };
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    match text.contains("\"edges\"") {
        true => dsrust_graph::CalibrateGraph::from_json(&text)?.to_document(),
        false => GraphDocument::from_json(&text),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|at| args.get(at + 1))
            .cloned()
    };
    let document = main_document(value("--document").as_deref())?;
    if args.iter().any(|arg| arg == "--document") && value("--document").is_none() {
        println!("{}", document.to_json());
        return Ok(());
    }

    let model = value("--model").unwrap_or_else(|| "openai/gpt-4o-mini".to_owned());
    let lm = Arc::new(LM::new(&model)?) as Arc<dyn DynChatModel>;
    let mut graph = Graph::from_document(&document, lm)?;
    let subject = value("--subject").unwrap_or_else(|| "winter mornings".to_owned());

    if args.iter().any(|arg| arg == "--optimize") {
        // What an optimizer can reach is what the graph declared — print it before and after, so a
        // run that rewrote nothing is visible rather than merely successful.
        println!("nodes     {:?}", names(&mut graph));
        let trainset = vec![
            example! { subject: json!(subject), haiku: "Snow settles slowly" }
                .with_inputs(["subject"]),
        ];
        BootstrapFewShot::new(exact_match)
            .compile(&mut graph, &trainset)
            .await?;
        let demos: usize = graph
            .named_predictors()
            .iter()
            .map(|found| found.demos.len())
            .sum();
        println!("demos     {demos} learned across the graph");
        return Ok(());
    }

    // Whatever the document's own input fields are — a Calibrate graph asks for `question`, the
    // built-in one for `subject`, and hardcoding either feeds the other a null.
    let fed: Vec<(String, Value)> = program_inputs(&document)
        .into_iter()
        .map(|name| (name, json!(subject)))
        .collect();
    println!(
        "fed       {:?}",
        fed.iter().map(|(name, _)| name).collect::<Vec<_>>()
    );
    let out = graph.forward(Example::new(fed)).await?;
    for (name, value) in out.example.fields() {
        println!("{name:9} {}", render(value));
    }
    Ok(())
}

/// The fields the program takes: every wire that reads from the program's own inputs.
fn program_inputs(document: &GraphDocument) -> Vec<String> {
    let mut named: Vec<String> = Vec::new();
    for node in &document.nodes {
        for wire in &node.inputs {
            if let dsrust_graph::Source::Input { field } = &wire.source
                && !named.contains(field)
            {
                named.push(field.clone());
            }
        }
    }
    named
}

fn names(graph: &mut Graph) -> Vec<String> {
    graph
        .named_predictors()
        .into_iter()
        .map(|found| found.name)
        .collect()
}

fn render(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}
