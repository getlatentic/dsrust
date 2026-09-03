//! Calibrate's real document, built into a program.
//!
//! The fixtures are copied from `calibrate-codegen/tests/fixtures/`. Three things about the shape
//! decide the translation, and each has a test: an edge renames a field between two modules, a
//! module's signature comes from its edges rather than from a declaration of its own, and the
//! program's answer is the OutputField layer.

use std::sync::Arc;

use dsrust::lm::DynChatModel;
use dsrust::serde_json::{Value, json};
use dsrust::{DummyLM, Example, Module, example};
use dsrust_graph::{CalibrateGraph, Graph, Source};

const MULTI: &str = include_str!("fixtures/multi_module_graph.json");

fn document() -> dsrust_graph::GraphDocument {
    CalibrateGraph::from_json(MULTI)
        .expect("the fixture parses")
        .to_document()
        .expect("it builds a program")
}

fn scripted() -> Arc<dyn DynChatModel> {
    Arc::new(DummyLM::new(std::iter::repeat_n(
        example! { answer: "Paris", summary: "It is Paris.", reasoning: "asked" },
        32,
    ))) as Arc<dyn DynChatModel>
}

/// Only the modules become steps: `Start`, `InputField` and `OutputField` are boundary and
/// ordering, not predictors.
#[test]
fn the_modules_are_the_steps() {
    let document = document();
    let ids: Vec<&str> = document.nodes.iter().map(|node| node.id.as_str()).collect();
    assert_eq!(ids, ["predict", "cot"]);
}

/// A module's signature comes from its edges: inputs from what arrives, outputs from what leaves.
/// In this fixture that makes `predict` a `question -> answer` and `cot` a `context -> summary`.
#[test]
fn a_signature_is_assembled_from_the_edges() {
    let document = document();
    let named = |at: usize| match &document.nodes[at].declared {
        dsrust_graph::Declared::Fields { inputs, outputs } => (
            inputs.iter().map(|f| f.name.clone()).collect::<Vec<_>>(),
            outputs.iter().map(|f| f.name.clone()).collect::<Vec<_>>(),
        ),
        other => panic!("expected assembled fields, got {other:?}"),
    };
    assert_eq!(
        named(0),
        (vec!["question".to_owned()], vec!["answer".to_owned()])
    );
    // `cot` is a ChainOfThought, so it answers its reasoning first — the leading field is the
    // whole of what makes it one. See `a_chain_of_thought_node_asks_for_its_reasoning`.
    assert_eq!(
        named(1),
        (
            vec!["context".to_owned()],
            vec!["reasoning".to_owned(), "summary".to_owned()]
        )
    );
}

/// **The rename.** `predict` produces `answer`; `cot` receives it as `context`. The wire is named
/// by the receiving end and fetches by the sending end — getting that backwards feeds a module a
/// field it never produced, silently, as a null.
#[test]
fn a_field_is_renamed_across_the_edge() {
    let document = document();
    let wire = &document.nodes[1].inputs[0];
    assert_eq!(wire.name, "context", "named by the receiving end");
    match &wire.source {
        Source::Node { node, field } => {
            assert_eq!(node, "predict");
            assert_eq!(field, "answer", "fetched by the sending end");
        }
        other => panic!("expected an upstream module, got {other:?}"),
    }
}

/// The program's own input comes from the `InputField` node, named by that node rather than by the
/// edge — the boundary port carries no `fieldName`.
#[test]
fn the_boundary_input_is_named_by_its_field_node() {
    let document = document();
    match &document.nodes[0].inputs[0].source {
        Source::Input { field } => assert_eq!(field, "question"),
        other => panic!("expected the program's input, got {other:?}"),
    }
}

/// The answer is the OutputField layer: each output names itself and takes its value from one
/// field of one module.
#[test]
fn the_answer_is_the_output_field_layer() {
    let document = document();
    assert_eq!(document.answers.len(), 1);
    let answer = &document.answers[0];
    assert_eq!(answer.name, "summary", "the output field's own name");
    assert_eq!(answer.node, "cot");
    assert_eq!(answer.field, "summary");
}

/// And it runs: two modules, the rename between them, answering under the output field's name.
#[tokio::test]
async fn the_real_document_runs() {
    let graph = Graph::from_document(&document(), scripted()).expect("builds");
    let out = graph
        .forward(Example::new([("question", json!("capital of France?"))]))
        .await
        .expect("runs");
    assert_eq!(
        out.get("summary").and_then(Value::as_str),
        Some("It is Paris.")
    );
}

/// Every module is reachable by an optimizer, named after its node id — so a rewritten instruction
/// can be shown against the box on the canvas it belongs to.
#[test]
fn the_optimizer_reaches_both_modules() {
    let mut graph = Graph::from_document(&document(), scripted()).expect("builds");
    graph
        .walk_covers_every_node()
        .expect("every module is reachable");
    let walked: Vec<String> = graph
        .named_predictors()
        .into_iter()
        .map(|found| found.name)
        .collect();
    assert_eq!(walked, ["predict", "cot"]);
}

/// With no `SignatureSpec`, instructions fall back to a line naming the node — which is what
/// Calibrate's current Python emitter synthesizes. A document that carries a real docstring gets
/// that instead, which is the more faithful of the two.
#[test]
fn instructions_fall_back_when_the_document_carries_none() {
    let mut graph = Graph::from_document(&document(), scripted()).expect("builds");
    let instructions: Vec<String> = graph
        .named_predictors()
        .into_iter()
        .map(|found| found.signature.instructions.clone())
        .collect();
    assert!(
        instructions.iter().all(|line| !line.trim().is_empty()),
        "a module with no instructions has nothing for an optimizer to rewrite: {instructions:?}"
    );
    assert!(instructions[0].contains("Predict"), "{:?}", instructions[0]);
}

/// The seed document names its fields at neither end of an edge: both are named by the boundary
/// field nodes they run between. A loader that only read `fieldName` would build nothing from it.
#[test]
fn a_document_that_names_no_field_on_its_edges_still_builds() {
    const SEED: &str = include_str!("fixtures/seed_field_node_graph.json");
    let document = CalibrateGraph::from_json(SEED)
        .expect("the fixture parses")
        .to_document()
        .expect("it builds a program");

    let ids: Vec<&str> = document.nodes.iter().map(|node| node.id.as_str()).collect();
    assert_eq!(ids, ["predict-answer"]);
    match &document.nodes[0].declared {
        dsrust_graph::Declared::Fields { inputs, outputs } => {
            assert_eq!(inputs[0].name, "question", "named by the InputField node");
            assert_eq!(outputs[0].name, "answer", "named by the OutputField node");
        }
        other => panic!("expected assembled fields, got {other:?}"),
    }
    assert_eq!(document.answers[0].name, "answer");
    assert_eq!(document.answers[0].node, "predict-answer");
}

/// A module's instructions come from the `SignatureSpec` its `signatureId` names.
///
/// Calibrate's current Python emitter synthesizes a generic line instead of reading this, which is
/// why a run against a document with no docstring reads its own signature back at you. Reading the
/// real one is the fix rather than a deviation.
#[test]
fn a_signature_spec_supplies_the_instructions() {
    const DOC: &str = include_str!("fixtures/signature_docstring_graph.json");
    let document = CalibrateGraph::from_json(DOC)
        .expect("parses")
        .to_document()
        .expect("builds");
    let instructions = document.nodes[0]
        .instructions
        .clone()
        .expect("the module carries instructions");
    assert!(
        instructions.starts_with("Answer the question with a concise, grounded response."),
        "expected the SignatureSpec docstring, got {instructions:?}"
    );
    // And it reaches the signature the model is actually asked with.
    let mut graph = Graph::from_document(&document, scripted()).expect("builds");
    assert_eq!(
        graph.named_predictors()[0].signature.instructions,
        instructions
    );
}

/// Two outputs from one module, fanned to two `OutputField` nodes — the case `answers: Vec<Answer>`
/// exists for. The program answers under both names, with the types each field node declared.
#[tokio::test]
async fn a_module_can_answer_with_several_fields() {
    const DOC: &str = include_str!("fixtures/multi_output_graph.json");
    let document = CalibrateGraph::from_json(DOC)
        .expect("parses")
        .to_document()
        .expect("builds");
    let named: Vec<&str> = document
        .answers
        .iter()
        .map(|answer| answer.name.as_str())
        .collect();
    assert_eq!(named, ["answer", "confidence"]);

    // `confidence` is declared `float` on its output field node, and the kind follows.
    let mut graph = Graph::from_document(&document, scripted()).expect("builds");
    let kinds: Vec<String> = graph.named_predictors()[0]
        .signature
        .outputs
        .iter()
        .map(|field| format!("{:?}", field.kind))
        .collect();
    assert_eq!(kinds, ["Str", "Float"]);

    let scored = Arc::new(DummyLM::new([
        example! { answer: "Paris", confidence: 0.9 },
    ])) as Arc<dyn DynChatModel>;
    let graph = Graph::from_document(&document, scored).expect("builds");
    let out = graph
        .forward(Example::new([("question", json!("capital of France?"))]))
        .await
        .expect("runs");
    assert_eq!(out.get("answer").and_then(Value::as_str), Some("Paris"));
    assert_eq!(out.get("confidence").and_then(Value::as_f64), Some(0.9));
}

/// **The precedence, where the two disagree.** An `OutputField` renamed to `verdict` beside a stale
/// edge still saying `answer`: Calibrate takes the node, so a rename propagates immediately rather
/// than waiting for the edge to catch up.
///
/// This is the mirror of the input rename test, and the opposite direction — an input is named by
/// its edge, an output by its node.
#[test]
fn an_output_field_node_outranks_a_stale_edge_name() {
    const DOC: &str = include_str!("fixtures/multi_output_graph.json");
    let mut raw: dsrust::serde_json::Value =
        dsrust::serde_json::from_str(DOC).expect("parses as json");
    // Rename the output field node; leave the edge saying the old name.
    raw["nodes"][3]["config"]["name"] = json!("verdict");
    let document = CalibrateGraph::from_json(&raw.to_string())
        .expect("parses")
        .to_document()
        .expect("builds");

    assert_eq!(
        document.answers[0].name, "verdict",
        "the renamed node names the program's answer"
    );
    assert_eq!(
        document.answers[0].field, "verdict",
        "and what is fetched from the module, so the two cannot drift apart"
    );
    match &document.nodes[0].declared {
        dsrust_graph::Declared::Fields { outputs, .. } => assert_eq!(
            outputs[0].name, "verdict",
            "the module declares the renamed field, not the edge's stale one"
        ),
        other => panic!("expected assembled fields, got {other:?}"),
    }
}

/// The mirror: an *input* is named by its edge, and a disagreeing `InputField` node does not
/// override it. Getting these two the same way round is the bug this pair exists to catch.
#[test]
fn an_input_edge_outranks_the_field_node() {
    const DOC: &str = include_str!("fixtures/multi_module_graph.json");
    let mut raw: dsrust::serde_json::Value =
        dsrust::serde_json::from_str(DOC).expect("parses as json");
    // The edge into `predict` says `question`; rename the InputField node beneath it.
    raw["nodes"][1]["config"]["name"] = json!("enquiry");
    let document = CalibrateGraph::from_json(&raw.to_string())
        .expect("parses")
        .to_document()
        .expect("builds");

    match &document.nodes[0].declared {
        dsrust_graph::Declared::Fields { inputs, .. } => assert_eq!(
            inputs[0].name, "question",
            "the edge names the module's input field"
        ),
        other => panic!("expected assembled fields, got {other:?}"),
    }
    match &document.nodes[0].inputs[0].source {
        Source::Input { field } => assert_eq!(
            field, "enquiry",
            "while the value still comes from the renamed program input"
        ),
        other => panic!("expected the program's input, got {other:?}"),
    }
}

/// A node kind this builder does not know stops the load, rather than being quietly left out.
///
/// Dropping it is what filtering to the known kinds does naturally, and it builds a program the
/// document does not describe: fewer steps, running less and optimizing less, with nothing saying
/// so. Measured before the check existed — renaming `cot`'s kind built a one-node program whose
/// answer pointed at the node that had been dropped.
#[test]
fn an_unknown_node_kind_stops_the_load() {
    const DOC: &str = include_str!("fixtures/multi_module_graph.json");
    let mut raw: dsrust::serde_json::Value =
        dsrust::serde_json::from_str(DOC).expect("parses as json");
    raw["nodes"][3]["kind"] = json!("Retrieve");

    let refused = match CalibrateGraph::from_json(&raw.to_string())
        .expect("parses")
        .to_document()
    {
        Ok(document) => panic!(
            "built a program of {} node(s) from a document with a kind it does not know",
            document.nodes.len()
        ),
        Err(why) => why.to_string(),
    };
    assert!(refused.contains("cot (Retrieve)"), "{refused}");
}

/// The guard against the failure this whole crate is about, on a real document: every module the
/// document declares is one the optimizer will reach.
///
/// The synthetic version of this lives in `graph.rs`'s tests. This one holds it against a document
/// Calibrate actually produces, which is where a drop would really happen.
#[test]
fn every_module_in_a_real_document_is_reachable() {
    for fixture in [
        include_str!("fixtures/multi_module_graph.json"),
        include_str!("fixtures/seed_field_node_graph.json"),
        include_str!("fixtures/signature_docstring_graph.json"),
        include_str!("fixtures/multi_output_graph.json"),
    ] {
        let document = CalibrateGraph::from_json(fixture)
            .expect("parses")
            .to_document()
            .expect("builds");
        let declared = document.nodes.len();
        let mut graph = Graph::from_document(&document, scripted()).expect("builds");
        graph
            .walk_covers_every_node()
            .unwrap_or_else(|why| panic!("{declared} module(s) declared: {why}"));
    }
}

/// A ChainOfThought node reasons; a Predict node does not.
///
/// dspy's ChainOfThought is not a different module — it is the same signature with a leading
/// `reasoning` output. The builder read every node the same way and never looked at its kind, so a
/// node the canvas draws as a ChainOfThought, and the exported Python emits as
/// `dspy.ChainOfThought`, ran here as a plain Predict: no reasoning field, no reasoning step, and
/// nothing anywhere saying the program that ran was not the program on screen.
#[test]
fn a_chain_of_thought_node_asks_for_its_reasoning() {
    let built = document();
    let outputs_of = |id: &str| -> Vec<String> {
        let node = built
            .nodes
            .iter()
            .find(|node| node.id == id)
            .expect("the fixture declares this node");
        match &node.declared {
            dsrust_graph::Declared::Fields { outputs, .. } => {
                outputs.iter().map(|field| field.name.clone()).collect()
            }
            other => panic!("expected declared fields, got {other:?}"),
        }
    };
    assert_eq!(
        outputs_of("cot").first().map(String::as_str),
        Some("reasoning"),
        "the reasoning field is prepended, as upstream prepends it"
    );
    assert!(
        !outputs_of("predict").contains(&"reasoning".to_owned()),
        "a Predict node must not grow one"
    );
}

/// A node kind this builder cannot honour is refused, not approximated.
///
/// ReAct carries tools and an iteration limit; CustomModule carries a Python
/// body. The translation drops both, so building either as a plain Predict runs
/// a different program than the canvas draws and the exported Python emits —
/// and says nothing. ChainOfThought had exactly this defect here until it grew
/// its reasoning field; these two have no field to add, so they stop.
#[test]
fn a_kind_this_builder_cannot_honour_is_refused() {
    let refuse = |kind: &str| -> String {
        let mut document: dsrust::serde_json::Value =
            dsrust::serde_json::from_str(MULTI).expect("the fixture parses");
        for node in document["nodes"].as_array_mut().expect("nodes") {
            if node["id"] == "cot" {
                node["kind"] = dsrust::serde_json::json!(kind);
            }
        }
        let text = document.to_string();
        match CalibrateGraph::from_json(&text)
            .expect("parses")
            .to_document()
        {
            Ok(_) => panic!("{kind} was built as something else instead of refused"),
            Err(refused) => refused.to_string(),
        }
    };

    let react = refuse("ReAct");
    assert!(react.contains("models no tools"), "{react}");
    assert!(
        react.contains("Python runtime"),
        "says where it does work: {react}"
    );

    let custom = refuse("CustomModule");
    assert!(custom.contains("body is Python"), "{custom}");
    assert!(
        custom.contains("Python runtime"),
        "says where it does work: {custom}"
    );
}

/// A ChainOfThought node renders the prompt dspy's ChainOfThought renders.
///
/// The reasoning field was declared as dspy 3.3's `Reasoning` type, which is
/// str-like in every way but one: its annotation is not literally `str`, so the
/// output-requirement hint fires and the line reads "`[[ ## reasoning ## ]]`
/// (must be formatted as a valid Python str)". dspy's own ChainOfThought
/// declares the rationale `str` and renders no such suffix.
///
/// Ten prompt tokens a call is the small half. The large half is that a model
/// told its reasoning must be a valid Python str starts quoting — and then
/// exact match fails on `"positive"`, so every score taken through a graph
/// carried a handicap the same program run through a plain prompt did not.
#[tokio::test]
async fn a_chain_of_thought_node_renders_no_type_suffix_on_its_reasoning() {
    use std::sync::Mutex;

    #[derive(Default)]
    struct Recording(Mutex<Vec<String>>);

    impl dsrust::lm::ChatModel for Recording {
        async fn forward(
            &self,
            request: &dsrust::lm::api::LmRequest,
        ) -> dsrust::anyhow::Result<dsrust::lm::LmResponse> {
            self.0.lock().expect("lock").push(
                request
                    .messages
                    .iter()
                    .filter_map(|message| message.text())
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            // Answer whichever node is asking: the first declares `answer`,
            // the ChainOfThought declares `reasoning` then `summary`. A reply
            // shaped for the wrong one fails to parse and the graph stops
            // before reaching the node under test.
            let asked: String = request
                .messages
                .iter()
                .filter_map(|message| message.text())
                .collect::<Vec<_>>()
                .join("\n");
            let reply = if asked.contains("1. `reasoning`") {
                "[[ ## reasoning ## ]]\nbecause\n\n[[ ## summary ## ]]\nfine\n\n[[ ## completed ## ]]"
            } else {
                "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"
            };
            Ok(dsrust::lm::LmResponse::text(reply))
        }
    }

    let lm = Arc::new(Recording::default());
    let graph = Graph::from_document(&document(), Arc::clone(&lm) as Arc<dyn DynChatModel>)
        .expect("the fixture builds");
    let _ = graph
        .forward(Example::new([("question", json!("hi"))]))
        .await;

    let asked = lm.0.lock().expect("lock").join("\n---\n");
    assert!(
        asked.contains("[[ ## reasoning ## ]]"),
        "the node reasons at all:\n{asked}"
    );
    // The exact shape the suffix takes, read off the rendered prompt rather
    // than guessed: the marker, then the hint. An assertion on a string that
    // never appears passes against the defect too.
    assert!(
        !asked.contains("reasoning ## ]]` (must be formatted"),
        "dspy's ChainOfThought renders no type suffix on its rationale:\n{asked}"
    );
}
