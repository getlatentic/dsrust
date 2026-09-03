//! What a runtime-wired program has to hold, and the one assertion that is easy to get wrong.
//!
//! The last test is the point of the crate. It asserts on what the optimizer *changed*, not on
//! `compile` returning `Ok` — because a graph module missing `named_predictors` compiles, runs,
//! optimizes, and reports success having rewritten nothing.

use std::sync::Arc;

use dsrust::lm::DynChatModel;
use dsrust::serde_json::{Value, json};
use dsrust::{BootstrapFewShot, DummyLM, Example, Module, exact_match, example};
use dsrust_graph::{Graph, GraphDocument};

fn scripted() -> Arc<dyn DynChatModel> {
    Arc::new(DummyLM::new(std::iter::repeat_n(
        example! { angle: "winter", haiku: "Paris" },
        64,
    ))) as Arc<dyn DynChatModel>
}

fn graph() -> Graph {
    Graph::from_document(&GraphDocument::example(), scripted()).expect("the document builds")
}

/// The wiring runs: the second node is fed from the first, and the program answers with the node
/// the document names.
#[tokio::test]
async fn the_wiring_is_the_forward() {
    let out = graph()
        .forward(Example::new([("subject", json!("winter mornings"))]))
        .await
        .expect("the graph runs");
    assert_eq!(out.get("haiku").and_then(Value::as_str), Some("Paris"));
}

/// Every node is reachable by an optimizer, named after the node — which is what lets a caller
/// show a rewritten instruction against the box on the canvas it belongs to.
#[test]
fn the_walk_reaches_every_node_by_name() {
    let mut graph = graph();
    let walked: Vec<String> = graph
        .named_predictors()
        .into_iter()
        .map(|found| found.name)
        .collect();
    assert_eq!(walked, ["plan", "write"]);
}

/// A node's instructions come from the document, so an optimizer has something to rewrite and a
/// caller has something to write back.
#[test]
fn the_document_carries_the_instructions() {
    let mut graph = graph();
    let instructions: Vec<String> = graph
        .named_predictors()
        .into_iter()
        .map(|found| found.signature.instructions.clone())
        .collect();
    assert_eq!(
        instructions,
        [
            "Pick one angle on the subject.",
            "Write a haiku on that angle."
        ]
    );
}

/// A document naming a node that does not exist fails at load, not at the first run.
#[test]
fn a_bad_document_is_refused_when_it_loads() {
    let mut document = GraphDocument::example();
    document.answers[0].node = "nowhere".to_owned();
    let refused = match Graph::from_document(&document, scripted()) {
        Ok(_) => panic!("a document naming no such node built anyway"),
        Err(refused) => refused.to_string(),
    };
    assert!(refused.contains("nowhere"), "{refused}");
}

/// **The one that matters.** A compile reaches the nodes and changes them.
///
/// Asserting `compile(...)` returned `Ok` would pass with `named_predictors` deleted — the
/// optimizer would walk an empty list, rewrite nothing, and report success. So the assertion is on
/// the demos the walk earned.
#[tokio::test]
async fn a_compile_actually_changes_the_graph() {
    let mut graph = graph();
    let trainset =
        vec![example! { subject: "winter mornings", haiku: "Paris" }.with_inputs(["subject"])];
    BootstrapFewShot::new(exact_match)
        .compile(&mut graph, &trainset)
        .await
        .expect("compiles");

    let demos: usize = graph
        .named_predictors()
        .iter()
        .map(|found| found.demos.len())
        .sum();
    assert!(
        demos > 0,
        "the optimizer reported success having rewritten nothing — which is what a graph module \
         missing `named_predictors` does"
    );

    // Counting them is not enough. A demo is earned from the *trace*, and a program whose trace
    // comes back empty falls through to the whole-program demo — so every node is taught the
    // program's own inputs and outputs instead of its own, and the count above is identical. The
    // first node answers `angle` and never sees `haiku`; the second is the other way round.
    assert_eq!(
        demo_fields(&mut graph, "plan"),
        // `augmented` is dspy's own marker on a bootstrapped demo.
        ["angle", "augmented", "subject"],
        "the first node is taught its own fields, and never the program's output"
    );
    assert_eq!(
        demo_fields(&mut graph, "write"),
        ["angle", "augmented", "haiku"],
        "the second node is taught its own fields, and never the program's input"
    );
}

/// The field names one node's demos carry, sorted.
fn demo_fields(graph: &mut Graph, node: &str) -> Vec<String> {
    graph
        .named_predictors()
        .into_iter()
        .find(|found| found.name == node)
        .expect("the node is walked")
        .demos
        .iter()
        .flat_map(|demo| demo.fields().map(|(name, _)| name.to_owned()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// The guard that turns the silent failure into a loud one, before a compile rather than after.
#[test]
fn the_walk_is_checked_before_it_is_trusted() {
    graph()
        .walk_covers_every_node()
        .expect("every node is reachable");
}

/// A node declared by its fields rather than by a string — the shape a canvas with a node per
/// field has, and the one a string signature cannot express once a field carries a custom type.
#[test]
fn a_node_can_declare_its_fields_instead_of_a_string() {
    let mut graph = graph();
    let signatures: Vec<(usize, usize)> = graph
        .named_predictors()
        .iter()
        .map(|found| (found.signature.inputs.len(), found.signature.outputs.len()))
        .collect();
    // `plan` is written as a string, `write` as fields; both arrive as one input and one output.
    assert_eq!(signatures, [(1, 1), (1, 1)]);

    // The field node's description reached the signature, which is what a prompt renders.
    let described = graph
        .named_predictors()
        .into_iter()
        .find(|found| found.name == "write")
        .expect("the write node");
    assert_eq!(described.signature.inputs[0].desc, "The angle to write on.");
}

/// A type the string spelling cannot express arrives as a structured field carrying its own
/// annotation — which is how a custom type reaches dspy.
#[test]
fn a_custom_type_travels_as_its_annotation() {
    use dsrust::signature::FieldKind;
    use dsrust_graph::{Declared, Field};

    let mut document = GraphDocument::example();
    document.nodes[1].declared = Declared::Fields {
        inputs: vec![Field {
            name: "angle".to_owned(),
            r#type: "str".to_owned(),
            description: String::new(),
        }],
        outputs: vec![Field {
            name: "citations".to_owned(),
            r#type: "list[Citation]".to_owned(),
            description: String::new(),
        }],
    };

    let mut graph = Graph::from_document(&document, scripted()).expect("builds");
    let kind = graph
        .named_predictors()
        .into_iter()
        .find(|found| found.name == "write")
        .expect("the write node")
        .signature
        .outputs[0]
        .kind
        .clone();
    match kind {
        FieldKind::Json(json) => assert_eq!(json.annotation, "list[Citation]"),
        other => panic!("expected a structured field, got {other:?}"),
    }
}

/// The graph reports itself to a callback, as a derived module would.
///
/// `#[derive(Module)]` is dspy's `Module.__call__` decorator: it opens `on_module_start` before the
/// body and closes `on_module_end` after. A hand-written `forward` *is* that entry, so omitting the
/// point makes the outermost program of a run silent — the inner `Predict`s still report, and a
/// listener sees steps happening inside a module that never started.
///
/// This crate shipped without it, and the app that took it as a reference inherited the hole. The
/// suite was green throughout: nothing here had ever registered a callback.
#[tokio::test]
async fn the_graph_reports_itself_to_a_callback() {
    use std::sync::Mutex;

    use dsrust::{CallId, Callback, Prediction};

    #[derive(Default)]
    struct Recorder {
        seen: Mutex<Vec<String>>,
    }

    impl Callback for Recorder {
        fn on_module_start(&self, _call: &CallId, module: &str, _inputs: &Example) {
            self.seen
                .lock()
                .expect("not poisoned")
                .push(format!("start:{module}"));
        }

        fn on_module_end(
            &self,
            _call: &CallId,
            answered: Result<&Prediction, &dsrust::anyhow::Error>,
        ) {
            self.seen
                .lock()
                .expect("not poisoned")
                .push(format!("end:{}", answered.is_ok()));
        }
    }

    let recorder = Arc::new(Recorder::default());
    dsrust::configure_callbacks(vec![recorder.clone() as Arc<dyn Callback>]);

    graph()
        .forward(Example::new([("subject", json!("winter mornings"))]))
        .await
        .expect("the graph answers");

    let seen = recorder.seen.lock().expect("not poisoned").clone();
    assert!(
        seen.contains(&"start:Graph".to_owned()),
        "the graph itself never opened a module point: {seen:?}"
    );
    assert!(
        seen.contains(&"end:true".to_owned()),
        "the graph itself never closed one: {seen:?}"
    );
}

/// One output feeding several consumers declares that field once, not once per consumer.
///
/// The builder derived a module's output fields one per *outgoing edge*, so a fan-out — an
/// ordinary shape; a timeline feeding three downstream modules — declared the same field once per
/// consumer. The signature became `-> answer, answer` and the adapter then demanded the model
/// answer `answer` twice, failing a perfectly good reply with "Expected to find output fields".
///
/// Nothing caught it: the walk and the wiring assertions both passed, because the structure *was*
/// right. Only running a compiled fan-out through an adapter showed it.
///
/// Inputs are deliberately left un-deduped: two edges feeding one input name is a document bug,
/// and it is better loud. Reported by the Calibrate session, which hit it on a six-module graph.
#[tokio::test]
async fn one_output_feeding_two_consumers_is_declared_once() {
    let document =
        dsrust_graph::CalibrateGraph::from_json(include_str!("fixtures/fan_out_graph.json"))
            .expect("the fixture parses")
            .to_document()
            .expect("it builds a program");
    // Every field this graph's three predictors ask for, so the run exercises the wiring rather
    // than the scripted model's vocabulary.
    let answers = Arc::new(DummyLM::new(std::iter::repeat_n(
        example! {
            answer: "cold", reasoning: "It is winter.",
            summary: "Paris in winter", brief: "Cold Paris"
        },
        64,
    ))) as Arc<dyn DynChatModel>;
    let mut graph = Graph::from_document(&document, answers).expect("the document builds");

    for predictor in graph.named_predictors() {
        let names: Vec<&str> = predictor
            .signature
            .outputs
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            names.len(),
            unique.len(),
            "node {:?} declares a field more than once: {names:?}",
            predictor.name
        );
    }

    graph
        .forward(Example::new([("question", json!("winter mornings"))]))
        .await
        .expect("a fan-out graph runs");
}

/// Two modules reading one input, both feeding a third — the fan-in shape.
///
/// From the Calibrate session's spec compiler, which generates these from a `*.spec.yaml`. Its
/// fixtures are worth more than a hand-written one: they are what its authoring surface actually
/// emits, so a shape it can express and this cannot is a real gap rather than a hypothetical.
#[tokio::test]
async fn a_fan_in_runs_and_walks_in_declaration_order() {
    let mut graph = calibrate_graph(include_str!("fixtures/support_triage_graph.json"));
    assert_eq!(
        walk(&mut graph),
        ["classify", "extract", "compose"],
        "declaration order is execution order"
    );

    let answered = graph
        .forward(Example::new([("ticket", json!("the printer is on fire"))]))
        .await
        .expect("a fan-in graph runs");
    // The run half is what catches what a structure check misses: the walk above passed on a graph
    // whose signature the adapter then rejected, which is how the fan-out bug got this far.
    assert!(answered.get("reply").is_some(), "{answered:?}");
    assert!(answered.get("priority").is_some(), "{answered:?}");
}

/// Six modules over four layers, two fan-ins, and one output read by three of them.
///
/// `timeline.events` feeds `correlate`, `investigate` and `postmortem` — the shape that found the
/// duplicate-field bug, here in the form a real spec produces rather than the two-consumer
/// fixture written to reproduce it.
#[tokio::test]
async fn a_deep_graph_with_a_three_way_fan_out_runs() {
    let mut graph = calibrate_graph(include_str!("fixtures/incident_review.graph.json"));
    assert_eq!(
        walk(&mut graph),
        [
            "triage",
            "timeline",
            "correlate",
            "investigate",
            "postmortem",
            "actions"
        ]
    );

    for predictor in graph.named_predictors() {
        let mut names: Vec<&str> = predictor
            .signature
            .outputs
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        let declared = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            declared,
            names.len(),
            "node {:?} declares a field more than once",
            predictor.name
        );
    }

    let answered = graph
        .forward(Example::new([(
            "report",
            json!("the disk filled at 03:00"),
        )]))
        .await
        .expect("a six-module graph runs");
    assert!(answered.get("draft").is_some(), "{answered:?}");
    assert!(answered.get("items").is_some(), "{answered:?}");
}

/// A Calibrate document, and a model that answers every field its modules ask for.
fn calibrate_graph(json: &str) -> Graph {
    let document = dsrust_graph::CalibrateGraph::from_json(json)
        .expect("the fixture parses")
        .to_document()
        .expect("it builds a program");
    let answers = Arc::new(DummyLM::new(std::iter::repeat_n(
        example! {
            reasoning: "Because.",
            category: "hardware", key_points: "smoke", reply: "We are on it", priority: "high",
            severity: "sev1", events: "03:00 disk full", hypotheses: "log rotation",
            root_cause: "no rotation", draft: "The disk filled.", items: "add rotation"
        },
        128,
    ))) as Arc<dyn DynChatModel>;
    Graph::from_document(&document, answers).expect("the document builds")
}

/// The node names, in the order the program walks them.
fn walk(graph: &mut Graph) -> Vec<String> {
    graph
        .named_predictors()
        .into_iter()
        .map(|predictor| predictor.name)
        .collect()
}

/// **The second one that matters.** A GEPA compile reaches the nodes and rewrites their
/// *instructions*.
///
/// The demo test above proves the walk is reachable and writable. It does not prove this: demos
/// are written by `BootstrapFewShot` through `NamedPredictor::demos`, while GEPA writes
/// instructions through `NamedPredictor::signature` after choosing a winner, keyed by the
/// predictor's **name** — and this crate renames every predictor to its node id on the way out.
/// A key that does not match writes nothing, and GEPA reports success having rewritten nothing:
/// the search runs, candidates win on the valset, and the student is handed back its seed.
///
/// It also pins the attribution this crate no longer supplies. `Graph` hand-wrote
/// `forward_traced` for a while because the trait's default dropped the trace; dsrust now records
/// each `Predict` call ambiently and names it from the walk, so the override was doing again what
/// the engine does. This test and the demo one above are what make that safe to rely on: against
/// an engine that stops attributing, both fail here rather than turning every optimize into a
/// silent no-op.
mod gepa_writes_back {
    use std::sync::Arc;

    use dsrust::anyhow::Result;
    use dsrust::lm::{ChatModel, DynChatModel, LmResponse, api};
    use dsrust::{Example, Feedback, GEPA, MetricContext, Module, Prediction, example};
    use dsrust_graph::{Graph, GraphDocument};

    const PROPOSAL: &str = "Answer with the single word Paris.";

    /// Answers correctly only when the instruction it was given carries the proposal, so the
    /// proposed program outscores the seed and the search has a reason to accept it.
    struct Task;

    impl ChatModel for Task {
        async fn forward(&self, request: &api::LmRequest) -> Result<LmResponse> {
            let asked: String = request
                .messages
                .iter()
                .filter_map(|message| message.text())
                .collect::<Vec<_>>()
                .join("\n");
            // Which node is asking, read off its declared output field rather than
            // off any word in the prompt — the proposal travels in the prompt too.
            let writing = asked.contains("Your output fields are:\n1. `haiku`");
            let (field, value) = if writing {
                (
                    "haiku",
                    if asked.contains(PROPOSAL) {
                        "Paris"
                    } else {
                        "wrong"
                    },
                )
            } else {
                ("angle", "an angle")
            };
            Ok(LmResponse::text(format!(
                "[[ ## {field} ## ]]\n{value}\n\n[[ ## completed ## ]]"
            )))
        }
    }

    /// Proposes the winning instruction, fenced the way the reflection tree expects.
    struct Reflector;

    impl ChatModel for Reflector {
        async fn forward(&self, _request: &api::LmRequest) -> Result<LmResponse> {
            Ok(LmResponse::text(format!("```\n{PROPOSAL}\n```")))
        }
    }

    fn metric(gold: &Example, pred: &Prediction, _: &MetricContext<'_>) -> Feedback {
        if gold.get("haiku") == pred.get("haiku") {
            Feedback::new(1.0, "Correct.")
        } else {
            Feedback::new(0.0, "Wrong haiku.")
        }
    }

    #[tokio::test]
    async fn a_gepa_compile_rewrites_the_graphs_instructions() {
        let task = Arc::new(Task) as Arc<dyn DynChatModel>;
        let mut graph =
            Graph::from_document(&GraphDocument::example(), task).expect("the document builds");
        let trainset = vec![
            example! { subject: "winter mornings", haiku: "Paris" }.with_inputs(["subject"]),
            example! { subject: "late trains", haiku: "Paris" }.with_inputs(["subject"]),
        ];

        GEPA::new(metric, Arc::new(Reflector))
            .max_metric_calls(40)
            .reflection_minibatch_size(2)
            .compile(&mut graph, &trainset, &trainset)
            .await
            .expect("compiles");

        let instructions: Vec<String> = graph
            .named_predictors()
            .iter()
            .map(|found| found.signature.instructions.clone())
            .collect();
        assert!(
            instructions.iter().any(|text| text == PROPOSAL),
            "GEPA reported success having rewritten nothing — the graph still holds {instructions:?}"
        );
    }
}

/// Rewriting a node's instruction changes what that node sends. The link between the walk and the
/// forward — without it an optimizer's every candidate runs as the seed, ties with its parent, and
/// is dropped.
#[tokio::test]
async fn a_rewritten_instruction_reaches_the_prompt() {
    use std::sync::Mutex;

    #[derive(Default)]
    struct Recording(Mutex<Vec<String>>);

    impl dsrust::lm::ChatModel for Recording {
        async fn forward(
            &self,
            request: &dsrust::lm::api::LmRequest,
        ) -> dsrust::anyhow::Result<dsrust::lm::LmResponse> {
            let asked: String = request
                .messages
                .iter()
                .filter_map(|message| message.text())
                .collect::<Vec<_>>()
                .join("\n");
            self.0.lock().expect("lock").push(asked);
            Ok(dsrust::lm::LmResponse::text(
                "[[ ## angle ## ]]\nx\n\n[[ ## completed ## ]]",
            ))
        }
    }

    let lm = Arc::new(Recording::default());
    let mut graph = Graph::from_document(
        &GraphDocument::example(),
        Arc::clone(&lm) as Arc<dyn DynChatModel>,
    )
    .expect("the document builds");

    for found in graph.named_predictors() {
        found.signature.instructions = "REWRITTEN".to_owned();
    }
    let _ = graph
        .forward(Example::new([("subject", json!("winter mornings"))]))
        .await;

    let asked = lm.0.lock().expect("lock").join("\n");
    assert!(
        asked.contains("REWRITTEN"),
        "the rewritten instruction never reached the model, so an optimizer's candidates all run \
         as the seed"
    );
}

/// A saved state for a differently-shaped program is named, not zipped on.
///
/// dspy's restore stops at the shorter of the two field lists on purpose, which is safe where a
/// program's shape is its source. Here a node keeps its id while changing kind — a `Predict` that
/// becomes a `ChainOfThought` grows a leading `reasoning` output — so every saved field lands one
/// position late and the prompt renders the next field's prefix over the reasoning.
#[test]
fn a_state_for_another_shape_is_named() {
    use dsrust::module::ProgramState;

    let mut graph = graph();
    let fitting: ProgramState = dsrust::serde_json::from_value(dsrust::serde_json::json!({
        "plan": {
            "traces": [], "train": [], "demos": [],
            "signature": { "instructions": "Pick one.", "fields": [
                { "prefix": "Subject:", "description": "${subject}" },
                { "prefix": "Angle:", "description": "${angle}" }
            ] }
        },
        "write": {
            "traces": [], "train": [], "demos": [],
            "signature": { "instructions": "Write it.", "fields": [
                { "prefix": "Angle:", "description": "${angle}" },
                { "prefix": "Haiku:", "description": "${haiku}" }
            ] }
        }
    }))
    .expect("the state parses");
    assert_eq!(
        graph.foreign_state(&fitting),
        None,
        "a state that fits is not foreign"
    );

    let mut short: dsrust::serde_json::Value =
        dsrust::serde_json::to_value(&fitting).expect("round-trips");
    short["write"]["signature"]["fields"] = dsrust::serde_json::json!([
        { "prefix": "Haiku:", "description": "${haiku}" }
    ]);
    let short: ProgramState = dsrust::serde_json::from_value(short).expect("parses");
    let complaint = graph.foreign_state(&short).expect("the shape disagrees");
    assert!(complaint.contains("write"), "{complaint}");
    assert!(complaint.contains("2 fields"), "{complaint}");
}
