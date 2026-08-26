//! COPRO against dspy's own, replaying an identical trace through both.
//!
//! Each case in `tests/conformance/optimize/copro.json` was produced by running dspy's COPRO
//! against a keyed `DummyLM` (see `scripts/generate_copro_fixture.py`). Here the crate's COPRO runs
//! against the same table, and every prompt it produces is compared to dspy's, in order, along with
//! the instruction it compiles to. A divergence is a bug in this crate until dspy is shown wrong —
//! it means the two optimizers made a different decision somewhere in the loop.
//!
//! The two-predictor case is the one that earns its keep: `all_candidates` accumulates across
//! rounds only when a program has more than one predictor, and the winning program is a single
//! snapshot, so the second predictor keeps its original instruction — the snapshot that scored
//! highest was taken before the second was ever changed.

use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

use super::{COPRO, CoproStats, DepthScores};
use crate::evaluate::exact_match;
use crate::example::{Example, Prediction};
use crate::lm::dummy::Asked;
use crate::module::{Forward, Module};
use crate::predict::Predict;
use crate::signature::Signature;
use crate::{DummyLM, input};

/// dspy's two-predictor `Pair`: the first drafts an answer, the second settles it. The composition
/// is what makes the second predictor read a `draft`, so a demo or instruction one earns is one the
/// other could not have.
#[derive(dsrust::Module)]
struct Pair {
    first: Predict,
    second: Predict,
}

impl Forward for Pair {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let drafted = self.first.forward(inputs).await?;
        let draft = drafted.get("draft").cloned().unwrap_or_default();
        self.second.forward(input! { draft: draft }).await
    }
}

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/optimize/copro.json");
    let text = std::fs::read_to_string(&path).expect("the copro golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

/// An example built from a fixture object, declaring `question` as its one input where present so a
/// trainset row scores and a keyed answer does not.
fn example(object: &Value) -> Example {
    let fields = object
        .as_object()
        .expect("object")
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()));
    let example = Example::new(fields);
    match object.get("question") {
        Some(_) => example.with_inputs(["question"]),
        None => example,
    }
}

/// The keyed model dspy was run against: each key answers whichever call carries it.
fn model(case: &Value) -> Arc<DummyLM> {
    let pairs = case["keyed"]
        .as_array()
        .expect("keyed")
        .iter()
        .map(|entry| {
            let key = entry["key"].as_str().expect("key").to_owned();
            (key, example(&entry["fields"]))
        });
    Arc::new(DummyLM::keyed(pairs))
}

fn predict(signature: &str, instruction: &str, model: Arc<DummyLM>) -> Predict {
    let mut signature: Signature = signature.parse().expect("parses");
    signature.instructions = instruction.to_owned();
    Predict::from_signature(signature).set_lm(model)
}

/// The student dspy compiled: a lone `Predict` for a `question -> answer` task, or the two-predictor
/// `Pair`, each carrying the case's starting instructions and answering through the shared model.
fn build(case: &Value, model: Arc<DummyLM>) -> Box<dyn Module> {
    let instructions = case["instructions"].as_array().expect("instructions");
    let text = |index: usize| instructions[index].as_str().expect("instruction");
    match case["module"].as_str().expect("module") {
        "predict" => Box::new(predict("question -> answer", text(0), model)),
        "pair" => Box::new(Pair {
            first: predict("question -> draft", text(0), model.clone()),
            second: predict("draft -> answer", text(1), model),
        }),
        other => panic!("the golden names a module the harness does not know: {other}"),
    }
}

/// Show the first difference rather than two walls of text.
fn assert_prompt(label: &str, expected: &str, actual: &str) {
    if expected == actual {
        return;
    }
    let at = expected
        .char_indices()
        .zip(actual.char_indices())
        .find(|((_, want), (_, got))| want != got)
        .map(|((index, _), _)| index)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    panic!(
        "{label} diverges from dspy\n  first difference at byte {at}\n\n  expected: {:?}\n  actual:   {:?}\n",
        &expected[at.saturating_sub(40)..(at + 60).min(expected.len())],
        &actual[at.saturating_sub(40)..(at + 60).min(actual.len())],
    );
}

/// Every prompt the crate produced against the trace dspy recorded — a task call's system carries
/// the instruction in force, a depth call's user carries the attempts, so matching both in order is
/// matching the whole loop.
fn assert_calls(case: &Value, asked: &[Asked]) {
    let expected = case["calls"].as_array().expect("calls");
    assert_eq!(
        asked.len(),
        expected.len(),
        "case {:?} made {} calls, dspy made {}",
        case["instructions"],
        asked.len(),
        expected.len()
    );
    for (index, (got, want)) in asked.iter().zip(expected).enumerate() {
        assert_prompt(
            &format!("system of call {index}"),
            want["system"].as_str().expect("system"),
            got.system(),
        );
        assert_prompt(
            &format!("user of call {index}"),
            want["user"].as_str().expect("user"),
            &got.last_message(),
        );
    }
}

fn compiled_instructions(module: &mut dyn Module) -> Vec<String> {
    module
        .named_predictors()
        .iter()
        .map(|predictor| predictor.signature.instructions.clone())
        .collect()
}

/// The five numbers a depth is summarised by, against Python's own `statistics`.
///
/// Separate from the run comparison because a run cannot discriminate them: the keyed model answers
/// by question, so every candidate at a depth scores the same and every recorded `std` is `0.0`
/// with `min == max`. A fixture built only from runs agrees by accident — it cannot tell `pstdev`
/// from the sample deviation, `average` from `max`, or a top-ten slice from taking everything.
/// These cases put spread in on purpose, including more than ten scores and ties across the cut.
#[test]
fn a_depth_is_summarised_the_way_python_summarises_it() {
    let fixture = fixture();
    let rows = fixture["summaries"].as_array().expect("summaries");
    assert!(!rows.is_empty(), "the golden records no summaries");
    for row in rows {
        let name = row["name"].as_str().expect("a name");
        let scores: Vec<f64> = row["scores"]
            .as_array()
            .expect("scores")
            .iter()
            .map(|value| value.as_f64().expect("a score"))
            .collect();

        let ours = DepthScores::of(0, &scores).expect("a non-empty list");
        assert_summary(name, "whole", &ours, &row["summary"]);

        // The other rule a run never reaches: `results_best` summarises the top ten of a
        // descending sort, so a set of fifteen is cut and ties across the cut keep their order.
        let mut sorted = scores.clone();
        let top = CoproStats::top_ten(&mut sorted);
        let theirs: Vec<f64> = row["top_ten"]
            .as_array()
            .expect("top_ten")
            .iter()
            .map(|value| value.as_f64().expect("a score"))
            .collect();
        assert_eq!(top, theirs.as_slice(), "{name}: the top ten themselves");
        let ours = DepthScores::of(0, top).expect("a non-empty list");
        assert_summary(name, "top_ten", &ours, &row["top_ten_summary"]);
    }
}

fn assert_summary(name: &str, which: &str, ours: &DepthScores, theirs: &Value) {
    for (label, mine, dspys) in [
        ("max", ours.max, theirs["max"].as_f64().expect("max")),
        (
            "average",
            ours.average,
            theirs["average"].as_f64().expect("average"),
        ),
        ("min", ours.min, theirs["min"].as_f64().expect("min")),
        ("std", ours.std, theirs["std"].as_f64().expect("std")),
    ] {
        assert!(
            (mine - dspys).abs() < 1e-12,
            "{name} ({which}) {label}: {mine} is not Python's {dspys}"
        );
    }
}

/// dspy's `track_stats` block, compared against the numbers it recorded for the same run.
///
/// The dicts upstream keys by `id(predictor)` are re-keyed to predictor position when the fixture
/// is written — the ids belong to a `student.deepcopy()` that is never returned, so order is the
/// only thing that crosses. `std` is `pstdev`, so a single-candidate depth is `0.0` rather than
/// undefined, and the scores are percentages because that is what `Evaluate.score` reports.
fn assert_tracked(case: &Value, stats: &CoproStats) {
    let recorded = &case["stats"];
    assert_eq!(
        stats.total_calls,
        recorded["total_calls"].as_u64().expect("total_calls") as usize,
        "total_calls for case {:?}",
        case["instructions"]
    );
    for (kind, ours) in [
        ("results_best", &stats.best),
        ("results_latest", &stats.latest),
    ] {
        let theirs = recorded[kind].as_array().expect("a list per predictor");
        assert_eq!(ours.len(), theirs.len(), "{kind}: predictors");
        for (predictor, (mine, dspys)) in ours.iter().zip(theirs).enumerate() {
            let rows = dspys.as_array().expect("a list per depth");
            assert_eq!(
                mine.len(),
                rows.len(),
                "{kind}: depths for predictor {predictor}"
            );
            for (at, (summary, row)) in mine.iter().zip(rows).enumerate() {
                let named = |key: &str| row[key].as_f64().expect("a number");
                assert_eq!(
                    summary.depth as f64,
                    named("depth"),
                    "{kind}[{predictor}][{at}] depth"
                );
                for (label, ours, theirs) in [
                    ("max", summary.max, named("max")),
                    ("average", summary.average, named("average")),
                    ("min", summary.min, named("min")),
                    ("std", summary.std, named("std")),
                ] {
                    assert!(
                        (ours - theirs).abs() < 1e-9,
                        "{kind}[{predictor}][{at}] {label}: {ours} is not dspy's {theirs}"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn copro_makes_the_decisions_dspy_makes() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the golden records no cases");
    for case in cases {
        let model = model(case);
        let trainset: Vec<Example> = case["trainset"]
            .as_array()
            .expect("trainset")
            .iter()
            .map(example)
            .collect();
        let mut module = build(case, model.clone());

        let stats = COPRO::new(exact_match)
            .breadth(case["breadth"].as_u64().expect("breadth") as usize)
            .depth(case["depth"].as_u64().expect("depth") as usize)
            .prompt_model(model.clone())
            .compile_traced(module.as_mut(), &trainset)
            .await
            .expect("compiles");

        assert_calls(case, &model.asked());
        assert_tracked(case, &stats);
        let expected: Vec<String> = case["final"]
            .as_array()
            .expect("final")
            .iter()
            .map(|value| value.as_str().expect("a final instruction").to_owned())
            .collect();
        assert_eq!(
            compiled_instructions(module.as_mut()),
            expected,
            "compiled instructions for case {:?}",
            case["instructions"]
        );
    }
}
