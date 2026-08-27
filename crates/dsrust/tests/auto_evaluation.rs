//! dspy's built-in LM judges, held to what dspy renders and what its arithmetic returns.
//!
//! `evaluate/auto_evaluation.py` is four signatures and one function, and every part of it is
//! prompt text or a number a metric reports. The instructions come from a class docstring, so
//! their line breaks are `inspect.cleandoc`'s; the field descriptions are sentences a model reads;
//! and `f1_score` clamps both arguments before taking the harmonic mean, which is the half a
//! reading skips.
//!
//! The golden is the real signatures rendered through the real `ChatAdapter` over a real
//! `ChainOfThought`, so what is compared is the prompt and not a description of it.

use dsrust::evaluate::auto;
use dsrust::signature::Signature;
use dsrust::{Adapter, ChainOfThought, ChatAdapter, Example};
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("conformance/evaluate/auto_evaluation.json"))
        .expect("the golden parses")
}

fn built(name: &str) -> Signature {
    match name {
        "SemanticRecallPrecision" => auto::semantic_recall_precision(),
        "DecompositionalSemanticRecallPrecision" => {
            auto::decompositional_semantic_recall_precision()
        }
        "AnswerCompleteness" => auto::answer_completeness(),
        "AnswerGroundedness" => auto::answer_groundedness(),
        other => panic!("the golden names a signature this crate does not build: {other}"),
    }
}

/// Every field, its description and its declared type, against dspy's own declaration.
#[test]
fn the_judge_signatures_declare_what_dspy_declares() {
    for case in golden()["signatures"].as_array().expect("signatures") {
        let name = case["name"].as_str().expect("a name");
        let ours = built(name);
        assert_eq!(
            ours.instructions,
            case["instructions"].as_str().expect("instructions"),
            "{name}: instructions"
        );
        let inputs: Vec<&str> = ours
            .inputs
            .iter()
            .map(|field| field.name.as_str())
            .collect();
        let theirs: Vec<&str> = case["inputs"]
            .as_array()
            .expect("inputs")
            .iter()
            .map(|field| field["name"].as_str().expect("a name"))
            .collect();
        assert_eq!(inputs, theirs, "{name}: input fields");

        for (ours, theirs) in ours
            .outputs
            .iter()
            .zip(case["outputs"].as_array().expect("outputs"))
        {
            assert_eq!(
                ours.name,
                theirs["name"].as_str().expect("a name"),
                "{name}: output name"
            );
            assert_eq!(
                ours.desc,
                theirs["desc"].as_str().expect("a desc"),
                "{name}: `{}` description",
                ours.name
            );
            assert_eq!(
                ours.kind.annotation(),
                theirs["annotation"].as_str().expect("an annotation"),
                "{name}: `{}` type",
                ours.name
            );
        }
        assert_eq!(
            ours.outputs.len(),
            case["outputs"].as_array().expect("outputs").len(),
            "{name}: output count"
        );
    }
}

/// And the prompt itself: the system message a `ChainOfThought` over each signature renders,
/// compared to dspy's byte for byte.
#[test]
fn the_judges_render_the_prompt_dspy_renders() {
    for case in golden()["signatures"].as_array().expect("signatures") {
        let name = case["name"].as_str().expect("a name");
        let reasoning = ChainOfThought::from_signature(built(name));
        let rendered = ChatAdapter::default()
            .format(reasoning.signature(), &[], &[])
            .expect("the judge signature renders");
        let theirs = case["chain_of_thought_messages"]
            .as_array()
            .expect("messages")[0]["content"]
            .as_str()
            .expect("dspy's system message");
        let ours = rendered
            .first()
            .expect("a system message")
            .text()
            .expect("text");
        assert_eq!(ours, theirs, "{name}: system message");
    }
}

/// The clamp and the zero guard, over dspy's own grid.
#[test]
fn f1_is_clamped_the_way_dspys_is() {
    let fixture = golden();
    let cases = fixture["f1_score"].as_array().expect("f1 cases");
    let mut clamped = 0;
    for case in cases {
        let precision = case["precision"].as_f64().expect("precision");
        let recall = case["recall"].as_f64().expect("recall");
        let theirs = case["f1"].as_f64().expect("f1");
        let ours = auto::f1_score(precision, recall);
        assert!(
            (ours - theirs).abs() < 1e-12,
            "f1_score({precision}, {recall}) is {ours}, dspy says {theirs}"
        );
        clamped += usize::from(!(0.0..=1.0).contains(&precision) || !(0.0..=1.0).contains(&recall));
    }
    assert!(
        clamped >= 20,
        "the out-of-range cases are what make this a test of the clamp, and there are {clamped}"
    );
}

/// The threshold arm — dspy's `score if trace is None else score >= self.threshold`.
#[test]
fn the_bootstrapping_arm_is_the_threshold() {
    let judge = auto::SemanticF1::new();
    assert_eq!(judge.threshold, auto::DEFAULT_THRESHOLD);
    assert!(!judge.accepts(0.65));
    assert!(judge.accepts(0.66), "dspy's arm is `>=`, not `>`");
    assert!(auto::SemanticF1::new().threshold(0.9).accepts(0.9));
    assert!(!auto::SemanticF1::new().threshold(0.9).accepts(0.89));
}

/// A judge is a `Metric`, which is the whole reason that trait exists — a caller can hand one to
/// `Evaluate` or to any optimizer exactly as they would a closure.
#[tokio::test]
async fn a_judge_is_a_metric() {
    fn takes_a_metric<M: dsrust::evaluate::Metric>(_: M) {}
    takes_a_metric(auto::SemanticF1::new());
    takes_a_metric(auto::CompleteAndGrounded::new());
    takes_a_metric(|_: &Example, _: &dsrust::Prediction| 1.0);
}

/// And a *borrowed* judge is one too, which is how an optimizer lends its metric to a scoring pass
/// without giving it up — the case `MetricRef` exists for, and the one a caller writing their own
/// optimizer hits first.
#[tokio::test]
async fn a_borrowed_metric_scores_what_the_owned_one_scores() {
    use dsrust::evaluate::{Metric, MetricRef};

    let judged = |example: &Example, prediction: &dsrust::Prediction| match example.get("answer")
        == prediction.get("answer")
    {
        true => 1.0,
        false => 0.0,
    };
    let example = Example::new([("answer", Value::from("blue"))]);
    let right = dsrust::Prediction::new(Example::new([("answer", Value::from("blue"))]), "");
    let wrong = dsrust::Prediction::new(Example::new([("answer", Value::from("red"))]), "");

    for (prediction, expected) in [(&right, 1.0), (&wrong, 0.0)] {
        assert_eq!(judged.score(&example, prediction).await, expected);
        assert_eq!(
            MetricRef(&judged).score(&example, prediction).await,
            expected,
            "a borrowed metric scores what the owned one scores"
        );
    }
    // And a judge behind it, which is the shape the optimizers pass.
    let judge = auto::SemanticF1::new();
    fn takes_a_metric<M: Metric>(_: M) {}
    takes_a_metric(MetricRef(&judge));
}
