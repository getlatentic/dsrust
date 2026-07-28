//! The answer metrics against dspy's own, value for value.
//!
//! `normalize_text` is what every metric agrees on, and it is where a port drifts quietly: the NFD
//! pass, the ASCII-only punctuation set, the `\b(a|an|the)\b` article regex and the whitespace
//! collapse each have an edge a reimplementation rounds off. The golden
//! (`tests/conformance/evaluate/metrics.json`, see `scripts/generate_metrics_fixture.py`) is what
//! upstream returned for inputs chosen at exactly those edges — accented text NFD splits, Unicode
//! punctuation the ASCII set does *not* strip, articles against word boundaries, the HotPotQA
//! labels, repeated tokens, and both sides empty.

use dsrust::evaluate::metrics::{
    em, em_score, f1, f1_score, hotpot_f1, hotpot_f1_score, normalize_text, precision_score,
};
use serde_json::Value;

fn golden() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/evaluate/metrics.json");
    let text = std::fs::read_to_string(&path).expect("the metrics golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

fn text(case: &Value, key: &str) -> String {
    case[key].as_str().expect(key).to_owned()
}

/// Every normalisation upstream recorded, character for character.
#[test]
fn it_normalises_the_text_dspy_normalises() {
    for case in golden()["normalize_text"].as_array().expect("cases") {
        let input = text(case, "text");
        assert_eq!(
            normalize_text(&input),
            text(case, "normalized"),
            "normalize_text({input:?})"
        );
    }
}

/// The pairwise metrics: exact match, token F1, the HotPotQA variant, and precision.
#[test]
fn it_scores_the_pairs_dspy_scores() {
    for case in golden()["pairs"].as_array().expect("cases") {
        let (prediction, truth) = (text(case, "prediction"), text(case, "truth"));
        let named = |metric| format!("{metric}({prediction:?}, {truth:?})");
        assert_eq!(
            em_score(&prediction, &truth),
            case["em_score"].as_bool().expect("em_score"),
            "{}",
            named("em_score")
        );
        for (ours, key) in [
            (f1_score(&prediction, &truth), "f1_score"),
            (hotpot_f1_score(&prediction, &truth), "hotpot_f1_score"),
            (precision_score(&prediction, &truth), "precision_score"),
        ] {
            let theirs = case[key].as_f64().expect(key);
            assert!(
                (ours - theirs).abs() < 1e-12,
                "{}: {ours} != {theirs}",
                named(key)
            );
        }
    }
}

/// The max-over-references metrics, which is how a metric grades an example with several
/// acceptable answers.
#[test]
fn it_scores_the_answer_sets_dspy_scores() {
    for case in golden()["sets"].as_array().expect("cases") {
        let prediction = text(case, "prediction");
        let answers: Vec<String> = case["answers"]
            .as_array()
            .expect("answers")
            .iter()
            .map(|answer| answer.as_str().expect("an answer").to_owned())
            .collect();
        let named = |metric| format!("{metric}({prediction:?}, {answers:?})");
        assert_eq!(em(&prediction, &answers), case["em"].as_bool().expect("em"), "{}", named("EM"));
        for (ours, key) in
            [(f1(&prediction, &answers), "f1"), (hotpot_f1(&prediction, &answers), "hotpot_f1")]
        {
            let theirs = case[key].as_f64().expect(key);
            assert!((ours - theirs).abs() < 1e-12, "{}: {ours} != {theirs}", named(key));
        }
    }
}
