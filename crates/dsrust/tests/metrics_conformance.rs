//! The answer metrics against dspy's own, value for value.
//!
//! `normalize_text` is what every metric agrees on, and it is where a port drifts quietly: the NFD
//! pass, the ASCII-only punctuation set, the `\b(a|an|the)\b` article regex and the whitespace
//! collapse each have an edge a reimplementation rounds off. The golden
//! (`tests/conformance/evaluate/metrics.json`, see `scripts/generate_metrics_fixture.py`) is what
//! upstream returned for inputs chosen at exactly those edges — accented text NFD splits, Unicode
//! punctuation the ASCII set does *not* strip, articles against word boundaries, the HotPotQA
//! labels, repeated tokens, and both sides empty.

use dsrust::evaluate::dpr;
use dsrust::evaluate::metrics::{
    answer_passage_match, em, em_score, f1, f1_score, hotpot_f1, hotpot_f1_score, normalize_text,
    precision_score,
};
use dsrust::example::{Example, Prediction};
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
        assert_eq!(
            em(&prediction, &answers),
            case["em"].as_bool().expect("em"),
            "{}",
            named("EM")
        );
        for (ours, key) in [
            (f1(&prediction, &answers), "f1"),
            (hotpot_f1(&prediction, &answers), "hotpot_f1"),
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

/// dspy's tokenizer, token for token, where a hand-written scanner would drift from its regex.
///
/// A run of letters, numbers and marks is one token and everything else printable is a token on
/// its own, so `北京市` is a single token and `a-b_c` is five. Separators and every "other" fall
/// out — a zero-width joiner, a private-use character and an unassigned codepoint leave no token
/// behind. Case is decided per token, which is why Greek keeps `ς` at a word's end but not alone.
#[test]
fn it_tokenises_the_passage_dspy_tokenises() {
    for case in golden()["dpr_normalize"].as_array().expect("cases") {
        let input = text(case, "text");
        let expected: Vec<String> = case["tokens"]
            .as_array()
            .expect("tokens")
            .iter()
            .map(|token| token.as_str().expect("a token").to_owned())
            .collect();
        assert_eq!(dpr::normalize(&input), expected, "DPR_normalize({input:?})");
    }
}

/// `answer_passage_match` over the passages, against what upstream scored.
///
/// Containment is asked of tokens, so two of these separate it from a substring search in opposite
/// directions: `北京市` is *not* in `北京市中心` because the run is one token, and `the` *is* in
/// `theatre` because the shared `normalize_text` strips the article first and an empty answer
/// matches anything.
#[test]
fn it_matches_the_passages_dspy_matches() {
    for case in golden()["passage_match"].as_array().expect("cases") {
        let example = Example::new([("answer", case["answer"].clone())]);
        let prediction = Prediction::new(
            Example::new([("context", case["context"].clone())]),
            String::new(),
        );
        let scored = case["score"].as_f64().expect("a score");
        assert_eq!(
            answer_passage_match(&example, &prediction),
            scored,
            "answer_passage_match({:?}, {:?})",
            case["answer"],
            case["context"]
        );
    }
}
