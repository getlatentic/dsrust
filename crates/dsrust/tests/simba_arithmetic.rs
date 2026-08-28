//! numpy's percentile and Python's rounding, against their own answers.
//!
//! SIMBA takes the 10th and 90th percentile of a mini-batch's scores and hands both to the rule
//! proposer as prompt text, and it picks its final candidate slate with `round(i * M / (N - 1))`.
//! Rust's `f64::round` disagrees with Python's on every half — away from zero against toward the
//! even neighbour — so `round(0.5)` is 1 here and 0 there, and a different program is returned.

use dsrust::optimize::simba::arithmetic;
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("conformance/optimize/simba_arithmetic.json"))
        .expect("the golden parses")
}

#[test]
fn the_percentiles_are_numpys() {
    for case in golden()["percentile"].as_array().expect("percentiles") {
        let sample: Vec<f64> = case["sample"]
            .as_array()
            .expect("a sample")
            .iter()
            .map(|value| value.as_f64().expect("a score"))
            .collect();
        let q = case["q"].as_f64().expect("a quantile");
        let theirs = case["value"].as_f64().expect("numpy's answer");
        let ours = arithmetic::percentile(&sample, q).expect("a non-empty sample");
        assert!(
            (ours - theirs).abs() < 1e-12,
            "percentile({sample:?}, {q}) is {ours}, numpy says {theirs}"
        );
    }
}

/// Every half, both signs. Rust's `f64::round` fails half of these.
#[test]
fn rounding_breaks_ties_toward_even_as_pythons_does() {
    let fixture = golden();
    let cases = fixture["round"].as_array().expect("round cases");
    let mut halves = 0;
    for case in cases {
        let x = case["x"].as_f64().expect("an input");
        let theirs = case["rounded"].as_i64().expect("Python's answer");
        assert_eq!(
            arithmetic::round_half_to_even(x),
            theirs,
            "round({x}) — Rust's own would say {}",
            x.round() as i64
        );
        halves += usize::from((x - x.trunc()).abs() == 0.5);
    }
    assert!(
        halves >= 8,
        "the halves are the test, and there are {halves}"
    );
    // And the disagreement is real rather than assumed.
    assert_ne!(arithmetic::round_half_to_even(0.5), 0.5_f64.round() as i64);
}

/// The slate itself, which is what those two combine into.
#[test]
fn the_final_slate_is_the_one_dspy_picks() {
    for case in golden()["final_slate"].as_array().expect("slates") {
        let winners = case["winners"].as_u64().expect("winners") as usize;
        let candidates = case["num_candidates"].as_u64().expect("candidates") as usize;
        let theirs: Vec<usize> = case["indices"]
            .as_array()
            .expect("indices")
            .iter()
            .map(|index| index.as_u64().expect("an index") as usize)
            .collect();
        assert_eq!(
            arithmetic::final_slate(winners, candidates),
            theirs,
            "slate for {winners} winners and {candidates} candidates"
        );
    }
}
