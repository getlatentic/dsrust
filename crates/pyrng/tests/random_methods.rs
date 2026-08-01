//! Every derived `random.Random` method against CPython's own answers.
//!
//! `pyrng` pinned the raw Mersenne-Twister stream and nothing built on it: a mutation run replaced
//! `random()`, `randint()` and `choice_index()` each with a **constant** and no test failed. Every
//! check that would have caught it lived in a consumer crate, and `pyrng` is published on its own.
//!
//! The calls are **interleaved against one generator per seed** rather than each taken from a fresh
//! one, because how far a call advances the stream matters as much as what it returns. A method that
//! answers correctly after consuming two words where CPython consumes one is right once and wrong
//! for the rest of the run — and every optimizer in this workspace reads a shared generator, so that
//! error surfaces as a different program compiled, far from its cause.

use pyrng::Random;
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/random_methods.json");
    let text = std::fs::read_to_string(&path).expect("the golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

fn numbers(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(|item| item.as_u64().expect("a number") as usize)
        .collect()
}

#[test]
fn every_derived_method_answers_what_cpython_answers() {
    let fixture = fixture();
    let runs = fixture["runs"].as_array().expect("runs");
    assert!(!runs.is_empty(), "the golden records no runs");

    for run in runs {
        let seed = run["seed"].as_u64().expect("seed");
        let mut rng = Random::seeded(seed);
        for (index, step) in run["steps"].as_array().expect("steps").iter().enumerate() {
            let call = step["call"].as_str().expect("a call");
            let expected = &step["value"];
            let where_ = format!("seed {seed}, step {index}: {call}");
            let words: Vec<&str> = call.split(' ').collect();
            match words.as_slice() {
                ["random"] => {
                    // Compared through Python's own repr, so a value differing in the last bit is a
                    // failure rather than something an epsilon would hide.
                    let ours = format!("{:?}", rng.random());
                    assert_eq!(ours, expected.as_str().expect("a float"), "{where_}");
                }
                ["randint", low, high] => {
                    let drawn = rng.randint(low.parse().expect("low"), high.parse().expect("high"));
                    assert_eq!(drawn, expected.as_u64().expect("an int"), "{where_}");
                }
                ["choice_index", size] => {
                    let drawn = rng.choice_index(size.parse().expect("size"));
                    assert_eq!(
                        drawn as u64,
                        expected.as_u64().expect("an index"),
                        "{where_}"
                    );
                }
                ["choices", weights, k] => {
                    let weights: Vec<f64> =
                        serde_json::from_str(weights).expect("the weights parse");
                    let drawn = rng.choices(&weights, k.parse().expect("k"));
                    assert_eq!(drawn, numbers(expected), "{where_}");
                }
                ["sample", n, k] => {
                    let n: usize = n.parse().expect("n");
                    let population: Vec<usize> = (0..n).collect();
                    let drawn = rng.sample(&population, k.parse().expect("k"));
                    assert_eq!(drawn, numbers(expected), "{where_}");
                }
                ["shuffle", n] => {
                    let mut items: Vec<usize> = (0..n.parse().expect("n")).collect();
                    rng.shuffle(&mut items);
                    assert_eq!(items, numbers(expected), "{where_}");
                }
                other => panic!("the golden records a call this test does not drive: {other:?}"),
            }
        }
    }
}

/// Two seeds that produced the same run would let a generator ignoring its seed pass everything
/// above. The generator refuses to write such a golden; this keeps a hand-edited one honest.
#[test]
fn the_runs_differ_between_seeds() {
    let fixture = fixture();
    let runs = fixture["runs"].as_array().expect("runs");
    let first = runs[0]["steps"].as_array().expect("steps");
    let second = runs[1]["steps"].as_array().expect("steps");
    let differing = first
        .iter()
        .zip(second)
        .filter(|(a, b)| a["value"] != b["value"])
        .count();
    assert!(
        differing >= first.len() / 2,
        "only {differing} of {} steps differ between the first two seeds",
        first.len()
    );
}
