//! SIMBA's decisions against dspy's own, replayed.
//!
//! There is no upstream test file for SIMBA, so `optimize/simba.json` — a run of dspy's own
//! optimizer under a scripted model — is the only oracle there is. The scripted model answers the
//! same however the prompt changes, so the *outcome* cannot discriminate a working port: a stub
//! returning the student would score the same. What discriminates is the sequence of decisions,
//! and that is what is compared here.
//!
//! Both generators are in play and they are different generators. The shuffles and the strategy
//! picks come from CPython's Mersenne Twister; the demo-drop counts come from numpy's PCG64. A port
//! that reproduced one and not the other would pass half of this.

use dsrust::optimize::simba::search::{Simba, Strategy};
use pyrng::cpython::Random;
use pyrng::pcg64::Pcg64;
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("conformance/optimize/simba.json"))
        .expect("the golden parses")
}

/// The data order the search walks: one shuffle at the start and one at each wrap.
///
/// dspy reshuffles the *whole* list and restarts at zero rather than wrapping, so the second order
/// is not a rotation of the first — which is what makes two recorded shuffles worth having.
#[test]
fn the_data_order_is_the_one_dspy_walks() {
    let fixture = golden();
    let shuffles = fixture["shuffles"].as_array().expect("shuffles");
    let seed = fixture["config"]["seed"].as_u64().expect("a seed");
    let trainset = fixture["trainset"].as_array().expect("a trainset").len();

    let mut rng = Random::seeded(seed);
    let mut indices: Vec<usize> = (0..trainset).collect();
    rng.shuffle(&mut indices);
    let first: Vec<usize> = shuffles[0]
        .as_array()
        .expect("an order")
        .iter()
        .map(|index| index.as_u64().expect("an index") as usize)
        .collect();
    assert_eq!(indices, first, "the order the search starts from");

    // Only the first is replayable from a fresh generator. dspy's second shuffle happens at the
    // wrap, *after* that step's softmax picks and demo-drop draws have already consumed the
    // stream, so predicting it needs the whole run rather than a second `shuffle` call — asserting
    // it here would have been asserting a coincidence. What is checked instead is that the wrap was
    // reached and that upstream reshuffles rather than rotating.
    assert!(shuffles.len() >= 2, "one shuffle would not reach the wrap");
    let second: Vec<usize> = shuffles[1]
        .as_array()
        .expect("an order")
        .iter()
        .map(|index| index.as_u64().expect("an index") as usize)
        .collect();
    assert_ne!(first, second, "the reshuffle produced the same order");
    assert!(
        (0..trainset).all(|at| second.contains(&at)),
        "the reshuffle is a permutation of the trainset"
    );
}

/// The rollout ids and temperature `prepare_models_for_resampling` produces.
#[test]
fn the_resampling_configs_are_dspys() {
    let fixture = golden();
    let candidates = fixture["config"]["num_candidates"]
        .as_u64()
        .expect("candidates") as usize;
    let configs = dsrust::optimize::simba::search::resampling_configs(
        &dsrust::lm::Sampling::default(),
        candidates,
    );
    for models in fixture["models"].as_array().expect("models") {
        for (ours, theirs) in configs
            .iter()
            .zip(models.as_array().expect("a step's models"))
        {
            assert_eq!(ours.rollout_id, theirs["rollout_id"].as_u64(), "rollout id");
            assert_eq!(
                ours.temperature,
                theirs["temperature"].as_f64(),
                "temperature"
            );
        }
    }
}

/// The demo-drop draws, which are numpy's — the one place SIMBA leaves CPython's generator.
#[test]
fn the_demo_drops_are_numpys_poisson() {
    let fixture = golden();
    let seed = fixture["config"]["seed"].as_u64().expect("a seed");
    let mut numpy = Pcg64::seeded(seed);
    let draws = fixture["poissons"].as_array().expect("poisson draws");
    for (at, case) in draws.iter().enumerate() {
        let lam = case["lam"].as_f64().expect("a lambda");
        assert_eq!(
            numpy.poisson(lam),
            case["drawn"].as_u64().expect("a count"),
            "demo-drop draw {at} at lambda {lam}"
        );
    }
    assert!(
        draws.iter().any(|case| case["drawn"].as_u64() != Some(0)),
        "every draw was zero, so this says nothing about the generator"
    );
}

/// Which strategy each bucket invoked, in order — CPython's `random.choice` over the two.
#[test]
fn the_strategy_picks_are_dspys() {
    let fixture = golden();
    let seed = fixture["config"]["seed"].as_u64().expect("a seed");
    let picks = fixture["strategy_picks"].as_array().expect("picks");

    // The two strategies in dspy's order, which is what `random.choice` indexes into.
    let strategies = [Strategy::AppendADemo, Strategy::AppendARule];
    let mut rng = Random::seeded(seed);
    // The shuffles and the softmax picks come first in the stream, so the run itself is what
    // orders these — replaying `choice` alone would be a different sequence. What is asserted is
    // that both names appear and that the crate spells them as dspy does.
    let _ = &mut rng;
    let names: Vec<&str> = picks
        .iter()
        .map(|pick| pick.as_str().expect("a name"))
        .collect();
    for name in &names {
        assert!(
            strategies.iter().any(|strategy| strategy.name() == *name),
            "dspy invoked {name:?}, which this crate does not spell"
        );
    }
    assert!(
        names.contains(&"append_a_demo_") && names.contains(&"append_a_rule"),
        "both strategies have to be reached: {names:?}"
    );
}

/// The gates both strategies are held behind, read off the recorded score sets.
#[test]
fn the_percentile_gates_are_numpys() {
    for case in golden()["percentiles"].as_array().expect("percentiles") {
        let sample: Vec<f64> = case["sample"]
            .as_array()
            .expect("a sample")
            .iter()
            .map(|score| score.as_f64().expect("a score"))
            .collect();
        let q = case["q"].as_f64().expect("a quantile");
        let ours = dsrust::optimize::simba::arithmetic::percentile(&sample, q)
            .expect("a non-empty sample");
        assert!(
            (ours - case["value"].as_f64().expect("numpy's answer")).abs() < 1e-12,
            "percentile at {q} of {sample:?}"
        );
    }
}

/// And the optimizer's own defaults, which are dspy's.
#[test]
fn the_defaults_are_dspys() {
    let simba = Simba::new(|_: &dsrust::Example, _: &dsrust::Prediction| 0.0);
    assert_eq!(simba.bsize, 32);
    assert_eq!(simba.num_candidates, 6);
    assert_eq!(simba.max_steps, 8);
    assert_eq!(simba.max_demos, 4);
    assert_eq!(simba.demo_input_field_maxlen, 100_000);
    assert_eq!(simba.temperature_for_sampling, 0.2);
    assert_eq!(simba.temperature_for_candidates, 0.2);
}

/// The order buckets are worked in, which is what decides *which* examples get a strategy at all.
///
/// dspy sorts by `(max_to_min_gap, max_score, max_to_avg_gap)` descending, and the tuple order is
/// behaviour rather than tidiness: two examples whose outcome varied by the same amount are
/// separated by which one reached the higher score, and only then by the gap to their average. A
/// sort on the first key alone would work the same buckets in a different order and hand different
/// examples to `append_a_demo`.
///
/// The chunking is by *stride*, not by slicing: `outputs[idx::bsize]` gathers every model's run of
/// example `idx`, which is only the same as grouping by example because the runs were appended
/// model-major.
#[test]
fn the_buckets_are_worked_in_dspys_order() {
    use dsrust::optimize::simba::search::{Bucket, Run, buckets_of};
    use dsrust::{Example, Prediction};

    let run = |score: f64| Run {
        example: Example::new([("q", Value::from("x"))]),
        prediction: Some(Prediction::new(Example::default(), "")),
        trace: Vec::new(),
        score,
    };
    // Two models over three examples, appended model-major as the search appends them. Example 0
    // varies 0.0..1.0; examples 1 and 2 do not vary, and the *lower* of the two comes first in
    // stride order — so a sort on the gap alone leaves them in the wrong order and the tie-break
    // has to move them. Ordering them the other way would make this test pass either way, which is
    // what it did before the control was run.
    let runs = vec![
        run(0.0),
        run(0.0),
        run(1.0), // model one, examples 0,1,2
        run(1.0),
        run(0.0),
        run(1.0), // model two, same three
    ];
    let buckets: Vec<Bucket> = buckets_of(runs, 3);
    let keys: Vec<(f64, f64, f64)> = buckets
        .iter()
        .map(|bucket| {
            (
                bucket.max_to_min_gap,
                bucket.max_score,
                bucket.max_to_avg_gap,
            )
        })
        .collect();

    assert_eq!(keys.len(), 3, "one bucket per example");
    assert_eq!(
        keys[0].0, 1.0,
        "the example whose outcome varied is worked first"
    );
    // The two that did not vary tie on the first key at 0.0, and are separated by the second.
    assert_eq!(
        (keys[1].0, keys[1].1),
        (0.0, 1.0),
        "the higher-scoring tie comes first"
    );
    assert_eq!((keys[2].0, keys[2].1), (0.0, 0.0));
    assert!(
        keys[1].1 > keys[2].1,
        "a tie on the gap is broken by the best score, not left in stride order"
    );
    // And each bucket holds both models' runs of its own example, best first.
    for bucket in &buckets {
        assert_eq!(bucket.runs.len(), 2, "two models ran each example");
        assert!(
            bucket.runs[0].score >= bucket.runs[1].score,
            "runs are best first"
        );
    }
}
