//! optuna's TPE over integer parameters, against whole sequences optuna proposed.
//!
//! Sequences rather than single draws: the pieces are held to their own grids elsewhere, and what
//! this adds is that they compose in optuna's order through one generator whose every draw advances
//! the same stream. A parameter drawn out of turn agrees on the first trial and diverges after.

use serde_json::Value;
use tpe::IntTpeSampler;

fn golden() -> Value {
    serde_json::from_str(include_str!("conformance/int_tpe.json"))
        .expect("the int-TPE golden is valid JSON")
}

/// The objective the golden was generated with: a fixed function of the parameters, so the seed
/// alone decides a run.
fn score(values: &[i64]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64 + 1.0) * ((v * 7).rem_euclid(5) as f64))
        .sum()
}

#[test]
fn every_recorded_sequence_is_proposed_again() {
    let golden = golden();
    assert_eq!(
        golden["n_startup_trials"].as_u64(),
        Some(10),
        "the startup count moved; the sampler's constant has to follow"
    );
    let cases = golden["cases"].as_array().expect("cases");
    assert!(cases.len() >= 12, "the golden lost cases: {}", cases.len());
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let seed = case["seed"].as_u64().expect("a seed") as u32;
        let parameters: Vec<(i64, i64)> = case["parameters"]
            .as_array()
            .expect("parameters")
            .iter()
            .map(|p| {
                (
                    p["low"].as_i64().expect("low"),
                    p["high"].as_i64().expect("high"),
                )
            })
            .collect();
        let expected: Vec<Vec<i64>> = case["trials"]
            .as_array()
            .expect("trials")
            .iter()
            .map(|values| {
                values
                    .as_array()
                    .expect("a trial")
                    .iter()
                    .map(|v| v.as_i64().expect("a value"))
                    .collect()
            })
            .collect();

        let mut sampler = IntTpeSampler::new(seed, parameters);
        for (index, want) in expected.iter().enumerate() {
            let asked = sampler.ask();
            assert_eq!(&asked, want, "{name} seed {seed}, trial {index}");
            let value = score(&asked);
            assert_eq!(
                value,
                case["scores"][index].as_f64().expect("a score"),
                "{name} seed {seed}: the recorded objective disagrees at trial {index}"
            );
            sampler.tell(asked, value);
        }
    }
}

/// A run that never reaches the tenth trial exercises only the startup sampler, so the two halves
/// are known to be tested apart as well as together.
#[test]
fn the_startup_half_is_reached_on_its_own() {
    let golden = golden();
    let short = golden["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["name"] == "one_parameter_through_startup_only")
        .expect("the short case");
    assert!(
        short["trials"].as_array().expect("trials").len() < 10,
        "that case now reaches the TPE half too"
    );
}
