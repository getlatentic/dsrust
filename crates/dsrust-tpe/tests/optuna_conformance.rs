//! `tpe`'s sampler against optuna's own, replaying identical studies.
//!
//! Each case in `tests/conformance/optuna_tpe.json` is a categorical study optuna ran: a fixed
//! table objective, a seed, and the exact sequence of trials optuna's `TPESampler(multivariate=True)`
//! produced. Here the crate's sampler runs the same study — ask a trial, look its score up in the
//! table, tell the sampler — and every trial is compared to optuna's. Matching them means the crate
//! proposes the trials optuna does, which is what dspy's MIPROv2 relies on.

use std::collections::HashMap;

use tpe::TpeSampler;

fn fixture() -> serde_json::Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/optuna_tpe.json");
    let text = std::fs::read_to_string(&path).expect("the optuna golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

/// The objective as a lookup: the table is keyed by the trial's categories joined with commas, which
/// is how the fixture stores what optuna optimized.
fn score(table: &HashMap<String, f64>, params: &[usize]) -> f64 {
    let key = params
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    *table
        .get(&key)
        .expect("every category combination is in the table")
}

#[test]
fn tpe_proposes_the_trials_optuna_proposes() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the golden records no cases");
    for case in cases {
        let seed = case["seed"].as_u64().expect("seed") as u32;
        let table: HashMap<String, f64> = case["table"]
            .as_object()
            .expect("table")
            .iter()
            .map(|(key, value)| (key.clone(), value.as_f64().expect("a score")))
            .collect();
        let expected = case["sequence"].as_array().expect("sequence");

        let mut sampler = TpeSampler::new(seed, parameters(case));
        for (trial, want) in expected.iter().enumerate() {
            let params = sampler.ask();
            let expected_params: Vec<usize> = want
                .as_array()
                .expect("trial params")
                .iter()
                .map(|value| value.as_u64().expect("a category") as usize)
                .collect();
            assert_eq!(
                params, expected_params,
                "trial {trial} for seed {seed} (cards {:?})",
                case["cards"]
            );
            let value = score(&table, &params);
            sampler.tell(params, value);
        }
    }
}

/// A case's parameters, named and in the order its objective suggests them.
///
/// The names are what the sampler sorts by once it leaves its random startup, so the cases carrying
/// dspy's own names — where `demos` sorts before the `instruction` suggested first — are the ones
/// that hold the two orders apart.
fn parameters(case: &serde_json::Value) -> Vec<(String, usize)> {
    let names = case["names"].as_array().expect("names");
    let cards = case["cards"].as_array().expect("cards");
    names
        .iter()
        .zip(cards)
        .map(|(name, count)| {
            (
                name.as_str().expect("a name").to_owned(),
                count.as_u64().expect("a cardinality") as usize,
            )
        })
        .collect()
}
