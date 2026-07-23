//! `tpe`'s MT19937 against numpy's own, replaying captured draws.
//!
//! Every vector in `tests/conformance/numpy_mt19937.json` was produced by running numpy (see the
//! generator note in the fixture's `source`). Matching them means this crate's generator is the one
//! optuna draws from — the precondition for the sampler above it reproducing optuna's trials.

use pyrng::RandomState;

fn fixture() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/numpy_mt19937.json");
    let text = std::fs::read_to_string(&path).expect("the numpy golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

#[test]
fn seeding_matches_numpys_init_genrand() {
    let fixture = fixture();
    for (seed, key) in fixture["seed_key"].as_object().expect("seed_key") {
        let seed: u32 = seed.parse().expect("a seed");
        let generator = RandomState::new(seed);
        let expected = key.as_array().expect("key words");
        for (index, word) in expected.iter().enumerate() {
            assert_eq!(
                u64::from(generator.state_word(index)),
                word.as_u64().expect("a word"),
                "state word {index} for seed {seed}"
            );
        }
    }
}

#[test]
fn random_sample_matches_numpy() {
    // Compared as raw IEEE-754 bits, not decimals: the draws must agree to the last bit, and a
    // decimal round-trip through JSON can shift that bit under a parser that rounds differently.
    let fixture = fixture();
    for (seed, values) in fixture["random_sample_bits"].as_object().expect("random_sample_bits") {
        let seed: u32 = seed.parse().expect("a seed");
        let mut generator = RandomState::new(seed);
        for (index, value) in values.as_array().expect("values").iter().enumerate() {
            let drawn = generator.random_sample().to_bits();
            assert_eq!(
                drawn,
                value.as_u64().expect("bit pattern"),
                "random_sample #{index} for seed {seed}"
            );
        }
    }
}

#[test]
fn choice_matches_numpy() {
    let fixture = fixture();
    for case in fixture["choice"].as_array().expect("choice cases") {
        let seed = case["seed"].as_u64().expect("seed") as u32;
        let probabilities: Vec<f64> = case["p"]
            .as_array()
            .expect("p")
            .iter()
            .map(|value| value.as_f64().expect("a probability"))
            .collect();
        let size = case["size"].as_u64().expect("size") as usize;
        let expected: Vec<usize> = case["out"]
            .as_array()
            .expect("out")
            .iter()
            .map(|value| value.as_u64().expect("an index") as usize)
            .collect();

        let mut generator = RandomState::new(seed);
        assert_eq!(
            generator.choice(&probabilities, size),
            expected,
            "choice for seed {seed}, n {}",
            probabilities.len()
        );
    }
}
