//! CPython's version-2 string seeding, against draws recorded from `random.Random(str)`.

use pyrng::Random;

#[test]
fn a_string_seed_appends_its_own_sha512_before_becoming_a_key() {
    let golden: serde_json::Value = serde_json::from_str(include_str!(
        "../../dsrust/tests/conformance/optimize/hasher.json"
    ))
    .expect("the hasher golden is valid JSON");
    let cases = golden["string_seeds"].as_array().expect("string_seeds");
    assert!(!cases.is_empty(), "the golden recorded no seeds");
    for case in cases {
        let seed = case["seed"].as_str().expect("seed");
        let mut rng = Random::from_seed_bytes(seed.as_bytes());
        for (draw, expected) in case["random"]
            .as_array()
            .expect("random")
            .iter()
            .enumerate()
        {
            assert_eq!(
                rng.random(),
                expected.as_f64().expect("a float"),
                "draw {draw} of random.Random({seed:?})"
            );
        }
        let mut fresh = Random::from_seed_bytes(seed.as_bytes());
        assert_eq!(
            fresh.below(7) as u64,
            case["below_7"][0].as_u64().expect("below_7"),
            "_randbelow(7) of random.Random({seed:?})"
        );
    }
}
