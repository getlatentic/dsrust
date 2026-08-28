//! PCG64's stream, against numpy's own.
//!
//! `default_rng` is not `RandomState`: one is PCG64 and the other MT19937, and they share no
//! values. The raw words are checked before the doubles because a seeding bug and an output-function
//! bug look identical one level up — numpy hashes an integer seed through `SeedSequence` into 256
//! bits, and a port that assigned the state directly would produce a perfectly good PCG64 stream
//! that is not numpy's.

use pyrng::pcg64::Pcg64;
use serde_json::Value;

fn golden() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/numpy_pcg64.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("the golden is committed"))
        .expect("it parses")
}

#[test]
fn the_raw_words_are_numpys() {
    let fixture = golden();
    for (seed, words) in fixture["raw_words"].as_object().expect("seeds") {
        let mut rng = Pcg64::seeded(seed.parse().expect("a seed"));
        for (at, word) in words.as_array().expect("words").iter().enumerate() {
            assert_eq!(
                rng.next_u64(),
                word.as_u64().expect("a 64-bit word"),
                "seed {seed}, word {at}"
            );
        }
    }
}

#[test]
fn the_doubles_are_numpys() {
    let fixture = golden();
    for (seed, doubles) in fixture["doubles"].as_object().expect("seeds") {
        let mut rng = Pcg64::seeded(seed.parse().expect("a seed"));
        for (at, expected) in doubles.as_array().expect("doubles").iter().enumerate() {
            assert_eq!(
                rng.next_double(),
                expected.as_f64().expect("a double"),
                "seed {seed}, draw {at}"
            );
        }
    }
}

/// `poisson` on both sides of numpy's branch at ten: the multiplication method below it and
/// transformed rejection at and above. The two consume the generator at different rates, so a
/// stream of twenty-four checks the branch as well as the values.
///
/// PTRS's third arm — the one that compares against `loggam` — takes 12 of the 24 draws at lambda
/// ten, so it is exercised. Perturbing a `loggam` coefficient by `1e-14` changes nothing here and
/// that is *not* a gap: the arm is an inequality whose two sides differ by far more than that. It
/// takes about `0.05` to flip one, which does fail this test. A control has to be the size of the
/// decision it is trying to change.
#[test]
fn the_poisson_draws_are_numpys() {
    let fixture = golden();
    let mut branches = (0, 0);
    for case in fixture["poisson"].as_array().expect("poisson streams") {
        let seed = case["seed"].as_u64().expect("a seed");
        let lam = case["lam"].as_f64().expect("a lambda");
        let mut rng = Pcg64::seeded(seed);
        for (at, expected) in case["draws"].as_array().expect("draws").iter().enumerate() {
            assert_eq!(
                rng.poisson(lam),
                expected.as_u64().expect("a count"),
                "seed {seed}, lam {lam}, draw {at}"
            );
        }
        match lam >= 10.0 {
            true => branches.1 += 1,
            false => branches.0 += 1,
        }
    }
    assert!(
        branches.0 > 0 && branches.1 > 0,
        "both branches have to be reached for this to be a test of the branch: {branches:?}"
    );
}

/// The stepper is PCG64 itself, not merely something that matches numpy on five seeds.
///
/// `rand_pcg::Lcg128Xsl64` is officially `pcg64`: same 128-bit LCG multiplier
/// (`0x2360ED051FC65DA44385DF649FCCF645`), same XSL-RR output, and its `new(state, stream)` does
/// what numpy's `pcg64_srandom_r` does — increment `(stream << 1) | 1`, step, add the state, step.
/// So handing it the two values numpy's `SeedSequence` derived must give numpy's stream.
///
/// This is what tells the two halves apart. If the seeding were wrong and the stepper right, this
/// test would still pass while the golden failed; if the stepper were wrong, both would fail. The
/// golden alone could not say which.
#[test]
fn agrees_with_rand_pcg() {
    use rand_core::Rng;

    for seed in [0u64, 1, 7, 42, 2024, 999_999] {
        let (state, stream) = Pcg64::seed_values(seed);
        let mut ours = Pcg64::seeded(seed);
        let mut theirs = rand_pcg::Lcg128Xsl64::new(state, stream);
        for at in 0..32 {
            assert_eq!(
                ours.next_u64(),
                theirs.next_u64(),
                "seed {seed}, word {at}: the hand-written stepper left PCG64"
            );
        }
    }
}
