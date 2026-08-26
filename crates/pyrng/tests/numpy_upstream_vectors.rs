//! The crate's generator against numpy's *own* test vectors.
//!
//! These are not values this project captured — they are the literals numpy asserts against in its
//! own suite, `numpy/random/tests/test_randomstate.py`, for the legacy `RandomState` seeded with
//! `1234567890`. That is the exact path optuna's `TPESampler` uses (`np.random.RandomState(seed)`),
//! so reproducing numpy's published expectations is reproducing what optuna draws from.
//!
//! numpy's other MT19937 vectors — the `data/mt19937-testset-*.csv` files driven by `test_direct.py`
//! — seed the *modern* `MT19937` bit generator through a `SeedSequence`, a different initialisation
//! that optuna does not take, so they are deliberately not reproduced here.

use pyrng::RandomState;

/// numpy `assert_array_almost_equal(actual, desired, decimal=15)`: `|actual - desired| < 1.5e-15`.
fn almost_equal(actual: f64, desired: f64) -> bool {
    (actual - desired).abs() < 1.5e-15
}

/// `numpy/random/tests/test_randomstate.py::TestRandomDist::test_random_sample` — the exact
/// assertion, run against this crate's generator. `random_sample((3, 2))` fills row-major, so the
/// six values are six draws in order.
///
/// Two bars are checked at once. `desired` is the literal numpy asserts, to its own 15-decimal
/// tolerance; `bits` is the raw IEEE-754 of the same values read straight out of numpy's compiled
/// C generator (`RandomState(1234567890).random_sample(6).view(uint64)`), so the equality below is
/// this crate reproducing numpy's C output to the last bit — stronger than numpy's own test.
#[test]
fn matches_numpys_own_random_sample_vector() {
    let desired = [
        0.61879477158567997,
        0.59162362775974664,
        0.88868358904449662,
        0.89165480011560816,
        0.4575674820298663,
        0.7781880808593471,
    ];
    let bits: [u64; 6] = [
        0x3fe3cd2ab15cae5f,
        0x3fe2ee94ac989b90,
        0x3fec701890ee043c,
        0x3fec886fa5ba2cae,
        0x3fdd48c91ec20188,
        0x3fe8e6eab0adb15a,
    ];
    let mut generator = RandomState::new(1234567890);
    for (index, (&expected, &exact)) in desired.iter().zip(&bits).enumerate() {
        let actual = generator.random_sample();
        assert!(
            almost_equal(actual, expected),
            "random_sample #{index}: numpy expects {expected:.17}, got {actual:.17}"
        );
        assert_eq!(
            actual.to_bits(),
            exact,
            "random_sample #{index}: not bit-identical to numpy's C output"
        );
    }
}

/// `numpy/random/tests/test_randomstate.py::TestRandomDist::test_choice_nonuniform_replace` —
/// `choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])` is expected to be `[1, 1, 2, 2]`, with replacement and
/// the given probabilities, which is the overload the sampler uses.
#[test]
fn matches_numpys_own_choice_vector() {
    let mut generator = RandomState::new(1234567890);
    let drawn = generator.choice(&[0.4, 0.4, 0.1, 0.1], 4);
    assert_eq!(
        drawn,
        vec![1, 1, 2, 2],
        "numpy expects choice -> [1, 1, 2, 2]"
    );
}
