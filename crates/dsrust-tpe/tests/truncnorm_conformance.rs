//! optuna's truncated-normal numerics, against a grid recorded from optuna itself.
//!
//! Every value here is compared exactly. These functions feed a Newton iteration and a density
//! ratio, so an approximation that is merely close changes which candidate the sampler picks.

use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("conformance/truncnorm.json"))
        .expect("the truncnorm golden is valid JSON")
}

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(number)
        .collect()
}

/// The golden tags a non-finite answer, because JSON has no spelling for one. They are real
/// answers — `log_gauss_mass(0, 0)` is `-inf` — so they are compared like any other.
fn number(value: &Value) -> f64 {
    match value {
        Value::String(tag) => match tag.as_str() {
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            "nan" => f64::NAN,
            other => panic!("unknown numeric tag {other:?}"),
        },
        other => other.as_f64().expect("a float"),
    }
}

/// Exact, including both non-finite spellings — two NaNs are never `==`, and the golden records
/// where optuna answers with one.
/// Exact on both non-finite spellings — two NaNs are never `==`, and the golden records where
/// optuna answers with one — and exact on a finite answer too, on the platform the golden was
/// recorded on.
///
/// Every function here ends in the platform's `erf`, `exp` or `log`, and that is optuna's own
/// arrangement: CPython's `math.erf` is the system's, so optuna's answers move with the libm under
/// it. The golden was recorded on macOS/arm64, where Apple's answers; glibc's and Windows' CRT
/// differ from it in the last bits, and `ndtri_exp`'s Newton iteration and `ppf`'s inversion
/// amplify that — measured at 128 ulps on `ndtr(-2.856)` under glibc and a relative 1e-10 on
/// `ppf(0.000001, 0, 10)` under the Windows CRT, which is 578,109 ulps of a number that small.
/// So elsewhere the comparison allows a relative 1e-9, far under what changes a Newton step or a
/// density ratio, and optuna on those platforms would differ from this golden by the same amount.
/// On the recording platform it allows nothing, and the sampler's own golden — which compares the
/// trials optuna proposes — is exact on all three.
#[track_caller]
fn same(ours: f64, expected: f64, what: &str) {
    if expected.is_nan() {
        assert!(ours.is_nan(), "{what}: expected NaN, got {ours}");
        return;
    }
    if RECORDING_PLATFORM || expected.is_infinite() || ours.is_infinite() || expected == 0.0 {
        assert_eq!(ours, expected, "{what}");
        return;
    }
    let apart = ((ours - expected) / expected).abs();
    assert!(
        apart <= 1e-9,
        "{what}: {ours} is {apart:e} away from {expected}, relatively"
    );
}

/// Where `tests/conformance/truncnorm.json` was recorded, and the platform libm it carries.
const RECORDING_PLATFORM: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

#[test]
fn the_normal_cdf_and_its_log_match_across_every_branch() {
    let golden = golden();
    let grid = floats(&golden["grid"]);
    assert!(grid.len() >= 50, "the grid shrank to {}", grid.len());
    for (name, expected, ours) in [
        (
            "ndtr",
            floats(&golden["ndtr"]),
            grid.iter()
                .map(|&x| tpe::truncnorm::ndtr(x))
                .collect::<Vec<_>>(),
        ),
        (
            "log_ndtr",
            floats(&golden["log_ndtr"]),
            grid.iter().map(|&x| tpe::truncnorm::log_ndtr(x)).collect(),
        ),
        (
            "norm_logpdf",
            floats(&golden["norm_logpdf"]),
            grid.iter()
                .map(|&x| tpe::truncnorm::norm_logpdf(x))
                .collect(),
        ),
    ] {
        for (index, (&expected, &ours)) in expected.iter().zip(ours.iter()).enumerate() {
            same(
                ours,
                expected,
                &format!("{name} at grid[{index}] = {}", grid[index]),
            );
        }
    }
}

#[test]
fn the_interval_mass_matches_in_all_three_cases() {
    for case in golden()["log_gauss_mass"].as_array().expect("cases") {
        let (a, b) = (
            case["a"].as_f64().expect("a"),
            case["b"].as_f64().expect("b"),
        );
        let expected = number(&case["value"]);
        let ours = tpe::truncnorm::log_gauss_mass(a, b);
        same(ours, expected, &format!("log_gauss_mass({a}, {b})"));
    }
}

#[test]
fn the_log_cdf_inverse_lands_where_optunas_newton_lands() {
    let golden = golden();
    let inputs = floats(&golden["ndtri_exp"]["inputs"]);
    let expected = floats(&golden["ndtri_exp"]["values"]);
    // One batch, as the golden recorded it: an element's answer depends on when its neighbours
    // converge, so inverting them one at a time gives different last bits for three of thirteen.
    let ours = tpe::truncnorm::ndtri_exp(&inputs);
    for ((&y, &want), &got) in inputs.iter().zip(expected.iter()).zip(ours.iter()) {
        same(got, want, &format!("ndtri_exp({y})"));
    }
}

#[test]
fn the_quantile_matches_on_both_sides() {
    for case in golden()["ppf"].as_array().expect("cases") {
        let q = case["q"].as_f64().expect("q");
        let a = case["a"].as_f64().unwrap_or(f64::NEG_INFINITY);
        let b = case["b"].as_f64().unwrap_or(f64::INFINITY);
        let expected = number(&case["value"]);
        // One case at a time, which is how the golden recorded them.
        let ours = tpe::truncnorm::ppf(&[q], &[a], &[b])[0];
        same(ours, expected, &format!("ppf({q}, {a}, {b})"));
    }
}

/// `logaddexp` has an equal-arguments arm no ordinary pair reaches — `logaddexp(x, x)` is
/// `x + ln 2`, not the general `big + log1p(exp(small - big))` — and an infinite argument short-
/// circuits rather than subtracting infinities.
#[test]
fn the_log_sum_handles_equal_and_infinite_arguments() {
    let golden = golden();
    let cases = golden["log_sum"].as_array().expect("log_sum");
    assert!(cases.len() >= 7, "the golden lost log_sum cases");
    let mut equal = 0;
    for case in cases {
        let left = case["left"].as_f64().unwrap_or(f64::NEG_INFINITY);
        let right = case["right"].as_f64().unwrap_or(f64::NEG_INFINITY);
        equal += usize::from(left == right);
        same(
            tpe::truncnorm::log_sum(left, right),
            number(&case["value"]),
            &format!("log_sum({left}, {right})"),
        );
    }
    assert!(
        equal >= 3,
        "only {equal} pair(s) are equal; the arm they exist for is untested"
    );
}

/// The same inversion one input at a time, which is the only way the initial guess is visible.
///
/// In a batch the Newton loop runs until *every* element has converged, and the extra steps take
/// both of the guess's regimes to the same answer. Alone, the switch at -5 decides the last bits —
/// so a mutant moving that boundary survives the batch and dies here.
#[test]
fn inverting_one_at_a_time_shows_the_guess() {
    for case in golden()["ndtri_exp_alone"]
        .as_array()
        .expect("ndtri_exp_alone")
    {
        let input = case["input"].as_f64().expect("an input");
        same(
            tpe::truncnorm::ndtri_exp(&[input])[0],
            number(&case["value"]),
            &format!("ndtri_exp([{input}])"),
        );
    }
}
