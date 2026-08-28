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
#[track_caller]
fn same(ours: f64, expected: f64, what: &str) {
    match expected.is_nan() {
        true => assert!(ours.is_nan(), "{what}: expected NaN, got {ours}"),
        false => assert_eq!(ours, expected, "{what}"),
    }
}

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
