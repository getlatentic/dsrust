//! The sampler against optuna in MIPROv2's exact usage: a baseline trial seeded before the search.
//!
//! MIPROv2 calls `study.add_trial(...)` with the default program's parameters and score, then
//! `study.optimize`. Reproduced here as `tell` before the first `ask` — the baseline must count
//! toward the startup trials and must not consume the startup generator, or the trajectory drifts.
//! The expected sequence was recorded from optuna 4.5.0, `TPESampler(seed=9, multivariate=True)`,
//! over `CategoricalDistribution(range(4))` and `range(3)` with the objective below.

use tpe::TpeSampler;

/// The deterministic objective optuna was run against: `(a==2)*30 + (b==1)*15 + 50`.
fn objective(params: &[usize]) -> f64 {
    let (a, b) = (params[0], params[1]);
    (a == 2) as u8 as f64 * 30.0 + (b == 1) as u8 as f64 * 15.0 + 50.0
}

#[test]
fn a_seeded_baseline_matches_optunas_add_trial() {
    let expected: [(usize, usize); 14] = [
        (1, 2),
        (2, 1),
        (2, 1),
        (3, 1),
        (1, 0),
        (1, 1),
        (2, 2),
        (1, 2),
        (0, 2),
        (2, 0),
        (2, 1),
        (2, 1),
        (0, 1),
        (2, 1),
    ];

    let mut sampler = TpeSampler::new(9, vec![("p0".to_owned(), 4), ("p1".to_owned(), 3)]);
    // MIPROv2's baseline: the all-zeros default program, told before any ask.
    let baseline = vec![0, 0];
    sampler.tell(baseline.clone(), objective(&baseline));

    for (trial, &(a, b)) in expected.iter().enumerate() {
        let params = sampler.ask();
        assert_eq!(
            (params[0], params[1]),
            (a, b),
            "trial {trial} diverges from optuna's post-baseline trajectory"
        );
        let score = objective(&params);
        sampler.tell(params, score);
    }
}
