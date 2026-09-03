//! optuna's truncated normal, which is what its TPE sampler draws an *integer* parameter through.
//!
//! A categorical parameter is a choice among unrelated options and optuna models it that way; an
//! integer one is a point on a line, and TPE fits a truncated normal to the trials it has and draws
//! from that. `BootstrapFewShotWithOptuna` asks for `suggest_int`, so reproducing its search means
//! reproducing this arithmetic and not merely its shape.
//!
//! optuna vendors the whole chain from SciPy rather than depending on it, so this is a port of a
//! port — and each function is pure, which is why every one of them is held to a grid recorded from
//! optuna itself in `tests/conformance/truncnorm.json`.
//!
//! `erf` comes from the platform, because that is where optuna's comes from. Its `_erf` module
//! carries FreeBSD's rational approximation for large arrays, but the array sizes TPE works at are
//! far below its 2000-element threshold, so what actually runs is CPython's `math.erf` — the
//! system's. The two disagree by an ULP where `erf` saturates, which is the kind of difference that
//! decides a comparison in a far tail.

use std::f64::consts::PI;

unsafe extern "C" {
    fn erf(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
}

/// `math.erf`, which optuna reaches for at the sizes TPE uses.
fn erf_(x: f64) -> f64 {
    // SAFETY: `erf` is a pure C function over one `double`, with no preconditions.
    unsafe { erf(x) }
}

fn erfc_(x: f64) -> f64 {
    // SAFETY: as `erf_`.
    unsafe { erfc(x) }
}

const NORM_PDF_LOG_C: f64 = 0.918_938_533_204_672_7; // ln(sqrt(2 * pi))
/// `sqrt(3) / pi`, the standard-logistic approximation's scale in `ndtri_exp`'s initial guess.
const NDTRI_EXP_APPROX_C: f64 = 0.551_328_895_421_792_1;

/// optuna's `_ndtr`: the standard normal CDF, through `erf`.
pub fn ndtr(a: f64) -> f64 {
    0.5 + 0.5 * erf_(a / std::f64::consts::SQRT_2)
}

/// optuna's `_ndtr_single`, which splits on the sign to keep the tail accurate. Reached only from
/// [`log_ndtr`], and not interchangeable with [`ndtr`] — they take different branches and differ in
/// the last bits.
///
/// The *second* comparison carries an equivalent mutant. At `x == 1/√2` — reached at `a == 1`,
/// which the grid holds — the middle and last arms agree to the last bit on this libm, both
/// answering `0.8413447460685429`, so moving the boundary changes nothing. The first comparison
/// does not: its two arms differ by an ULP at `a == -1`, and the grid catches that one.
fn ndtr_single(a: f64) -> f64 {
    let x = a / std::f64::consts::SQRT_2;
    let half_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    if x < -half_sqrt2 {
        0.5 * erfc_(-x)
    } else if x < half_sqrt2 {
        0.5 + 0.5 * erf_(x)
    } else {
        1.0 - 0.5 * erfc_(x)
    }
}

/// optuna's `_log_ndtr_single`: the log CDF, by an asymptotic series once the tail is far enough
/// out that the CDF itself has underflowed.
pub fn log_ndtr(a: f64) -> f64 {
    if a > 6.0 {
        return -ndtr_single(-a);
    }
    if a > -20.0 {
        return ndtr_single(a).ln();
    }
    let log_lhs = -0.5 * a * a - (-a).ln() - 0.5 * (2.0 * PI).ln();
    let mut last_total = 0.0f64;
    let mut right_hand_side = 1.0f64;
    let mut numerator = 1.0f64;
    let mut denom_factor = 1.0f64;
    let denom_cons = 1.0 / (a * a);
    let mut sign = 1.0f64;
    // Upstream loops until the term stops moving the sum. The bound is the range's rather than a
    // comparison of its own: a `while i < limit` is two mutants nothing can catch, since the series
    // converges in a handful of terms and no input reaches the limit. The convergence test is still
    // upstream's, and is what actually ends the loop.
    for step in 1..=1000u32 {
        if (last_total - right_hand_side).abs() <= f64::EPSILON {
            break;
        }
        let i = f64::from(step);
        last_total = right_hand_side;
        sign = -sign;
        denom_factor *= denom_cons;
        numerator *= 2.0 * i - 1.0;
        right_hand_side += sign * numerator * denom_factor;
    }
    log_lhs + right_hand_side.ln()
}

/// optuna's `_norm_logpdf`.
pub fn norm_logpdf(x: f64) -> f64 {
    -(x * x) / 2.0 - NORM_PDF_LOG_C
}

pub fn log_sum(log_p: f64, log_q: f64) -> f64 {
    // `np.logaddexp`, whose branch keeps the larger term outside the exponential.
    //
    // The comparison below carries an equivalent mutant: equal arguments have already returned, so
    // it only ever sees two different values and `>` and `>=` pick the same one.
    if log_p == log_q {
        return log_p + std::f64::consts::LN_2;
    }
    let (big, small) = match log_p > log_q {
        true => (log_p, log_q),
        false => (log_q, log_p),
    };
    if small == f64::NEG_INFINITY {
        return big;
    }
    big + (small - big).exp().ln_1p()
}

fn log_diff(log_p: f64, log_q: f64) -> f64 {
    log_p + (-(log_q - log_p).exp()).ln_1p()
}

/// optuna's `_log_gauss_mass`: the log probability between two standardised bounds.
///
/// Three cases, and the middle one is not the sum of the other two: adding the two tails cancels
/// catastrophically as the mass approaches one, so upstream writes it as `log1p(-Φ(a) - Φ(-b))`.
pub fn log_gauss_mass(a: f64, b: f64) -> f64 {
    if b <= 0.0 {
        log_diff(log_ndtr(b), log_ndtr(a))
    } else if a > 0.0 {
        // The right tail is the left tail of the mirrored interval.
        log_diff(log_ndtr(-a), log_ndtr(-b))
    } else {
        (-ndtr(a) - ndtr(-b)).ln_1p()
    }
}

/// optuna's `_ndtri_exp`: the `x` with `log_ndtr(x) == y`, by Newton from a closed-form guess.
///
/// **Over a slice, because the answer for one input depends on the others.** Upstream runs the
/// Newton step on the whole array and breaks only once *every* element has converged, so an element
/// that settled early keeps taking steps while its neighbours catch up — and those extra steps move
/// its last bits. Three of the thirteen recorded inputs answer differently alone than in company,
/// which is why there is no scalar form of this to be tempted by.
///
/// The guess has three regimes and the loop's tolerance is relative, not absolute; both decide the
/// result exactly.
pub fn ndtri_exp(ys: &[f64]) -> Vec<f64> {
    // Close to zero, invert the *other* tail first: `log_ndtr` is better conditioned on the left.
    let flipped: Vec<bool> = ys.iter().map(|&y| y > -1e-2).collect();
    let z: Vec<f64> = ys
        .iter()
        .zip(&flipped)
        .map(|(&y, &flip)| match flip {
            true => (-y.exp_m1()).ln(),
            false => y,
        })
        .collect();
    let mut x: Vec<f64> = z
        .iter()
        .map(|&z| match z < -5.0 {
            true => -(-2.0 * (z + NORM_PDF_LOG_C)).sqrt(),
            false => -NDTRI_EXP_APPROX_C * (-z).exp_m1().ln(),
        })
        .collect();
    for _ in 0..100 {
        let mut settled = true;
        for (x, &z) in x.iter_mut().zip(&z) {
            let log_ndtr_x = log_ndtr(*x);
            let log_norm_pdf_x = -0.5 * *x * *x - NORM_PDF_LOG_C;
            // `exp(log_ndtr - log_pdf)` rather than the ratio, which upstream notes is what keeps
            // the far tail from dividing two underflowed numbers.
            let dx = (log_ndtr_x - z) * (log_ndtr_x - log_norm_pdf_x).exp();
            *x -= dx;
            // Upstream's relative tolerance. The comparison carries an equivalent mutant: it can
            // only matter when the step is *exactly* `1e-8` times the value, which no input reaches.
            settled &= dx.abs() < 1e-8 * x.abs();
        }
        if settled {
            break;
        }
    }
    x.iter()
        .zip(&flipped)
        .map(|(&x, &flip)| match flip {
            true => -x,
            false => x,
        })
        .collect()
}

/// optuna's `ppf`: the quantile of each truncated normal bounded by `a` and `b`.
///
/// Over slices for the reason [`ndtri_exp`] is, and split the way upstream splits: the `a < 0`
/// entries are inverted together and the rest are inverted together, mirrored first. So which group
/// an entry lands in changes not only its formula but which batch its Newton loop shares.
pub fn ppf(q: &[f64], a: &[f64], b: &[f64]) -> Vec<f64> {
    let log_mass: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| log_gauss_mass(a, b))
        .collect();
    let mut out = vec![f64::NAN; q.len()];
    for left in [true, false] {
        let group: Vec<usize> = (0..q.len()).filter(|&i| (a[i] < 0.0) == left).collect();
        if group.is_empty() {
            continue;
        }
        let logs: Vec<f64> = group
            .iter()
            .map(|&i| match left {
                true => log_sum(log_ndtr(a[i]), q[i].ln() + log_mass[i]),
                false => log_sum(log_ndtr(-b[i]), (-q[i]).ln_1p() + log_mass[i]),
            })
            .collect();
        for (&i, &value) in group.iter().zip(&ndtri_exp(&logs)) {
            out[i] = match left {
                true => value,
                false => -value,
            };
        }
    }
    // Upstream writes these over the computed values rather than short-circuiting, so a bound of
    // exactly zero or one is the bound itself however the arithmetic came out.
    for i in 0..q.len() {
        if q[i] == 0.0 {
            out[i] = a[i];
        }
        if q[i] == 1.0 {
            out[i] = b[i];
        }
        if a[i] == b[i] {
            out[i] = f64::NAN;
        }
    }
    out
}
