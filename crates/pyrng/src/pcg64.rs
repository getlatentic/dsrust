//! numpy's `default_rng` — PCG64, which is not the generator [`RandomState`](crate::numpy) uses.
//!
//! `np.random.RandomState` is MT19937 and is what optuna's TPE draws from. `np.random.default_rng`
//! is PCG64: a 128-bit linear congruential generator whose state is permuted on the way out
//! (XSL-RR — xor the halves, then rotate by the top six bits). They share no stream at all, so a
//! caller reproducing `default_rng` needs this one.
//!
//! Seeded through numpy's `SeedSequence`, not by assignment: an integer seed is hashed into 256
//! bits of entropy and the first 128 become the state, the next 128 the increment. Reproducing the
//! draws means reproducing that hash, so it is here rather than skipped — a port that seeded the
//! state directly would produce a valid PCG64 stream and the wrong one.
//!
//! Held to `tests/conformance/numpy_pcg64.json`, recorded from numpy itself.

/// PCG64's multiplier, `setseq_dxsm`'s cheap constant is not this one — numpy 1.x/2.x `PCG64`
/// uses the full 128-bit LCG multiplier.
const MULTIPLIER: u128 = 47026247687942121848144207491837523525;

/// numpy's `SeedSequence` with an integer entropy and no spawn key.
pub(crate) mod seed_sequence {
    /// The three constants of numpy's `hashmix`, from `_entropy.pyx`.
    const INIT_A: u32 = 0x43b0d7e5;
    const MULT_A: u32 = 0x931e8875;
    const INIT_B: u32 = 0x8b51f9dd;
    const MULT_B: u32 = 0x58f38ded;
    const MIX_MULT_L: u32 = 0xca01f9dd;
    const MIX_MULT_R: u32 = 0x4973f715;
    const XSHIFT: u32 = 16;

    fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
        let mut value = value ^ *hash_const;
        *hash_const = hash_const.wrapping_mul(MULT_A);
        value = value.wrapping_mul(*hash_const);
        value ^= value >> XSHIFT;
        value
    }

    fn mix(x: u32, y: u32) -> u32 {
        let result = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        result ^ (result >> XSHIFT)
    }

    /// `SeedSequence(entropy).generate_state(n, uint32)`, for an integer entropy and no pool size
    /// beyond numpy's default of four.
    pub(crate) fn generate_state(entropy: u64, words: usize) -> Vec<u32> {
        // The entropy is spread into 32-bit words, little end first, as `_coerce_to_uint32_array`
        // does for a Python int.
        // Written as a bounded walk over the two words a `u64` has rather than a loop that shifts
        // until the value runs out: the loop is the same two iterations, and a mutant that stops
        // the shift advancing should fail a test rather than spin.
        let mut source: Vec<u32> = (0..2)
            .map(|word| ((entropy >> (32 * word)) & 0xffff_ffff) as u32)
            .collect();
        while source.len() > 1 && *source.last().expect("two words") == 0 {
            source.pop();
        }

        let mut pool = [0u32; 4];
        let mut hash_const = INIT_A;
        for (at, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(source.get(at).copied().unwrap_or(0), &mut hash_const);
        }
        for at in 0..pool.len() {
            for other in 0..pool.len() {
                if at != other {
                    pool[other] = mix(pool[other], hashmix(pool[at], &mut hash_const));
                }
            }
        }
        for extra in source.iter().skip(pool.len()) {
            for slot in &mut pool {
                *slot = mix(*slot, hashmix(*extra, &mut hash_const));
            }
        }

        let mut state = vec![0u32; words];
        let mut hash_const = INIT_B;
        for (at, slot) in state.iter_mut().enumerate() {
            let mut data = pool[at % pool.len()];
            data ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data = data.wrapping_mul(hash_const);
            data ^= data >> XSHIFT;
            *slot = data;
        }
        state
    }
}

/// numpy's `PCG64` bit generator.
#[derive(Clone, Debug)]
pub struct Pcg64 {
    state: u128,
    increment: u128,
}

impl Pcg64 {
    /// The `(state, stream)` numpy's `SeedSequence` derives for a seed, before any stepping.
    ///
    /// numpy asks it for four *64-bit* words — eight 32-bit ones paired up, low half first — and
    /// builds each 128-bit value as `first << 64 | second`. So the pair order is little-endian and
    /// the pair-of-pairs order is big-endian, and reading all four words one way gives a valid
    /// PCG64 stream that is not numpy's.
    ///
    /// Public because it is the half that belongs to *numpy* rather than to PCG: handing these two
    /// values to an independent PCG64 must produce the same stream, which is what
    /// `agrees_with_rand_pcg` asserts.
    pub fn seed_values(seed: u64) -> (u128, u128) {
        let words = seed_sequence::generate_state(seed, 8);
        let read = |at: usize| -> u128 {
            let low = |at: usize| (words[at] as u64) | ((words[at + 1] as u64) << 32);
            ((low(at) as u128) << 64) | (low(at + 2) as u128)
        };
        (read(0), read(4))
    }

    /// `np.random.default_rng(seed)`, seeded through `SeedSequence` as numpy does.
    pub fn seeded(seed: u64) -> Self {
        let (state, stream) = Self::seed_values(seed);
        // numpy seeds the increment first — `(stream << 1) | 1` — then advances once, then adds
        // the initial state and advances again. Seeding the state directly would give a valid
        // PCG64 stream and the wrong one.
        let increment = (stream << 1) | 1;
        let mut generator = Self {
            state: 0,
            increment,
        };
        generator.step();
        generator.state = generator.state.wrapping_add(state);
        generator.step();
        generator
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.increment);
    }

    /// One 64-bit word — numpy's `pcg64_next64`, whose output function is XSL-RR.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let state = self.state;
        let xored = ((state >> 64) as u64) ^ (state as u64);
        let rotation = (state >> 122) as u32;
        xored.rotate_right(rotation)
    }

    /// numpy's `next_double`: the top 53 bits of one word, scaled — the same `>> 11` and
    /// `/ 9007199254740992.0` `RandomState` uses, over a different stream.
    pub fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

/// numpy's `random_poisson`, which branches at `lam == 10`.
///
/// Below ten it is the multiplication method: draw uniforms and multiply until the running product
/// falls at or under `exp(-lam)`, counting the draws before that. At ten and above it is transformed
/// rejection (PTRS), which draws in *pairs* and may reject — so the two branches consume the
/// generator at different rates, and picking the wrong one desynchronises every later draw as well
/// as returning a different number.
impl Pcg64 {
    pub fn poisson(&mut self, lam: f64) -> u64 {
        if lam >= 10.0 {
            return self.poisson_ptrs(lam);
        }
        if lam == 0.0 {
            return 0;
        }
        let enlam = (-lam).exp();
        let mut product = 1.0;
        // numpy multiplies uniforms until the product drops below `exp(-lam)`. The count is
        // unbounded in principle and never large in practice — at the branch's own ceiling of
        // `lam < 10` the mean is under ten and the tail falls off geometrically. The bound is here
        // because a draw a mutant pins at one would otherwise never let the product fall.
        for drawn in 0..10_000u64 {
            product *= self.next_double();
            if product <= enlam {
                return drawn;
            }
        }
        panic!("ten thousand uniforms multiplied without falling below exp(-{lam})")
    }

    /// numpy's `random_poisson_ptrs`: Hörmann's transformed rejection, two uniforms per attempt.
    ///
    /// Two of the three arms are *squeezes* — shortcuts that decide a point without evaluating the
    /// density, and that are correct because the region each one claims sits strictly inside what
    /// the density test would have decided anyway. `us >= 0.07 && v <= vr` accepts only points the
    /// third arm also accepts (checked over 360,000 draws: no acceptance outside it), and
    /// `us < 0.013 && v > us` rejects only proposals so far into the tail that the density rejects
    /// them too — at `lam` ten, `us` that small puts `k` near 27, whose log density is about -21
    /// against a left-hand side that cannot fall below -7.
    ///
    /// That is why nine mutants on this function survive a corpus that reaches every arm tens of
    /// thousands of times: **narrowing a squeeze is unobservable by construction.** All four ways
    /// to break `vr` make it smaller or negative, which only sends work to the density test; the
    /// two ways to shrink `us < 0.013` and the two that shrink `v > us` toward equality do the
    /// same. Only a mutant that moves a squeeze *across* the density test can show up: reversing
    /// `v > us` does, since the points it then rejects are the small-`v` ones the density accepts.
    /// The `k < 0` guard is not a squeeze at all — it drops proposals outside the support — so
    /// widening it to `k <= 0` or narrowing it to `k == 0` changes a real decision. All three are
    /// caught, by the deep streams the golden carries for exactly this.
    ///
    /// The ninth, `||` weakened to `&&`, is the same principle by a different route: a negative
    /// `k` that no longer retries falls through to the density test, where `loggam` of a
    /// non-positive argument is NaN and the comparison is false, so it retries anyway having
    /// consumed the same two uniforms.
    fn poisson_ptrs(&mut self, lam: f64) -> u64 {
        let slam = lam.sqrt();
        let loglam = lam.ln();
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.02483 * b;
        let invalpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        // Rejection sampling, so the attempt count is unbounded in principle. Hörmann's method
        // accepts well over ninety per cent of attempts, which makes ten thousand rejections in a
        // row impossible in practice and a stalled draw the only way to reach the bound.
        for _ in 0..10_000u32 {
            let u = self.next_double() - 0.5;
            let v = self.next_double();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
            if us >= 0.07 && v <= vr {
                return k as u64;
            }
            if k < 0.0 || (us < 0.013 && v > us) {
                continue;
            }
            if v.ln() + invalpha.ln() - (a / (us * us) + b).ln()
                <= -lam + k * loglam - loggam(k + 1.0)
            {
                return k as u64;
            }
        }
        panic!("ten thousand rejections drawing a poisson at lambda {lam}")
    }
}

/// numpy's `loggam`: `ln Γ(x)` by its own series, reproduced rather than taken from a crate.
///
/// PTRS compares against this, so a different implementation of the same mathematical function
/// changes which draws are accepted — the last digit decides a rejection.
///
/// Held twice over, because the two hold different things. The poisson streams hold it to numpy
/// exactly and are the fidelity claim; what they do not reach is the tail of the series, since the
/// acceptance test is an inequality whose sides are far apart. The smallest margin over 66,427
/// comparisons swept for this is `4.3e-06`, so a coefficient worth less than that moves no draw,
/// and only the second is worth more — `1.6e-05`, and the golden now carries the one stream in
/// 350,000 where it decides. `tests/conformance/loggam.json` holds the rest, against CPython's
/// `lgamma`, which asks whether the series is ln Γ at all rather than whether it matches numpy.
///
/// Three lines survive both, and the reasons are worth keeping because none is a near miss. `A[9]`
/// seeds the Horner evaluation and is multiplied by `x2` nine times before it meets `A[0]`; with
/// `x0` never below seven that is a factor of `49^-9`, and flipping its sign changes the returned
/// `f64` by exactly zero. `n` is a free shift, since `ln Γ(x) = ln Γ(x + n) - Σ ln(x + i)` holds
/// for any non-negative `n` — computing it from `7 + x` rather than `7 - x` costs `7e-15` against
/// two implementations that agree to `8.9e-16`. (`7 / x` is *not* free and is caught: it can leave
/// `x0` at five, where the series is worth `2e-14`.) And disabling the early return for one and two
/// gives `5.6e-16` there instead of the exact zero, below both bounds.
fn loggam(x: f64) -> f64 {
    // numpy's coefficients, spelled as numpy spells them. Two carry a digit past what an `f64`
    // keeps, and rounding them to what it keeps would be editing the transcription to please a
    // lint — the same literal, but no longer the same text as the source it is checked against.
    #[allow(clippy::excessive_precision)]
    const A: [f64; 10] = [
        8.333333333333333e-02,
        -2.777777777777778e-03,
        7.936507936507937e-04,
        -5.952380952380952e-04,
        8.417508417508418e-04,
        -1.917526917526918e-03,
        6.410256410256410e-03,
        -2.955065359477124e-02,
        1.796443723688307e-01,
        -1.39243221690590e+00,
    ];
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    let mut x0 = x;
    let mut n = 0i64;
    if x <= 7.0 {
        n = (7.0 - x) as i64;
        x0 = x + n as f64;
    }
    let x2 = 1.0 / (x0 * x0);
    let mut gl0 = A[9];
    for coefficient in A.iter().take(9).rev() {
        gl0 *= x2;
        gl0 += coefficient;
    }
    let mut gl = gl0 / x0 + 0.5 * (2.0 * std::f64::consts::PI).ln() + (x0 - 0.5) * x0.ln() - x0;
    if x <= 7.0 {
        for _ in 1..=n {
            gl -= (x0 - 1.0).ln();
            x0 -= 1.0;
        }
    }
    gl
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// `loggam` against CPython's `lgamma` — a second implementation of the same function, the way
    /// `rand_pcg` is a second implementation of the stepper.
    ///
    /// The poisson streams hold `loggam` to numpy exactly, which is the fidelity claim. They do not
    /// hold its coefficients to much: it is read through an inequality whose sides are usually far
    /// apart, so a coefficient worth less than the smallest margin moves no draw. This asks the
    /// sharper question instead — whether the series is ln Γ at all — and catches the fourth and
    /// sixth coefficients, which poisson would need some `10^8` draws to reach.
    #[test]
    fn the_series_agrees_with_an_independent_ln_gamma() {
        let golden: Value = serde_json::from_str(include_str!("../tests/conformance/loggam.json"))
            .expect("the loggam golden is valid JSON");
        let tolerance = golden["relative_tolerance"].as_f64().expect("a tolerance");
        let values = golden["values"].as_array().expect("values");
        assert!(
            values.len() >= 130,
            "the golden lost values: {}",
            values.len()
        );
        let mut worst = 0.0f64;
        for value in values {
            let x = value["x"].as_f64().expect("an x");
            let expected = value["lgamma"].as_f64().expect("an lgamma");
            let relative = (loggam(x) - expected).abs() / expected.abs().max(1.0);
            assert!(
                relative <= tolerance,
                "loggam({x}) is {} against lgamma's {expected}, a relative {relative:e} past \
                 {tolerance:e}",
                loggam(x)
            );
            worst = worst.max(relative);
        }
        // The two differ by four ULPs — `8.88e-16`, the same value on both sides, since `ln` is
        // the same libm call. The bound above is a hundred times that, and this asserts the
        // hundred is still there, so a bound that has quietly gone slack fails rather than passes
        // wider. Ten is the slack this guard allows itself: closer than that to the tolerance and
        // the tolerance is no longer describing anything.
        assert!(
            worst < tolerance / 10.0,
            "the two implementations now differ by {worst:e}, close enough to the {tolerance:e} \
             bound that it has stopped being headroom"
        );
    }
}
