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
        let mut source: Vec<u32> = Vec::new();
        let mut rest = entropy;
        while rest > 0 {
            source.push((rest & 0xffff_ffff) as u32);
            rest >>= 32;
        }
        if source.is_empty() {
            source.push(0);
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
            for slot in 0..pool.len() {
                pool[slot] = mix(pool[slot], hashmix(*extra, &mut hash_const));
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
        let mut drawn = 0u64;
        let mut product = 1.0;
        loop {
            product *= self.next_double();
            if product > enlam {
                drawn += 1;
            } else {
                return drawn;
            }
        }
    }

    /// numpy's `random_poisson_ptrs`: Hörmann's transformed rejection, two uniforms per attempt.
    fn poisson_ptrs(&mut self, lam: f64) -> u64 {
        let slam = lam.sqrt();
        let loglam = lam.ln();
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.02483 * b;
        let invalpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
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
    }
}

/// numpy's `loggam`: `ln Γ(x)` by its own series, reproduced rather than taken from a crate.
///
/// PTRS compares against this, so a different implementation of the same mathematical function
/// changes which draws are accepted — the last digit decides a rejection.
fn loggam(x: f64) -> f64 {
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
