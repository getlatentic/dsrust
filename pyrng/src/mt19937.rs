//! The MT19937 generator CPython's `random` and numpy's `RandomState` both draw from.
//!
//! Reproduced word for word rather than stood in for. The same 624-word state, twist and temper
//! serve both — they differ only in how they *seed* it ([`Mt19937::from_word`] vs
//! [`Mt19937::from_key`]) and what they read out of it (see [`crate::cpython`] and [`crate::numpy`]).

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// The raw generator: the state and the position within it, with the two Python seedings and the
/// reads each flavour builds on.
pub struct Mt19937 {
    state: [u32; N],
    /// At `N` the state is spent and must twist.
    next: usize,
}

impl Mt19937 {
    /// `init_genrand`: Knuth's recurrence from a single word, the seed itself word zero. numpy's
    /// `RandomState(int)` seeds exactly this, and CPython's `init_by_array` starts from it.
    pub fn from_word(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for at in 1..N {
            let previous = state[at - 1];
            state[at] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(at as u32);
        }
        Self { state, next: N }
    }

    /// CPython `init_by_array`: seed from a key of words, which is what `random.Random(int)` reaches.
    ///
    /// The key is never empty: CPython spends one word on a seed of zero rather than none.
    pub fn from_key(key: &[u32]) -> Self {
        debug_assert!(!key.is_empty(), "CPython spends a word even on a seed of zero");
        let mut this = Self::from_word(19_650_218);
        let (mut at, mut key_at) = (1usize, 0usize);
        // CPython walks the longer of the state and the key. A seed narrower than 624 words can
        // never be the longer one, so this reads as `N` for every seed reachable here; it is
        // written as upstream writes it because the seed type is what makes that true.
        for _ in 0..N.max(key.len()) {
            let previous = this.state[at - 1];
            this.state[at] = (this.state[at] ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[key_at])
                .wrapping_add(key_at as u32);
            at = step(&mut this.state, at);
            key_at = (key_at + 1) % key.len();
        }
        for _ in 0..N - 1 {
            let previous = this.state[at - 1];
            this.state[at] = (this.state[at]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(at as u32);
            at = step(&mut this.state, at);
        }
        // Leaves the state's one distinguishing bit set, which is what makes the period full.
        this.state[0] = UPPER_MASK;
        this
    }

    /// One word of the raw state, for holding a seeding to numpy's `get_state()` directly rather
    /// than only through the draws that follow it.
    pub fn state_word(&self, index: usize) -> u32 {
        self.state[index]
    }

    /// One tempered 32-bit word, twisting the state forward when the block is spent — CPython's
    /// `genrand_uint32` and numpy's `rk_random`.
    pub fn next_u32(&mut self) -> u32 {
        if self.next >= N {
            self.twist();
        }
        let drawn = self.state[self.next];
        self.next += 1;
        temper(drawn)
    }

    fn twist(&mut self) {
        for at in 0..N {
            let joined = (self.state[at] & UPPER_MASK) | (self.state[(at + 1) % N] & LOWER_MASK);
            let odd = if joined & 1 == 0 { 0 } else { MATRIX_A };
            self.state[at] = self.state[(at + M) % N] ^ (joined >> 1) ^ odd;
        }
        self.next = 0;
    }

    /// CPython `getrandbits(k)`: `k` random bits, filled a word at a time from the low end.
    ///
    /// Stops at 64 bits, where Python would carry on into an arbitrary-width integer. Callers ask
    /// for the bit width of a population size, so nothing here reaches that far.
    pub fn getrandbits(&mut self, bits: u32) -> u64 {
        debug_assert!(bits <= 64, "wider than a u64 needs Python's arbitrary-width integer");
        let mut drawn = 0u64;
        let mut shift = 0u32;
        let mut owed = bits;
        while owed > 0 {
            let word = self.next_u32();
            // The last word is shifted down to leave exactly the bits still owed.
            let word = if owed < 32 { word >> (32 - owed) } else { word };
            drawn |= u64::from(word) << shift;
            shift += 32;
            owed = owed.saturating_sub(32);
        }
        drawn
    }

    /// `genrand_res53` / `rk_double`: a 53-bit double in `[0, 1)`, the high 27 bits of one word and
    /// the high 26 of the next. CPython's `random()` and numpy's `random_sample` share this.
    pub fn random_double(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }
}

/// Advance the write cursor CPython's `init_by_array` uses, wrapping the state's last word to its
/// first when it laps.
fn step(state: &mut [u32; N], at: usize) -> usize {
    if at + 1 >= N {
        state[0] = state[N - 1];
        return 1;
    }
    at + 1
}

fn temper(mut word: u32) -> u32 {
    word ^= word >> 11;
    word ^= (word << 7) & 0x9d2c_5680;
    word ^= (word << 15) & 0xefc6_0000;
    word ^ (word >> 18)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPython `Lib/test/test_random.py::test_strong_reference_implementation`.
    ///
    /// Not CPython's numbers: CPython carries these to check *itself* against the original
    /// Matsumoto-Nishimura C implementation, seeding `init_by_array` with a four-word key and taking
    /// the last ten of two thousand `genrand_res53` draws at full 53-bit precision. Borrowing them
    /// puts this port against that same reference, and is the only check here that seeds with more
    /// than one word — what makes the key walking in [`Mt19937::from_key`] observable.
    #[test]
    fn matches_the_original_c_implementations_published_draws() {
        let expected = [
            0x000e_ab32_58d2_231f,
            0x001b_89db_3152_77a5,
            0x001d_b622_a551_8016,
            0x000b_7f9a_f0d5_75bf,
            0x0002_9e4c_4db8_2240,
            0x0004_9618_92f5_d673,
            0x0002_b291_598e_4589,
            0x0011_3883_82c1_5694,
            0x0002_dad9_77c9_e1fe,
            0x0019_1d96_d4d3_34c6,
        ];
        let mut generator = Mt19937::from_key(&[61731, 24903, 614, 42143]);
        let drawn: Vec<u64> = (0..2000)
            .map(|_| {
                let (high, low) = (generator.next_u32(), generator.next_u32());
                u64::from(high >> 5) * (1 << 26) + u64::from(low >> 6)
            })
            .collect();
        assert_eq!(drawn[drawn.len() - 10..], expected);
    }
}
