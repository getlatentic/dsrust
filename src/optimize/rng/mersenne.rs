//! CPython's Mersenne Twister: the generator `random.Random` draws every number from.
//!
//! Reproduced word for word rather than stood in for. dspy seeds `random.Random(0)`, so the
//! examples an optimizer keeps are whichever ones this generator picks; a merely deterministic
//! stand-in gives a reproducible compile that is not dspy's compile, and the two cannot then be
//! compared demo for demo.

/// Words of state the recurrence carries.
const N: usize = 624;
/// How far ahead the recurrence reaches when it twists one word.
const M: usize = 397;
/// The polynomial the twist folds in for an odd word.
const MATRIX_A: u32 = 0x9908_b0df;
/// The one bit taken from a word, and the thirty-one taken from its neighbour.
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub(super) struct Mersenne {
    state: [u32; N],
    /// How far into `state` the next draw reads. At `N` the state is spent and must twist.
    next: usize,
}

impl Mersenne {
    /// CPython `init_by_array`, which is what seeding with an integer reaches.
    ///
    /// The key is never empty: CPython spends one word on a seed of zero rather than none.
    pub(super) fn from_key(key: &[u32]) -> Self {
        debug_assert!(
            !key.is_empty(),
            "CPython spends a word even on a seed of zero"
        );
        let mut this = Self::from_word(19_650_218);
        let (mut at, mut key_at) = (1usize, 0usize);
        // CPython walks the longer of the state and the key. A seed narrower than 624 words can
        // never be the longer one, so this reads as `N` for every seed reachable here; it is
        // written as upstream writes it because the seed type is what makes that true, not the
        // algorithm.
        for _ in 0..N.max(key.len()) {
            let previous = this.state[at - 1];
            this.state[at] = (this.state[at]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[key_at])
            .wrapping_add(key_at as u32);
            at = Self::step(&mut this.state, at);
            key_at = (key_at + 1) % key.len();
        }
        for _ in 0..N - 1 {
            let previous = this.state[at - 1];
            this.state[at] = (this.state[at]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(at as u32);
            at = Self::step(&mut this.state, at);
        }
        // Leaves the state's one distinguishing bit set, which is what makes the period full.
        this.state[0] = UPPER_MASK;
        this
    }

    /// Advance the seeding walk, wrapping past the end by carrying the last word to the front.
    fn step(state: &mut [u32; N], at: usize) -> usize {
        if at + 1 >= N {
            state[0] = state[N - 1];
            return 1;
        }
        at + 1
    }

    /// CPython `init_genrand`: Knuth's recurrence, which seeds the seeding.
    fn from_word(seed: u32) -> Self {
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

    /// CPython `genrand_uint32`: one 32-bit word.
    fn next_word(&mut self) -> u32 {
        if self.next >= N {
            self.twist();
        }
        let drawn = self.state[self.next];
        self.next += 1;
        temper(drawn)
    }

    /// Refill the state, one word at a time, in place.
    fn twist(&mut self) {
        for at in 0..N {
            let joined = (self.state[at] & UPPER_MASK) | (self.state[(at + 1) % N] & LOWER_MASK);
            let odd = if joined & 1 == 0 { 0 } else { MATRIX_A };
            // Wraps back into words this pass has already rewritten, which the recurrence
            // intends: the second half of the state twists against the freshly written first.
            self.state[at] = self.state[(at + M) % N] ^ (joined >> 1) ^ odd;
        }
        self.next = 0;
    }

    /// CPython `getrandbits(k)`: `k` random bits, filled a word at a time from the low end.
    ///
    /// Stops at 64 bits, where Python would carry on into an arbitrary-width integer. Callers
    /// ask for the bit width of a population size, so nothing here reaches that far.
    pub(super) fn getrandbits(&mut self, bits: u32) -> u64 {
        debug_assert!(
            bits <= 64,
            "wider than a u64 needs Python's arbitrary-width integer"
        );
        let mut drawn = 0u64;
        let mut shift = 0u32;
        let mut owed = bits;
        while owed > 0 {
            let word = self.next_word();
            // The last word is shifted down to leave exactly the bits still owed.
            let word = if owed < 32 { word >> (32 - owed) } else { word };
            drawn |= u64::from(word) << shift;
            shift += 32;
            owed = owed.saturating_sub(32);
        }
        drawn
    }
}

/// Scramble one word of raw state. The recurrence on its own leaves structure a test can see;
/// these shifts are what make the output equidistributed.
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
    /// These are not CPython's numbers. CPython carries them to check *itself* against the
    /// original Matsumoto-Nishimura C implementation, seeding `init_by_array` with a four-word
    /// key and taking the last ten of two thousand draws at full 53-bit precision. Borrowing
    /// them puts this port against that same reference rather than against a recording of the
    /// interpreter that happens to be installed.
    ///
    /// It is also the only check here that seeds with more than one word, which is what makes
    /// the key walking in [`Mersenne::from_key`] observable at all.
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

        let mut generator = Mersenne::from_key(&[61731, 24903, 614, 42143]);
        // `genrand_res53`, the draw behind `random.random()`: two words spliced into one 53-bit
        // fraction. Compared as that fraction's numerator, so nothing rests on float formatting.
        let drawn: Vec<u64> = (0..2000)
            .map(|_| {
                let (high, low) = (generator.next_word(), generator.next_word());
                u64::from(high >> 5) * (1 << 26) + u64::from(low >> 6)
            })
            .collect();

        assert_eq!(drawn[drawn.len() - 10..], expected);
    }
}
