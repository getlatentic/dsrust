//! Deterministic randomness, without pulling in a random-number crate.
//!
//! dspy seeds `random.Random(0)` so a compile is reproducible, then draws from that one
//! generator repeatedly: two predictors filled in the same loop get different demos because the
//! generator advanced between them. Both halves are decisions a port has to keep — the fixed
//! seed, and the advancing.
//!
//! What is not on the table is reproducing CPython's Mersenne Twister bit for bit; no Rust crate
//! does, and this one takes no dependency to try. A compile here is therefore reproducible
//! rather than identical to dspy's: same shapes, same sizes, different picks. Said out loud
//! because a caller comparing the two implementations would otherwise expect the same demos.

/// A seeded generator standing in for one `random.Random` instance.
pub(super) struct Rng(u64);

impl Rng {
    /// dspy's `random.Random(seed)`.
    pub(super) fn seeded(seed: u64) -> Self {
        // Zero is a fixed point of xorshift, so the seed is offset past it rather than used raw.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound.max(1) as u64) as usize
    }

    /// `random.shuffle`: Fisher-Yates walked from the end, the direction CPython walks it.
    pub(super) fn shuffle<T>(&mut self, items: &mut [T]) {
        for position in (1..items.len()).rev() {
            items.swap(position, self.below(position + 1));
        }
    }

    /// `random.sample`: `k` distinct members in the order they were drawn — neither a prefix of
    /// the population nor a sorted subset, which is what separates a sample from a take.
    ///
    /// Python raises when `k` exceeds the population; this clamps instead. Every dspy call site
    /// already takes that minimum itself, so the two agree wherever upstream can reach.
    pub(super) fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        let mut pool = population.to_vec();
        (0..k.min(pool.len()))
            .map(|_| {
                // CPython's pool branch: draw one, then backfill the hole from the tail.
                let drawn = self.below(pool.len());
                pool.swap_remove(drawn)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digits() -> Vec<usize> {
        (0..8).collect()
    }

    #[test]
    fn the_same_seed_replays_the_same_run() {
        let draw = |seed| {
            let mut rng = Rng::seeded(seed);
            let mut items = digits();
            rng.shuffle(&mut items);
            (items, rng.sample(&digits(), 3))
        };
        assert_eq!(draw(0), draw(0));
    }

    #[test]
    fn a_shuffle_keeps_every_member_and_moves_some() {
        let mut items = digits();
        Rng::seeded(0).shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, digits());
        assert_ne!(
            items,
            digits(),
            "a shuffle that reorders nothing is not one"
        );
    }

    #[test]
    fn a_sample_draws_distinct_members() {
        let drawn = Rng::seeded(0).sample(&digits(), 5);
        let mut sorted = drawn.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 5, "drew a member twice: {drawn:?}");
    }

    /// dspy asks for `min(k, len(population))` everywhere; clamping here means a caller that
    /// forgets gets the whole population rather than a panic on the modulo.
    #[test]
    fn a_sample_larger_than_the_population_yields_the_population() {
        let drawn = Rng::seeded(0).sample(&digits(), 99);
        assert_eq!(drawn.len(), digits().len());
    }

    /// The property `LabeledFewShot` and `_train` both lean on: one generator, drawn twice,
    /// gives two different answers. A generator reseeded per draw would hand every predictor
    /// the same demos.
    #[test]
    fn successive_draws_from_one_generator_differ() {
        let mut rng = Rng::seeded(0);
        let first = rng.sample(&digits(), 4);
        let second = rng.sample(&digits(), 4);
        assert_ne!(first, second);
    }

    #[test]
    fn an_empty_population_samples_to_nothing() {
        assert!(Rng::seeded(0).sample::<usize>(&[], 3).is_empty());
    }
}
