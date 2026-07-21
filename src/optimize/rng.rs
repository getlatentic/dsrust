//! `random.Random`, ported so a compile draws the examples dspy's compile draws.
//!
//! Which examples an optimizer keeps *is* its output, and here that output is whatever these
//! draws select — so the generator and both sampling algorithms are reproduced exactly, and held
//! to `tests/conformance/rng/cpython_random.json`. dspy reuses one generator across a walk, so
//! the advancing between draws is as load-bearing as the seed.

mod mersenne;

use std::collections::HashSet;

use mersenne::Mersenne;

pub(super) struct Rng(Mersenne);

impl Rng {
    /// dspy's `random.Random(seed)`.
    pub(super) fn seeded(seed: u64) -> Self {
        Self(Mersenne::from_key(&key(seed)))
    }

    /// CPython `_randbelow`: draw the bound's bit width, and redraw until it lands in range.
    ///
    /// The rejections are observable — a discarded draw still advances the generator — so a
    /// remainder here would shift every number that follows, not just skew this one.
    fn below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0, "defined for a bound above zero, as CPython's is");
        let bits = usize::BITS - bound.leading_zeros();
        loop {
            let drawn = self.0.getrandbits(bits) as usize;
            if drawn < bound {
                return drawn;
            }
        }
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
        let wanted = k.min(population.len());
        if population.len() <= setsize(wanted) {
            self.sample_from_pool(population, wanted)
        } else {
            self.sample_by_index(population, wanted)
        }
    }

    /// CPython's small-population branch: draw from a pool, backfilling each hole from the tail.
    fn sample_from_pool<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        let size = population.len();
        let mut pool = population.to_vec();
        (0..k)
            .map(|taken| {
                let at = self.below(size - taken);
                let chosen = pool[at].clone();
                pool[at] = pool[size - taken - 1].clone();
                chosen
            })
            .collect()
    }

    /// CPython's large-population branch: remember which indices were drawn and redraw past
    /// them. It consumes a different number of draws than the pool branch for the same `k`, so
    /// the threshold between them decides the answer and has to be reproduced, not chosen.
    fn sample_by_index<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        let mut drawn = HashSet::with_capacity(k);
        (0..k)
            .map(|_| {
                let mut at = self.below(population.len());
                while !drawn.insert(at) {
                    at = self.below(population.len());
                }
                population[at].clone()
            })
            .collect()
    }
}

/// CPython `random_seed`: an integer seed spread over 32-bit words, least significant first.
///
/// Zero takes one zero word rather than none, which is the case dspy actually seeds with.
fn key(seed: u64) -> Vec<u32> {
    let mut key = vec![seed as u32];
    let mut rest = seed >> 32;
    while rest > 0 {
        key.push(rest as u32);
        rest >>= 32;
    }
    key
}

/// The population size at which CPython stops tracking a pool and starts tracking drawn indices.
///
/// Upstream writes `21 + 4 ** ceil(log(k * 3, 4))`, the smallest power of four at or above
/// `k * 3`. A loop reaches the same number without floating point, and cannot round the other
/// way: that would need `k * 3` to be a power of four, and no power of four is divisible by
/// three. `scripts/generate_rng_fixture.py` asserts the two agree rather than leaving it argued.
fn setsize(k: usize) -> usize {
    if k <= 5 {
        return 21;
    }
    let mut table = 1usize;
    while table < k * 3 {
        table *= 4;
    }
    21 + table
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::Value;

    /// The draws CPython makes, recorded by running it. Regenerate with
    /// `scripts/generate_rng_fixture.py`.
    fn golden() -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/rng/cpython_random.json");
        let text = std::fs::read_to_string(&path).expect("the CPython golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    fn cases(section: &str) -> Vec<Value> {
        golden()[section]
            .as_array()
            .unwrap_or_else(|| panic!("the golden has a {section} section"))
            .clone()
    }

    fn number(case: &Value, field: &str) -> usize {
        case[field].as_u64().expect("a number") as usize
    }

    fn integers(case: &Value, field: &str) -> Vec<usize> {
        case[field]
            .as_array()
            .expect("a list")
            .iter()
            .map(|value| value.as_u64().expect("a number") as usize)
            .collect()
    }

    #[test]
    fn draws_the_bits_cpython_draws() {
        for case in cases("getrandbits") {
            let seed = number(&case, "seed") as u64;
            let bits = number(&case, "bits") as u32;
            let mut rng = Rng::seeded(seed);
            let expected: Vec<u64> = case["draws"]
                .as_array()
                .expect("a list")
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .expect("a string")
                        .parse()
                        .expect("an integer")
                })
                .collect();
            let actual: Vec<u64> = expected.iter().map(|_| rng.0.getrandbits(bits)).collect();
            assert_eq!(actual, expected, "getrandbits({bits}) from seed {seed}");
        }
    }

    #[test]
    fn lands_below_a_bound_where_cpython_lands() {
        for case in cases("randbelow") {
            let seed = number(&case, "seed") as u64;
            let bound = number(&case, "bound");
            let expected = integers(&case, "draws");
            let mut rng = Rng::seeded(seed);
            let actual: Vec<usize> = expected.iter().map(|_| rng.below(bound)).collect();
            assert_eq!(actual, expected, "_randbelow({bound}) from seed {seed}");
        }
    }

    #[test]
    fn shuffles_into_cpythons_order() {
        for case in cases("shuffle") {
            let seed = number(&case, "seed") as u64;
            let size = number(&case, "population");
            let mut items: Vec<usize> = (0..size).collect();
            Rng::seeded(seed).shuffle(&mut items);
            assert_eq!(
                items,
                integers(&case, "result"),
                "shuffle({size}) from seed {seed}"
            );
        }
    }

    #[test]
    fn samples_the_members_cpython_samples() {
        for case in cases("sample") {
            let seed = number(&case, "seed") as u64;
            let size = number(&case, "population");
            let k = number(&case, "k");
            let population: Vec<usize> = (0..size).collect();
            let drawn = Rng::seeded(seed).sample(&population, k);
            assert_eq!(
                drawn,
                integers(&case, "result"),
                "sample({size}, {k}) from seed {seed}, threshold {}",
                number(&case, "setsize")
            );
        }
    }

    /// The canonical `mt19937ar.c` sequence: `init_by_array` over the key that implementation
    /// publishes, then a thousand consecutive draws.
    ///
    /// This is the one check here that does not trace back to a locally installed interpreter.
    /// The sequence is what every Mersenne Twister is measured against, and this fixture's copy
    /// was confirmed against an independent transcription of the 2002 reference before landing.
    #[test]
    fn reproduces_the_published_reference_sequence() {
        let reference = &golden()["reference"];
        let key: Vec<u32> = reference["key"]
            .as_array()
            .expect("a key")
            .iter()
            .map(|word| word.as_u64().expect("a word") as u32)
            .collect();
        let expected: Vec<u64> = reference["draws"]
            .as_array()
            .expect("a list")
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .expect("a string")
                    .parse()
                    .expect("an integer")
            })
            .collect();

        let mut generator = Mersenne::from_key(&key);
        let drawn: Vec<u64> = expected.iter().map(|_| generator.getrandbits(32)).collect();
        assert_eq!(
            drawn.len(),
            1000,
            "the reference publishes a thousand draws"
        );
        assert_eq!(drawn, expected);
    }

    /// The threshold decides which sampling algorithm runs, and the two draw differently. A
    /// fixture either side of it is what makes that visible, so guard the pair.
    #[test]
    fn covers_both_sampling_branches() {
        let branches: Vec<bool> = cases("sample")
            .iter()
            .map(|case| number(case, "population") <= setsize(number(case, "k")))
            .collect();
        assert!(branches.contains(&true), "no fixture takes the pool branch");
        assert!(
            branches.contains(&false),
            "no fixture takes the drawn-index branch"
        );
    }

    #[test]
    fn agrees_with_cpython_on_the_sampling_threshold() {
        for case in cases("sample") {
            assert_eq!(
                setsize(number(&case, "k")),
                number(&case, "setsize"),
                "threshold for k={}",
                number(&case, "k")
            );
        }
    }

    #[test]
    fn spreads_a_seed_over_words_the_way_cpython_does() {
        assert_eq!(key(0), vec![0], "a seed of zero still spends a word");
        assert_eq!(key(42), vec![42]);
        assert_eq!(key(1 << 32), vec![0, 1], "least significant word first");
    }

    /// CPython seeds by walking the longer of its state and the key. Nothing here can make the
    /// key the longer one, which is why that branch is never exercised — pinned so that widening
    /// the seed type has to come back through this test.
    #[test]
    fn a_seed_never_outruns_the_generators_state() {
        assert!(
            key(u64::MAX).len() < 624,
            "a wider seed would reach CPython's other branch"
        );
    }
}
