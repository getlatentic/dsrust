//! CPython's `random.Random`, ported so a compile draws the examples dspy's compile draws.
//!
//! Which examples an optimizer keeps *is* its output, and here that output is whatever these draws
//! select — so the generator and both config algorithms are reproduced exactly, and held to
//! `tests/conformance/cpython_random.json`. dspy reuses one generator across a walk, so the
//! advancing between draws is as load-bearing as the seed.

use std::collections::HashSet;

use crate::mt19937::Mt19937;

/// CPython `random.Random`: seed it as `random.Random(int)`, then draw.
pub struct Random(Mt19937);

impl Random {
    /// dspy's `random.Random(seed)`.
    pub fn seeded(seed: u64) -> Self {
        Self(Mt19937::from_key(&key(seed)))
    }

    /// CPython `_randbelow`: draw the bound's bit width, and redraw until it lands in range.
    ///
    /// The rejections are observable — a discarded draw still advances the generator — so a
    /// remainder here would shift every number that follows, not just skew this one.
    pub fn below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0, "defined for a bound above zero, as CPython's is");
        let bits = usize::BITS - bound.leading_zeros();
        // CPython's rejection loop, bounded. `getrandbits(bits)` spans at most twice `bound`, so a
        // draw is accepted with probability above one half and a thousand consecutive rejections
        // has probability below 2^-1000. The bound is not there for that: it is there because the
        // loop's termination rests on the comparison being the right way round, and reversing it
        // spins a core for the full mutation timeout instead of failing a test.
        for _ in 0..1000 {
            let drawn = self.0.getrandbits(bits) as usize;
            if drawn < bound {
                return drawn;
            }
        }
        panic!("a thousand draws of {bits} bits all landed at or above {bound}")
    }

    /// CPython `random.random()`: a float in `[0, 1)` from 53 bits, which is `genrand_res53`.
    ///
    /// The one draw that is not an integer, and the one gepa's epsilon-greedy selector opens with.
    pub fn random(&mut self) -> f64 {
        self.0.random_double()
    }

    /// CPython `random.choice(seq)`: the element at `_randbelow(len)`. Returned as the index so a
    /// caller can pick from any sequence.
    pub fn choice_index(&mut self, len: usize) -> usize {
        self.below(len)
    }

    /// CPython `random.choices(population, weights=..., k=k)`: `k` weighted draws, each the index
    /// `bisect_right(cumulative_weights, random() * total)`. Returned as indices so a caller picks
    /// from its own population.
    ///
    /// Draws through `random()` — the 53-bit double — not `_randbelow`, so this consumes the
    /// generator differently from [`Self::choice_index`]; a caller mixing the two on one generator
    /// only reproduces dspy if it mixes them in the same order. Weights need not sum to one; they
    /// are accumulated and the draw scaled by the total, exactly as CPython does.
    pub fn choices(&mut self, weights: &[f64], k: usize) -> Vec<usize> {
        let cumulative: Vec<f64> = weights
            .iter()
            .scan(0.0, |sum, weight| {
                *sum += weight;
                Some(*sum)
            })
            .collect();
        let total = cumulative.last().copied().unwrap_or(0.0);
        // CPython caps the index at the last element so a draw that rounds up to the total still
        // lands in the population. `len() - 1` against `len() / 1` needs `random_double() * total`
        // to reach `total`, which takes a rounding edge at a large total rather than an ordinary
        // draw — `random_double()` is strictly below one.
        let hi = cumulative.len() - 1;
        (0..k)
            .map(|_| {
                let target = self.0.random_double() * total;
                bisect_right(&cumulative, target, hi)
            })
            .collect()
    }

    /// CPython `random.randint(low, high)` (inclusive): `low + _randbelow(high - low + 1)`.
    pub fn randint(&mut self, low: u64, high: u64) -> u64 {
        low + self.below((high - low + 1) as usize) as u64
    }

    /// `random.shuffle`: Fisher-Yates walked from the end, the direction CPython walks it.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for position in (1..items.len()).rev() {
            items.swap(position, self.below(position + 1));
        }
    }

    /// `random.sample`: `k` distinct members in the order they were drawn — neither a prefix of the
    /// population nor a sorted subset, which is what separates a sample from a take.
    ///
    /// Python raises when `k` exceeds the population; this clamps instead. Every dspy call site
    /// already takes that minimum itself, so the two agree wherever upstream can reach.
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
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

    /// CPython's large-population branch: remember which indices were drawn and redraw past them.
    /// It consumes a different number of draws than the pool branch for the same `k`, so the
    /// threshold between them decides the answer and has to be reproduced, not chosen.
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

/// Python's `bisect_right(a, x, 0, hi)`: the leftmost position at which `x` could be inserted to
/// keep `a` sorted and stay to the right of any equal value, bounded above by `hi`.
///
/// `hi` is `len - 1`, matching CPython's `random.choices`, which caps the index at the last
/// element so a draw that rounds up to the total still lands in the population.
fn bisect_right(sorted: &[f64], x: f64, hi: usize) -> usize {
    // `partition_point` rather than a hand-rolled halving loop. The loop was right, and its
    // termination rested on `(low + high) / 2` staying strictly between the bounds — so changing
    // the `/` or the `+` did not fail a test, it hung the suite for the full timeout. The
    // predicate is `!(x < v)` rather than `v <= x` to keep CPython's comparison exactly: it tests
    // `x < a[mid]` and takes the else branch otherwise, so a NaN in the population falls right,
    // where `v <= x` would send it left.
    // See `numpy::searchsorted_right`: the `<=` spelling needs a draw exactly on a boundary.
    sorted[..hi].partition_point(|&v| !(x < v))
}

/// CPython `random_seed`: an integer seed spread over 32-bit words, least significant first.
///
/// Zero takes one zero word rather than none, which is the case dspy actually seeds with.
fn key(seed: u64) -> Vec<u32> {
    // A `u64` is two words, so the "spread over 32-bit words" loop can run at most once — and
    // written as a loop its exit depended on the shift reducing `rest`, which a mutation reversing
    // the comparison turned into a hang rather than a failure.
    let mut key = vec![seed as u32];
    let high = seed >> 32;
    if high > 0 {
        key.push(high as u32);
    }
    key
}

/// The population size at which CPython stops tracking a pool and starts tracking drawn indices.
///
/// Upstream writes `21 + 4 ** ceil(log(k * 3, 4))`, the smallest power of four at or above `k * 3`.
/// A loop reaches the same number without floating point, and cannot round the other way: that
/// would need `k * 3` to be a power of four, and no power of four is divisible by three.
fn setsize(k: usize) -> usize {
    if k <= 5 {
        return 21;
    }
    // The smallest power of four at or above `k * 3`, without a loop whose exit depends on the
    // multiply growing the value. A power of four is a power of two with an even exponent, so
    // rounding `k * 3` up to a power of two and then up again to an even exponent is the same
    // number — and cannot round the wrong way, since no power of four is divisible by three.
    let two = (k * 3).next_power_of_two();
    let table = if two.trailing_zeros() % 2 == 0 {
        two
    } else {
        two << 1
    };
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
            .join("tests/conformance/cpython_random.json");
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
            let mut rng = Random::seeded(seed);
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
            let mut rng = Random::seeded(seed);
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
            Random::seeded(seed).shuffle(&mut items);
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
            let drawn = Random::seeded(seed).sample(&population, k);
            assert_eq!(
                drawn,
                integers(&case, "result"),
                "sample({size}, {k}) from seed {seed}, threshold {}",
                number(&case, "setsize")
            );
        }
    }

    #[test]
    fn draws_the_weighted_choices_cpython_draws() {
        for case in cases("choices") {
            let seed = number(&case, "seed") as u64;
            let weights: Vec<f64> = case["weights"]
                .as_array()
                .expect("weights")
                .iter()
                .map(|value| value.as_f64().expect("a number"))
                .collect();
            let k = number(&case, "k");
            let drawn = Random::seeded(seed).choices(&weights, k);
            assert_eq!(
                drawn,
                integers(&case, "result"),
                "choices({weights:?}, k={k}) from seed {seed}"
            );
        }
    }

    /// CPython's own `test_guaranteed_stable` (`Lib/test/test_random.py`): `random()` after seeding
    /// with `3456147`, compared to the values CPython's test suite hardcodes as "guaranteed to stay
    /// the same across versions of python". Unlike the generated golden, these are CPython's own
    /// published known answer, not this machine's interpreter — the strongest anchor for `random()`,
    /// the 53-bit double, which the rest of the golden exercises only through `choices`.
    #[test]
    fn draws_the_double_cpythons_own_test_pins() {
        let case = &golden()["cpython_guaranteed_stable"];
        let seed = case["seed"].as_u64().expect("a seed");
        let expected: Vec<u64> = case["random_bits"]
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
        let mut rng = Random::seeded(seed);
        let drawn: Vec<u64> = expected
            .iter()
            .map(|_| rng.0.random_double().to_bits())
            .collect();
        assert_eq!(
            drawn, expected,
            "random() after Random({seed}), against CPython's own test"
        );
    }

    /// The canonical `mt19937ar.c` sequence: `init_by_array` over the key that implementation
    /// publishes, then a thousand consecutive draws — the one check here that does not trace back to
    /// a locally installed interpreter.
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
        let mut generator = Mt19937::from_key(&key);
        let drawn: Vec<u64> = expected.iter().map(|_| generator.getrandbits(32)).collect();
        assert_eq!(
            drawn.len(),
            1000,
            "the reference publishes a thousand draws"
        );
        assert_eq!(drawn, expected, "mt19937ar.c reference sequence");
    }

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

    /// CPython seeds by walking the longer of its state and the key. Nothing here can make the key
    /// the longer one, which is why that branch is never exercised — pinned so that widening the
    /// seed type has to come back through this test.
    #[test]
    fn a_seed_never_outruns_the_generators_state() {
        assert!(
            key(u64::MAX).len() < 624,
            "a wider seed would reach CPython's other branch"
        );
    }
}
