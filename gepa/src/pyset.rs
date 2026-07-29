//! CPython's `set` of non-negative ints, faithful to its iteration order.
//!
//! GEPA's merge selection draws against `list(set(...))` and `list(set_a & set_b)` — the candidate
//! pair through `rng.sample`, the common ancestor through `rng.choices` — so *which* element the
//! shared generator picks depends on the order the set yields. That order is not `sorted`: it is
//! the walk of an open-addressing hash table, which agrees with sorted for keys 0-7 and diverges
//! above (`{8, 13, 5}` yields `8, 5, 13`). Leaning on `sorted` would pass a small run and diverge
//! the moment a real one grew past eight candidates, so the table is reproduced instead — the
//! probe sequence, the load-factor resize, and the slot-order iteration — and held to
//! `tests/conformance/pyset.json`, whose cases cross the boundary on purpose.
//!
//! Add-only: merge never removes from these sets, so there is no dummy slot to model. Keys are the
//! candidate indices, all non-negative, and `hash(i) == i` for those — the reason a plain integer
//! is the whole key.

/// CPython constants (`setobject.c`).
const MINSIZE: usize = 8;
const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;

/// A set of non-negative integers with CPython's iteration order.
#[derive(Clone, Debug)]
pub struct PyIntSet {
    /// `Some(key)` for a filled slot, `None` for an empty one. Length is always a power of two.
    table: Vec<Option<usize>>,
    /// One less than the table length, the mask CPython probes with.
    mask: usize,
    /// Filled slots. With no deletions this equals the element count, but CPython compares it,
    /// not the count, against the resize threshold, so it is tracked as its own value.
    fill: usize,
}

impl PyIntSet {
    pub fn new() -> Self {
        Self {
            table: vec![None; MINSIZE],
            mask: MINSIZE - 1,
            fill: 0,
        }
    }

    /// The set of `keys`, inserted in order — CPython's `set(iterable)`.
    pub fn from_keys(keys: impl IntoIterator<Item = usize>) -> Self {
        let mut set = Self::new();
        for key in keys {
            set.add(key);
        }
        set
    }

    pub fn contains(&self, key: usize) -> bool {
        self.slot_of(key)
            .is_none_or(|slot| self.table[slot].is_some())
    }

    /// `a & b`: CPython iterates the smaller operand — the right one on a size tie — and keeps the
    /// keys the larger also holds, in that iteration order.
    pub fn intersection(&self, other: &Self) -> Self {
        let (smaller, larger) = match other.len() > self.len() {
            true => (self, other),
            false => (other, self),
        };
        Self::from_keys(smaller.iter().filter(|&key| larger.contains(key)))
    }

    /// The elements in CPython's iteration order: the filled slots, front to back.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.table.iter().filter_map(|slot| *slot)
    }

    pub fn to_vec(&self) -> Vec<usize> {
        self.iter().collect()
    }

    pub fn len(&self) -> usize {
        self.fill
    }

    pub fn is_empty(&self) -> bool {
        self.fill == 0
    }

    /// Insert `key`, growing the table when CPython would. Present keys are a no-op.
    pub fn add(&mut self, key: usize) {
        let Some(slot) = self.slot_of(key) else {
            return; // already present
        };
        self.table[slot] = Some(key);
        self.fill += 1;
        // CPython resizes when `fill*5 >= mask*3` — against the mask, not the table length, so the
        // trigger differs from a size-based one from 16 slots up. The new table is sized to the
        // smallest power of two above `used*4` (elements never reach 50000 here).
        if self.fill * 5 >= self.mask * 3 {
            self.resize(self.fill * 4);
        }
    }

    /// The slot `key` belongs in, or `None` when it is already there. CPython probes the initial
    /// slot, then a window of `LINEAR_PROBES` consecutive slots while they fit under the mask,
    /// then perturbs to a new region — the sequence that decides where a colliding key lands, and
    /// so the order the table yields.
    fn slot_of(&self, key: usize) -> Option<usize> {
        let mut perturb = key;
        let mut i = key & self.mask;
        loop {
            match self.table[i] {
                None => return Some(i),
                Some(present) if present == key => return None,
                Some(_) => {}
            }
            if i + LINEAR_PROBES <= self.mask {
                for offset in 1..=LINEAR_PROBES {
                    match self.table[i + offset] {
                        None => return Some(i + offset),
                        Some(present) if present == key => return None,
                        Some(_) => {}
                    }
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & self.mask;
        }
    }

    /// Grow to the smallest power of two above `minused`, re-inserting every key in slot order —
    /// which is where the resize can reorder the set, since a key rehashes against the new mask.
    fn resize(&mut self, minused: usize) {
        let mut size = MINSIZE;
        while size <= minused {
            size <<= 1;
        }
        let old = std::mem::replace(&mut self.table, vec![None; size]);
        self.mask = size - 1;
        self.fill = 0;
        for key in old.into_iter().flatten() {
            let slot = self
                .slot_of(key)
                .expect("a fresh table has room for every old key");
            self.table[slot] = Some(key);
            self.fill += 1;
        }
    }
}

impl Default for PyIntSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn golden() -> Value {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/pyset.json");
        let text = std::fs::read_to_string(&path).expect("the pyset golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    fn ints(value: &Value) -> Vec<usize> {
        value
            .as_array()
            .expect("a list")
            .iter()
            .map(|v| v.as_u64().expect("an int") as usize)
            .collect()
    }

    #[test]
    fn builds_in_cpythons_iteration_order() {
        for case in golden()["build"].as_array().expect("build cases") {
            let input = ints(&case["input"]);
            let set = PyIntSet::from_keys(input.iter().copied());
            assert_eq!(
                set.to_vec(),
                ints(&case["order"]),
                "set({input:?}) iteration order"
            );
        }
    }

    #[test]
    fn intersects_the_way_cpython_intersects() {
        for case in golden()["intersection"]
            .as_array()
            .expect("intersection cases")
        {
            let a = PyIntSet::from_keys(ints(&case["a"]));
            let b = PyIntSet::from_keys(ints(&case["b"]));
            assert_eq!(
                a.intersection(&b).to_vec(),
                ints(&case["order"]),
                "set({:?}) & set({:?})",
                ints(&case["a"]),
                ints(&case["b"])
            );
        }
    }

    #[test]
    fn membership_and_dedup_hold() {
        let set = PyIntSet::from_keys([40, 8, 72, 8, 40, 1, 99]);
        assert_eq!(set.len(), 5, "duplicates fold");
        assert!(set.contains(72) && set.contains(1));
        assert!(!set.contains(73));
    }
}
