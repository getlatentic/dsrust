//! What the parser returns: Python's `JSONReturnType`, with the distinctions Python makes.
//!
//! `dict | list | str | float | int | bool | None` — so `7` and `7.0` are different answers, and a
//! key's position is where it was first assigned rather than where it was last written. Both are
//! observable through `json.dumps`, which is what a caller comparing against `json_repair` sees.

use std::fmt;

/// A parsed JSON value, in the shapes Python's `json` produces.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Python's `None`, JSON's `null`.
    Null,
    /// `true` or `false`, which Python keeps distinct from `0` and `1` by type.
    Bool(bool),
    /// A Python `int` that fits a machine word.
    Int(i64),
    /// A Python `int` that does not, held as the digits `int.__repr__` would print. Python's
    /// integers are unbounded and the parser reaches them by reading a long run of digits, so
    /// narrowing here would silently answer a different number than `json_repair` does.
    BigInt(String),
    /// A Python `float`. Kept apart from [`Value::Int`], so `7` and `7.0` are different answers.
    Float(f64),
    /// A Python `str`.
    Str(String),
    /// A Python `list`.
    Array(Vec<Value>),
    /// A Python `dict`, ordered by first assignment.
    Object(Object),
}

impl Value {
    /// Python's `bool(value)`: empty containers, empty strings, zero and `None` are false.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(flag) => *flag,
            Value::Int(number) => *number != 0,
            // A big integer reached this branch by being too large for a word, so it is not zero.
            Value::BigInt(_) => true,
            Value::Float(number) => *number != 0.0,
            Value::Str(text) => !text.is_empty(),
            Value::Array(items) => !items.is_empty(),
            Value::Object(fields) => !fields.is_empty(),
        }
    }

    /// `ObjectComparer.is_strictly_empty`: an empty container, where `None` and `0` are not.
    pub fn is_strictly_empty(&self) -> bool {
        match self {
            Value::Str(text) => text.is_empty(),
            Value::Array(items) => items.is_empty(),
            Value::Object(fields) => fields.is_empty(),
            _ => false,
        }
    }

    /// `ObjectComparer.is_same_object`: same type, same shape, values ignored.
    pub fn is_same_shape(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Object(left), Value::Object(right)) => {
                left.len() == right.len()
                    && left.iter().all(|(key, value)| {
                        right
                            .get(key)
                            .is_some_and(|other| value.is_same_shape(other))
                    })
            }
            (Value::Array(left), Value::Array(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(value, other)| value.is_same_shape(other))
            }
            // Python compares `type(obj1) is not type(obj2)` first, and `bool` is not `int`.
            (Value::Bool(_), Value::Bool(_)) => true,
            (Value::Int(_) | Value::BigInt(_), Value::Int(_) | Value::BigInt(_)) => true,
            (Value::Float(_), Value::Float(_)) => true,
            (Value::Str(_), Value::Str(_)) => true,
            (Value::Null, Value::Null) => true,
            _ => false,
        }
    }

    /// A field of this value, when it is an object; `None` for every other shape.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(fields) => fields.get(key),
            _ => None,
        }
    }

    /// Whether this is the empty string, the comparison `parse_object` and `repair_json` make.
    pub fn is_empty_string(&self) -> bool {
        matches!(self, Value::Str(text) if text.is_empty())
    }

    /// Python's `==`, which crosses the numeric types: `1 == 1.0` and `True == 1` are both true.
    /// Structural [`PartialEq`] keeps them apart, and both meanings are wanted — this one only
    /// where the library compares against `const`, `enum` or a literal.
    pub fn python_eq(&self, other: &Value) -> bool {
        match (self.as_python_number(), other.as_python_number()) {
            (Some(left), Some(right)) => left == right,
            (None, None) => match (self, other) {
                (Value::Array(left), Value::Array(right)) => {
                    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.python_eq(b))
                }
                (Value::Object(left), Value::Object(right)) => {
                    left.len() == right.len()
                        && left.iter().all(|(key, value)| {
                            right.get(key).is_some_and(|other| value.python_eq(other))
                        })
                }
                _ => self == other,
            },
            _ => false,
        }
    }

    /// The numeric value Python would compare, for the types it compares numerically.
    fn as_python_number(&self) -> Option<f64> {
        match self {
            Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
            Value::Int(number) => Some(*number as f64),
            Value::Float(number) => Some(*number),
            Value::BigInt(digits) => digits.parse().ok(),
            _ => None,
        }
    }
}

/// An object, ordered by first assignment the way a Python `dict` is.
///
/// The entries carry the order; the index exists only once an object is big enough to need one. A
/// dict's lookup is O(1), and a bare `Vec` held a linear scan per insert and per get — building a
/// four-thousand-key object took eight million string comparisons, measured as valid JSON parsing
/// eight times slower per byte than its small-object twin. Indexing *every* object swung the cost
/// the other way: most objects a model emits hold a handful of keys, the map made `Value` more
/// than twice its size, and the depth-240 recursion tests overflowed their stack on the fatter
/// frames. So the index is boxed, absent below [`INDEX_AT`], and derived state throughout: every
/// mutation leaves it exactly the positions of `entries`, and equality reads the entries alone.
#[derive(Clone, Debug, Default)]
pub struct Object {
    entries: Vec<(String, Value)>,
    index: Option<Box<std::collections::HashMap<String, usize>>>,
}

/// How many entries an object holds before lookups earn a map. Sixteen linear string compares are
/// cheaper than hashing for the objects a model usually emits.
const INDEX_AT: usize = 16;

impl PartialEq for Object {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

impl Object {
    /// An object with no entries.
    pub fn new() -> Self {
        Self::default()
    }

    /// The position of `key`, through the index when one exists.
    fn position(&self, key: &str) -> Option<usize> {
        match &self.index {
            Some(map) => map.get(key).copied(),
            None => self
                .entries
                .iter()
                .position(|(existing, _)| existing == key),
        }
    }

    /// Assigns `value` to `key`. A key already present keeps the position it was first given, as a
    /// Python `dict` does.
    pub fn insert(&mut self, key: String, value: Value) {
        match self.position(&key) {
            Some(at) => self.entries[at].1 = value,
            None => {
                if let Some(map) = &mut self.index {
                    map.insert(key.clone(), self.entries.len());
                }
                self.entries.push((key, value));
                if self.index.is_none() && self.entries.len() > INDEX_AT {
                    self.index = Some(Box::new(
                        self.entries
                            .iter()
                            .enumerate()
                            .map(|(at, (key, _))| (key.clone(), at))
                            .collect(),
                    ));
                }
            }
        }
    }

    /// Inserts every entry of `other`, as `dict.update` does. A key already present keeps its
    /// position and takes the new value.
    pub fn update(&mut self, other: Object) {
        for (key, value) in other.entries {
            self.insert(key, value);
        }
    }

    /// The value stored under `key`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.position(key).map(|at| &self.entries[at].1)
    }

    /// `dict.pop(key, None)`: the value if it was there, and the key gone from the order.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let at = self.position(key)?;
        let (_, value) = self.entries.remove(at);
        if let Some(map) = &mut self.index {
            map.remove(key);
            for position in map.values_mut() {
                if *position > at {
                    *position -= 1;
                }
            }
        }
        Some(value)
    }

    /// Whether `key` is present.
    pub fn contains_key(&self, key: &str) -> bool {
        self.position(key).is_some()
    }

    /// How many entries the object holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the object holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The key assigned a position most recently.
    pub fn last_key(&self) -> Option<&str> {
        self.entries.last().map(|(key, _)| key.as_str())
    }

    /// The value stored under `key`, to modify in place.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        let at = self.position(key)?;
        Some(&mut self.entries[at].1)
    }

    /// Every entry, in the order its key was first assigned.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    /// Every key, in the order it was first assigned.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(key, _)| key.as_str())
    }
}

impl FromIterator<(String, Value)> for Object {
    fn from_iter<I: IntoIterator<Item = (String, Value)>>(entries: I) -> Self {
        let mut object = Object::new();
        for (key, value) in entries {
            object.insert(key, value);
        }
        object
    }
}

impl IntoIterator for Object {
    type Item = (String, Value);
    type IntoIter = std::vec::IntoIter<(String, Value)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// Python's `str(value)`, which is `repr` for every container and for `None`/`True`, and the text
/// itself for a string. Reached only by the message naming a schema `type` that is not one.
pub(crate) fn python_str(value: &Value) -> String {
    match value {
        Value::Str(text) => text.clone(),
        other => python_repr(other),
    }
}

/// Python's `repr(value)`: single-quoted strings, `None`/`True`/`False`, `[a, b]`, `{'k': v}`.
pub(crate) fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Int(number) => number.to_string(),
        Value::BigInt(digits) => digits.clone(),
        Value::Float(_) => crate::dump::dumps(value, true),
        Value::Str(text) => format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(items) => {
            let rendered: Vec<String> = items.iter().map(python_repr).collect();
            format!("[{}]", rendered.join(", "))
        }
        Value::Object(fields) => {
            let rendered: Vec<String> = fields
                .iter()
                .map(|(key, item)| {
                    format!(
                        "{}: {}",
                        python_repr(&Value::Str(key.to_owned())),
                        python_repr(item)
                    )
                })
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
    }
}

impl fmt::Display for Value {
    /// `json.dumps(value)` with its default arguments, `ensure_ascii` among them.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&crate::dump::dumps(self, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hybrid is invisible from outside on purpose — a linear scan and a map answer every
    /// query identically — so its threshold and bookkeeping are checked from inside, where
    /// `index.is_some()` is a fact rather than an inference. Sixteen of its mutants survived the
    /// whole suite for exactly this reason.
    #[test]
    fn the_index_appears_at_the_threshold_and_not_before() {
        let mut object = Object::new();
        for n in 0..INDEX_AT {
            object.insert(format!("k{n}"), Value::Int(n as i64));
        }
        assert!(object.index.is_none(), "sixteen entries scan linearly");
        object.insert("crossing".into(), Value::Int(-1));
        assert!(object.index.is_some(), "the seventeenth builds the map");
        for n in 0..INDEX_AT {
            assert_eq!(object.get(&format!("k{n}")), Some(&Value::Int(n as i64)));
        }
        assert_eq!(object.get("crossing"), Some(&Value::Int(-1)));
        // A key inserted after the crossing goes through the map arm.
        object.insert("later".into(), Value::Int(99));
        assert_eq!(object.get("later"), Some(&Value::Int(99)));
        object.insert("k3".into(), Value::Int(33));
        assert_eq!(
            object.get("k3"),
            Some(&Value::Int(33)),
            "replacement through the map"
        );
        assert_eq!(object.len(), INDEX_AT + 2);
    }

    #[test]
    fn removing_from_an_indexed_object_keeps_every_position_true() {
        let mut object = Object::new();
        for n in 0..20 {
            object.insert(format!("k{n}"), Value::Int(n as i64));
        }
        assert!(object.index.is_some());
        assert_eq!(object.remove("k7"), Some(Value::Int(7)));
        assert_eq!(object.remove("k7"), None, "gone means gone");
        for n in (0..20).filter(|&n| n != 7) {
            assert_eq!(
                object.get(&format!("k{n}")),
                Some(&Value::Int(n as i64)),
                "k{n} after the removal"
            );
        }
        let keys: Vec<&str> = object.keys().collect();
        assert_eq!(keys[6], "k6");
        assert_eq!(keys[7], "k8", "the order closes over the gap");
    }

    #[test]
    fn objects_compare_by_their_entries() {
        let mut left = Object::new();
        left.insert("a".into(), Value::Int(1));
        let mut right = Object::new();
        right.insert("a".into(), Value::Int(2));
        assert_ne!(left, right);
        assert_ne!(left, Object::new());
    }

    #[test]
    fn into_iter_yields_the_entries_in_order() {
        let mut object = Object::new();
        object.insert("b".into(), Value::Int(2));
        object.insert("a".into(), Value::Int(1));
        let entries: Vec<(String, Value)> = object.into_iter().collect();
        assert_eq!(
            entries,
            vec![("b".into(), Value::Int(2)), ("a".into(), Value::Int(1))]
        );
    }

    #[test]
    fn a_reassigned_key_keeps_the_position_it_was_first_given() {
        let mut object = Object::new();
        object.insert("a".into(), Value::Int(1));
        object.insert("b".into(), Value::Int(2));
        object.insert("a".into(), Value::Int(3));
        assert_eq!(object.keys().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(object.get("a"), Some(&Value::Int(3)));
    }

    #[test]
    fn python_equality_crosses_the_numeric_types_and_structural_equality_does_not() {
        assert!(Value::Int(1).python_eq(&Value::Float(1.0)));
        assert!(Value::Bool(true).python_eq(&Value::Int(1)));
        assert_ne!(Value::Int(1), Value::Float(1.0));
        assert!(!Value::Str("1".into()).python_eq(&Value::Int(1)));
    }

    #[test]
    fn shape_comparison_keeps_bool_and_int_apart_as_python_types_do() {
        assert!(!Value::Bool(true).is_same_shape(&Value::Int(1)));
        assert!(Value::Int(1).is_same_shape(&Value::Int(99)));
    }

    #[test]
    fn truthiness_follows_python_rather_than_null_checking() {
        assert!(!Value::Int(0).is_truthy());
        assert!(!Value::Array(vec![]).is_truthy());
        assert!(!Value::Str(String::new()).is_truthy());
        assert!(Value::Float(-1.0).is_truthy());
        assert!(!Value::Float(0.0).is_truthy());
    }
}
