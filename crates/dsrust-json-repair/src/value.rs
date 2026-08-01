//! What the parser returns: Python's `JSONReturnType`, with the distinctions Python makes.
//!
//! `dict | list | str | float | int | bool | None` — so `7` and `7.0` are different answers, and a
//! key's position is where it was first assigned rather than where it was last written. Both are
//! observable through `json.dumps`, which is what a caller comparing against `json_repair` sees.

use std::fmt;

/// A parsed JSON value, in the shapes Python's `json` produces.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// A Python `int` that fits a machine word.
    Int(i64),
    /// A Python `int` that does not, held as the digits `int.__repr__` would print. Python's
    /// integers are unbounded and the parser reaches them by reading a long run of digits, so
    /// narrowing here would silently answer a different number than `json_repair` does.
    BigInt(String),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
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

    /// A field of this value, when it is an object. `None` for every other shape, which is
    /// what reading a schema node's keyword needs.
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
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Object(Vec<(String, Value)>);

impl Object {
    pub fn new() -> Self {
        Self::default()
    }

    /// `obj[key] = value`. A key that is already present keeps the position it was first given.
    pub fn insert(&mut self, key: String, value: Value) {
        match self.0.iter_mut().find(|(existing, _)| *existing == key) {
            Some(entry) => entry.1 = value,
            None => self.0.push((key, value)),
        }
    }

    /// `dict.update(other)`.
    pub fn update(&mut self, other: Object) {
        for (key, value) in other.0 {
            self.insert(key, value);
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0
            .iter()
            .find(|(existing, _)| existing == key)
            .map(|(_, value)| value)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The last key assigned a position, which `_merge_object_array_continuation` reads.
    pub fn last_key(&self) -> Option<&str> {
        self.0.last().map(|(key, _)| key.as_str())
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.0
            .iter_mut()
            .find(|(existing, _)| existing == key)
            .map(|(_, value)| value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.0.iter().map(|(key, value)| (key.as_str(), value))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(key, _)| key.as_str())
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
        self.0.into_iter()
    }
}

impl fmt::Display for Value {
    /// `json.dumps(value)` with its default arguments, which is what `repair_json` returns.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&crate::dump::dumps(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
