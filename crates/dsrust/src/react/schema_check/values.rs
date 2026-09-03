//! A JSON value as python-jsonschema sees it: its draft 2020-12 type checker, and the `equal` that
//! keeps `True` apart from `1`.

use serde_json::Value;

/// `TypeChecker.is_type` for the draft 2020-12 checker: a bool is never a number, and a float with
/// no fractional part is an integer.
pub(super) fn is_type(value: &Value, kind: &str) -> bool {
    match kind {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "number" => value.is_number(),
        "integer" => is_integer(value),
        _ => false,
    }
}

fn is_integer(value: &Value) -> bool {
    match value {
        Value::Number(number) => {
            number.is_i64()
                || number.is_u64()
                || number
                    .as_f64()
                    .is_some_and(|float| float.is_finite() && float.fract() == 0.0)
        }
        _ => false,
    }
}

/// `_utils.equal`: strings compare as strings, sequences and mappings element-wise, and anything
/// else only after a bool has been made unequal to every number.
pub(super) fn equal(one: &Value, two: &Value) -> bool {
    match (one, two) {
        (Value::String(_), _) | (_, Value::String(_)) => one == two,
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len() && left.iter().zip(right).all(|(a, b)| equal(a, b))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .all(|(key, value)| right.get(key).is_some_and(|other| equal(value, other)))
        }
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Bool(_), _) | (_, Value::Bool(_)) => false,
        (Value::Number(a), Value::Number(b)) => number_equal(a, b),
        _ => one == two,
    }
}

fn number_equal(a: &serde_json::Number, b: &serde_json::Number) -> bool {
    match (a.as_i64(), b.as_i64()) {
        (Some(x), Some(y)) => x == y,
        _ => a.as_f64() == b.as_f64(),
    }
}

/// `_utils.uniq`: no two elements equal under [`equal`].
pub(super) fn uniq(items: &[Value]) -> bool {
    items
        .iter()
        .enumerate()
        .all(|(i, a)| items[i + 1..].iter().all(|b| !equal(a, b)))
}

/// Python's `len` of a str: code points.
pub(super) fn char_count(text: &str) -> usize {
    text.chars().count()
}

pub(super) fn as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        _ => None,
    }
}
