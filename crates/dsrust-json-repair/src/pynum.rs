//! Python's `int()` and `float()`, for the two places the library calls them on text it scanned.
//!
//! Both can fail, and both callers treat failure as an answer rather than an error: `parse_number`
//! returns the text it read when neither conversion works, and CPython's scanner refuses the input.

use crate::value::Value;

/// `int(text)` for text already known to be digits, keeping every one of them.
///
/// Python's integers are unbounded, so a long run of digits is an exact value rather than a float.
pub(crate) fn python_int(text: &str) -> Value {
    match text.parse::<i64>() {
        Ok(number) => Value::Int(number),
        Err(_) => Value::BigInt(normalize(text)),
    }
}

/// `int.__repr__` of the parsed value: no leading zeros, no `+`, one spelling of zero.
fn normalize(text: &str) -> String {
    let (sign, digits) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", text.strip_prefix('+').unwrap_or(text)),
    };
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        return "0".to_owned();
    }
    format!("{sign}{trimmed}")
}

/// `int(text)`, raising as Python does when the text is not an integer.
pub(crate) fn try_python_int(text: &str) -> Option<Value> {
    let digits = text.strip_prefix(['-', '+']).unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(python_int(text))
}

/// `float(text)`, raising as Python does. The callers only reach this with characters drawn from
/// `parse_number`'s own set, which is where Rust's grammar and Python's agree.
pub(crate) fn try_python_float(text: &str) -> Option<f64> {
    if text.is_empty()
        || text
            .bytes()
            .any(|byte| !matches!(byte, b'0'..=b'9' | b'.' | b'e' | b'E' | b'-' | b'+'))
    {
        return None;
    }
    text.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_integer_too_wide_for_a_word_keeps_its_digits() {
        assert_eq!(python_int("7"), Value::Int(7));
        assert_eq!(
            python_int("99999999999999999999"),
            Value::BigInt("99999999999999999999".into()),
        );
        assert_eq!(
            python_int("-00012345678901234567890"),
            Value::BigInt("-12345678901234567890".into())
        );
        // Leading zeros do not make a number wide: this one still fits a word.
        assert_eq!(python_int("-0000000000000000000000"), Value::Int(0));
    }

    #[test]
    fn the_conversions_refuse_what_python_refuses() {
        assert_eq!(try_python_int("007"), Some(Value::Int(7)));
        assert_eq!(try_python_int("1-2"), None);
        assert_eq!(try_python_int(""), None);
        assert_eq!(try_python_int("1.0"), None);
        assert_eq!(try_python_float("1."), Some(1.0));
        assert_eq!(try_python_float(".5"), Some(0.5));
        assert_eq!(try_python_float("1.2.3"), None);
        assert_eq!(try_python_float("--1"), None);
        // Not reachable from `parse_number`'s character set, and refused rather than read as
        // Rust's `f64::from_str` would read it.
        assert_eq!(try_python_float("inf"), None);
    }
}
