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

/// Python's shared numeric-literal-from-string rules, as ASCII.
///
/// `int()` and `float()` both strip surrounding whitespace, take an optional sign, accept any
/// Unicode *decimal* digit, and allow `_` only with a digit on each side. `coerce_scalar` hands
/// these arbitrary strings out of a parsed value, so none of that is hypothetical: `int("١٢٣")` is
/// 123 and `float(" 1_0 ")` is 10.0.
fn as_ascii_literal(text: &str) -> Option<std::borrow::Cow<'_, str>> {
    let trimmed = text.trim_matches(crate::pychar::is_space);
    if trimmed.is_empty() {
        return None;
    }
    // Nearly every number a parse hands over is already the ASCII the conversion would rebuild —
    // borrow it back rather than allocate a copy. `_` and non-ASCII digits take the owned path.
    if trimmed.bytes().all(|byte| byte.is_ascii() && byte != b'_') {
        return Some(std::borrow::Cow::Borrowed(trimmed));
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let mut out = String::with_capacity(chars.len());
    for (at, &ch) in chars.iter().enumerate() {
        if ch == '_' {
            // Between two digits, and nowhere else: `1_0` is ten, `_1`, `1_` and `1__0` are not.
            let flanked = at > 0
                && crate::pychar::is_decimal(chars[at - 1])
                && chars
                    .get(at + 1)
                    .copied()
                    .is_some_and(crate::pychar::is_decimal);
            if !flanked {
                return None;
            }
            continue;
        }
        match crate::pychar::decimal_value(ch) {
            Some(value) => out.push(char::from_digit(value, 10).expect("a decimal digit")),
            None => out.push(ch),
        }
    }
    Some(std::borrow::Cow::Owned(out))
}

/// `int(text)`, raising as Python does.
pub(crate) fn try_python_int(text: &str) -> Option<Value> {
    let literal = as_ascii_literal(text)?;
    let digits = literal.strip_prefix(['-', '+']).unwrap_or(&literal);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(python_int(&literal))
}

/// `float(text)`, raising as Python does — including `inf`, `infinity` and `nan` in any case,
/// which Rust's own parser spells the same way, and rejecting `0x`/`0b`, which it also does.
pub(crate) fn try_python_float(text: &str) -> Option<f64> {
    as_ascii_literal(text)?.parse().ok()
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
    }

    #[test]
    fn the_conversions_accept_everything_python_accepts() {
        // `parse_number` only ever hands these ASCII from its own character set, but
        // `coerce_scalar` hands them any string out of a parsed value — so all of this is reachable
        // through a schema, and every line below was a refusal until it was measured.
        assert_eq!(
            try_python_int("١٢٣"),
            Some(Value::Int(123)),
            "Arabic-Indic digits"
        );
        assert_eq!(
            try_python_int("１２３"),
            Some(Value::Int(123)),
            "fullwidth digits"
        );
        assert_eq!(try_python_int("1_000"), Some(Value::Int(1000)));
        assert_eq!(try_python_int(" \t12\n "), Some(Value::Int(12)));
        assert_eq!(try_python_int("+5"), Some(Value::Int(5)));
        assert_eq!(try_python_float("inf"), Some(f64::INFINITY));
        assert_eq!(try_python_float("-INFINITY"), Some(f64::NEG_INFINITY));
        assert!(try_python_float("NaN").is_some_and(f64::is_nan));
        assert_eq!(try_python_float("  1.5  "), Some(1.5));
        assert_eq!(try_python_float("1_0.0_1"), Some(10.01));
        assert_eq!(try_python_float("١٢٣.٤"), Some(123.4));

        // And still refuse what Python refuses: an underscore needs a digit on each side, `²` is a
        // digit to `isdigit` but not to `int()`, and neither takes a base prefix.
        assert_eq!(try_python_int("_1"), None);
        assert_eq!(try_python_int("1_"), None);
        assert_eq!(try_python_int("1__0"), None);
        assert_eq!(try_python_float("1_.0"), None);
        assert_eq!(try_python_int("²"), None);
        assert_eq!(try_python_int("0x1f"), None);
        assert_eq!(try_python_float("0b101"), None);
        assert_eq!(try_python_int("inf"), None, "an integer is not infinite");
    }
}
