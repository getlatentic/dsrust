//! CPython's `json` module, which decides whether any repair happens at all.
//!
//! `repair_json` opens with `json.loads(json_str)` and takes the result when it succeeds, and
//! `_try_parse_valid_json_value` re-enters through `JSONDecoder().raw_decode`. Getting either
//! boundary wrong moves inputs between the two parsers, so this is CPython's grammar rather than
//! a general-purpose one: `NaN`, `Infinity` and `-Infinity` are values; whitespace is four
//! characters and not Unicode's; an integer keeps every digit it was written with; and a leading
//! zero, a trailing comma or a raw control character inside a string is a refusal.

use crate::value::{Object, Value};

/// A refusal. CPython distinguishes several messages; every caller here swallows all of them.
#[derive(Debug)]
pub(crate) struct NotJson;

/// The input, with its reads counted. The count is debug-only, as `Parser::get_char_at`'s is.
///
/// This scanner walks its own cursor rather than the parser's, so the read counter that turned the
/// repair parser's hangs into failures never saw it: seven mutants here were scored as timeouts,
/// each a loop whose index stopped advancing. One forward pass reads each position a small constant
/// number of times — a token is consumed once, with a peek or two around it — so the budget is
/// linear where the parser's is quadratic, and no scan that ends can reach it.
struct Text<'a> {
    chars: &'a [char],
    #[cfg(debug_assertions)]
    reads: std::cell::Cell<u64>,
    #[cfg(debug_assertions)]
    budget: u64,
}

impl<'a> Text<'a> {
    fn new(chars: &'a [char]) -> Self {
        Self {
            chars,
            #[cfg(debug_assertions)]
            reads: std::cell::Cell::new(0),
            #[cfg(debug_assertions)]
            budget: (chars.len() as u64)
                .saturating_mul(8)
                .saturating_add(1 << 12),
        }
    }

    fn get(&self, index: usize) -> Option<&char> {
        #[cfg(debug_assertions)]
        {
            let read = self.reads.get() + 1;
            self.reads.set(read);
            assert!(
                read <= self.budget,
                "{read} reads over {} characters of strict JSON — the scan is not advancing",
                self.chars.len()
            );
        }
        self.chars.get(index)
    }

    fn get_range(&self, range: std::ops::Range<usize>) -> Option<&[char]> {
        #[cfg(debug_assertions)]
        self.reads.set(self.reads.get() + 4);
        self.chars.get(range)
    }

    fn slice(&self, range: std::ops::Range<usize>) -> &[char] {
        &self.chars[range]
    }

    fn len(&self) -> usize {
        self.chars.len()
    }
}

type Scan = Result<(Value, usize), NotJson>;

/// `json.loads(text)`: one value, with nothing but whitespace around it.
pub(crate) fn loads(text: &[char]) -> Result<Value, NotJson> {
    let text = Text::new(text);
    let text = &text;
    let start = skip_whitespace(text, 0);
    let (value, end) = scan_once(text, start)?;
    let end = skip_whitespace(text, end);
    if end != text.len() {
        return Err(NotJson);
    }
    Ok(value)
}

/// `JSONDecoder().raw_decode(text)`: one value, and where it ended. Trailing text is the caller's.
pub(crate) fn raw_decode(text: &[char]) -> Scan {
    scan_once(&Text::new(text), 0)
}

/// `json.decoder.WHITESPACE`, which is these four characters and not `str.isspace()`.
fn skip_whitespace(text: &Text<'_>, mut index: usize) -> usize {
    while matches!(text.get(index), Some(' ' | '\t' | '\n' | '\r')) {
        index += 1;
    }
    index
}

fn literal(text: &Text<'_>, index: usize, word: &str) -> bool {
    word.chars()
        .enumerate()
        .all(|(offset, expected)| text.get(index + offset) == Some(&expected))
}

fn scan_once(text: &Text<'_>, index: usize) -> Scan {
    match text.get(index) {
        Some('"') => scan_string(text, index + 1),
        Some('{') => scan_object(text, index + 1),
        Some('[') => scan_array(text, index + 1),
        Some('n') if literal(text, index, "null") => Ok((Value::Null, index + 4)),
        Some('t') if literal(text, index, "true") => Ok((Value::Bool(true), index + 4)),
        Some('f') if literal(text, index, "false") => Ok((Value::Bool(false), index + 5)),
        _ => scan_number(text, index),
    }
}

/// The number branch, and the three literals CPython reaches only after it.
fn scan_number(text: &Text<'_>, index: usize) -> Scan {
    if let Some((value, end)) = match_number(text, index) {
        return Ok((value, end));
    }
    if literal(text, index, "NaN") {
        return Ok((Value::Float(f64::NAN), index + 3));
    }
    if literal(text, index, "Infinity") {
        return Ok((Value::Float(f64::INFINITY), index + 8));
    }
    if literal(text, index, "-Infinity") {
        return Ok((Value::Float(f64::NEG_INFINITY), index + 9));
    }
    Err(NotJson)
}

/// `json.decoder.NUMBER_RE`: `-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`. The integer part refusing a
/// leading zero is what makes `01` two tokens rather than one number.
fn match_number(text: &Text<'_>, index: usize) -> Option<(Value, usize)> {
    let mut end = index;
    if text.get(end) == Some(&'-') {
        end += 1;
    }
    let integer_start = end;
    match text.get(end) {
        Some('0') => end += 1,
        Some(digit) if digit.is_ascii_digit() => {
            while text.get(end).is_some_and(char::is_ascii_digit) {
                end += 1;
            }
        }
        _ => return None,
    }
    if end == integer_start {
        return None;
    }

    let mut is_float = false;
    if text.get(end) == Some(&'.') && text.get(end + 1).is_some_and(char::is_ascii_digit) {
        end += 1;
        while text.get(end).is_some_and(char::is_ascii_digit) {
            end += 1;
        }
        is_float = true;
    }
    if let Some('e' | 'E') = text.get(end) {
        let mut exponent = end + 1;
        if let Some('+' | '-') = text.get(exponent) {
            exponent += 1;
        }
        if text.get(exponent).is_some_and(char::is_ascii_digit) {
            while text.get(exponent).is_some_and(char::is_ascii_digit) {
                exponent += 1;
            }
            end = exponent;
            is_float = true;
        }
    }

    let literal: String = text.slice(index..end).iter().collect();
    let value = if is_float {
        Value::Float(literal.parse().ok()?)
    } else {
        crate::pynum::python_int(&literal)
    };
    Some((value, end))
}

fn scan_object(text: &Text<'_>, index: usize) -> Scan {
    let mut object = Object::new();
    let mut end = skip_whitespace(text, index);
    if text.get(end) == Some(&'}') {
        return Ok((Value::Object(object), end + 1));
    }
    if text.get(end) != Some(&'"') {
        return Err(NotJson);
    }
    loop {
        let (key, after_key) = scan_string(text, end + 1)?;
        let Value::Str(key) = key else {
            return Err(NotJson);
        };
        end = skip_whitespace(text, after_key);
        if text.get(end) != Some(&':') {
            return Err(NotJson);
        }
        end = skip_whitespace(text, end + 1);
        let (value, after_value) = scan_once(text, end)?;
        object.insert(key, value);
        end = skip_whitespace(text, after_value);
        match text.get(end) {
            Some('}') => return Ok((Value::Object(object), end + 1)),
            Some(',') => end = skip_whitespace(text, end + 1),
            _ => return Err(NotJson),
        }
        if text.get(end) != Some(&'"') {
            return Err(NotJson);
        }
    }
}

fn scan_array(text: &Text<'_>, index: usize) -> Scan {
    let mut items = Vec::new();
    let mut end = skip_whitespace(text, index);
    if text.get(end) == Some(&']') {
        return Ok((Value::Array(items), end + 1));
    }
    loop {
        let (value, after_value) = scan_once(text, end)?;
        items.push(value);
        end = skip_whitespace(text, after_value);
        match text.get(end) {
            Some(']') => return Ok((Value::Array(items), end + 1)),
            Some(',') => end = skip_whitespace(text, end + 1),
            _ => return Err(NotJson),
        }
    }
}

/// `py_scanstring` with `strict=True`: a raw character below `\x20` ends the parse.
fn scan_string(text: &Text<'_>, index: usize) -> Scan {
    let mut out = String::new();
    let mut end = index;
    loop {
        let Some(&ch) = text.get(end) else {
            return Err(NotJson);
        };
        end += 1;
        match ch {
            '"' => return Ok((Value::Str(out), end)),
            '\\' => {
                let Some(&escape) = text.get(end) else {
                    return Err(NotJson);
                };
                end += 1;
                match escape {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let (ch, after) = scan_unicode_escape(text, end)?;
                        out.push(ch);
                        end = after;
                    }
                    _ => return Err(NotJson),
                }
            }
            ch if (ch as u32) < 0x20 => return Err(NotJson),
            ch => out.push(ch),
        }
    }
}

/// `\uXXXX`, joining a surrogate pair when one follows.
///
/// A high surrogate that is *not* followed by a low one stays a lone surrogate in Python, which a
/// Rust `char` cannot hold — see the note on [`crate::LONE_SURROGATE`].
fn scan_unicode_escape(text: &Text<'_>, index: usize) -> Result<(char, usize), NotJson> {
    let first = four_hex_digits(text, index)?;
    let mut end = index + 4;
    if (0xd800..0xdc00).contains(&first)
        && text.get(end) == Some(&'\\')
        && text.get(end + 1) == Some(&'u')
        && let Ok(second) = four_hex_digits(text, end + 2)
        && (0xdc00..0xe000).contains(&second)
    {
        end += 6;
        let combined = 0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00);
        return Ok((char::from_u32(combined).ok_or(NotJson)?, end));
    }
    Ok((char::from_u32(first).unwrap_or(crate::LONE_SURROGATE), end))
}

fn four_hex_digits(text: &Text<'_>, index: usize) -> Result<u32, NotJson> {
    let digits: String = text
        .get_range(index..index + 4)
        .ok_or(NotJson)?
        .iter()
        .collect();
    if digits.len() != 4 || !digits.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(NotJson);
    }
    u32::from_str_radix(&digits, 16).map_err(|_| NotJson)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Value, NotJson> {
        loads(&text.chars().collect::<Vec<_>>())
    }

    #[test]
    fn the_values_json_has_no_spelling_for_are_still_values_here() {
        assert!(matches!(parse("NaN"), Ok(Value::Float(number)) if number.is_nan()));
        assert_eq!(parse("-Infinity").unwrap(), Value::Float(f64::NEG_INFINITY));
    }

    #[test]
    fn what_cpython_refuses_and_a_lenient_parser_would_not() {
        assert!(parse("{\"a\": 1,}").is_err(), "a trailing comma");
        assert!(parse("01").is_err(), "a leading zero");
        assert!(parse("\"a\u{1}b\"").is_err(), "a raw control character");
        assert!(parse("{'a': 1}").is_err(), "single quotes");
        assert!(parse("1 2").is_err(), "extra data");
        assert!(
            parse("\u{a0}1").is_err(),
            "a non-breaking space is not JSON whitespace"
        );
    }

    #[test]
    fn an_integer_keeps_every_digit_it_was_written_with() {
        assert_eq!(parse("123").unwrap(), Value::Int(123));
        assert_eq!(
            parse("123456789012345678901234567890").unwrap(),
            Value::BigInt("123456789012345678901234567890".into()),
        );
        assert_eq!(parse("1.0").unwrap(), Value::Float(1.0));
    }

    #[test]
    fn a_surrogate_pair_joins_and_a_lone_one_cannot() {
        assert_eq!(parse(r#""🙂""#).unwrap(), Value::Str("\u{1f642}".into()));
        assert_eq!(
            parse(r#""\ud800""#).unwrap(),
            Value::Str(crate::LONE_SURROGATE.to_string()),
        );
    }

    #[test]
    fn raw_decode_stops_at_the_end_of_the_value_and_reports_where() {
        let text: Vec<char> = "{\"a\": 1} trailing".chars().collect();
        let (value, end) = raw_decode(&text).expect("an object");
        assert_eq!(end, 8);
        assert!(matches!(value, Value::Object(_)));
    }
}
