//! The scanner of `strict_json.rs`, over bytes.
//!
//! The char scanner exists because the *repair* parser indexes by code point, as Python does, and
//! `raw_decode` answers with a position inside that world. The whole-input fast path has no such
//! caller: it either takes the value or hands the text to the repair parser untouched. Scanning
//! bytes there skips decoding the input into a `Vec<char>` — four bytes a character and an
//! allocation before a single byte is judged — and copies string spans in bulk instead of pushing
//! characters one at a time.
//!
//! Byte scanning is UTF-8-transparent for this grammar: every byte the scanner branches on — the
//! quotes, the backslash, the brackets, digits, whitespace, controls below 0x20 — is ASCII, and no
//! byte of a multi-byte UTF-8 sequence is below 0x80, so a lead or continuation byte can never be
//! mistaken for one of them.
//!
//! `tests/scanner_agreement.rs` holds the two scanners to the same answer over every committed
//! input, which is what makes a parallel grammar tolerable at all.

use super::NotJson;
use crate::value::{Object, Value};

type Scan = Result<(Value, usize), NotJson>;

/// The input, with its reads counted in debug builds — the same guard the char scanner carries,
/// for the same reason: a loop whose index stops advancing must fail, not hang.
struct Text<'a> {
    text: &'a str,
    #[cfg(debug_assertions)]
    reads: std::cell::Cell<u64>,
    #[cfg(debug_assertions)]
    budget: u64,
}

impl<'a> Text<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            #[cfg(debug_assertions)]
            reads: std::cell::Cell::new(0),
            #[cfg(debug_assertions)]
            budget: (text.len() as u64)
                .saturating_mul(8)
                .saturating_add(1 << 12),
        }
    }

    fn get(&self, index: usize) -> Option<u8> {
        #[cfg(debug_assertions)]
        {
            let read = self.reads.get() + 1;
            self.reads.set(read);
            assert!(
                read <= self.budget,
                "{read} reads over {} bytes of strict JSON — the scan is not advancing",
                self.text.len()
            );
        }
        self.text.as_bytes().get(index).copied()
    }

    fn slice(&self, range: std::ops::Range<usize>) -> &'a str {
        &self.text[range]
    }

    fn len(&self) -> usize {
        self.text.len()
    }
}

/// `json.loads(text)`: one value, with nothing but whitespace around it.
pub(crate) fn loads(text: &str) -> Result<Value, NotJson> {
    let text = Text::new(text);
    let start = skip_whitespace(&text, 0);
    let (value, end) = scan_once(&text, start)?;
    let end = skip_whitespace(&text, end);
    if end != text.len() {
        return Err(NotJson);
    }
    Ok(value)
}

/// `JSONDecoder().raw_decode(text)`: one value, and where it ended — in bytes, which for the
/// ASCII caller is the same number as code points. Trailing text is the caller's.
pub(crate) fn raw_decode(text: &str) -> Scan {
    scan_once(&Text::new(text), 0)
}

/// `json.decoder.WHITESPACE`, which is these four characters and not `str.isspace()`.
fn skip_whitespace(text: &Text<'_>, mut index: usize) -> usize {
    while matches!(text.get(index), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        index += 1;
    }
    index
}

fn literal(text: &Text<'_>, index: usize, word: &str) -> bool {
    word.bytes()
        .enumerate()
        .all(|(offset, expected)| text.get(index + offset) == Some(expected))
}

fn scan_once(text: &Text<'_>, index: usize) -> Scan {
    match text.get(index) {
        Some(b'"') => scan_string(text, index + 1),
        Some(b'{') => scan_object(text, index + 1),
        Some(b'[') => scan_array(text, index + 1),
        Some(b'n') if literal(text, index, "null") => Ok((Value::Null, index + 4)),
        Some(b't') if literal(text, index, "true") => Ok((Value::Bool(true), index + 4)),
        Some(b'f') if literal(text, index, "false") => Ok((Value::Bool(false), index + 5)),
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
    if text.get(end) == Some(b'-') {
        end += 1;
    }
    let integer_start = end;
    match text.get(end) {
        Some(b'0') => end += 1,
        Some(digit) if digit.is_ascii_digit() => {
            while text.get(end).is_some_and(|byte| byte.is_ascii_digit()) {
                end += 1;
            }
        }
        _ => return None,
    }
    if end == integer_start {
        return None;
    }

    let mut is_float = false;
    if text.get(end) == Some(b'.') && text.get(end + 1).is_some_and(|byte| byte.is_ascii_digit()) {
        end += 1;
        while text.get(end).is_some_and(|byte| byte.is_ascii_digit()) {
            end += 1;
        }
        is_float = true;
    }
    if let Some(b'e' | b'E') = text.get(end) {
        let mut exponent = end + 1;
        if let Some(b'+' | b'-') = text.get(exponent) {
            exponent += 1;
        }
        if text.get(exponent).is_some_and(|byte| byte.is_ascii_digit()) {
            while text.get(exponent).is_some_and(|byte| byte.is_ascii_digit()) {
                exponent += 1;
            }
            end = exponent;
            is_float = true;
        }
    }

    let literal = text.slice(index..end);
    let value = if is_float {
        Value::Float(literal.parse().ok()?)
    } else {
        crate::pynum::python_int(literal)
    };
    Some((value, end))
}

fn scan_object(text: &Text<'_>, index: usize) -> Scan {
    let mut object = Object::new();
    let mut end = skip_whitespace(text, index);
    if text.get(end) == Some(b'}') {
        return Ok((Value::Object(object), end + 1));
    }
    if text.get(end) != Some(b'"') {
        return Err(NotJson);
    }
    loop {
        let (key, after_key) = scan_string(text, end + 1)?;
        let Value::Str(key) = key else {
            return Err(NotJson);
        };
        end = skip_whitespace(text, after_key);
        if text.get(end) != Some(b':') {
            return Err(NotJson);
        }
        end = skip_whitespace(text, end + 1);
        let (value, after_value) = scan_once(text, end)?;
        object.insert(key, value);
        end = skip_whitespace(text, after_value);
        match text.get(end) {
            Some(b'}') => return Ok((Value::Object(object), end + 1)),
            Some(b',') => end = skip_whitespace(text, end + 1),
            _ => return Err(NotJson),
        }
        if text.get(end) != Some(b'"') {
            return Err(NotJson);
        }
    }
}

fn scan_array(text: &Text<'_>, index: usize) -> Scan {
    let mut items = Vec::new();
    let mut end = skip_whitespace(text, index);
    if text.get(end) == Some(b']') {
        return Ok((Value::Array(items), end + 1));
    }
    loop {
        let (value, after_value) = scan_once(text, end)?;
        items.push(value);
        end = skip_whitespace(text, after_value);
        match text.get(end) {
            Some(b']') => return Ok((Value::Array(items), end + 1)),
            Some(b',') => end = skip_whitespace(text, end + 1),
            _ => return Err(NotJson),
        }
    }
}

/// `py_scanstring` with `strict=True`: a raw byte below 0x20 ends the parse.
///
/// The span between escapes is copied in one `push_str` rather than a character at a time, which
/// is most of what this scanner is for — a long string is one memcpy here.
fn scan_string(text: &Text<'_>, index: usize) -> Scan {
    let mut out = String::new();
    let mut span_start = index;
    let mut end = index;
    loop {
        let Some(byte) = text.get(end) else {
            return Err(NotJson);
        };
        match byte {
            b'"' => {
                out.push_str(text.slice(span_start..end));
                return Ok((Value::Str(out), end + 1));
            }
            b'\\' => {
                out.push_str(text.slice(span_start..end));
                end += 1;
                let Some(escape) = text.get(end) else {
                    return Err(NotJson);
                };
                end += 1;
                match escape {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let (ch, after) = scan_unicode_escape(text, end)?;
                        out.push(ch);
                        end = after;
                    }
                    _ => return Err(NotJson),
                }
                span_start = end;
            }
            byte if byte < 0x20 => return Err(NotJson),
            _ => end += 1,
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
        && text.get(end) == Some(b'\\')
        && text.get(end + 1) == Some(b'u')
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
    let mut value = 0u32;
    for offset in 0..4 {
        let byte = text.get(index + offset).ok_or(NotJson)?;
        let digit = (byte as char).to_digit(16).ok_or(NotJson)?;
        value = value * 16 + digit;
    }
    Ok(value)
}
