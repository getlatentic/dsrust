//! `json.dumps(value)` with its default arguments, which is what `repair_json` returns.
//!
//! Not `serde_json::to_string`: Python separates with `", "` and `": "`, escapes every code point
//! outside `\x20`-`\x7e`, writes `NaN`/`Infinity` where JSON has no spelling at all, and renders a
//! float with `float.__repr__` — shortest round-trip, switching to an exponent outside a window
//! that is Python's rather than Rust's.

use crate::value::Value;

/// The text `json.dumps` would produce for `value`.
///
/// `ascii` is `json.dumps`'s `ensure_ascii`: on, every code point outside `\x20`-`\x7e` leaves as
/// an escape; off, only the characters JSON itself requires do.
pub(crate) fn dumps(value: &Value, ascii: bool) -> String {
    let mut out = String::new();
    write(value, ascii, &mut out);
    out
}

fn write(value: &Value, ascii: bool, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(number) => {
            use std::fmt::Write;
            let _ = write!(out, "{number}");
        }
        Value::BigInt(digits) => out.push_str(digits),
        Value::Float(number) => out.push_str(&float_repr(*number)),
        Value::Str(text) => write_string(text, ascii, out),
        Value::Array(items) => {
            out.push('[');
            for (position, item) in items.iter().enumerate() {
                if position > 0 {
                    out.push_str(", ");
                }
                write(item, ascii, out);
            }
            out.push(']');
        }
        Value::Object(fields) => {
            out.push('{');
            for (position, (key, item)) in fields.iter().enumerate() {
                if position > 0 {
                    out.push_str(", ");
                }
                write_string(key, ascii, out);
                out.push_str(": ");
                write(item, ascii, out);
            }
            out.push('}');
        }
    }
}

/// `py_encode_basestring_ascii` when `ascii` is set, and `py_encode_basestring` when it is not —
/// the two escaping tables `json.dumps` chooses between on `ensure_ascii`.
fn write_string(text: &str, ascii: bool, out: &mut String) {
    out.push('"');
    let bytes = text.as_bytes();
    let mut span_start = 0;
    let mut at = 0;
    // One forward pass, so the steps cannot exceed the bytes plus a constant. Four mutants held
    // this loop spinning for the full timeout before the guard; the scanners already count their
    // reads for the same reason, and a writer's cursor is no different.
    #[cfg(debug_assertions)]
    let mut steps: u64 = 0;
    // Already carrying the bound the lint asks for: the debug step counter above panics on a
    // stalled walk, and the release build inherits the loop the counter proved finite.
    // ast-grep-ignore: cursor-arithmetic-loop
    while at < bytes.len() {
        #[cfg(debug_assertions)]
        {
            steps += 1;
            assert!(
                steps <= bytes.len() as u64 + 8,
                "{steps} steps over {} bytes — the writer is not advancing",
                bytes.len()
            );
        }
        let byte = bytes[at];
        // The span runs until a byte the table escapes. With `ensure_ascii` off that is exactly
        // `py_encode_basestring`'s `[\x00-\x1f\\"]` — JSON requires the controls escaped
        // whichever way the flag is set — and with it on, everything past ASCII joins the list.
        // Multi-byte UTF-8 never holds a byte below 0x80, so the byte test cannot fire inside one.
        if byte >= 0x20 && byte != b'"' && byte != b'\\' && (!ascii || byte < 0x7f) {
            at += 1;
            continue;
        }
        out.push_str(&text[span_start..at]);
        if byte < 0x80 && (byte < 0x7f || ascii) {
            // DEL sits past `~` and before the multi-byte world: `ensure_ascii` writes it
            // `\u007f` like any other character outside `' '..='~'`.
            match byte {
                b'\\' => out.push_str("\\\\"),
                b'"' => out.push_str("\\\""),
                0x08 => out.push_str("\\b"),
                0x0c => out.push_str("\\f"),
                b'\n' => out.push_str("\\n"),
                b'\r' => out.push_str("\\r"),
                b'\t' => out.push_str("\\t"),
                _ => push_u16_escape(byte as u32, out),
            }
            at += 1;
        } else {
            let ch = text[at..]
                .chars()
                .next()
                .expect("a span boundary is a char boundary");
            let code_point = ch as u32;
            if code_point < 0x10000 {
                push_u16_escape(code_point, out);
            } else {
                let offset = code_point - 0x10000;
                push_u16_escape(0xd800 + (offset >> 10), out);
                push_u16_escape(0xdc00 + (offset & 0x3ff), out);
            }
            at += ch.len_utf8();
        }
        span_start = at;
    }
    out.push_str(&text[span_start..]);
    out.push('"');
}

/// `\uXXXX`, without the `format!` machinery — the escape is four hex digits, not a formatter.
fn push_u16_escape(code_point: u32, out: &mut String) {
    // Four nibbles hold sixteen bits; anything wider must arrive as a surrogate pair, and writing
    // it here would truncate silently rather than fail.
    debug_assert!(
        code_point < 0x10000,
        "{code_point:#x} does not fit one escape"
    );
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push('\\');
    out.push('u');
    for shift in [12, 8, 4, 0] {
        out.push(HEX[((code_point >> shift) & 0xf) as usize] as char);
    }
}

/// `float.__repr__`, except for the two values JSON cannot spell and `json.dumps` names anyway.
fn float_repr(number: f64) -> String {
    if number.is_nan() {
        return "NaN".to_owned();
    }
    if number.is_infinite() {
        return if number > 0.0 {
            "Infinity"
        } else {
            "-Infinity"
        }
        .to_owned();
    }
    if number == 0.0 {
        return if number.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_owned();
    }

    let (sign, digits, decimal_point) = shortest_digits(number);
    // CPython's `format_float_short` leaves the fixed form for an exponent below -4 or above 16,
    // so `1e15` prints in full and `1e16` does not.
    let body = if decimal_point <= -4 || decimal_point > 16 {
        exponential(&digits, decimal_point)
    } else {
        fixed(&digits, decimal_point)
    };
    format!("{sign}{body}")
}

/// The shortest round-tripping digits, and where the decimal point sits relative to them: the
/// value is `0.<digits> * 10^decimal_point`.
fn shortest_digits(number: f64) -> (&'static str, String, i32) {
    let rendered = format!("{number:e}");
    let (mantissa, exponent) = rendered.split_once('e').expect("`{:e}` writes an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` writes a decimal exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    (sign, digits, exponent + 1)
}

fn exponential(digits: &str, decimal_point: i32) -> String {
    let exponent = decimal_point - 1;
    let (leading, rest) = digits.split_at(1);
    let mantissa = if rest.is_empty() {
        leading.to_owned()
    } else {
        format!("{leading}.{rest}")
    };
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exponent.abs())
}

fn fixed(digits: &str, decimal_point: i32) -> String {
    let point = decimal_point as usize;
    if decimal_point <= 0 {
        let zeros = "0".repeat(decimal_point.unsigned_abs() as usize);
        format!("0.{zeros}{digits}")
    } else if point >= digits.len() {
        let zeros = "0".repeat(point - digits.len());
        format!("{digits}{zeros}.0")
    } else {
        let (whole, fraction) = digits.split_at(point);
        format!("{whole}.{fraction}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Object;

    fn dumps_ascii(value: &Value) -> String {
        dumps(value, true)
    }

    #[test]
    fn floats_switch_to_an_exponent_where_python_switches() {
        assert_eq!(float_repr(1e15), "1000000000000000.0");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(1e-4), "0.0001");
        assert_eq!(float_repr(1e-5), "1e-05");
        assert_eq!(float_repr(1.5e-5), "1.5e-05");
        assert_eq!(float_repr(1e100), "1e+100");
        assert_eq!(float_repr(1.0 / 3.0), "0.3333333333333333");
        assert_eq!(float_repr(-0.0), "-0.0");
        assert_eq!(float_repr(2.5), "2.5");
        assert_eq!(float_repr(f64::INFINITY), "Infinity");
        assert_eq!(float_repr(f64::NEG_INFINITY), "-Infinity");
        assert!(float_repr(f64::NAN) == "NaN");
        // Neither is JSON, and `json.dumps` writes both anyway.
    }

    #[test]
    fn strings_leave_as_ascii_the_way_ensure_ascii_does() {
        assert_eq!(dumps_ascii(&Value::Str("\u{e9}".into())), "\"\\u00e9\"");
        assert_eq!(
            dumps_ascii(&Value::Str("\u{1f642}".into())),
            "\"\\ud83d\\ude42\""
        );
        assert_eq!(
            dumps_ascii(&Value::Str("a\"b\\c\n".into())),
            r#""a\"b\\c\n""#
        );
        assert_eq!(dumps_ascii(&Value::Str("\u{7f}".into())), "\"\\u007f\"");
    }

    #[test]
    fn ensure_ascii_off_leaves_the_characters_json_does_not_require_escaping() {
        // The escapes JSON *requires* still happen; only the `\\uXXXX` fallback stops.
        assert_eq!(dumps(&Value::Str("统一码".into()), false), "\"统一码\"");
        assert_eq!(
            dumps(&Value::Str("统一码".into()), true),
            "\"\\u7edf\\u4e00\\u7801\""
        );
        assert_eq!(dumps(&Value::Str("a\"b\n".into()), false), r#""a\"b\n""#);
        assert_eq!(dumps(&Value::Str("\u{7f}".into()), false), "\"\u{7f}\"");
        // A control character stays escaped either way; only the fallback for ordinary code points
        // is what `ensure_ascii` turns off.
        assert_eq!(dumps(&Value::Str("\u{1}".into()), false), "\"\\u0001\"");
    }

    #[test]
    fn every_escape_json_dumps_has_a_short_name_for() {
        // `\b`, `\f` and `\r` reach no value in the conformance corpus — a model does not write
        // them and `json.dumps` is the only thing that ever names them — so without this each is a
        // match arm nothing distinguishes from the `\u00XX` fallback.
        assert_eq!(
            dumps_ascii(&Value::Str("\u{8}\u{c}\r\t".into())),
            r#""\b\f\r\t""#
        );
    }

    #[test]
    fn the_boundary_between_an_escape_and_a_surrogate_pair_is_where_python_puts_it() {
        // U+FFFF is the last code point with one escape and U+10000 the first with two, which is
        // the only pair that tells `code_point < 0x10000` from `<=`.
        assert_eq!(dumps_ascii(&Value::Str("\u{ffff}".into())), "\"\\uffff\"");
        assert_eq!(
            dumps_ascii(&Value::Str("\u{10000}".into())),
            "\"\\ud800\\udc00\""
        );
    }

    #[test]
    fn containers_carry_pythons_separators() {
        let object: Object = [
            ("a".to_owned(), Value::Int(1)),
            ("b".to_owned(), Value::Null),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            dumps_ascii(&Value::Object(object)),
            r#"{"a": 1, "b": null}"#
        );
        assert_eq!(
            dumps_ascii(&Value::Array(vec![Value::Int(1), Value::Int(2)])),
            "[1, 2]"
        );
    }
}
