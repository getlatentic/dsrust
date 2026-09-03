//! Python's `repr`, for the values that reach a prompt as Python source rather than as JSON.
//!
//! Several prompts show a value the way Python would print it — a tool's arguments, an RLM
//! submission, the dataset samples MIPROv2's proposer is asked to describe. The difference is
//! visible to the model: `{'city': 'Paris'}` rather than `{"city":"Paris"}`, `None` rather than
//! `null`, `True` rather than `true`. Matching it keeps the prompt bytes identical.
//!
//! One home rather than one per caller: there were two copies before this, one handling containers
//! and one only scalars, and the third caller is what made that worth fixing. A fourth appeared in
//! `adapter/types/` on 2026-08-26, quoting strings a way Python does not, and came back here.

use serde_json::Value;

/// Python's `repr` of a JSON value.
pub fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(text) => quoted(text),
        Value::Array(items) => format!("[{}]", joined(items.iter().map(repr))),
        Value::Object(fields) => format!(
            "{{{}}}",
            joined(
                fields
                    .iter()
                    .map(|(key, value)| format!("{}: {}", quoted(key), repr(value)))
            )
        ),
        number => number.to_string(),
    }
}

/// Python's `repr` of a string.
///
/// Single quotes normally — but a string containing a `'` and no `"` is quoted with `"` instead, and
/// then the apostrophe is *not* escaped: `repr("it's")` is `"it's"`, not `'it\'s'`. Only when the
/// string holds both does CPython fall back to single quotes and escape. Backslashes are always
/// doubled, whichever quote is chosen.
///
/// Measured against CPython, after this escaped unconditionally and disagreed with `repr` on every
/// string carrying an apostrophe — which reached a prompt anywhere a tool's arguments or an RLM
/// submission was shown.
///
/// Control characters are escaped as CPython escapes them — `\n`, `\r`, `\t` by name and the rest
/// of the C0 range plus DEL as `\xNN`. That is not cosmetic where the result is *source*: a raw
/// newline inside `'…'` is an unterminated string, so a value carrying one produced a syntax error
/// inside the sandbox rather than a value.
///
/// **Falls short of `repr` for non-ASCII non-printables** — NBSP, format characters, unassigned
/// code points — which CPython escapes and this passes through. The string means the same either
/// way; the bytes differ. Closing it needs `str.isprintable()` as a table taken from the
/// interpreter, the way `pychar_data.txt` holds the four predicates `json_repair` branches on.
pub(crate) fn quoted(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ if character == quote => {
                out.push('\\');
                out.push(character);
            }
            _ if character.is_control() && character.is_ascii() => {
                out.push_str(&format!("\\x{:02x}", character as u32));
            }
            _ => out.push(character),
        }
    }
    out.push(quote);
    out
}

/// Python's `str` of a JSON value — what an f-string interpolating the value itself prints.
///
/// The whole difference from [`repr`] is the bare string: `str("x")` is `x` where `repr("x")` is
/// `'x'`. A container prints its elements with `repr` either way, so everything else defers.
pub(crate) fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => repr(other),
    }
}

/// Python's `type(value)` as an f-string prints it — `<class 'int'>`.
///
/// A JSON number is an `int` only when it has no fractional part on the wire, which is the same
/// distinction `serde_json::Number` draws and the one Python's parser draws reading the same bytes.
pub(crate) fn type_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "<class 'NoneType'>",
        Value::Bool(_) => "<class 'bool'>",
        Value::Number(number) if number.is_f64() => "<class 'float'>",
        Value::Number(_) => "<class 'int'>",
        Value::String(_) => "<class 'str'>",
        Value::Array(_) => "<class 'list'>",
        Value::Object(_) => "<class 'dict'>",
    }
}

/// Python's `repr` of a tuple, whose one-element form carries a trailing comma.
pub(crate) fn tuple(values: &[Value]) -> String {
    let members: Vec<String> = values.iter().map(repr).collect();
    match members.len() {
        1 => format!("({},)", members[0]),
        _ => format!("({})", members.join(", ")),
    }
}

fn joined(parts: impl Iterator<Item = String>) -> String {
    parts.collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The four literals that differ from JSON's, which is the whole reason this exists.
    #[test]
    fn the_scalars_python_spells_differently() {
        assert_eq!(repr(&Value::Null), "None");
        assert_eq!(repr(&json!(true)), "True");
        assert_eq!(repr(&json!(false)), "False");
        assert_eq!(repr(&json!("Paris")), "'Paris'");
    }

    /// CPython's quoting rule, measured rather than assumed: an apostrophe alone switches the
    /// quotes to `"` and is then *not* escaped; a string holding both quotes falls back to `'` and
    /// escapes. This escaped unconditionally before, and disagreed with `repr` on every string
    /// carrying an apostrophe.
    #[test]
    fn the_quote_python_picks_depends_on_what_the_string_holds() {
        assert_eq!(repr(&json!("it's")), r#""it's""#);
        assert_eq!(repr(&json!(r#"say "hi""#)), r#"'say "hi"'"#);
        assert_eq!(repr(&json!("both ' and \"")), r#"'both \' and "'"#);
        assert_eq!(repr(&json!("plain")), "'plain'");
    }

    /// A backslash is always doubled, whichever quote was chosen.
    #[test]
    fn a_backslash_is_always_doubled() {
        assert_eq!(repr(&json!(r"a\b")), r"'a\\b'");
        assert_eq!(repr(&json!(r"it's a\b")), r#""it's a\\b""#);
    }

    /// Keys are quoted too — a Python dict's keys are strings and print as strings.
    #[test]
    fn a_mapping_quotes_its_keys() {
        assert_eq!(
            repr(&json!({"city": "Paris", "n": 2})),
            "{'city': 'Paris', 'n': 2}"
        );
    }

    /// A one-element tuple carries the comma that tells it from a parenthesised value.
    #[test]
    fn a_single_element_tuple_keeps_its_comma() {
        assert_eq!(tuple(&[json!("yes")]), "('yes',)");
        assert_eq!(tuple(&[json!("yes"), json!(1)]), "('yes', 1)");
    }

    /// `str` differs from `repr` on exactly one type, and a container is not it.
    #[test]
    fn a_string_prints_bare_alone_and_quoted_inside_a_container() {
        assert_eq!(text(&json!("not a citation")), "not a citation");
        assert_eq!(text(&json!(["not a citation"])), "['not a citation']");
        assert_eq!(text(&json!({"unrelated": 1})), "{'unrelated': 1}");
    }

    #[test]
    fn a_number_is_an_int_only_without_a_fractional_part() {
        assert_eq!(type_of(&json!(5)), "<class 'int'>");
        assert_eq!(type_of(&json!(5.5)), "<class 'float'>");
        assert_eq!(type_of(&json!("x")), "<class 'str'>");
        assert_eq!(type_of(&json!({})), "<class 'dict'>");
    }
}
