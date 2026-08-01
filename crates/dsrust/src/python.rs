//! Python's `repr`, for the values that reach a prompt as Python source rather than as JSON.
//!
//! Several prompts show a value the way Python would print it — a tool's arguments, an RLM
//! submission, the dataset samples MIPROv2's proposer is asked to describe. The difference is
//! visible to the model: `{'city': 'Paris'}` rather than `{"city":"Paris"}`, `None` rather than
//! `null`, `True` rather than `true`. Matching it keeps the prompt bytes identical.
//!
//! One home rather than one per caller: there were two copies before this, one handling containers
//! and one only scalars, and the third caller is what made that worth fixing.

use serde_json::Value;

/// Python's `repr` of a JSON value.
pub(crate) fn repr(value: &Value) -> String {
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
            _ if character == quote => {
                out.push('\\');
                out.push(character);
            }
            _ => out.push(character),
        }
    }
    out.push(quote);
    out
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
}
