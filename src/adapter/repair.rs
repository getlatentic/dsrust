//! Reading a reply written as a Python literal rather than as strict JSON.
//!
//! dspy hands every reply value to json-repair before validating it against the declared
//! type, which is what lets a model that answered in Python's own syntax still satisfy the
//! signature. `{'score': 123_456.789}` is the case upstream pins: single-quoted keys and a
//! digit-grouped float, neither of which serde_json accepts.

use serde_json::Value;
use std::iter::Peekable;
use std::str::Chars;

/// The value a Python literal denotes, or `None` when the text is not one.
///
/// Text serde_json already accepts is left unrepaired: it has a reader downstream, and
/// rewriting a well-formed reply could only lose fidelity.
pub(crate) fn python_literal(text: &str) -> Option<Value> {
    if serde_json::from_str::<Value>(text).is_ok() {
        return None;
    }
    serde_json::from_str(&to_json(text)?).ok()
}

/// Rewrite the parts of Python's literal syntax that JSON spells differently, and leave the
/// rest — structure, whitespace, punctuation — to serde_json, which then judges the result.
fn to_json(text: &str) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(&next) = chars.peek() {
        match next {
            '\'' | '"' => copy_string(&mut chars, &mut out)?,
            '0'..='9' | '-' | '+' | '.' => copy_number(&mut chars, &mut out),
            next if next.is_alphabetic() => copy_word(&mut chars, &mut out)?,
            _ => {
                out.push(next);
                chars.next();
            }
        }
    }
    Some(out)
}

/// A quoted run in either of Python's two quote styles, re-emitted in JSON's one. Only the
/// delimiter changes meaning between the dialects, so the content carries over as written.
fn copy_string(chars: &mut Peekable<Chars>, out: &mut String) -> Option<()> {
    let quote = chars.next()?;
    out.push('"');
    loop {
        match chars.next()? {
            found if found == quote => break,
            // Python escapes its own delimiter; JSON has no `\'`, and the bare quote is legal.
            '\\' if chars.peek() == Some(&'\'') => out.push(chars.next()?),
            '\\' => {
                out.push('\\');
                out.push(chars.next()?);
            }
            '"' => out.push_str("\\\""),
            found => out.push(found),
        }
    }
    out.push('"');
    Some(())
}

/// A numeric literal with Python's digit-group underscores dropped. What the grouping hides
/// is still a number for serde_json to judge, so nothing else about the token is touched.
fn copy_number(chars: &mut Peekable<Chars>, out: &mut String) {
    while let Some(&next) = chars.peek() {
        match next {
            '_' => {}
            '.' | '+' | '-' => out.push(next),
            next if next.is_ascii_alphanumeric() => out.push(next),
            _ => break,
        }
        chars.next();
    }
}

/// Python's spellings of the three keywords. A bare word that is not one of them means the
/// text is prose or some other dialect, and reading further would be guesswork.
fn copy_word(chars: &mut Peekable<Chars>, out: &mut String) -> Option<()> {
    let mut word = String::new();
    while let Some(&next) = chars.peek() {
        if !next.is_alphanumeric() && next != '_' {
            break;
        }
        word.push(next);
        chars.next();
    }
    out.push_str(match word.as_str() {
        "True" | "true" => "true",
        "False" | "false" => "false",
        "None" | "null" => "null",
        _ => return None,
    });
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn digit_group_underscores_read_as_the_number_they_hide() {
        assert_eq!(
            python_literal("{'score': 123_456.789}"),
            Some(json!({ "score": 123_456.789 }))
        );
        assert_eq!(
            python_literal(r#"{"counts": [1_000, -2_0, 1_0e1]}"#),
            Some(json!({ "counts": [1000, -20, 100.0] }))
        );
    }

    #[test]
    fn underscores_inside_strings_and_keys_survive() {
        assert_eq!(
            python_literal("{'a_b': 'x_y'}"),
            Some(json!({ "a_b": "x_y" }))
        );
    }

    #[test]
    fn python_quotes_and_keywords_read_as_json_ones() {
        assert_eq!(
            python_literal("{'ok': True, 'bad': False, 'gone': None}"),
            Some(json!({ "ok": true, "bad": false, "gone": null }))
        );
        assert_eq!(
            python_literal(r#"{'say': 'it\'s "here"'}"#),
            Some(json!({ "say": "it's \"here\"" }))
        );
    }

    #[test]
    fn strict_json_is_left_for_the_reader_downstream() {
        assert_eq!(python_literal(r#"{"a": 1}"#), None);
        assert_eq!(python_literal(r#""hello""#), None);
    }

    #[test]
    fn prose_and_other_dialects_are_not_guessed_at() {
        assert_eq!(python_literal("no json here"), None);
        assert_eq!(python_literal("{unquoted: 1}"), None);
        assert_eq!(python_literal("{'a': 1"), None);
    }
}
