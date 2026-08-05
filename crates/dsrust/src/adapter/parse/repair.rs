//! Reading a reply that is not strict JSON: json-repair first, then Python's literal syntax.
//!
//! dspy hands every reply value to json-repair before validating it against the declared type,
//! which is what lets a model that answered in Python's own syntax still satisfy the signature.
//! [`loads`] is that library, reproduced in [`json_repair`]; [`python_literal`] stands in for the
//! `ast.literal_eval` upstream falls back to when json-repair found nothing at all.

use serde_json::{Value, json};
use std::iter::Peekable;
use std::str::Chars;

/// `json_repair.loads(text)`, in the shapes this crate speaks.
///
/// Two of Python's shapes have no serde_json spelling and are converted rather than carried:
///
/// - an integer wider than `i64`/`u64` becomes a float, which is what serde_json already does
///   reading one out of ordinary JSON, so the two paths agree;
/// - a non-finite float becomes the text `str()` gives it — `nan`, `inf`, `-inf` — because JSON
///   has no literal for one and `parse_value` under a `str` annotation produces exactly that.
///   A numeric field answered `Infinity` therefore reads as text here and as a float upstream.
pub(crate) fn loads(text: &str) -> Result<Value, json_repair::Error> {
    json_repair::loads(text).map(from_repaired)
}

fn from_repaired(value: json_repair::Value) -> Value {
    use json_repair::Value as Repaired;
    match value {
        Repaired::Null => Value::Null,
        Repaired::Bool(flag) => Value::Bool(flag),
        Repaired::Int(number) => Value::from(number),
        Repaired::BigInt(digits) => digits.parse::<f64>().map_or(Value::Null, Value::from),
        Repaired::Float(number) if number.is_finite() => Value::from(number),
        Repaired::Float(number) if number.is_nan() => Value::from("nan"),
        Repaired::Float(number) => Value::from(if number > 0.0 { "inf" } else { "-inf" }),
        Repaired::Str(text) => Value::String(text),
        Repaired::Array(items) => Value::Array(items.into_iter().map(from_repaired).collect()),
        Repaired::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, from_repaired(value)))
                .collect(),
        ),
    }
}

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

/// Rewrite the parts of Python's literal syntax that JSON spells differently, and close any
/// container the writer left open, leaving the rest for serde_json to judge.
fn to_json(text: &str) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut open: Vec<char> = Vec::new();
    while let Some(&next) = chars.peek() {
        match next {
            '\'' | '"' => copy_string(&mut chars, &mut out)?,
            '0'..='9' | '-' | '+' | '.' => copy_number(&mut chars, &mut out),
            next if next.is_alphabetic() => copy_word(&mut chars, &mut out)?,
            '[' | '{' => {
                open.push(next);
                out.push(next);
                chars.next();
            }
            ']' | '}' => {
                close_through(&mut open, next, &mut out)?;
                chars.next();
            }
            _ => {
                out.push(next);
                chars.next();
            }
        }
    }
    // A reply cut short leaves its containers open; closing them reads the value the writer
    // was part-way through rather than discarding everything it did say.
    while let Some(opener) = open.pop() {
        drop_dangling_comma(&mut out);
        out.push(closer(opener));
    }
    Some(out)
}

/// Emit `found`, closing any container opened inside the one it belongs to.
///
/// A model that writes `[{"a": 1]` closed the list while its object was still open. Upstream's
/// repairer reads that as the object ending too, which is the only reading that keeps the value
/// — so a closer that does not match the innermost container closes the inner ones first.
fn close_through(open: &mut Vec<char>, found: char, out: &mut String) -> Option<()> {
    let wanted = match found {
        ']' => '[',
        _ => '{',
    };
    // A closer with nothing open is damage this cannot read; serde_json will reject it.
    let depth = open.iter().rposition(|opener| *opener == wanted)?;
    while open.len() > depth + 1 {
        let inner = open.pop()?;
        drop_dangling_comma(out);
        out.push(closer(inner));
    }
    open.pop();
    drop_dangling_comma(out);
    out.push(found);
    Some(())
}

/// Drop a comma left hanging before a container closes.
///
/// A model that writes `{"query": "cats",}` separated a member it then did not write. JSON has no
/// reading for that and serde_json refuses the whole value; upstream's repairer takes the members
/// that are there, which is the only reading that keeps any of them.
fn drop_dangling_comma(out: &mut String) {
    let kept = out.trim_end();
    if let Some(without) = kept.strip_suffix(',') {
        out.truncate(without.len());
    }
}

fn closer(opener: char) -> char {
    match opener {
        '[' => ']',
        _ => '}',
    }
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
    match word.as_str() {
        "True" | "true" => out.push_str("true"),
        "False" | "false" => out.push_str("false"),
        "None" | "null" => out.push_str("null"),
        // A bare word is a model writing JSON from memory and forgetting the quotes, on a key
        // or on a value. json-repair reads it as the string it plainly is, and a reply this
        // close to right is worth more read than refused.
        name => out.push_str(&format!("{}", json!(name))),
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A comma separating a member the writer never wrote. serde_json refuses the whole value;
    /// upstream's repairer keeps the members that are there. The expected values are json-repair's
    /// own, for the same inputs.
    #[test]
    fn a_comma_left_hanging_before_a_closer_is_dropped() {
        for (written, meant) in [
            (r#"{"query": "cats",}"#, json!({ "query": "cats" })),
            ("[1, 2,]", json!([1, 2])),
            (r#"{"a": [1,2,],}"#, json!({ "a": [1, 2] })),
        ] {
            assert_eq!(python_literal(written), Some(meant), "{written}");
        }
    }

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
    fn prose_is_not_guessed_at() {
        // Words with no structure around them are not a value that lost its punctuation.
        assert_eq!(python_literal("no json here"), None);
    }

    #[test]
    fn a_word_that_lost_its_quotes_is_read_as_the_string_it_is() {
        // A model writing JSON from memory drops quotes on a key or a value; json-repair reads
        // both, and a reply this close to right is worth more read than refused.
        assert_eq!(
            python_literal("{unquoted: 1}"),
            Some(json!({ "unquoted": 1 }))
        );
        assert_eq!(python_literal("{a: b}"), Some(json!({ "a": "b" })));
    }

    #[test]
    fn a_container_left_open_is_closed_rather_than_discarded() {
        // A model that stops mid-structure has still said most of what it meant, and upstream's
        // repairer reads it that way.
        assert_eq!(python_literal("{'a': 1"), Some(json!({ "a": 1 })));
        assert_eq!(python_literal("[1, 2"), Some(json!([1, 2])));
    }

    #[test]
    fn a_closer_that_skips_a_level_closes_what_it_skipped() {
        // `[{'name': 'x', 'args': {'city': 'Paris'}]` is upstream's own tool-call case: the
        // list closes while the object is still open, and the only reading that keeps the
        // value ends the object there too.
        assert_eq!(
            python_literal("[{'name': 'get_weather', 'args': {'city': 'Paris'}]"),
            Some(json!([{ "name": "get_weather", "args": { "city": "Paris" } }]))
        );
    }

    #[test]
    fn a_stray_closer_is_still_refused() {
        // Nothing was open, so this is damage with no reading rather than a truncation.
        assert_eq!(python_literal("}'a': 1}"), None);
    }
}
