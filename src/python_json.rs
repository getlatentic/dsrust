//! Serializing a JSON value the way Python's `json.dumps` prints it.
//!
//! dspy renders every structured field value through `json.dumps`, whose default separators put
//! a space after each comma and colon. serde_json's compact form emits neither, so a value that
//! round-trips through `Value::to_string` reaches the model as different bytes than upstream
//! sends — on every list- or object-valued field in every prompt.

use std::io;

use serde::Serialize;
use serde_json::Value;
use serde_json::ser::{Formatter, Serializer};

/// The `json.dumps` defaults: `", "` between items, `": "` after a key. Escaping, number shape
/// and the absence of padding inside brackets already agree, so the separators are the only
/// departure from serde_json's compact form.
struct DumpsDefaults;

impl Formatter for DumpsDefaults {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        match first {
            true => Ok(()),
            false => writer.write_all(b", "),
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        match first {
            true => Ok(()),
            false => writer.write_all(b", "),
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(b": ")
    }
}

/// dspy passes `ensure_ascii=False`, so non-ASCII text stays as itself rather than becoming
/// `\uXXXX` escapes — which is serde_json's behaviour already.
pub(crate) fn dumps(value: &Value) -> String {
    let mut out = Vec::new();
    let mut serializer = Serializer::with_formatter(&mut out, DumpsDefaults);
    // A `Value` is always valid JSON and a `Vec` never fails a write, so neither step can fail.
    value
        .serialize(&mut serializer)
        .expect("a Value serializes into a Vec without failing");
    String::from_utf8(out).expect("serde_json writes UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Every expectation here was read off dspy 3.2.1's own `json.dumps(value, ensure_ascii=False)`.
    #[test]
    fn separators_carry_a_space_the_way_json_dumps_does() {
        assert_eq!(dumps(&json!(["a", "b"])), r#"["a", "b"]"#);
        assert_eq!(dumps(&json!({"a": 1, "b": 2})), r#"{"a": 1, "b": 2}"#);
    }

    /// Keys are alphabetical here so the assertion turns purely on nesting: serde_json's `Map`
    /// sorts, where a Python dict keeps insertion order, and that gap is not this test's subject.
    #[test]
    fn nesting_spaces_every_level() {
        assert_eq!(
            dumps(&json!({"alpha": {"inner": [1, 2, {"k": "v"}]}, "beta": {}, "gamma": []})),
            r#"{"alpha": {"inner": [1, 2, {"k": "v"}]}, "beta": {}, "gamma": []}"#
        );
    }

    #[test]
    fn an_empty_container_has_no_separator_to_space() {
        assert_eq!(dumps(&json!([])), "[]");
        assert_eq!(dumps(&json!({})), "{}");
    }

    #[test]
    fn non_ascii_survives_unescaped() {
        assert_eq!(
            dumps(&json!(["héllo", "日本語", "emoji 🎉"])),
            r#"["héllo", "日本語", "emoji 🎉"]"#
        );
        assert_eq!(
            dumps(&json!({"ké": "vá", "nested": ["ü", null, true, false]})),
            r#"{"ké": "vá", "nested": ["ü", null, true, false]}"#
        );
    }

    #[test]
    fn escaping_matches_python() {
        assert_eq!(
            dumps(&json!({"s": "quote\" back\\ nl\n tab\t"})),
            r#"{"s": "quote\" back\\ nl\n tab\t"}"#
        );
    }
}
