//! The words a custom type uses when it will not read a value.
//!
//! Upstream's `validate_input` raises f-strings that interpolate the offending value or its type —
//! `f"Received invalid value for \`dspy.Reasoning\`: {data}"`, `f"...received type: {type(data)}"`.
//! Matching those means rendering an incoming `serde_json::Value` the way Python would, which is
//! not how `Display` renders it: `Display` on a JSON string keeps the quotes and Python's `str`
//! does not, and a JSON object has no `{'key': 1}` spelling at all.
//!
//! Its own module because every type in this directory refuses in the same vocabulary, and a copy
//! per type is how three of them came to render a value three different ways.

use serde_json::Value;

/// Python's `str(value)` — what an f-string interpolating the value itself prints.
///
/// A string prints bare, and a container prints its *elements* with `repr`, which is why this and
/// [`python_repr`] call each other rather than one being written in terms of the other.
pub(super) fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => python_repr(other),
    }
}

/// Python's `repr(value)`, which is what a container prints for each of its elements.
fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(items) => {
            let rendered: Vec<String> = items.iter().map(python_repr).collect();
            format!("[{}]", rendered.join(", "))
        }
        Value::Object(fields) => {
            let rendered: Vec<String> = fields
                .iter()
                .map(|(key, item)| {
                    format!(
                        "'{}': {}",
                        key.replace('\\', "\\\\").replace('\'', "\\'"),
                        python_repr(item)
                    )
                })
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
    }
}

/// Python's `type(value)` as an f-string prints it — `<class 'int'>`.
///
/// A JSON number is an `int` only when it has no fractional part on the wire, which is the same
/// distinction `serde_json::Number` draws and the one Python's parser draws reading the same bytes.
pub(super) fn python_type(value: &Value) -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The one difference that matters between `str` and `repr`, and the reason both exist here:
    /// a bare string interpolates without quotes, but the same string inside a container keeps them.
    #[test]
    fn a_string_prints_bare_alone_and_quoted_inside_a_container() {
        assert_eq!(python_str(&json!("not a citation")), "not a citation");
        assert_eq!(python_str(&json!(["not a citation"])), "['not a citation']");
        assert_eq!(python_str(&json!({"unrelated": 1})), "{'unrelated': 1}");
    }

    /// Python's literals, which are not JSON's.
    #[test]
    fn the_three_python_literals_are_not_spelled_as_json_spells_them() {
        assert_eq!(python_str(&json!(null)), "None");
        assert_eq!(python_str(&json!(true)), "True");
        assert_eq!(python_str(&json!(false)), "False");
    }

    #[test]
    fn a_number_is_an_int_only_without_a_fractional_part() {
        assert_eq!(python_type(&json!(5)), "<class 'int'>");
        assert_eq!(python_type(&json!(5.5)), "<class 'float'>");
        assert_eq!(python_type(&json!("x")), "<class 'str'>");
        assert_eq!(python_type(&json!({})), "<class 'dict'>");
    }
}
