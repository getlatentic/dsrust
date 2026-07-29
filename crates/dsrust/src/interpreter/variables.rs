//! dspy `_inject_variables`: the caller's values, as Python assignments above the model's code.
//!
//! This is how a module puts its inputs where generated code can reach them — `RLM` passes its
//! input fields every turn, so `SUBMIT(sum(numbers))` can see `numbers`. Upstream writes each as a
//! Python literal rather than as JSON, so a name lands as the value the model asked for and not as
//! a string it has to parse.

use anyhow::{Result, bail};
use serde_json::{Map, Value};

/// Names Python refuses, so an assignment to one is caught here rather than as a syntax error
/// inside the sandbox. `json` is upstream's own addition: the injected preamble may import it.
const RESERVED: [&str; 36] = [
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield", "json",
];

/// Whether Python would accept this as a name.
fn is_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|first| first.is_alphabetic() || first == '_')
        && characters.all(|rest| rest.is_alphanumeric() || rest == '_')
}

/// One value as the Python literal upstream writes.
///
/// JSON and Python agree on numbers and on strings; they part on the three constants, and that is
/// the whole of the difference. Containers are rendered recursively rather than through
/// `serde_json`'s printer, because a nested `true` inside an array is the same problem.
fn literal(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => Value::String(text.clone()).to_string(),
        Value::Array(items) => {
            format!(
                "[{}]",
                items.iter().map(literal).collect::<Vec<_>>().join(", ")
            )
        }
        Value::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    literal(&Value::String(key.clone())),
                    literal(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The code with each variable assigned above it, or the code unchanged when there are none.
pub(super) fn prepended(code: &str, variables: &Map<String, Value>) -> Result<String> {
    if variables.is_empty() {
        return Ok(code.to_owned());
    }
    for name in variables.keys() {
        if !is_identifier(name) || RESERVED.contains(&name.as_str()) {
            bail!("Invalid variable name: '{name}'");
        }
    }
    let assignments: Vec<String> = variables
        .iter()
        .map(|(name, value)| format!("{name} = {}", literal(value)))
        .collect();
    Ok(format!("{}\n{code}", assignments.join("\n")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn injected(pairs: Value) -> Result<String> {
        prepended("print(x)", pairs.as_object().expect("an object"))
    }

    /// JSON's three constants are not Python's, and a `true` reaching the sandbox is a `NameError`
    /// rather than a wrong answer — which makes it the one substitution that must not be missed.
    #[test]
    fn the_constants_are_pythons_and_not_jsons() {
        let written = injected(json!({ "a": true, "b": false, "c": null })).expect("injects");
        assert!(written.contains("a = True"), "{written}");
        assert!(written.contains("b = False"), "{written}");
        assert!(written.contains("c = None"), "{written}");
    }

    /// And inside a container too, which is where a printer that only fixed the top level would
    /// still hand Python a `true`.
    #[test]
    fn a_constant_nested_in_a_container_is_converted_as_well() {
        let written =
            injected(json!({ "flags": [true, null], "at": { "on": false } })).expect("injects");
        assert!(written.contains("flags = [True, None]"), "{written}");
        assert!(written.contains(r#"at = {"on": False}"#), "{written}");
    }

    /// A name Python would refuse is refused here, in upstream's wording, rather than arriving as a
    /// syntax error from inside the sandbox where the caller cannot see which name caused it.
    #[test]
    fn a_name_python_would_refuse_is_named_here() {
        for name in ["class", "not an identifier", "9lives", "json"] {
            let refused = prepended("pass", json!({ name: 1 }).as_object().expect("an object"))
                .expect_err("refused");
            assert_eq!(
                refused.to_string(),
                format!("Invalid variable name: '{name}'")
            );
        }
    }

    /// No variables means the code is untouched — not the code with a blank line above it, which
    /// would move every line number in a traceback the model reads.
    #[test]
    fn no_variables_leaves_the_code_alone() {
        assert_eq!(
            prepended("print(1)", &Map::new()).expect("injects"),
            "print(1)"
        );
    }

    /// Strings keep JSON's escaping, which Python reads the same way.
    #[test]
    fn a_string_keeps_its_escaping() {
        let written = injected(json!({ "s": "a\"b\nc" })).expect("injects");
        assert!(written.starts_with(r#"s = "a\"b\nc""#), "{written}");
    }
}
