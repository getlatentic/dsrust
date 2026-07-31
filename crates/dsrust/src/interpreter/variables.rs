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

/// Pyodide's FFI dies at exactly 128MB, so upstream sends anything over 100MB through the
/// sandbox's filesystem instead of through the code. The number is dspy's `LARGE_VAR_THRESHOLD`.
const LARGE_VALUE: usize = 100 * 1024 * 1024;

/// One execution's code, and whatever has to reach the sandbox before it runs.
#[derive(Debug)]
pub(super) struct Prepared {
    pub(super) code: String,
    /// dspy's `_pending_large_vars`: `(name, JSON)` pairs, each written to
    /// `/tmp/dspy_vars/<name>.json` by an `inject_var` request the code then reads back.
    pub(super) large: Vec<(String, String)>,
}

/// The code with each variable assigned above it, or the code unchanged when there are none.
pub(super) fn prepared(code: &str, variables: &Map<String, Value>) -> Result<Prepared> {
    if variables.is_empty() {
        return Ok(Prepared {
            code: code.to_owned(),
            large: Vec::new(),
        });
    }
    for name in variables.keys() {
        if !is_identifier(name) || RESERVED.contains(&name.as_str()) {
            bail!("Invalid variable name: '{name}'");
        }
    }

    let mut small = Vec::new();
    let mut reads = Vec::new();
    let mut large = Vec::new();
    for (name, value) in variables {
        // The threshold is measured against the *literal*, which is what would have gone into the
        // code, while the payload crosses as JSON. Upstream uses the two encodings the same way.
        let written = literal(value);
        if written.len() > LARGE_VALUE {
            reads.push(format!(
                "{name} = json.loads(open('/tmp/dspy_vars/{name}.json').read())"
            ));
            large.push((name.clone(), value.to_string()));
        } else {
            small.push(format!("{name} = {written}"));
        }
    }

    // `import json` only when something needs it, which is upstream's condition rather than a
    // tidier unconditional one — an extra import would change the line numbers in a traceback.
    let mut assignments = Vec::new();
    if !large.is_empty() {
        assignments.push("import json".to_owned());
    }
    assignments.extend(small);
    assignments.extend(reads);
    Ok(Prepared {
        code: format!("{}\n{code}", assignments.join("\n")),
        large,
    })
}

#[cfg(test)]
mod tests {
    /// The committed table, generated from CPython and dspy by
    /// `scripts/generate_constants_fixture.py`.
    fn tables() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/constants/tables.json");
        let text = std::fs::read_to_string(&path).expect("the constants golden is committed");
        serde_json::from_str(&text).expect("the golden parses")
    }

    /// dspy holds no list: it asks `keyword.iskeyword(name) or name == "json"`. The crate holds
    /// one, so the two agree exactly as long as the list is CPython's — compared as a set, since
    /// membership is all that is observable and the order here is the crate's own.
    ///
    /// A keyword a later CPython adds would reach upstream's predicate and not this list. The
    /// golden records the interpreter it was generated under so that surfaces as a diff here
    /// rather than as a syntax error inside the sandbox much later.
    #[test]
    fn the_refused_names_are_pythons_keywords_and_dspys_json() {
        let tables = tables();
        let recorded: std::collections::BTreeSet<&str> = tables["refused_variable_names"]
            .as_array()
            .expect("refused names")
            .iter()
            .map(|name| name.as_str().expect("a name"))
            .collect();
        let ours: std::collections::BTreeSet<&str> = RESERVED.iter().copied().collect();
        assert_eq!(ours, recorded);
    }

    use super::*;
    use serde_json::json;

    fn injected(pairs: Value) -> Result<String> {
        Ok(prepared("print(x)", pairs.as_object().expect("an object"))?.code)
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
            let refused = prepared("pass", json!({ name: 1 }).as_object().expect("an object"))
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
        let untouched = prepared("print(1)", &Map::new()).expect("injects");
        assert_eq!(untouched.code, "print(1)");
        assert!(untouched.large.is_empty());
    }

    /// Strings keep JSON's escaping, which Python reads the same way.
    #[test]
    fn a_string_keeps_its_escaping() {
        let written = injected(json!({ "s": "a\"b\nc" })).expect("injects");
        assert!(written.starts_with(r#"s = "a\"b\nc""#), "{written}");
    }

    /// A value too big for Pyodide's FFI goes through the sandbox's filesystem instead of through
    /// the code, and the code reads it back. Sending it inline crashes the sandbox at 128MB.
    #[test]
    fn a_value_over_the_threshold_travels_as_a_file() {
        let big = Value::String("x".repeat(LARGE_VALUE + 1));
        let mut given = Map::new();
        given.insert("small".to_owned(), json!(1));
        given.insert("huge".to_owned(), big.clone());

        let out = prepared("print(huge)", &given).expect("prepares");
        assert_eq!(out.large.len(), 1, "only the big one");
        assert_eq!(out.large[0].0, "huge");
        assert_eq!(
            out.large[0].1,
            big.to_string(),
            "the payload crosses as JSON"
        );
        assert!(out.code.starts_with("import json\n"), "{}", &out.code[..40]);
        assert!(
            out.code.contains("small = 1"),
            "the small one is still inline"
        );
        assert!(
            out.code
                .contains("huge = json.loads(open('/tmp/dspy_vars/huge.json').read())"),
            "the big one is read back"
        );
    }

    /// And a small one brings no import with it, since an extra line moves every line number in a
    /// traceback the model is shown.
    #[test]
    fn a_small_value_brings_no_import() {
        let out = prepared(
            "print(x)",
            json!({ "x": 1 }).as_object().expect("an object"),
        )
        .expect("prepares");
        assert_eq!(out.code, "x = 1\nprint(x)");
        assert!(out.large.is_empty());
    }
}
