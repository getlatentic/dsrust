//! What the sandbox is told before it runs anything: the tools it may call, and how `SUBMIT` looks.
//!
//! dspy sends one `register` request carrying both, and `runner.js` generates a Python `def` per
//! entry from it. That is why a parameter list travels rather than a JSON schema: the sandbox is
//! writing a function signature, so it needs names, and types only where a name can carry one.

use serde_json::{Map, Value, json};

use crate::interpreter::OutputField;
use crate::react::Tool;

/// The Python names dspy's `SIMPLE_TYPES` covers, keyed by the JSON-schema type that means each.
///
/// Anything outside this list is registered without a type, exactly as upstream drops an annotation
/// it cannot spell in a signature — a `Union` or an `Optional` reaches the sandbox as a bare name.
fn python_type(schema: &Value) -> Option<&'static str> {
    Some(match schema.get("type").and_then(Value::as_str)? {
        "string" => "str",
        "integer" => "int",
        "number" => "float",
        "boolean" => "bool",
        "array" => "list",
        "object" => "dict",
        "null" => "NoneType",
        _ => return None,
    })
}

/// One tool's parameters, read off what [`Tool::args`] answers with.
///
/// That is dspy's `Tool.args` — a map of argument name to *that argument's* schema, not an object
/// schema wrapping them. There is no `required` list in it, so an argument is optional exactly when
/// its schema states a `default`, which is how a generated `def` ends up with the same optional
/// arguments the tool has.
fn parameters(args: &Value) -> Vec<Value> {
    let Some(arguments) = args.as_object() else {
        return Vec::new();
    };
    arguments
        .iter()
        .map(|(name, schema)| {
            let mut described = Map::new();
            described.insert("name".to_owned(), json!(name));
            if let Some(spelled) = python_type(schema) {
                described.insert("type".to_owned(), json!(spelled));
            }
            if let Some(default) = schema.get("default") {
                described.insert("default".to_owned(), default.clone());
            }
            Value::Object(described)
        })
        .collect()
}

/// One output field as dspy sends it: a type only where the generated `def` can carry one.
fn described(field: &OutputField) -> Value {
    match &field.python_type {
        Some(spelled) => json!({ "name": field.name, "type": spelled }),
        None => json!({ "name": field.name }),
    }
}

/// The `register` request's params, or `None` when there is nothing to say.
///
/// Upstream skips the round trip entirely in that case rather than sending an empty registration,
/// and a sandbox told nothing keeps the default single-argument `SUBMIT`.
pub(super) fn params(tools: &[std::sync::Arc<dyn Tool>], outputs: &[OutputField]) -> Option<Value> {
    let mut asked = Map::new();
    if !tools.is_empty() {
        asked.insert(
            "tools".to_owned(),
            Value::Array(
                tools
                    .iter()
                    .map(|tool| json!({ "name": tool.name(), "parameters": parameters(tool.args()) }))
                    .collect(),
            ),
        );
    }
    if !outputs.is_empty() {
        asked.insert(
            "outputs".to_owned(),
            Value::Array(outputs.iter().map(described).collect()),
        );
    }
    (!asked.is_empty()).then(|| Value::Object(asked))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::react::FnTool;
    use std::sync::Arc;

    fn tool(name: &'static str, args: Value) -> Arc<dyn Tool> {
        Arc::new(FnTool::new(name, "a tool", args, |_| Ok(String::new())))
    }

    /// `Tool::args` is dspy's `Tool.args`: the argument map itself. Reading it as an object schema
    /// and looking for `properties` finds nothing, which registers every tool as taking no
    /// arguments — a `def` the model then calls wrongly.
    #[test]
    fn an_argument_carries_its_python_type() {
        let described = parameters(&json!({ "city": { "type": "string" } }));
        assert_eq!(described, vec![json!({ "name": "city", "type": "str" })]);
    }

    /// An argument stating a default is the optional one, since this shape carries no `required`
    /// list to say otherwise.
    #[test]
    fn an_argument_with_a_default_keeps_it() {
        let described = parameters(&json!({ "units": { "type": "string", "default": "metric" } }));
        assert_eq!(
            described,
            vec![json!({ "name": "units", "type": "str", "default": "metric" })]
        );
    }

    /// A type with no Python spelling is registered without one, as upstream drops an annotation
    /// it cannot write into a signature rather than guessing at a name.
    #[test]
    fn a_type_python_cannot_spell_is_registered_without_one() {
        let described = parameters(&json!({ "anything": { "anyOf": [{ "type": "string" }] } }));
        assert_eq!(described, vec![json!({ "name": "anything" })]);
    }

    /// An object schema handed here by mistake reads as one argument named `properties`, which is
    /// nonsense rather than an empty list — so the shape is worth asserting on directly.
    #[test]
    fn the_argument_map_is_not_an_object_schema() {
        let described = parameters(&json!({ "type": "object", "properties": { "a": {} } }));
        assert!(
            described.iter().any(|p| p["name"] == json!("properties")),
            "read as arguments, which is what an object schema would wrongly become: {described:?}"
        );
    }

    /// Nothing to register means no round trip at all, which leaves the sandbox on its default
    /// single-argument `SUBMIT`.
    #[test]
    fn nothing_to_say_sends_nothing() {
        assert!(params(&[], &[]).is_none());
    }

    /// Tools and output fields travel together in the one request upstream sends.
    #[test]
    fn tools_and_outputs_travel_in_one_request() {
        let asked = params(
            &[tool("weather", json!({ "city": { "type": "string" } }))],
            &[
                OutputField::named("answer"),
                OutputField {
                    name: "score".to_owned(),
                    python_type: Some("int".to_owned()),
                },
            ],
        )
        .expect("something to say");
        assert_eq!(asked["tools"][0]["name"], json!("weather"));
        assert_eq!(
            asked["outputs"],
            json!([{ "name": "answer" }, { "name": "score", "type": "int" }])
        );
    }
}
