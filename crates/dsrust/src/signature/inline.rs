//! A schema with its references resolved, the way dspy hands one to a tool.
//!
//! dspy's `_resolve_json_schema_reference` walks a pydantic schema, replaces every `$ref` with the
//! `$defs` entry it points at, and drops the `$defs` block — so a tool's argument map carries whole
//! types rather than pointers into a document the model was never shown.
//!
//! schemars can inline while it generates, and that is not the same thing: it drops the name of
//! every type it inlines, where pydantic titles a model wherever it appears. A `Date` parameter is
//! `{..., "title": "Date", "type": "object"}` upstream and would be untitled here — in a map that
//! is rendered into the ReAct roster as "It takes arguments {args}".
//!
//! So the references are generated and then resolved here, which keeps the name the definition was
//! filed under.

use serde_json::{Map, Value};

/// Every `$ref` replaced by what it points at, and the `$defs` block dropped.
pub(crate) fn resolved(schema: Value) -> Value {
    let Some(definitions) = definitions(&schema) else {
        return schema;
    };
    let mut resolved = resolve(schema.clone(), &definitions, 0);
    if let Some(object) = resolved.as_object_mut() {
        object.remove("$defs");
        object.remove("definitions");
    }
    resolved
}

fn definitions(schema: &Value) -> Option<Map<String, Value>> {
    let object = schema.as_object()?;
    let block = object.get("$defs").or_else(|| object.get("definitions"))?;
    Some(block.as_object()?.clone())
}

/// A reference cycle is a type that contains itself, which cannot be inlined at all. Upstream
/// recurses without a guard and raises; stopping at the depth of the deepest possible acyclic
/// expansion leaves the `$ref` in place instead, which is a schema a reader can still follow.
const DEEPEST: usize = 64;

fn resolve(value: Value, definitions: &Map<String, Value>, depth: usize) -> Value {
    if depth > DEEPEST {
        return value;
    }
    match value {
        Value::Object(object) => match reference(&object) {
            Some(name) => match definitions.get(&name) {
                Some(target) => {
                    let mut expanded = resolve(target.clone(), definitions, depth + 1);
                    // pydantic titles a model with its own name wherever it appears, and the name
                    // is the key the definition is filed under.
                    if let Some(map) = expanded.as_object_mut()
                        && map.contains_key("properties")
                    {
                        map.entry("title").or_insert(Value::String(name));
                    }
                    expanded
                }
                None => Value::Object(object),
            },
            None => Value::Object(
                object
                    .into_iter()
                    .map(|(key, value)| (key, resolve(value, definitions, depth + 1)))
                    .collect(),
            ),
        },
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| resolve(item, definitions, depth + 1))
                .collect(),
        ),
        other => other,
    }
}

/// The name a `$ref` points at, which is its last path segment.
fn reference(object: &Map<String, Value>) -> Option<String> {
    let target = object.get("$ref")?.as_str()?;
    Some(target.rsplit('/').next()?.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The name survives the expansion, which is the whole reason this does not use schemars'
    /// own inlining.
    #[test]
    fn an_expanded_reference_keeps_the_name_it_was_filed_under() {
        let resolved = resolved(json!({
            "type": "object",
            "properties": { "from": { "$ref": "#/$defs/Origin" } },
            "$defs": { "Origin": { "type": "object", "properties": { "city": { "type": "string" } } } },
        }));
        assert_eq!(resolved["properties"]["from"]["title"], json!("Origin"));
        assert_eq!(
            resolved["properties"]["from"]["properties"]["city"]["type"],
            json!("string")
        );
        assert_eq!(resolved.get("$defs"), None);
    }

    /// A definition that is not a model is a scalar or a container, and pydantic titles neither.
    #[test]
    fn an_expanded_scalar_gains_no_title() {
        let resolved = resolved(json!({
            "type": "object",
            "properties": { "kind": { "$ref": "#/$defs/Kind" } },
            "$defs": { "Kind": { "type": "string" } },
        }));
        assert_eq!(resolved["properties"]["kind"], json!({ "type": "string" }));
    }

    /// A type containing itself cannot be expanded, and must not spin.
    #[test]
    fn a_cycle_stops_rather_than_recurring_forever() {
        let resolved = resolved(json!({
            "$ref": "#/$defs/Node",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": { "next": { "$ref": "#/$defs/Node" } },
                },
            },
        }));
        assert_eq!(resolved["title"], json!("Node"));
    }
}
