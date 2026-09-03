//! Closing a JSON schema for the Responses API — dspy's `_close_object_schemas`.
//!
//! The Responses API refuses a `json_schema` format whose object schemas leave
//! `additionalProperties` unspecified, and neither pydantic's `model_json_schema()` nor a schema a
//! caller wrote by hand supplies it. So every object in the tree gets `false` on the way out,
//! except the ones that already said something — a `dict[str, T]` field declares a value schema
//! there and must keep it.
//!
//! The recursion is the whole of the rule. It follows **JSON Schema keyword positions only**, so a
//! schema-shaped value sitting in a `default`, an `examples` or a `const` is data rather than a
//! subschema and is never rewritten. Walking every nested object instead would quietly edit a
//! caller's default value — a field defaulting to `{"type": "object"}` would come back defaulting
//! to `{"type": "object", "additionalProperties": false}`, which is a different program.

use serde_json::Value;

/// Keywords holding one subschema.
const ONE: [&str; 8] = [
    "items",
    "additionalProperties",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
];

/// Keywords holding a list of subschemas.
const MANY: [&str; 5] = ["items", "prefixItems", "anyOf", "oneOf", "allOf"];

/// Keywords holding a map of name to subschema.
const NAMED: [&str; 4] = ["properties", "patternProperties", "$defs", "definitions"];

/// Every object schema in the tree given an explicit `additionalProperties`, leaving alone the ones
/// that already have one — and leaving alone anything that is not in a subschema position.
pub fn close_object_schemas(schema: &Value) -> Value {
    let Some(fields) = schema.as_object() else {
        return schema.clone();
    };
    let mut closed = fields.clone();
    // `items` is in both lists, as upstream's is: it holds one subschema in draft 2020-12 and a
    // list in the older tuple form, and whichever it is here is what gets walked.
    for key in ONE {
        if let Some(Value::Object(_)) = closed.get(key) {
            let sub = close_object_schemas(&closed[key]);
            closed.insert(key.to_owned(), sub);
        }
    }
    for key in MANY {
        if let Some(Value::Array(items)) = closed.get(key) {
            let walked = items.iter().map(close_object_schemas).collect();
            closed.insert(key.to_owned(), Value::Array(walked));
        }
    }
    for key in NAMED {
        if let Some(Value::Object(named)) = closed.get(key) {
            let walked = named
                .iter()
                .map(|(name, sub)| (name.clone(), close_object_schemas(sub)))
                .collect();
            closed.insert(key.to_owned(), Value::Object(walked));
        }
    }
    if closed.get("type") == Some(&Value::String("object".to_owned()))
        && !closed.contains_key("additionalProperties")
    {
        closed.insert("additionalProperties".to_owned(), Value::Bool(false));
    }
    Value::Object(closed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn every_object_in_the_tree_is_closed() {
        let closed = close_object_schemas(&json!({
            "type": "object",
            "properties": {
                "nested": { "type": "object", "properties": { "a": { "type": "string" } } },
                "listed": { "type": "array", "items": { "type": "object" } },
                "either": { "anyOf": [{ "type": "object" }, { "type": "null" }] },
            },
            "$defs": { "Ref": { "type": "object" } },
        }));
        assert_eq!(closed["additionalProperties"], json!(false));
        assert_eq!(
            closed["properties"]["nested"]["additionalProperties"],
            json!(false)
        );
        assert_eq!(
            closed["properties"]["nested"]["properties"]["a"].get("additionalProperties"),
            None,
            "a string is not an object"
        );
        assert_eq!(
            closed["properties"]["listed"]["items"]["additionalProperties"],
            json!(false)
        );
        // The branch a nullable field becomes: reached through `anyOf`, which the older
        // `enforce_required` walk does not follow at all.
        assert_eq!(
            closed["properties"]["either"]["anyOf"][0]["additionalProperties"],
            json!(false)
        );
        assert_eq!(closed["$defs"]["Ref"]["additionalProperties"], json!(false));
    }

    /// A `dict[str, T]` field declares a value schema under `additionalProperties`; overwriting it
    /// with `false` would forbid exactly the keys the field exists to carry.
    #[test]
    fn a_declared_additional_properties_is_kept_and_walked() {
        let closed = close_object_schemas(&json!({
            "type": "object",
            "additionalProperties": { "type": "object" },
        }));
        assert_eq!(
            closed["additionalProperties"],
            json!({ "type": "object", "additionalProperties": false }),
            "kept as a schema, and closed in its own right"
        );
    }

    /// The reason the walk is keyword-directed: a schema-shaped *value* is data. A field defaulting
    /// to `{"type": "object"}` must still default to exactly that.
    #[test]
    fn a_schema_shaped_default_is_a_value_and_is_left_alone() {
        let closed = close_object_schemas(&json!({
            "type": "object",
            "properties": {
                "payload": { "type": "object", "default": { "type": "object" } },
            },
        }));
        assert_eq!(
            closed["properties"]["payload"]["default"],
            json!({ "type": "object" }),
            "the default is the caller's value, not a subschema"
        );
        assert_eq!(
            closed["properties"]["payload"]["additionalProperties"],
            json!(false),
            "the field itself is still closed"
        );
    }

    #[test]
    fn something_that_is_not_a_schema_comes_back_unchanged() {
        for value in [json!(true), json!([1, 2]), json!("text"), json!(null)] {
            assert_eq!(close_object_schemas(&value), value);
        }
    }
}
