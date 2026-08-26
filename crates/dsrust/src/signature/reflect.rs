//! A declared type's own structure, read off its JSON schema.
//!
//! [`BamlAdapter`](crate::BamlAdapter) states a type rather than a schema of it — `title: string`
//! over four lines instead of a `{"type": "object", "properties": …}` blob — and to do that it
//! needs the shape, not the schema. dspy gets it by walking the pydantic annotation at run time.
//! Rust has no run-time annotation to walk, so it is taken from the one description of the type
//! that already exists: the `schemars` schema every signature field is required to have.
//!
//! Without this a Rust-declared signature reached BAML as the bare word `json`, which is the one
//! thing that adapter exists not to send.
//!
//! The tree this builds is the same one `crates/dsrs-bridge/python/reflect.py` produces from a pydantic model,
//! because [`baml::notation`](crate::adapter::baml) reads exactly one shape whichever side built
//! it.

use serde_json::{Value, json};

use super::json_field_schema;

/// The structure of `T`, in the form an adapter that states types can render.
pub fn json_field_reflection<T: schemars::JsonSchema>() -> Value {
    from_schema(&json_field_schema::<T>())
}

/// The same, for a schema already in hand.
fn from_schema(schema: &Value) -> Value {
    let mut models = Vec::new();
    let declared = node(schema, &mut models);
    json!({ "type": declared, "models": models })
}

/// One node of the tree, pushing any object it meets onto `models` and referring to it by index.
fn node(schema: &Value, models: &mut Vec<Value>) -> Value {
    if let Some(members) = schema.get("enum").and_then(Value::as_array) {
        return json!({ "kind": "literal", "members": members });
    }
    if let Some(arms) = union_arms(schema) {
        return union(arms, models);
    }
    match declared_type(schema) {
        Some("string") => json!({ "kind": "str" }),
        Some("integer") => json!({ "kind": "int" }),
        Some("number") => json!({ "kind": "float" }),
        Some("boolean") => json!({ "kind": "bool" }),
        Some("array") => json!({ "kind": "list", "of": items(schema, models) }),
        Some("object") => object(schema, models),
        // A schema that names no type describes anything — `serde_json::Value`, and what
        // pydantic spells `Any`. Upstream prints the name rather than pretending to a shape.
        _ => json!({ "kind": "named", "name": "any" }),
    }
}

/// A schema's `type`, ignoring the null arm of an optional — that is read by [`union_arms`].
fn declared_type(schema: &Value) -> Option<&str> {
    match &schema["type"] {
        Value::String(name) => Some(name.as_str()),
        Value::Array(names) => names
            .iter()
            .filter_map(Value::as_str)
            .find(|name| *name != "null"),
        _ => None,
    }
}

/// The arms of a choice, however the schema spells one: `anyOf`, `oneOf`, or a `type` array.
///
/// `None` when the schema is not a choice at all, which is what keeps a plain `{"type": "string"}`
/// from being wrapped in a one-armed union nobody wrote.
fn union_arms(schema: &Value) -> Option<Vec<Value>> {
    for key in ["anyOf", "oneOf"] {
        if let Some(arms) = schema[key].as_array() {
            return Some(arms.clone());
        }
    }
    let names = schema["type"].as_array()?;
    let arms: Vec<Value> = names
        .iter()
        .filter_map(Value::as_str)
        .map(|name| json!({ "type": name }))
        .collect();
    (arms.len() > 1).then_some(arms)
}

/// A choice, with the null arm lifted out — upstream renders `Optional[T]` as `T or null`
/// rather than as a union with a null in it.
fn union(arms: Vec<Value>, models: &mut Vec<Value>) -> Value {
    let optional = arms.iter().any(is_null);
    let of: Vec<Value> = arms
        .iter()
        .filter(|arm| !is_null(arm))
        .map(|arm| node(arm, models))
        .collect();
    // `Option<T>` needs no case of its own: a two-armed choice with a null arm leaves one type in
    // `of` and `optional` true, which is what the general form already writes. It *had* one, and
    // the two arms were byte-identical — which is why both mutants of its guard survived, and how
    // a branch that can never be told from its neighbour announces itself.
    json!({ "kind": "union", "of": of, "optional": optional })
}

fn is_null(schema: &Value) -> bool {
    schema["type"] == json!("null")
}

fn items(schema: &Value, models: &mut Vec<Value>) -> Value {
    match schema.get("items") {
        Some(items) => node(items, models),
        // An array that does not say what it holds holds anything.
        None => json!({ "kind": "named", "name": "any" }),
    }
}

/// An object is either a record with declared members or a map keyed by string, and the two are
/// told apart the way serde writes them: a map has values under `additionalProperties` and no
/// members of its own.
fn object(schema: &Value, models: &mut Vec<Value>) -> Value {
    let properties = schema.get("properties").and_then(Value::as_object);
    if properties.is_none() {
        // No `is_object` guard: `node` answers "any" for a schema that declares no type, which a
        // bare `true`/`false` `additionalProperties` is — so the guard could not change an answer
        // and its mutant survived. Letting `node` decide removes the branch rather than testing it.
        let value = schema.get("additionalProperties").map_or_else(
            || json!({ "kind": "named", "name": "any" }),
            |values| node(values, models),
        );
        // serde keys every map it writes by a string, so there is no other key type to find.
        return json!({ "kind": "dict", "key": { "kind": "str" }, "value": value });
    }
    let at = model(schema, models);
    json!({ "kind": "model", "model": at })
}

/// Record a model and answer with its index.
///
/// The slot is taken before the members are walked, so a type reaching itself refers back to the
/// index it already has rather than recursing forever. The renderer stops such a cycle where it
/// finds it.
fn model(schema: &Value, models: &mut Vec<Value>) -> usize {
    let at = models.len();
    models.push(Value::Null);
    let fields: Vec<Value> = schema["properties"]
        .as_object()
        .map(|members| {
            members
                .iter()
                .map(|(name, member)| {
                    json!({
                        "name": name,
                        "desc": member.get("description"),
                        // A Rust field is keyed on the wire by the one name serde writes, so
                        // there is no second spelling for an alias to carry.
                        "alias": Value::Null,
                        "type": node(member, models),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    models[at] = json!({ "doc": schema.get("description"), "fields": fields });
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every JSON-schema scalar name maps to the kind BAML prints, and an unnamed one is `any`.
    /// The `number` and `boolean` arms were both deletable without a test noticing.
    #[test]
    fn every_scalar_type_name_maps_to_its_kind() {
        let mut models = Vec::new();
        for (declared, kind) in [
            ("string", "str"),
            ("integer", "int"),
            ("number", "float"),
            ("boolean", "bool"),
        ] {
            assert_eq!(
                node(&json!({ "type": declared }), &mut models),
                json!({ "kind": kind }),
                "for {declared}"
            );
        }
        assert_eq!(
            node(&json!({}), &mut models),
            json!({ "kind": "named", "name": "any" }),
            "a schema naming no type describes anything"
        );
    }

    /// A `type` written as an array. One name is not a choice — it is that type, read through
    /// `declared_type`'s Array arm — while two names are, and the null arm is lifted into
    /// `optional` rather than becoming an arm of its own. Three mutants lived in this gap: the
    /// Array arm, its null filter, and `union_arms`' arity test.
    #[test]
    fn a_type_array_is_a_type_when_it_names_one_and_a_choice_when_it_names_two() {
        let mut models = Vec::new();
        assert_eq!(
            node(&json!({ "type": ["string"] }), &mut models),
            json!({ "kind": "str" }),
            "one name is not a union"
        );
        assert_eq!(
            node(&json!({ "type": ["string", "null"] }), &mut models),
            json!({ "kind": "union", "of": [{ "kind": "str" }], "optional": true }),
            "the null arm becomes optionality"
        );
        assert_eq!(
            node(&json!({ "type": ["string", "integer"] }), &mut models),
            json!({
                "kind": "union",
                "of": [{ "kind": "str" }, { "kind": "int" }],
                "optional": false,
            })
        );
    }
    use schemars::JsonSchema;

    fn tree<T: JsonSchema>() -> Value {
        json_field_reflection::<T>()
    }

    #[derive(JsonSchema)]
    #[allow(dead_code)]
    /// A gift idea.
    struct Gift {
        title: String,
        why: Option<String>,
        rank: u8,
    }

    #[test]
    fn a_struct_becomes_a_model_whose_members_keep_their_order() {
        let tree = tree::<Gift>();
        assert_eq!(tree["type"], json!({ "kind": "model", "model": 0 }));

        let model = &tree["models"][0];
        assert_eq!(model["doc"], "A gift idea.", "the doc comment travels");
        let names: Vec<&str> = model["fields"]
            .as_array()
            .expect("members")
            .iter()
            .map(|field| field["name"].as_str().expect("a name"))
            .collect();
        assert_eq!(
            names,
            ["title", "why", "rank"],
            "declaration order, not alphabetical"
        );
    }

    #[test]
    fn the_scalars_map_onto_their_own_kinds() {
        let model = &tree::<Gift>()["models"][0];
        assert_eq!(model["fields"][0]["type"], json!({ "kind": "str" }));
        assert_eq!(model["fields"][2]["type"], json!({ "kind": "int" }));
    }

    /// `Option<T>` is a choice of two of which one is null, and upstream renders that as the one
    /// type plus `or null` rather than as a union naming null.
    #[test]
    fn an_option_is_one_arm_marked_optional() {
        let model = &tree::<Gift>()["models"][0];
        assert_eq!(
            model["fields"][1]["type"],
            json!({ "kind": "union", "of": [{ "kind": "str" }], "optional": true })
        );
    }

    #[test]
    fn a_vec_becomes_a_list_of_whatever_it_holds() {
        #[derive(JsonSchema)]
        #[allow(dead_code)]
        struct Wrap {
            gifts: Vec<Gift>,
        }
        let tree = tree::<Wrap>();
        let outer = &tree["models"][0]["fields"][0]["type"];
        assert_eq!(outer["kind"], "list");
        assert_eq!(
            outer["of"],
            json!({ "kind": "model", "model": 1 }),
            "the element is a model of its own"
        );
        assert_eq!(tree["models"][1]["doc"], "A gift idea.");
    }

    /// A map and a record are both `object` in a schema; what tells them apart is that a map
    /// declares no members of its own.
    #[test]
    fn a_map_becomes_a_dict_rather_than_a_model() {
        #[derive(JsonSchema)]
        #[allow(dead_code)]
        struct Wrap {
            tags: std::collections::HashMap<String, i32>,
        }
        let field = &tree::<Wrap>()["models"][0]["fields"][0]["type"];
        assert_eq!(
            *field,
            json!({ "kind": "dict", "key": { "kind": "str" }, "value": { "kind": "int" } })
        );
    }

    #[test]
    fn a_unit_enum_becomes_the_set_of_its_members() {
        #[derive(JsonSchema)]
        #[allow(dead_code)]
        enum Mood {
            Calm,
            Wry,
        }
        #[derive(JsonSchema)]
        #[allow(dead_code)]
        struct Wrap {
            mood: Mood,
        }
        let field = &tree::<Wrap>()["models"][0]["fields"][0]["type"];
        assert_eq!(field["kind"], "literal");
        assert_eq!(field["members"], json!(["Calm", "Wry"]));
    }

    /// A field whose type says nothing about itself is named rather than given a shape it does
    /// not have.
    #[test]
    fn an_untyped_value_is_named_rather_than_shaped() {
        #[derive(JsonSchema)]
        #[allow(dead_code)]
        struct Wrap {
            anything: serde_json::Value,
        }
        let field = &tree::<Wrap>()["models"][0]["fields"][0]["type"];
        assert_eq!(*field, json!({ "kind": "named", "name": "any" }));
    }

    #[test]
    fn a_members_description_travels_with_it() {
        #[derive(JsonSchema)]
        #[allow(dead_code)]
        struct Wrap {
            /// who it is for
            recipient: String,
        }
        let field = &tree::<Wrap>()["models"][0]["fields"][0];
        assert_eq!(field["desc"], "who it is for");
    }
}
