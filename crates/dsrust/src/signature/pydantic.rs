//! A schemars schema, in pydantic's dialect and dspy's key order.
//!
//! dspy prints a structured field's schema straight into the prompt:
//!
//! ```python
//! schema = pydantic.TypeAdapter(field_type).json_schema()
//! schema = move_type_to_front(schema)   # every map: "type" first, then alphabetical
//! f"must adhere to the JSON schema: {json.dumps(schema, ensure_ascii=False)}"
//! ```
//!
//! So the schema is not an implementation detail — it is prompt text, and two generators that
//! both emit valid JSON Schema still write different prompts. Measured across thirteen type
//! shapes, schemars and pydantic agreed on seven and disagreed on six, in four ways:
//!
//! * schemars refines a number's `format` with its Rust width (`"int64"`, `"double"`), which
//!   pydantic never writes — though both write `format` for the semantic types, and dropping
//!   those too would lose the `"date-time"` dspy prints for a `datetime`;
//! * pydantic titles every property and every named model, and schemars titles none of them;
//! * a nullable is `{"type": ["number", "null"]}` in one and `{"anyOf": [...]}` in the other;
//! * schemars was asked to inline, where pydantic hoists a named type into `$defs` and points a
//!   `$ref` at it.
//!
//! This is the translation, and the ordering pass that follows it. Both are mechanical; what is
//! not obvious is that they matter at all, which is why the divergence went unnoticed until a real
//! DSPy tutorial was rendered on both sides and diffed.

use serde_json::{Map, Value, json};

/// One schemars schema as dspy would have printed it in a signature's field note.
pub(crate) fn as_dspy_prints_it(schema: Value) -> Value {
    ordered(translated(schema, Where::Root))
}

/// One schemars schema as pydantic itself would have printed it, for the places dspy passes a
/// schema through without its own ordering pass — a tool's argument map is the one that reaches a
/// prompt.
///
/// The difference from [`as_dspy_prints_it`] is only the order, and the order is prompt text.
pub(crate) fn as_pydantic_prints_it(schema: Value) -> Value {
    pydantic_order(translated(schema, Where::Root))
}

/// Where a schema sits, which decides whether it is titled.
#[derive(Clone, PartialEq)]
enum Where {
    /// The whole schema. Titled only when it is a model or an enumeration — pydantic gives a
    /// container no title, and schemars invents one (`Array_of_Entity`).
    Root,
    /// A named type under `$defs`, which pydantic titles with the name it is filed under.
    Definition,
    /// One entry of `properties`, titled after its field name — unless it is a bare `$ref`, which
    /// pydantic leaves alone.
    Property(String),
    /// Anywhere else: an `items`, an `anyOf` arm, an `additionalProperties`. Never titled.
    Nested,
}

fn translated(schema: Value, at: Where) -> Value {
    // A keyword whose value is a *list* of schemas — `prefixItems`, `anyOf`, `allOf` — is as much
    // a place a schema lives as a property is. Returning early here left every one of them in
    // schemars' dialect: a `(String, i64)` tuple kept the `int64` width inside its `prefixItems`.
    if let Value::Array(items) = schema {
        return Value::Array(
            items
                .into_iter()
                .map(|item| translated(item, Where::Nested))
                .collect(),
        );
    }
    let Value::Object(mut object) = schema else {
        return schema;
    };
    // schemars states the dialect it wrote; pydantic does not, and dspy prints what pydantic wrote.
    object.remove("$schema");
    reformat(&mut object);

    let nullable = split_nullable(&mut object);
    let mut translated_object = Map::new();
    for (key, value) in object {
        let translated_value = match key.as_str() {
            "properties" => properties(value),
            "$defs" | "definitions" => definitions(value),
            _ => translated(value, Where::Nested),
        };
        translated_object.insert(key, translated_value);
    }
    default_none(&mut translated_object);
    if let Some(anyof) = nullable {
        // `{"type": ["T", "null"]}` is the same schema as `{"anyOf": [{"type": "T"}, …]}` and a
        // different string. pydantic writes the second.
        translated_object.insert("anyOf".to_owned(), anyof);
    }
    titled(translated_object, at)
}

/// pydantic's `"default": null`, on the fields a Rust program spelled `Option<T>`.
///
/// The two languages agree on the meaning and disagree on what they write down. `Optional[str] =
/// None` is not required and defaults to `None`; a bare `Optional[str]` is required and has no
/// default. serde's `Option<T>` is the first — a missing field deserializes to `None` — and
/// schemars writes only half of it, leaving the field out of `required` and the default unsaid.
///
/// So a Rust optional rendered a schema no Python program produces, in the one place where the
/// schema is read by a model rather than a validator.
fn default_none(object: &mut Map<String, Value>) {
    let required: Vec<&str> = object
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let optional: Vec<String> = object
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .iter()
                .filter(|(name, schema)| !required.contains(&name.as_str()) && accepts_null(schema))
                .map(|(name, _)| name.clone())
                .collect()
        })
        .unwrap_or_default();
    let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) else {
        return;
    };
    for name in optional {
        let Some(schema) = properties.get_mut(&name).and_then(Value::as_object_mut) else {
            continue;
        };
        schema.entry("default").or_insert(Value::Null);
    }
}

/// Whether a translated property has a null arm, which is what `Option<T>` became.
fn accepts_null(schema: &Value) -> bool {
    schema
        .get("anyOf")
        .and_then(Value::as_array)
        .is_some_and(|arms| {
            arms.iter()
                .any(|arm| arm.get("type") == Some(&json!("null")))
        })
}

/// The widths schemars gives a number, which pydantic states only as `integer` or `number`. Every
/// other `format` is a semantic type both libraries name, and dspy prints it.
const RUST_WIDTHS: [&str; 12] = [
    "int8", "int16", "int32", "int64", "int128", "uint8", "uint16", "uint32", "uint64", "uint128",
    "float", "double",
];

/// `format` in pydantic's vocabulary: the widths dropped, and the one name the two libraries spell
/// differently made pydantic's.
///
/// A blanket drop here read as correct against integers and floats, and silently cost every
/// `datetime`, `date`, `uuid` and `ipv4` the keyword dspy prints for it.
fn reformat(object: &mut Map<String, Value>) {
    let Some(format) = object.get("format").and_then(Value::as_str) else {
        return;
    };
    match format {
        width if RUST_WIDTHS.contains(&width) => {
            object.remove("format");
        }
        // chrono's name for a clock time; pydantic's `datetime.time` is `"time"`.
        "partial-time" => {
            object.insert("format".to_owned(), json!("time"));
        }
        _ => {}
    }
}

/// pydantic's title for a schema in this position, if it has one.
fn titled(mut object: Map<String, Value>, at: Where) -> Value {
    match at {
        // A model or an enumeration keeps the name schemars gave it; a container loses the one
        // schemars invented, because pydantic never names a container.
        Where::Root => {
            if !object.contains_key("properties") && !object.contains_key("enum") {
                object.remove("title");
            }
        }
        Where::Definition => {}
        Where::Property(name) => {
            // A property that is only a `$ref` carries nothing beside it upstream.
            if !object.contains_key("$ref") {
                object.insert("title".to_owned(), json!(title_case(&name)));
            }
        }
        Where::Nested => {
            object.remove("title");
        }
    }
    Value::Object(object)
}

fn properties(value: Value) -> Value {
    let Value::Object(object) = value else {
        return value;
    };
    Value::Object(
        object
            .into_iter()
            .map(|(name, schema)| {
                let translated = translated(schema, Where::Property(name.clone()));
                (name, translated)
            })
            .collect(),
    )
}

fn definitions(value: Value) -> Value {
    let Value::Object(object) = value else {
        return value;
    };
    Value::Object(
        object
            .into_iter()
            .map(|(name, schema)| {
                let mut translated = translated(schema, Where::Definition);
                // pydantic titles a definition with the name it is filed under.
                if let Some(map) = translated.as_object_mut() {
                    map.insert("title".to_owned(), json!(name.clone()));
                }
                (name, translated)
            })
            .collect(),
    )
}

/// `{"type": ["T", "null"]}` as pydantic's `anyOf`, with everything that qualified the type moving
/// into the arm it qualified.
///
/// Moving only `type` leaves the rest describing the union: `Option<Vec<String>>` became
/// `{"anyOf": [{"type": "array"}, {"type": "null"}], "items": {"type": "string"}}`, which says an
/// array of anything or a null, beside a stray `items`. pydantic writes
/// `{"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}`, and the
/// difference reaches the model as a weaker contract, not only as different bytes.
///
/// [`WRAPPER_KEYS`] is what stays behind, because it describes the *field* rather than its type.
fn split_nullable(object: &mut Map<String, Value>) -> Option<Value> {
    let types = object.get("type")?.as_array()?.clone();
    if types.len() != 2 || !types.iter().any(|kind| kind.as_str() == Some("null")) {
        return None;
    }
    let carried = types
        .iter()
        .find(|kind| kind.as_str() != Some("null"))?
        .clone();
    object.remove("type");
    let mut arm = Map::new();
    arm.insert("type".to_owned(), carried);
    for key in qualifiers(object) {
        if let Some(value) = object.remove(&key) {
            arm.insert(key, value);
        }
    }
    Some(json!([Value::Object(arm), json!({ "type": "null" })]))
}

/// What pydantic writes beside an `anyOf` rather than inside one of its arms: these describe the
/// field, and every other keyword describes the type.
const WRAPPER_KEYS: [&str; 4] = ["title", "default", "description", "$defs"];

/// The keys that qualified the type that is moving into the arm.
fn qualifiers(object: &Map<String, Value>) -> Vec<String> {
    object
        .keys()
        .filter(|key| !WRAPPER_KEYS.contains(&key.as_str()))
        .cloned()
        .collect()
}

/// Python's `str.title()` over a field name with underscores as spaces: `entity_type` reads
/// `Entity Type`, and — because `title()` lowercases the rest of every word — `HTTPCode` reads
/// `Httpcode`. Matching that oddity is the difference between two prompts.
fn title_case(name: &str) -> String {
    name.split('_')
        .map(|word| {
            let mut characters = word.chars();
            match characters.next() {
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &characters.as_str().to_lowercase()
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// dspy's `move_type_to_front`: every map's keys with `type` first, then in order.
///
/// Not cosmetic. The schema is serialized into the prompt, so the key order *is* prompt text, and
/// upstream states this is "for LLM readability/adherence" — the model is meant to read the type
/// first.
fn ordered(schema: Value) -> Value {
    match schema {
        Value::Object(object) => {
            let mut keys: Vec<String> = object.keys().cloned().collect();
            keys.sort_by(|a, b| (a != "type", a).cmp(&(b != "type", b)));
            let mut sorted = Map::new();
            for key in keys {
                let value = object.get(&key).cloned().unwrap_or(Value::Null);
                sorted.insert(key, ordered(value));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(ordered).collect()),
        other => other,
    }
}

/// pydantic's own order: keys alphabetical, and `properties` left in the order the model declared
/// its fields.
///
/// dspy applies `move_type_to_front` on top of this for a field note and nothing at all for a
/// tool's arguments, so this is what a tool roster prints.
fn pydantic_order(schema: Value) -> Value {
    match schema {
        Value::Object(object) => {
            let mut keys: Vec<String> = object.keys().cloned().collect();
            keys.sort();
            let mut sorted = Map::new();
            for key in keys {
                let value = object.get(&key).cloned().unwrap_or(Value::Null);
                // A field name is not a keyword to sort; pydantic keeps declaration order here,
                // and schemars is asked to as well.
                let ordered_value = match key.as_str() {
                    "properties" => declared_order(value),
                    _ => pydantic_order(value),
                };
                sorted.insert(key, ordered_value);
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(pydantic_order).collect()),
        other => other,
    }
}

/// The properties map: its own keys untouched, each schema under them still ordered.
fn declared_order(properties: Value) -> Value {
    match properties {
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .map(|(name, schema)| (name, pydantic_order(schema)))
                .collect(),
        ),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The oddity worth pinning: Python's `title()` lowercases everything after the first letter.
    #[test]
    fn the_title_is_pythons() {
        assert_eq!(title_case("entity_type"), "Entity Type");
        assert_eq!(title_case("alpha"), "Alpha");
        assert_eq!(title_case("HTTPCode"), "Httpcode");
        assert_eq!(title_case("x"), "X");
    }

    /// `type` leads, and the rest follow in order — which is what a caller reads in the prompt.
    #[test]
    fn type_leads_and_the_rest_are_ordered() {
        let ordered = ordered(json!({ "title": "T", "properties": {}, "type": "object" }));
        let keys: Vec<&str> = ordered
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["type", "properties", "title"]);
    }

    /// A width is schemars' own and goes; a semantic format is pydantic's too and stays. Dropping
    /// both is the bug this splits, and it renders a `datetime` as a bare string.
    #[test]
    fn a_width_goes_and_a_semantic_format_stays() {
        for width in ["int64", "uint8", "double", "float"] {
            let translated = as_dspy_prints_it(json!({ "type": "integer", "format": width }));
            assert_eq!(translated, json!({ "type": "integer" }), "for {width}");
        }
        for kept in ["date-time", "date", "uuid", "ipv4", "binary", "duration"] {
            let translated = as_dspy_prints_it(json!({ "type": "string", "format": kept }));
            assert_eq!(
                translated,
                json!({ "type": "string", "format": kept }),
                "for {kept}"
            );
        }
    }

    /// An optional carries the default pydantic writes for it, and a required field does not.
    #[test]
    fn a_rust_optional_defaults_to_null() {
        let translated = as_dspy_prints_it(json!({
            "type": "object",
            "properties": {
                "here": { "type": "string" },
                "maybe": { "type": ["string", "null"] },
            },
            "required": ["here"],
        }));
        let properties = &translated["properties"];
        assert_eq!(properties["maybe"]["default"], json!(null));
        assert!(
            properties["maybe"].get("default").is_some(),
            "an optional says so"
        );
        assert!(
            properties["here"].get("default").is_none(),
            "a required field has no default"
        );
    }

    /// A nullable that upstream would have made required keeps no default it was not given.
    #[test]
    fn a_required_nullable_has_no_default() {
        let translated = as_dspy_prints_it(json!({
            "type": "object",
            "properties": { "maybe": { "type": ["string", "null"] } },
            "required": ["maybe"],
        }));
        assert!(translated["properties"]["maybe"].get("default").is_none());
    }

    /// A schema inside a list of them is translated too, which is where a tuple's members live.
    #[test]
    fn a_schema_in_a_list_of_them_is_translated() {
        let translated = as_dspy_prints_it(json!({
            "type": "array",
            "prefixItems": [
                { "type": "string" },
                { "type": "integer", "format": "int64" },
            ],
        }));
        assert_eq!(
            translated["prefixItems"][1],
            json!({ "type": "integer" }),
            "a width inside prefixItems is still schemars'"
        );
    }

    /// What qualified the type moves into the arm it qualified; what described the field stays.
    ///
    /// Moving only `type` left `{"anyOf": [{"type": "array"}, …], "items": …}` — an array of
    /// anything or a null, beside a stray `items`. That is a weaker contract than upstream states,
    /// not merely different bytes.
    #[test]
    fn a_nullables_qualifiers_move_into_its_arm() {
        // At a property, where pydantic writes the title beside the `anyOf` and the default under
        // it — the position that distinguishes a field's keywords from its type's.
        let translated = as_dspy_prints_it(json!({
            "type": "object",
            "properties": {
                "tags": { "type": ["array", "null"], "items": { "type": "string" } },
            },
        }));
        assert_eq!(
            translated["properties"]["tags"],
            json!({
                "anyOf": [
                    { "type": "array", "items": { "type": "string" } },
                    { "type": "null" },
                ],
                "default": null,
                "title": "Tags",
            })
        );
    }

    /// The same for a semantic format, which qualifies the string it sits on.
    #[test]
    fn a_nullable_string_carries_its_format_inside() {
        let translated =
            as_dspy_prints_it(json!({ "type": ["string", "null"], "format": "date-time" }));
        assert_eq!(
            translated,
            json!({
                "anyOf": [{ "type": "string", "format": "date-time" }, { "type": "null" }]
            })
        );
    }

    /// chrono and pydantic name a clock time differently, and dspy prints pydantic's name.
    #[test]
    fn a_clock_time_takes_pydantics_name() {
        let translated = as_dspy_prints_it(json!({ "type": "string", "format": "partial-time" }));
        assert_eq!(translated, json!({ "type": "string", "format": "time" }));
    }

    /// A nullable becomes pydantic's `anyOf`, and loses the `type` it was spelled with.
    #[test]
    fn a_nullable_becomes_an_any_of() {
        let translated =
            as_dspy_prints_it(json!({ "type": ["number", "null"], "format": "double" }));
        assert_eq!(
            translated,
            json!({ "anyOf": [{ "type": "number" }, { "type": "null" }] })
        );
    }
}
