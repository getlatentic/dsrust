//! `parser_schema.py`: what a schema says about an object or an array, read once.
//!
//! Upstream keeps this apart from `schema_repair.py` because the *parser* needs it — `parse_object`
//! and `parse_array` resolve a schema before they read a member — while the repair pass needs the
//! repairer as well. The same split holds here.

use crate::value::Value;
use crate::{Error, Result, Schema};

use super::SchemaRepairer;

/// A property's schema and where it came from, resolved once per object.
pub(crate) struct ObjectSchemaConfig {
    pub(crate) properties: Vec<(String, Value)>,
    pub(crate) pattern_properties: Vec<(String, Value)>,
    pub(crate) additional_properties: Option<Value>,
    pub(crate) required: Vec<String>,
}

pub(crate) struct ArraySchemaConfig {
    pub(crate) items_schema: Option<Value>,
    pub(crate) additional_items: Option<Value>,
}

/// Whether `type` names `wanted`, as a string or as one of a list.
pub(crate) fn declares_type(schema: &Value, wanted: &str) -> bool {
    match schema.get("type") {
        Some(Value::Str(declared)) => declared == wanted,
        Some(Value::Array(declared)) => declared
            .iter()
            .any(|item| matches!(item, Value::Str(name) if name == wanted)),
        _ => false,
    }
}

pub(crate) fn object_schema_config(schema: &Schema) -> crate::error::Result<ObjectSchemaConfig> {
    Ok(ObjectSchemaConfig {
        properties: entries(schema.get("properties")),
        pattern_properties: entries(schema.get("patternProperties")),
        additional_properties: schema.get("additionalProperties").cloned(),
        required: required_keys(schema.get("required"))?,
    })
}

/// Upstream's `set(schema.get("required", []))`, with `set()`'s own refusals.
///
/// `set()` iterates anything iterable and hashes what it yields, so a string is taken per
/// character, a dict contributes its keys, and a scalar raises `TypeError` — as does a list
/// holding a list or a dict, which cannot be hashed. Both errors are foreign to `repair_json`, so
/// a union does not swallow them the way it swallows a failed branch; the surviving mutant on that
/// guard is how this function's silent `filter_map` was found.
///
/// A non-string member that *is* hashable — `required: [5]` — is kept by Python's set and then
/// salvage-fills a non-string key into the object, which this crate's string-keyed `Object` cannot
/// hold. That corner stays unimplemented and unclaimed: such members are dropped here, and no
/// corpus case records the shape.
fn required_keys(declared: Option<&Value>) -> crate::error::Result<Vec<String>> {
    let not_iterable = |value: &Value| {
        crate::error::Error::type_error(&format!(
            "'{}' object is not iterable",
            crate::schema::container::type_name(value)
        ))
    };
    match declared {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => {
            let mut keys = Vec::new();
            for item in items {
                match item {
                    Value::Array(_) | Value::Object(_) => {
                        return Err(crate::error::Error::type_error(&format!(
                            "unhashable type: '{}'",
                            crate::schema::container::type_name(item)
                        )));
                    }
                    Value::Str(name) if !keys.contains(name) => keys.push(name.clone()),
                    _ => {}
                }
            }
            Ok(keys)
        }
        Some(Value::Str(text)) => {
            let mut keys = Vec::new();
            for ch in text.chars() {
                let key = ch.to_string();
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
            Ok(keys)
        }
        Some(Value::Object(fields)) => Ok(fields.keys().map(str::to_owned).collect()),
        Some(other) => Err(not_iterable(other)),
    }
}

pub(crate) fn array_schema_config(schema: &Schema) -> ArraySchemaConfig {
    ArraySchemaConfig {
        items_schema: schema.get("items").cloned(),
        additional_items: schema.get("additionalItems").cloned(),
    }
}

/// A schema sub-object's entries, or nothing when it is not an object at all.
fn entries(node: Option<&Value>) -> Vec<(String, Value)> {
    match node {
        Some(Value::Object(fields)) => fields
            .iter()
            .map(|(key, value)| (key.to_owned(), value.clone()))
            .collect(),
        _ => Vec::new(),
    }
}

/// Whether the repairer guides this object, and the property table if so.
pub(crate) fn resolve_parser_object_schema(
    repairer: Option<&SchemaRepairer>,
    schema: Option<&Schema>,
) -> Result<(bool, Option<Schema>, Option<ObjectSchemaConfig>)> {
    let Some((repairer, resolved)) = guided_schema(repairer, schema)? else {
        return Ok((false, schema.cloned(), None));
    };
    if !repairer.is_object_schema(Some(&resolved)) {
        return Ok((false, Some(resolved), None));
    }
    let config = object_schema_config(&resolved)?;
    Ok((true, Some(resolved), Some(config)))
}

pub(crate) fn resolve_parser_array_schema(
    repairer: Option<&SchemaRepairer>,
    schema: Option<&Schema>,
) -> Result<(bool, Option<ArraySchemaConfig>)> {
    let Some((repairer, resolved)) = guided_schema(repairer, schema)? else {
        return Ok((false, None));
    };
    if !repairer.is_array_schema(Some(&resolved)) {
        return Ok((false, None));
    }
    Ok((true, Some(array_schema_config(&resolved))))
}

/// The shared opening of both resolvers: nothing to guide with, or a resolved schema.
fn guided_schema<'a>(
    repairer: Option<&'a SchemaRepairer>,
    schema: Option<&Schema>,
) -> Result<Option<(&'a SchemaRepairer, Schema)>> {
    let Some(repairer) = repairer else {
        return Ok(None);
    };
    // `schema not in (None, True)` upstream. A null schema is Python's `None` too, and
    // `resolve_schema` below is the one place that says so — naming it again here would be a
    // branch nothing can reach.
    if matches!(schema, None | Some(Value::Bool(true))) {
        return Ok(None);
    }
    let resolved = repairer.resolve_schema(schema)?;
    match resolved {
        Value::Bool(false) => Err(Error::new("Schema does not allow any values.")),
        Value::Bool(true) => Ok(None),
        resolved => Ok(Some((repairer, resolved))),
    }
}

/// The schema for item `idx`, and whether the schema forbids the item outright.
pub(crate) fn resolve_array_item_schema(
    config: Option<&ArraySchemaConfig>,
    idx: usize,
) -> Result<(Option<Schema>, bool)> {
    let Some(config) = config else {
        return Ok((None, false));
    };
    match &config.items_schema {
        Some(Value::Array(schemas)) => match schemas.get(idx) {
            Some(raw) => Ok((Some(as_schema(raw)?), false)),
            None => match &config.additional_items {
                Some(Value::Bool(false)) => Ok((None, true)),
                Some(node @ Value::Object(_)) => Ok((Some(node.clone()), false)),
                _ => Ok((Some(Value::Bool(true)), false)),
            },
        },
        Some(node @ Value::Object(_)) => Ok((Some(node.clone()), false)),
        _ => Ok((Some(Value::Bool(true)), false)),
    }
}

pub(crate) fn as_schema(node: &Value) -> Result<Schema> {
    match node {
        Value::Object(_) | Value::Bool(_) | Value::Null => Ok(node.clone()),
        _ => Err(Error::new("Schema must be an object.")),
    }
}
