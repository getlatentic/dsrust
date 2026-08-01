//! `schema_repair` and `parser_schema`: a schema guiding the repair, not merely judging it.
//!
//! A schema turns a parse into a negotiation — a missing property is filled from its `default`, a
//! number that should be a string is coerced, a `oneOf` is tried branch by branch until one
//! validates. Which is the seam: *validating* is not this library's code. Upstream imports
//! `jsonschema`, a separate package, and raises when it is absent. So does this — the check is a
//! [`SchemaValidator`] a caller plugs in, and with none plugged in the repairer answers exactly as
//! a Python environment without `jsonschema` does.

pub(crate) mod coerce;
pub(crate) mod container;
pub(crate) mod object;
pub(crate) mod pattern;
pub(crate) mod validate;

use std::cell::RefCell;
use std::rc::Rc;

use crate::parser::{LogEntry, Parser};
use crate::value::Value;
use crate::{Error, LogSink, Result, Schema};

pub use validate::SchemaValidator;

/// How hard the repairer tries before giving up on a value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchemaRepairMode {
    /// Upstream's default: a value that cannot be repaired is an error.
    #[default]
    Standard,
    /// Best-effort: drop what will not fit, fill what is missing, map a list onto an object.
    Salvage,
}

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

pub struct SchemaRepairer {
    root_schema: Schema,
    log: LogSink,
    pub(crate) mode: SchemaRepairMode,
    validator: Option<Rc<dyn SchemaValidator>>,
    /// Schemas currently being resolved, so a `$ref` cycle is reported rather than followed.
    resolving: RefCell<Vec<String>>,
}

impl SchemaRepairer {
    pub fn new(root_schema: Schema, log: LogSink, mode: SchemaRepairMode) -> Self {
        Self {
            root_schema,
            log,
            mode,
            validator: None,
            resolving: RefCell::new(Vec::new()),
        }
    }

    /// Plugs in the check upstream imports `jsonschema` for.
    pub fn with_validator(mut self, validator: Rc<dyn SchemaValidator>) -> Self {
        self.validator = Some(validator);
        self
    }

    pub(crate) fn validator(&self) -> Option<&dyn SchemaValidator> {
        self.validator.as_deref()
    }

    pub(crate) fn log(&self, text: &str, path: &str) {
        if let Some(log) = &self.log {
            log.borrow_mut().push(LogEntry {
                text: text.to_owned(),
                context: path.to_owned(),
            });
        }
    }

    /// A schema with its `$ref` chain followed. `None` means "anything", which is `True`.
    pub(crate) fn resolve_schema(&self, schema: Option<&Schema>) -> Result<Schema> {
        let Some(schema) = schema else {
            return Ok(Value::Bool(true));
        };
        if let Value::Bool(flag) = schema {
            return Ok(Value::Bool(*flag));
        }
        let Value::Object(_) = schema else {
            return Err(Error::definition("Schema must be an object."));
        };

        let mut current = schema.clone();
        self.resolving.borrow_mut().clear();
        while let Some(reference) = current.get("$ref") {
            let Value::Str(reference) = reference else {
                return Err(Error::definition("$ref must be a string."));
            };
            let reference = reference.clone();
            if self.resolving.borrow().contains(&reference) {
                return Err(Error::definition(&format!(
                    "Circular $ref detected: {reference}"
                )));
            }
            self.resolving.borrow_mut().push(reference.clone());
            current = self.resolve_ref(&reference)?;
            if matches!(current, Value::Bool(_)) {
                return Ok(current);
            }
        }
        Ok(current)
    }

    fn resolve_ref(&self, reference: &str) -> Result<Schema> {
        let Some(pointer) = reference.strip_prefix("#/") else {
            return Err(Error::definition(&format!("Unsupported $ref: {reference}")));
        };
        let mut current = &self.root_schema;
        for part in pointer.split('/') {
            let resolved_part = part.replace("~1", "/").replace("~0", "~");
            match current.get(&resolved_part) {
                Some(next) => current = next,
                None => {
                    return Err(Error::definition(&format!(
                        "Unresolvable $ref: {reference}"
                    )));
                }
            }
        }
        match current {
            Value::Object(_) | Value::Bool(_) => Ok(current.clone()),
            _ => Err(Error::definition(&format!(
                "Unresolvable $ref: {reference}"
            ))),
        }
    }

    pub(crate) fn is_object_schema(&self, schema: Option<&Schema>) -> bool {
        let Ok(schema) = self.resolve_schema(schema) else {
            return false;
        };
        if !matches!(schema, Value::Object(_)) {
            return false;
        }
        if declares_type(&schema, "object") {
            return true;
        }
        [
            "properties",
            "patternProperties",
            "additionalProperties",
            "required",
        ]
        .iter()
        .any(|key| schema.get(key).is_some())
    }

    pub(crate) fn is_array_schema(&self, schema: Option<&Schema>) -> bool {
        let Ok(schema) = self.resolve_schema(schema) else {
            return false;
        };
        if !matches!(schema, Value::Object(_)) {
            return false;
        }
        declares_type(&schema, "array") || schema.get("items").is_some()
    }

    /// A list can stand in for an object only where the schema wants an object and not an array.
    pub(crate) fn can_salvage_list_as_object(&self, schema: &Schema) -> bool {
        self.allows_schema_type(schema, "object") && !self.allows_schema_type(schema, "array")
    }

    fn allows_schema_type(&self, schema: &Schema, schema_type: &str) -> bool {
        match schema.get("type") {
            Some(Value::Str(declared)) => declared == schema_type,
            Some(Value::Array(declared)) => declared
                .iter()
                .any(|item| matches!(item, Value::Str(name) if name == schema_type)),
            _ if schema_type == "object" => self.is_object_schema(Some(schema)),
            _ => self.is_array_schema(Some(schema)),
        }
    }
}

/// Whether `type` names `wanted`, as a string or as one of a list.
fn declares_type(schema: &Value, wanted: &str) -> bool {
    match schema.get("type") {
        Some(Value::Str(declared)) => declared == wanted,
        Some(Value::Array(declared)) => declared
            .iter()
            .any(|item| matches!(item, Value::Str(name) if name == wanted)),
        _ => false,
    }
}

pub(crate) fn object_schema_config(schema: &Schema) -> ObjectSchemaConfig {
    ObjectSchemaConfig {
        properties: entries(schema.get("properties")),
        pattern_properties: entries(schema.get("patternProperties")),
        additional_properties: schema.get("additionalProperties").cloned(),
        required: match schema.get("required") {
            Some(Value::Array(items)) => items
                .iter()
                .filter_map(|item| match item {
                    Value::Str(name) => Some(name.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        },
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
    let config = object_schema_config(&resolved);
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

fn as_schema(node: &Value) -> Result<Schema> {
    match node {
        Value::Object(_) | Value::Bool(_) | Value::Null => Ok(node.clone()),
        _ => Err(Error::new("Schema must be an object.")),
    }
}

/// The parser's side of the seam: the handful of calls `parse_object` and `parse_array` make.
impl Parser {
    pub(crate) fn repairer_salvages(&self) -> bool {
        self.schema_repairer
            .as_ref()
            .is_some_and(|repairer| repairer.mode == SchemaRepairMode::Salvage)
    }

    pub(crate) fn repairer_log(&self, text: &str, path: &str) {
        if let Some(repairer) = &self.schema_repairer {
            repairer.log(text, path);
        }
    }

    pub(crate) fn repair_value(
        &self,
        value: Value,
        schema: Option<Schema>,
        path: &str,
    ) -> Result<Value> {
        let repairer = self
            .schema_repairer
            .clone()
            .expect("only called on the guided path");
        repairer.repair_value(Some(value), schema.as_ref(), path)
    }

    pub(crate) fn repair_missing_value(&self, schema: Option<Schema>, path: &str) -> Result<Value> {
        let repairer = self
            .schema_repairer
            .clone()
            .expect("only called on the guided path");
        repairer.repair_value(None, schema.as_ref(), path)
    }

    pub(crate) fn salvage_schema_expects_an_object(&self, schema: Option<&Schema>) -> bool {
        let Some(repairer) = &self.schema_repairer else {
            return false;
        };
        repairer.mode == SchemaRepairMode::Salvage
            && matches!(schema, Some(Value::Object(_)))
            && repairer.is_object_schema(schema)
            && !repairer.is_array_schema(schema)
    }

    /// The schema for one key: its own, one matched by pattern, or whatever
    /// `additionalProperties` allows — and whether the key is forbidden outright.
    pub(crate) fn resolve_object_property_schema(
        &self,
        guided: bool,
        config: Option<&ObjectSchemaConfig>,
        key: &str,
    ) -> Result<(Option<Schema>, Vec<Option<Schema>>, bool)> {
        let (true, Some(config)) = (guided, config) else {
            return Ok((None, Vec::new(), false));
        };
        if let Some((_, declared)) = config.properties.iter().find(|(name, _)| name == key) {
            return Ok((Some(as_schema(declared)?), Vec::new(), false));
        }

        let (matched, unsupported) =
            pattern::match_pattern_properties(&config.pattern_properties, key);
        for unsupported_pattern in unsupported {
            self.repairer_log(
                &format!(
                    "Skipped unsupported patternProperties regex '{unsupported_pattern}' while parsing object key '{key}'"
                ),
                key,
            );
        }
        if let Some((primary, rest)) = matched.split_first() {
            let extra = rest
                .iter()
                .map(|schema| as_schema(schema).map(Some))
                .collect::<Result<Vec<_>>>()?;
            return Ok((Some(as_schema(primary)?), extra, false));
        }

        match &config.additional_properties {
            Some(Value::Bool(false)) => Ok((None, Vec::new(), true)),
            Some(node @ Value::Object(_)) => Ok((Some(node.clone()), Vec::new(), false)),
            _ => Ok((Some(Value::Bool(true)), Vec::new(), false)),
        }
    }

    /// `_finalize_object`: required properties must be there, and absent optionals take defaults.
    pub(crate) fn finalize_object(
        &self,
        mut obj: crate::value::Object,
        config: Option<ObjectSchemaConfig>,
        path: &str,
    ) -> Result<Value> {
        let (Some(repairer), Some(config)) = (self.schema_repairer.as_ref(), config) else {
            return Ok(Value::Object(obj));
        };
        let missing: Vec<&String> = config
            .required
            .iter()
            .filter(|key| !obj.contains_key(key))
            .collect();
        if !missing.is_empty() && repairer.mode != SchemaRepairMode::Salvage {
            let names: Vec<&str> = missing.iter().map(|key| key.as_str()).collect();
            return Err(Error::new(&format!(
                "Missing required properties at {path}: {}",
                names.join(", ")
            )));
        }
        for (key, prop_schema) in &config.properties {
            if obj.contains_key(key) || config.required.contains(key) {
                continue;
            }
            if let Some(default) = prop_schema.get("default") {
                let key_path = format!("{path}.{key}");
                obj.insert(
                    key.clone(),
                    repairer.copy_json_value(default, &key_path, "default")?,
                );
                repairer.log("Inserted default value for missing property", &key_path);
            }
        }
        Ok(Value::Object(obj))
    }
}
