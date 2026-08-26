//! `schema_repair` and `parser_schema`: a schema guiding the repair, not merely judging it.
//!
//! A schema turns a parse into a negotiation — a missing property is filled from its `default`, a
//! number that should be a string is coerced, a `oneOf` is tried branch by branch until one
//! validates. Which is the seam: *validating* is not this library's code. Upstream imports
//! `jsonschema`, a separate package, and raises when it is absent. So does this — the check is a
//! [`SchemaValidator`] a caller plugs in, and with none plugged in the repairer answers exactly as
//! a Python environment without `jsonschema` does.

pub(crate) mod coerce;
pub(crate) mod config;
pub(crate) mod container;
pub(crate) mod object;
pub(crate) mod pattern;
pub(crate) mod validate;

use std::rc::Rc;

use crate::parser::{LogEntry, Parser};
use crate::value::Value;
use crate::{Error, LogSink, Result, Schema};

pub use validate::{SchemaValidator, ValidationError};

pub(crate) use config::{
    ArraySchemaConfig, ObjectSchemaConfig, array_schema_config, as_schema, declares_type,
    object_schema_config, resolve_array_item_schema, resolve_parser_array_schema,
    resolve_parser_object_schema,
};

/// How hard the repairer tries before giving up on a value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchemaRepairMode {
    /// Upstream's default: a value that cannot be repaired is an error.
    #[default]
    Standard,
    /// Best-effort: drop what will not fit, fill what is missing, map a list onto an object.
    Salvage,
}

pub struct SchemaRepairer {
    root_schema: Schema,
    log: LogSink,
    pub(crate) mode: SchemaRepairMode,
    validator: Option<Rc<dyn SchemaValidator>>,
}

impl SchemaRepairer {
    pub fn new(root_schema: Schema, log: LogSink, mode: SchemaRepairMode) -> Self {
        Self {
            root_schema,
            log,
            mode,
            validator: None,
        }
    }

    /// Plugs in the check upstream imports `jsonschema` for.
    ///
    /// Keeps the `with_` prefix, where a builder setter in this workspace normally drops it: the
    /// unprefixed name is the reader below. Same reason `LM::with_capabilities` keeps its.
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
        // Python has one `None` for both "no schema" and a schema that *is* null, and
        // `resolve_schema` answers `True` — anything allowed — for it. Rust tells the two apart,
        // so a `properties: {"a": null}` would otherwise refuse where upstream parses.
        let Some(schema) = schema.filter(|schema| !matches!(schema, Value::Null)) else {
            return Ok(Value::Bool(true));
        };
        if let Value::Bool(flag) = schema {
            return Ok(Value::Bool(*flag));
        }
        let Value::Object(_) = schema else {
            return Err(Error::definition("Schema must be an object."));
        };

        // Upstream remembers `id(schema_dict)` — the node it is *leaving* — not the reference it
        // is about to follow, so it notices a cycle one hop later and names the other pointer.
        // `#/x -> #/y -> #/x` is "Circular $ref detected: #/$defs/y", not `#/$defs/x`. A pointer
        // stands in for identity here: within one document the same pointer is the same node.
        let mut current = schema.clone();
        let mut produced_by: Option<String> = None;
        let mut visited: Vec<Option<String>> = Vec::new();
        while let Some(reference) = current.get("$ref") {
            let Value::Str(reference) = reference else {
                return Err(Error::definition("$ref must be a string."));
            };
            let reference = reference.clone();
            if visited.contains(&produced_by) {
                return Err(Error::definition(&format!(
                    "Circular $ref detected: {reference}"
                )));
            }
            visited.push(produced_by);
            current = self.resolve_ref(&reference)?;
            produced_by = Some(reference);
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
            // The *parser's* log, not the repairer's: upstream writes this one with `self.log`
            // from inside `JSONParser`, so its context is the window around the cursor rather
            // than the schema path. The near-identical message in `repair_extra_properties` is
            // the repairer's and does carry a path.
            self.log(&format!(
                "Skipped unsupported patternProperties regex '{unsupported_pattern}' while parsing object key '{key}'"
            ));
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
