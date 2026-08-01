//! The check this library does not implement.
//!
//! Upstream's `SchemaRepairer` imports `jsonschema` — a separate PyPI package — and raises
//! `ValueError("jsonschema is required when using schema-aware repair.")` when it is not
//! installed. Reproducing a JSON Schema validator here would be porting a *fourth* library and
//! guessing at which draft, so the check is a seam instead: plug one in, or get upstream's answer
//! for an environment that has none.

use crate::value::Value;
use crate::{Error, Result};

/// Why a validator would not pass a value.
///
/// The two are not interchangeable. `json_repair` catches `ValueError` in six places, and
/// `jsonschema`'s `ValidationError` is re-raised as one — so a union branch that fails is simply
/// the next branch's turn. A validator that cannot *read* the schema raises something else
/// entirely, which nothing catches and which reaches the caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// The value does not satisfy the schema.
    Invalid(String),
    /// The schema itself could not be read — `jsonschema` answers `{"type": "date"}` this way.
    Unreadable(String),
}

impl ValidationError {
    fn into_error(self) -> Error {
        match self {
            ValidationError::Invalid(message) => Error::new(&message),
            ValidationError::Unreadable(message) => Error::foreign(&message),
        }
    }
}

/// Whether a value satisfies a schema. One implementation of `jsonschema`'s two entry points.
pub trait SchemaValidator {
    /// `validator.is_valid(value)`.
    fn is_valid(&self, value: &Value, schema: &Value)
    -> std::result::Result<bool, ValidationError>;

    /// `validator.validate(value)`, whose message becomes the `ValueError`'s.
    fn validate(&self, value: &Value, schema: &Value) -> std::result::Result<(), ValidationError>;
}

/// `_prepare_schema_for_validation_node`: draft-4 tuple `items` rewritten to 2020-12.
///
/// This is `json_repair`'s own code, not `jsonschema`'s, and it runs before *every* validator call
/// — so the schema a validator receives is never the one the caller wrote. Skipping it does not
/// merely validate differently: `{"items": [...]}` under 2020-12 means "every item matches this
/// schema", and the schema is a list, so `jsonschema` raises `AttributeError` reaching for `$id`
/// on it. Python never sees that because it prepares first.
pub(crate) fn prepare_for_validation(node: &Value) -> Value {
    match node {
        Value::Array(items) => Value::Array(items.iter().map(prepare_for_validation).collect()),
        Value::Object(fields) => {
            let mut normalized: crate::value::Object = fields
                .iter()
                .map(|(key, value)| (key.to_owned(), prepare_for_validation(value)))
                .collect();
            let Some(Value::Array(tuple)) = normalized.get("items").cloned() else {
                return Value::Object(normalized);
            };
            normalized.remove("items");
            normalized.insert("prefixItems".to_owned(), Value::Array(tuple));
            // `additionalItems` becomes the *new* `items`, which is what 2020-12 calls the rule for
            // everything past the tuple. Only `false` and an object schema carry over; `true` and a
            // missing key both mean "anything", which 2020-12 spells by leaving `items` out.
            match normalized.remove("additionalItems") {
                Some(Value::Bool(false)) => {
                    normalized.insert("items".to_owned(), Value::Bool(false));
                }
                Some(extra @ Value::Object(_)) => {
                    normalized.insert("items".to_owned(), extra);
                }
                _ => {}
            }
            Value::Object(normalized)
        }
        scalar => scalar.clone(),
    }
}

impl super::SchemaRepairer {
    pub(crate) fn is_valid(&self, value: &Value, schema: Option<&Value>) -> Result<bool> {
        match self.resolve_schema(schema)? {
            Value::Bool(true) => Ok(true),
            Value::Bool(false) => Ok(false),
            resolved => self
                .require_validator()?
                .is_valid(value, &prepare_for_validation(&resolved))
                .map_err(ValidationError::into_error),
        }
    }

    pub(crate) fn validate(&self, value: &Value, schema: Option<&Value>) -> Result<()> {
        match self.resolve_schema(schema)? {
            Value::Bool(true) => Ok(()),
            Value::Bool(false) => Err(Error::new("Schema does not allow any values.")),
            resolved => self
                .require_validator()?
                .validate(value, &prepare_for_validation(&resolved))
                .map_err(ValidationError::into_error),
        }
    }

    fn require_validator(&self) -> Result<&dyn SchemaValidator> {
        match self.validator() {
            Some(validator) => Ok(validator),
            None => Err(Error::new(
                "jsonschema is required when using schema-aware repair.",
            )),
        }
    }
}
