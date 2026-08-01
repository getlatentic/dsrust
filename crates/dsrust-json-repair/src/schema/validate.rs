//! The check this library does not implement.
//!
//! Upstream's `SchemaRepairer` imports `jsonschema` — a separate PyPI package — and raises
//! `ValueError("jsonschema is required when using schema-aware repair.")` when it is not
//! installed. Reproducing a JSON Schema validator here would be porting a *fourth* library and
//! guessing at which draft, so the check is a seam instead: plug one in, or get upstream's answer
//! for an environment that has none.

use crate::value::Value;
use crate::{Error, Result};

/// Whether a value satisfies a schema. One implementation of `jsonschema`'s two entry points.
pub trait SchemaValidator {
    /// `validator.is_valid(value)`.
    fn is_valid(&self, value: &Value, schema: &Value) -> bool;

    /// `validator.validate(value)`, whose message becomes the `ValueError`'s.
    fn validate(&self, value: &Value, schema: &Value) -> std::result::Result<(), String>;
}

impl super::SchemaRepairer {
    pub(crate) fn is_valid(&self, value: &Value, schema: Option<&Value>) -> Result<bool> {
        match self.resolve_schema(schema)? {
            Value::Bool(true) => Ok(true),
            Value::Bool(false) => Ok(false),
            resolved => Ok(self.require_validator()?.is_valid(value, &resolved)),
        }
    }

    pub(crate) fn validate(&self, value: &Value, schema: Option<&Value>) -> Result<()> {
        match self.resolve_schema(schema)? {
            Value::Bool(true) => Ok(()),
            Value::Bool(false) => Err(Error::new("Schema does not allow any values.")),
            resolved => self
                .require_validator()?
                .validate(value, &resolved)
                .map_err(|message| Error::new(&message)),
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
