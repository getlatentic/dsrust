//! What a schema does to a scalar: coerce it, fill it in, or refuse it.
//!
//! The coercions are one-directional and specific — a number becomes a string but a string does
//! not become a boolean unless it is one of ten spellings, and `true` is never an integer even
//! though Python would say `True == 1`.

use crate::schema::{SchemaRepairMode, SchemaRepairer};
use crate::value::Value;
use crate::{Error, Result};

impl SchemaRepairer {
    /// `_coerce_scalar`.
    pub(crate) fn coerce_scalar(
        &self,
        value: &Value,
        schema_type: &str,
        path: &str,
    ) -> Result<Value> {
        match schema_type {
            "string" => match value {
                Value::Str(_) => Ok(value.clone()),
                Value::Int(_) | Value::BigInt(_) | Value::Float(_) => {
                    self.log("Coerced number to string", path);
                    Ok(Value::Str(number_as_python_str(value)))
                }
                _ => Err(Error::new(&format!("Expected string at {path}."))),
            },
            "integer" => self.coerce_integer(value, path),
            "number" => self.coerce_number(value, path),
            "boolean" => self.coerce_boolean(value, path),
            "null" => match value {
                Value::Null => Ok(Value::Null),
                _ => Err(Error::new(&format!("Expected null at {path}."))),
            },
            _ => Err(Error::definition(&format!(
                "Unsupported schema type {schema_type} at {path}."
            ))),
        }
    }

    fn coerce_integer(&self, value: &Value, path: &str) -> Result<Value> {
        let refuse = || Error::new(&format!("Expected integer at {path}."));
        match value {
            Value::Bool(_) => Err(refuse()),
            Value::Int(_) | Value::BigInt(_) => Ok(value.clone()),
            Value::Float(number) => {
                if number.fract() != 0.0 || !number.is_finite() {
                    return Err(refuse());
                }
                self.log("Coerced number to integer", path);
                Ok(Value::Int(*number as i64))
            }
            Value::Str(text) => {
                if let Some(number) = crate::pynum::try_python_int(text) {
                    self.log("Coerced string to integer", path);
                    return Ok(number);
                }
                let number = crate::pynum::try_python_float(text).ok_or_else(refuse)?;
                if number.fract() != 0.0 || !number.is_finite() {
                    return Err(refuse());
                }
                self.log("Coerced number to integer", path);
                Ok(Value::Int(number as i64))
            }
            _ => Err(refuse()),
        }
    }

    fn coerce_number(&self, value: &Value, path: &str) -> Result<Value> {
        let refuse = || Error::new(&format!("Expected number at {path}."));
        match value {
            Value::Bool(_) => Err(refuse()),
            Value::Int(_) | Value::BigInt(_) | Value::Float(_) => Ok(value.clone()),
            Value::Str(text) => {
                let number = crate::pynum::try_python_float(text).ok_or_else(refuse)?;
                self.log("Coerced string to number", path);
                Ok(Value::Float(number))
            }
            _ => Err(refuse()),
        }
    }

    fn coerce_boolean(&self, value: &Value, path: &str) -> Result<Value> {
        if let Value::Bool(flag) = value {
            return Ok(Value::Bool(*flag));
        }
        if let Value::Str(text) = value {
            let lowered = text.to_lowercase();
            if matches!(lowered.as_str(), "true" | "yes" | "y" | "on" | "1") {
                self.log("Coerced string to boolean", path);
                return Ok(Value::Bool(true));
            }
            if matches!(lowered.as_str(), "false" | "no" | "n" | "off" | "0") {
                self.log("Coerced string to boolean", path);
                return Ok(Value::Bool(false));
            }
        }
        if let Value::Int(number @ (0 | 1)) = value {
            self.log("Coerced number to boolean", path);
            return Ok(Value::Bool(*number == 1));
        }
        if let Value::Float(number) = value
            && (*number == 0.0 || *number == 1.0)
        {
            self.log("Coerced number to boolean", path);
            return Ok(Value::Bool(*number == 1.0));
        }
        Err(Error::new(&format!("Expected boolean at {path}.")))
    }

    /// `_apply_enum_const`: `const` and `enum` compared with Python's `==`, which crosses `1` and
    /// `1.0` and would accept `True` for `1`.
    pub(crate) fn apply_enum_const(
        &self,
        value: Value,
        schema: &Value,
        path: &str,
    ) -> Result<Value> {
        if let Some(expected) = schema.get("const")
            && !value.python_eq(expected)
        {
            return Err(Error::new(&format!(
                "Value at {path} does not match const."
            )));
        }
        if let Some(Value::Array(allowed)) = schema.get("enum")
            && !allowed.iter().any(|candidate| value.python_eq(candidate))
        {
            return Err(Error::new(&format!("Value at {path} does not match enum.")));
        }
        Ok(value)
    }

    /// `_fill_missing`: what stands in for a member the parser never found.
    pub(crate) fn fill_missing(&self, schema: &Value, path: &str) -> Result<Value> {
        if let Some(constant) = schema.get("const") {
            self.log("Filled missing value with const", path);
            return self.copy_json_value(constant, path, "const");
        }
        if let Some(allowed) = schema.get("enum") {
            let first = first_enum_value(allowed, path)?;
            self.log("Filled missing value with first enum value", path);
            return self.copy_json_value(&first, path, "enum");
        }
        if let Some(default) = schema.get("default") {
            self.log("Filled missing value with default", path);
            return self.copy_json_value(default, path, "default");
        }

        if let Some(Value::Array(types)) = schema.get("type") {
            for schema_type in types {
                let mut narrowed = schema.clone();
                if let Value::Object(fields) = &mut narrowed {
                    fields.insert("type".to_owned(), schema_type.clone());
                }
                if let Ok(filled) = self.fill_missing(&narrowed, path) {
                    return Ok(filled);
                }
            }
            return Err(Error::new(&format!(
                "Cannot infer missing value at {path}."
            )));
        }

        // `if expected_type is None:` guards the inference upstream, so a `type` that is present
        // but is not a string — `{"type": 7}` — blocks it and reaches no branch at all. A `_` arm
        // here would infer `object` from a stray `properties` and fill where Python refuses.
        let expected_type = match schema.get("type") {
            Some(Value::Str(name)) => Some(name.clone()),
            Some(_) => None,
            None if self.is_object_schema(Some(schema)) => Some("object".to_owned()),
            None if self.is_array_schema(Some(schema)) => Some("array".to_owned()),
            None => None,
        };
        self.empty_value_for(expected_type.as_deref(), schema, path)
    }

    fn empty_value_for(&self, expected: Option<&str>, schema: &Value, path: &str) -> Result<Value> {
        match expected {
            Some("string") => {
                self.log("Filled missing value with empty string", path);
                Ok(Value::Str(String::new()))
            }
            Some("integer" | "number") => {
                self.log("Filled missing value with 0", path);
                Ok(Value::Int(0))
            }
            Some("boolean") => {
                self.log("Filled missing value with false", path);
                Ok(Value::Bool(false))
            }
            Some("array") => {
                if schema.get("minItems").is_some_and(Value::is_truthy) {
                    let min = schema.get("minItems").expect("just checked");
                    return Err(Error::new(&format!(
                        "Array at {path} requires at least {min} items."
                    )));
                }
                self.log("Filled missing value with empty array", path);
                Ok(Value::Array(Vec::new()))
            }
            Some("object") => {
                if schema.get("minProperties").is_some_and(Value::is_truthy) {
                    let min = schema.get("minProperties").expect("just checked");
                    return Err(Error::new(&format!(
                        "Object at {path} requires at least {min} properties."
                    )));
                }
                self.log("Filled missing value with empty object", path);
                Ok(Value::Object(crate::value::Object::new()))
            }
            Some("null") => {
                self.log("Filled missing value with null", path);
                Ok(Value::Null)
            }
            _ => Err(Error::new(&format!(
                "Cannot infer missing value at {path}."
            ))),
        }
    }

    /// `_fill_missing_required_for_salvage`: a required property with nothing to put in it.
    pub(crate) fn fill_missing_required_for_salvage(
        &self,
        schema: Option<&Value>,
        path: &str,
    ) -> Result<Option<Value>> {
        let Ok(resolved @ Value::Object(_)) = self.resolve_schema(schema) else {
            return Ok(None);
        };
        for (key, label) in [("default", "default"), ("const", "const")] {
            if let Some(value) = resolved.get(key) {
                return Ok(Some(self.copy_json_value(value, path, label)?));
            }
        }
        if let Some(Value::Array(allowed)) = resolved.get("enum")
            && let Some(first) = allowed.first()
        {
            return Ok(Some(self.copy_json_value(first, path, "enum")?));
        }

        // The same `is None` guard as `fill_missing`, and the same trap.
        let expected_type = match resolved.get("type") {
            Some(Value::Str(name)) => Some(name.clone()),
            Some(_) => None,
            None if self.is_array_schema(Some(&resolved)) => Some("array".to_owned()),
            None if self.is_object_schema(Some(&resolved)) => Some("object".to_owned()),
            None => None,
        };
        match expected_type.as_deref() {
            Some("array") if !resolved.get("minItems").is_some_and(Value::is_truthy) => {
                Ok(Some(Value::Array(Vec::new())))
            }
            Some("object") if !resolved.get("minProperties").is_some_and(Value::is_truthy) => {
                Ok(Some(Value::Object(crate::value::Object::new())))
            }
            _ => Ok(None),
        }
    }

    /// `_copy_json_value`: a schema's own literal, checked for being JSON at all.
    pub(crate) fn copy_json_value(&self, value: &Value, path: &str, label: &str) -> Result<Value> {
        match value {
            Value::Array(items) => Ok(Value::Array(
                items
                    .iter()
                    .enumerate()
                    .map(|(idx, item)| self.copy_json_value(item, &format!("{path}[{idx}]"), label))
                    .collect::<Result<Vec<_>>>()?,
            )),
            Value::Object(fields) => {
                let mut copied = crate::value::Object::new();
                for (key, item) in fields.iter() {
                    copied.insert(
                        key.to_owned(),
                        self.copy_json_value(item, &format!("{path}.{key}"), label)?,
                    );
                }
                Ok(Value::Object(copied))
            }
            scalar => Ok(scalar.clone()),
        }
    }

    pub(crate) fn salvages(&self) -> bool {
        self.mode == SchemaRepairMode::Salvage
    }
}

/// `enum_values[0]` after `if not enum_values: raise`.
///
/// Upstream subscripts whatever `enum` holds rather than requiring a list, so a *string* enum
/// yields its first character. Anything truthy that Python cannot subscript by `0` raises a
/// `TypeError` or a `KeyError`, neither of which `except ValueError` catches — hence a foreign
/// error rather than a refusal.
fn first_enum_value(allowed: &Value, path: &str) -> Result<Value> {
    if !allowed.is_truthy() {
        return Err(Error::new(&format!("Enum at {path} has no values.")));
    }
    match allowed {
        Value::Array(items) => Ok(items[0].clone()),
        Value::Str(text) => Ok(Value::Str(
            text.chars()
                .next()
                .expect("truthy is non-empty")
                .to_string(),
        )),
        other => Err(Error::type_error(&format!(
            "'{}' object is not subscriptable",
            crate::schema::container::type_name(other)
        ))),
    }
}

/// `str(value)` for the numeric types, which is `repr` for a float and the digits for an int.
fn number_as_python_str(value: &Value) -> String {
    match value {
        Value::Int(number) => number.to_string(),
        Value::BigInt(digits) => digits.clone(),
        other => other.to_string(),
    }
}
