//! `repair_value` and the two containers, which is where a schema does most of its work.
//!
//! An object drops what the schema forbids and fills what it allows; an array is rebuilt item by
//! item against `items` or a tuple of them; a union tries each branch and keeps the first that
//! validates. A string that *holds* JSON is unwrapped, because a model asked for an object often
//! answers with one inside quotes.

use crate::schema::{SchemaRepairer, array_schema_config};
use crate::value::Value;
use crate::{Error, Result};

/// What the parser found. `None` is upstream's `MISSING_VALUE`: a member with no value at all.
pub(crate) type Maybe = Option<Value>;

impl SchemaRepairer {
    /// `repair_value`: the whole schema pass over one node.
    pub(crate) fn repair_value(
        &self,
        value: Maybe,
        schema: Option<&Value>,
        path: &str,
    ) -> Result<Value> {
        let schema = self.resolve_schema(schema)?;
        match &schema {
            Value::Bool(true) => return Ok(normalize_missing_values(value)),
            Value::Bool(false) => return Err(Error::new("Schema does not allow any values.")),
            // An empty schema object constrains nothing, as `if not schema` says upstream.
            Value::Object(fields) if fields.is_empty() => {
                return Ok(normalize_missing_values(value));
            }
            _ => {}
        }

        let Some(value) = value else {
            return self.fill_missing(&schema, path);
        };

        if let Some(Value::Array(subschemas)) = schema.get("allOf") {
            let subschemas = subschemas.clone();
            let Some(first) = subschemas.first() else {
                return Ok(normalize_missing_values(Some(value)));
            };
            let mut repaired = self.repair_value(Some(value), Some(first), path)?;
            for subschema in &subschemas[1..] {
                repaired = self.repair_value(Some(repaired), Some(subschema), path)?;
            }
            return Ok(repaired);
        }
        for keyword in ["oneOf", "anyOf"] {
            if let Some(Value::Array(subschemas)) = schema.get(keyword) {
                return self.repair_union(&value, &subschemas.clone(), path);
            }
        }

        let expected_type = match schema.get("type") {
            Some(declared) => Some(declared.clone()),
            None if self.is_object_schema(Some(&schema)) => Some(Value::Str("object".to_owned())),
            None if self.is_array_schema(Some(&schema)) => Some(Value::Str("array".to_owned())),
            None => None,
        };

        if let Some(Value::Array(types)) = &expected_type {
            return self.repair_type_union(&value, &types.clone(), &schema, path);
        }
        let repaired = match &expected_type {
            Some(Value::Str(name)) => self.repair_by_type(value, name, &schema, path)?,
            _ => normalize_missing_values(Some(value)),
        };
        self.apply_enum_const(repaired, &schema, path)
    }

    fn repair_by_type(
        &self,
        value: Value,
        schema_type: &str,
        schema: &Value,
        path: &str,
    ) -> Result<Value> {
        match schema_type {
            "array" => self.repair_array(value, schema, path),
            "object" => self.repair_object(value, schema, path),
            _ => self.coerce_scalar(&value, schema_type, path),
        }
    }

    /// `oneOf`/`anyOf`: the first branch that both repairs and validates.
    fn repair_union(&self, value: &Value, schemas: &[Value], path: &str) -> Result<Value> {
        let mut last_error = None;
        for subschema in schemas {
            let candidate = self
                .repair_value(Some(value.clone()), Some(subschema), path)
                .and_then(|candidate| {
                    self.validate(&candidate, Some(subschema))?;
                    Ok(candidate)
                });
            match candidate {
                Ok(candidate) => return Ok(candidate),
                Err(error) if error.is_definition() => return Err(error),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| Error::new("No schema matched the value.")))
    }

    /// A `type` that lists several: the same value read as each in turn.
    fn repair_type_union(
        &self,
        value: &Value,
        types: &[Value],
        schema: &Value,
        path: &str,
    ) -> Result<Value> {
        let mut last_error = None;
        for schema_type in types {
            let Value::Str(schema_type) = schema_type else {
                continue;
            };
            let mut branch_schema = schema.clone();
            if let Value::Object(fields) = &mut branch_schema {
                fields.insert("type".to_owned(), Value::Str(schema_type.clone()));
            }
            // The structural schema stays whole for the repair, and only the validation narrows.
            let candidate = self
                .repair_by_type(value.clone(), schema_type, schema, path)
                .and_then(|candidate| self.apply_enum_const(candidate, &branch_schema, path))
                .and_then(|candidate| {
                    self.validate(&candidate, Some(&branch_schema))?;
                    Ok(candidate)
                });
            match candidate {
                Ok(candidate) => return Ok(candidate),
                Err(error) if error.is_definition() => return Err(error),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| Error::new("No schema type matched the value.")))
    }

    /// A string that holds a container, unwrapped — and in salvage mode, repaired first.
    pub(crate) fn load_json_string_container(
        &self,
        value: Value,
        wants_object: bool,
        path: &str,
        unwrap_log: &str,
        salvage_log: &str,
    ) -> Value {
        let Value::Str(text) = &value else {
            return value;
        };
        let holds_wanted = |parsed: &Value| match wants_object {
            true => matches!(parsed, Value::Object(_)),
            false => matches!(parsed, Value::Array(_)),
        };
        match crate::strict_json::loads(&text.chars().collect::<Vec<_>>()) {
            Ok(parsed) if holds_wanted(&parsed) => {
                self.log(unwrap_log, path);
                parsed
            }
            Ok(_) => value,
            Err(_) if !self.salvages() => value,
            Err(_) => {
                let repaired = crate::Repair::new().skip_json_loads(true).loads(text);
                match repaired {
                    Ok(repaired) if holds_wanted(&repaired) => {
                        self.log(salvage_log, path);
                        repaired
                    }
                    _ => value,
                }
            }
        }
    }

    fn repair_array(&self, value: Value, schema: &Value, path: &str) -> Result<Value> {
        let value = self.load_json_string_container(
            value,
            false,
            path,
            "Unwrapped JSON string to array to match schema",
            "Repaired malformed JSON string to array to match schema",
        );
        let mut items = match value {
            Value::Array(items) => items,
            other => {
                self.log("Wrapped value in array to match schema", path);
                vec![normalize_missing_values(Some(other))]
            }
        };
        let config = array_schema_config(schema);

        match &config.items_schema {
            Some(Value::Array(tuple)) => {
                items = self.repair_tuple_items(items, tuple, &config, path)?;
            }
            Some(items_schema) => {
                let mut repaired = Vec::new();
                for (idx, item) in items.into_iter().enumerate() {
                    let item_path = format!("{path}[{idx}]");
                    if let Some(value) =
                        self.repair_or_drop(item, Some(items_schema), &item_path)?
                    {
                        repaired.push(value);
                    }
                }
                items = repaired;
            }
            None => {}
        }

        if let Some(min_items) = schema.get("minItems")
            && let Some(min) = as_count(min_items)
            && items.len() < min
        {
            return Err(Error::new(&format!(
                "Array at {path} does not meet minItems."
            )));
        }
        Ok(Value::Array(items))
    }

    /// `items` given as a list: one schema per position, and a rule for the rest.
    fn repair_tuple_items(
        &self,
        items: Vec<Value>,
        tuple: &[Value],
        config: &crate::schema::ArraySchemaConfig,
        path: &str,
    ) -> Result<Vec<Value>> {
        let mut repaired = Vec::new();
        let mut items = items.into_iter();
        for (idx, item_schema) in tuple.iter().enumerate() {
            let Some(item) = items.next() else { break };
            let item_path = format!("{path}[{idx}]");
            if let Some(value) = self.repair_or_drop(item, Some(item_schema), &item_path)? {
                repaired.push(value);
            }
        }
        for (offset, item) in items.enumerate() {
            let idx = tuple.len() + offset;
            let item_path = format!("{path}[{idx}]");
            match &config.additional_items {
                Some(extra @ Value::Object(_)) => {
                    if let Some(value) = self.repair_or_drop(item, Some(extra), &item_path)? {
                        repaired.push(value);
                    }
                }
                Some(Value::Bool(true)) | None => {
                    repaired.push(normalize_missing_values(Some(item)))
                }
                _ => self.log("Dropped extra array item not covered by schema", &item_path),
            }
        }
        Ok(repaired)
    }

    /// An item that will not fit: an error, or a drop when salvaging.
    fn repair_or_drop(
        &self,
        item: Value,
        schema: Option<&Value>,
        item_path: &str,
    ) -> Result<Option<Value>> {
        match self.repair_value(Some(item), schema, item_path) {
            Ok(value) => Ok(Some(value)),
            Err(error) if error.is_definition() || !self.salvages() => Err(error),
            Err(_) => {
                self.log("Dropped invalid array item while salvaging", item_path);
                Ok(None)
            }
        }
    }
}

/// `normalize_missing_values`: the sentinel becomes the empty string, everything else stands.
pub(crate) fn normalize_missing_values(value: Maybe) -> Value {
    value.unwrap_or_else(|| Value::Str(String::new()))
}

/// `type(value).__name__`, for the one error message that prints it.
pub(crate) fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Int(_) | Value::BigInt(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

pub(crate) fn as_count(value: &Value) -> Option<usize> {
    match value {
        Value::Int(number) => usize::try_from(*number).ok(),
        Value::Float(number) if *number >= 0.0 => Some(*number as usize),
        _ => None,
    }
}
