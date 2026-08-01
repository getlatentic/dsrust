//! What a schema does to an object: fill, coerce, drop, and — when salvaging — accept a list.
//!
//! The declared properties are repaired in the schema's order rather than the value's, so the
//! result reads the way the schema does. Everything else is matched against `patternProperties`
//! and then `additionalProperties`, and dropped when neither allows it.

use crate::schema::SchemaRepairer;
use crate::schema::container::{as_count, normalize_missing_values, type_name};
use crate::schema::object_schema_config;
use crate::schema::pattern::match_pattern_properties;
use crate::value::{Object, Value};
use crate::{Error, Result};

impl SchemaRepairer {
    pub(crate) fn repair_object(&self, value: Value, schema: &Value, path: &str) -> Result<Value> {
        let value = self.salvage_list_as_object(value, schema, path)?;
        let value = self.load_json_string_container(
            value,
            true,
            path,
            "Unwrapped JSON string to object to match schema",
            "Repaired malformed JSON string to object to match schema",
        );
        let Value::Object(mut fields) = value else {
            return Err(Error::new(&format!(
                "Expected object at {path}, got {}.",
                type_name(&value)
            )));
        };
        let config = object_schema_config(schema);

        if self.salvages() && !config.required.is_empty() {
            for key in &config.required {
                if fields.contains_key(key) {
                    continue;
                }
                let Some(prop_schema) = config.properties.iter().find(|(name, _)| name == key)
                else {
                    continue;
                };
                let key_path = format!("{path}.{key}");
                if let Some(filled) =
                    self.fill_missing_required_for_salvage(Some(&prop_schema.1), &key_path)?
                {
                    fields.insert(key.clone(), filled);
                    self.log(
                        "Filled missing required property while salvaging",
                        &key_path,
                    );
                }
            }
        }

        let missing: Vec<&str> = config
            .required
            .iter()
            .filter(|key| !fields.contains_key(key))
            .map(String::as_str)
            .collect();
        if !missing.is_empty() {
            return Err(Error::new(&format!(
                "Missing required properties at {path}: {}",
                missing.join(", ")
            )));
        }

        let mut repaired = Object::new();
        for (key, prop_schema) in &config.properties {
            let key_path = format!("{path}.{key}");
            if let Some(present) = fields.get(key) {
                repaired.insert(
                    key.clone(),
                    self.repair_value(Some(present.clone()), Some(prop_schema), &key_path)?,
                );
            } else if prop_schema.get("default").is_some() && !config.required.contains(key) {
                let default = prop_schema.get("default").expect("just checked");
                repaired.insert(
                    key.clone(),
                    self.copy_json_value(default, &key_path, "default")?,
                );
                self.log("Inserted default value for missing property", &key_path);
            }
        }
        self.repair_extra_properties(&fields, &config, &mut repaired, path)?;

        if let Some(min_properties) = schema.get("minProperties")
            && let Some(min) = as_count(min_properties)
            && repaired.len() < min
        {
            return Err(Error::new(&format!(
                "Object at {path} does not meet minProperties."
            )));
        }
        Ok(Value::Object(repaired))
    }

    /// Keys the schema did not name: matched against `patternProperties`, then
    /// `additionalProperties`, and dropped when neither allows them.
    fn repair_extra_properties(
        &self,
        fields: &Object,
        config: &crate::schema::ObjectSchemaConfig,
        repaired: &mut Object,
        path: &str,
    ) -> Result<()> {
        for (key, raw_value) in fields.iter() {
            if config.properties.iter().any(|(name, _)| name == key) {
                continue;
            }
            let key_path = format!("{path}.{key}");
            let (matched, unsupported) = match_pattern_properties(&config.pattern_properties, key);
            for pattern in unsupported {
                self.log(
                    &format!("Skipped unsupported patternProperties regex '{pattern}'"),
                    &key_path,
                );
            }
            if let Some((first, rest)) = matched.split_first() {
                let mut value =
                    self.repair_value(Some(raw_value.clone()), Some(first), &key_path)?;
                for prop_schema in rest {
                    value = self.repair_value(Some(value), Some(prop_schema), &key_path)?;
                }
                repaired.insert(key.to_owned(), value);
                continue;
            }
            match &config.additional_properties {
                Some(extra @ Value::Object(_)) => {
                    let value =
                        self.repair_value(Some(raw_value.clone()), Some(extra), &key_path)?;
                    repaired.insert(key.to_owned(), value);
                }
                Some(Value::Bool(true)) | None => {
                    repaired.insert(
                        key.to_owned(),
                        normalize_missing_values(Some(raw_value.clone())),
                    );
                }
                _ => self.log("Dropped extra property not covered by schema", &key_path),
            }
        }
        Ok(())
    }

    /// Salvage only: a list where an object was wanted, mapped onto the declared properties in
    /// order — or, at the root, a single-item wrapper unwrapped.
    fn salvage_list_as_object(&self, value: Value, schema: &Value, path: &str) -> Result<Value> {
        if !self.salvages() {
            return Ok(value);
        }
        let Value::Array(items) = &value else {
            return Ok(value);
        };
        if !self.can_salvage_list_as_object(schema) {
            return Ok(value);
        }
        if let Some(mapped) = self.map_list_to_object(items, schema, path)? {
            return Ok(mapped);
        }
        if path == "$" && items.len() == 1 && matches!(items.first(), Some(Value::Object(_))) {
            self.log(
                "Unwrapped single-item root array to object while salvaging",
                path,
            );
            return Ok(items[0].clone());
        }
        Ok(value)
    }

    fn map_list_to_object(
        &self,
        items: &[Value],
        schema: &Value,
        path: &str,
    ) -> Result<Option<Value>> {
        let Some(Value::Object(properties)) = schema.get("properties") else {
            return Ok(None);
        };
        if properties.is_empty() || items.len() != properties.len() {
            return Ok(None);
        }
        let mut mapped = Object::new();
        for (item, (key, prop_schema)) in items.iter().zip(properties.iter()) {
            let key_path = format!("{path}.{key}");
            match self.repair_value(Some(item.clone()), Some(prop_schema), &key_path) {
                Ok(value) => mapped.insert(key.to_owned(), value),
                Err(error) if error.is_definition() => return Err(error),
                Err(_) => return Ok(None),
            }
        }
        self.log("Mapped array to object by schema property order", path);
        Ok(Some(Value::Object(mapped)))
    }
}
