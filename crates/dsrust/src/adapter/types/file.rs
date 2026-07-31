//! dspy `adapters/types/file.py`: the `File` type.

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value, json};

use super::base::{Formatted, Type, serialized};

/// dspy's `File`: a file by data URI, provider file id, or name, rendered as a `file` content
/// block. At least one of the three must be present.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct File {
    /// A `data:<mime>;base64,<data>` URI.
    pub file_data: Option<String>,
    /// A provider-side file id.
    pub file_id: Option<String>,
    pub filename: Option<String>,
}

impl File {
    /// A file by its data URI.
    pub fn from_data(file_data: impl Into<String>) -> Self {
        Self {
            file_data: Some(file_data.into()),
            ..Self::default()
        }
    }

    /// A file by a provider-side id.
    pub fn from_id(file_id: impl Into<String>) -> Self {
        Self {
            file_id: Some(file_id.into()),
            ..Self::default()
        }
    }

    pub fn filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    fn is_empty(&self) -> bool {
        self.file_data.is_none() && self.file_id.is_none() && self.filename.is_none()
    }
}

impl Type for File {
    /// dspy `File.format`: one `file` block carrying only the fields that are set, in dspy's order.
    fn format(&self) -> Formatted {
        let mut file = Map::new();
        if let Some(file_data) = &self.file_data {
            file.insert("file_data".to_owned(), json!(file_data));
        }
        if let Some(file_id) = &self.file_id {
            file.insert("file_id".to_owned(), json!(file_id));
        }
        if let Some(filename) = &self.filename {
            file.insert("filename".to_owned(), json!(filename));
        }
        Formatted::Blocks(vec![json!({ "type": "file", "file": Value::Object(file) })])
    }
}

impl Serialize for File {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&serialized(self))
    }
}

impl<'de> Deserialize<'de> for File {
    /// dspy `File.validate`: a mapping carrying at least one of `file_data`, `file_id`, `filename`.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Value::Object(mut map) = Value::deserialize(deserializer)? else {
            return Err(de::Error::custom("`dspy.File` must be a mapping"));
        };
        let take = |map: &mut Map<String, Value>, key: &str| match map.remove(key) {
            Some(Value::String(value)) => Some(value),
            _ => None,
        };
        let file = File {
            file_data: take(&mut map, "file_data"),
            file_id: take(&mut map, "file_id"),
            filename: take(&mut map, "filename"),
        };
        if file.is_empty() {
            return Err(de::Error::custom(
                "Value of `dspy.File` must contain at least one of: file_data, file_id, or filename",
            ));
        }
        Ok(file)
    }
}

impl schemars::JsonSchema for File {
    /// The serialized form is a string — the sentinel-wrapped block — so an output field carries a
    /// string's schema.
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "File".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        String::json_schema(generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::base::{CUSTOM_TYPE_END, CUSTOM_TYPE_START};

    /// The block carries only the fields that are set, in dspy's order, wrapped in the sentinels.
    #[test]
    fn it_renders_only_the_fields_that_are_set() {
        let file = File::from_id("file-1").filename("a.txt");
        assert_eq!(
            file.format(),
            Formatted::Blocks(vec![
                json!({ "type": "file", "file": { "file_id": "file-1", "filename": "a.txt" } })
            ])
        );
        assert_eq!(
            serde_json::to_value(&file).expect("serializes"),
            json!(format!(
                "{CUSTOM_TYPE_START}{}{CUSTOM_TYPE_END}",
                r#"[{"type":"file","file":{"file_id":"file-1","filename":"a.txt"}}]"#
            ))
        );
    }

    #[test]
    fn it_requires_at_least_one_field() {
        let one: File =
            serde_json::from_value(json!({ "file_data": "data:text/plain;base64,QQ==" }))
                .expect("parses");
        assert_eq!(
            one.file_data.as_deref(),
            Some("data:text/plain;base64,QQ==")
        );
        assert!(serde_json::from_value::<File>(json!({})).is_err());
        assert!(serde_json::from_value::<File>(json!({ "other": "x" })).is_err());
    }
}
