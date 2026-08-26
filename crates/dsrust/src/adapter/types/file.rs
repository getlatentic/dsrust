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

    /// dspy `encode_file_to_dict`, string branch: a `data:` URI and nothing else.
    ///
    /// **Refuses a path or a URL**, naming the factory that reads one. A `File` has no reference
    /// form — a provider is handed the bytes or a file id it already holds — so a locator here is
    /// a caller reaching for the wrong door, and keeping it would put a local path on the wire.
    pub fn parse(source: impl AsRef<str>) -> anyhow::Result<Self> {
        let source = source.as_ref();
        if !source.starts_with("data:") {
            anyhow::bail!(
                "String file inputs must be data URIs, received: {source}. \
                 Load local files with File.from_path()."
            );
        }
        Ok(Self::from_data(source))
    }

    /// dspy `File(bytes)`: raw bytes the caller already holds.
    ///
    /// `application/octet-stream`, because nothing here names the type — there is no filename to
    /// guess from and, unlike an image, no signature worth trusting across the formats a `File`
    /// carries. Upstream's default, and its own test asserts exactly this prefix.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Self {
        Self::from_bytes_as(bytes, "application/octet-stream")
    }

    /// The same, naming the media type — upstream's `mime_type=`.
    ///
    /// Bytes a caller holds often *are* known: a PDF they just rendered, a CSV they just wrote. The
    /// media type is baked into the `data:` URI and is what a provider decides how to read them by,
    /// so being able to say so is the difference between a document and an opaque blob.
    pub fn from_bytes_as(bytes: impl AsRef<[u8]>, media_type: impl AsRef<str>) -> Self {
        Self::from_data(crate::resource::data_uri(
            media_type.as_ref(),
            &crate::resource::encode(bytes.as_ref()),
        ))
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

    /// dspy `File.from_path`: read a local file, encode it as a `data:` URI, and keep its name.
    ///
    /// The name is kept because it is the only thing telling a model what the file *is* once the
    /// bytes are base64 — upstream sends it beside the data for that reason. `application/octet-
    /// stream` where the suffix names nothing, which is what a caller gets for an extensionless
    /// file and is also what `.bin` means.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        Self::from_path_as(
            path,
            crate::resource::media_type_for(path, "application/octet-stream"),
        )
    }

    /// The same, naming the media type rather than guessing it — upstream's `mime_type=`.
    ///
    /// For a file whose suffix lies, or has none: the media type is baked into the `data:` URI and
    /// is what a provider decides how to read the bytes by, so a `.dat` that is really a PDF has to
    /// be able to say so. Upstream's `filename=` override is the builder
    /// [`filename`](Self::filename) that was already here.
    pub fn from_path_as(
        path: impl AsRef<std::path::Path>,
        media_type: impl AsRef<str>,
    ) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let encoded = crate::resource::read_base64(path)?;
        let file = Self::from_data(crate::resource::data_uri(media_type.as_ref(), &encoded));
        Ok(match path.file_name().and_then(|name| name.to_str()) {
            Some(name) => file.filename(name),
            None => file,
        })
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

    /// Against dspy's own answer for the same bytes: `File.from_path` on a `.txt` holding
    /// `file bytes` gives `data:text/plain;base64,ZmlsZSBieXRlcw==` and keeps the basename.
    #[test]
    fn from_path_encodes_the_bytes_and_keeps_the_name_dspy_keeps() {
        let path = std::env::temp_dir().join("dsrs_file_from_path.txt");
        std::fs::write(&path, b"file bytes").expect("writes");
        let file = File::from_path(&path).expect("reads");
        assert_eq!(
            file.file_data.as_deref(),
            Some("data:text/plain;base64,ZmlsZSBieXRlcw==")
        );
        assert_eq!(file.filename.as_deref(), Some("dsrs_file_from_path.txt"));
        let _ = std::fs::remove_file(&path);
    }

    /// A suffix the table does not know is `application/octet-stream`, which is upstream's
    /// fallback and what an extensionless file gets.
    #[test]
    fn an_unknown_suffix_falls_back_to_opaque_bytes() {
        let path = std::env::temp_dir().join("dsrs_file_from_path_unknown");
        std::fs::write(&path, b"file bytes").expect("writes");
        let file = File::from_path(&path).expect("reads");
        assert!(
            file.file_data
                .as_deref()
                .expect("data")
                .starts_with("data:application/octet-stream;base64,"),
            "{file:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Bytes with nothing naming their type are opaque — measured against `dspy.File(b"file
    /// bytes")`, which gives exactly this, and upstream's own test asserts this prefix.
    #[test]
    fn bytes_become_an_opaque_data_uri() {
        assert_eq!(
            File::from_bytes(b"file bytes").file_data.as_deref(),
            Some("data:application/octet-stream;base64,ZmlsZSBieXRlcw==")
        );
    }

    /// A locator is refused in upstream's words. A `File` has no reference form, so keeping one
    /// would put a local path on the wire under a key a provider reads as content.
    #[test]
    fn a_string_that_is_not_a_data_uri_is_refused() {
        let why = File::parse("/etc/passwd").expect_err("refused").to_string();
        assert_eq!(
            why,
            "String file inputs must be data URIs, received: /etc/passwd. \
             Load local files with File.from_path()."
        );
        assert_eq!(
            File::parse("data:text/plain;base64,QQ==")
                .expect("a data uri")
                .file_data
                .as_deref(),
            Some("data:text/plain;base64,QQ==")
        );
    }

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
