//! dspy `adapters/types/image.py`: the `Image` type.

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Value, json};

use super::base::{Formatted, Type, serialized};

/// dspy's `Image`: an image by URL or base64 data URI, rendered as an `image_url` content block.
///
/// dspy's constructor also accepts raw bytes, a PIL image, or a remote URL to download — each
/// encoded to a data URI first. Those are Python objects a Rust caller does not hold; here the
/// value is the `url` (an `http(s)`/`gs` URL, a local path, or a `data:` URI) as given.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Image {
    pub url: String,
}

impl Image {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }
}

impl Type for Image {
    /// dspy `Image.format`: one `image_url` block carrying the URL.
    fn format(&self) -> Formatted {
        Formatted::Blocks(vec![json!({ "type": "image_url", "image_url": { "url": self.url } })])
    }
}

impl Serialize for Image {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&serialized(self))
    }
}

impl<'de> Deserialize<'de> for Image {
    /// dspy accepts a bare URL string or the legacy `{"url": ...}` mapping.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::String(url) => Ok(Self::new(url)),
            Value::Object(mut map) => match map.remove("url") {
                Some(Value::String(url)) => Ok(Self::new(url)),
                _ => Err(de::Error::custom("`url` field is required for `dspy.Image`")),
            },
            other => Err(de::Error::custom(format!("Received invalid value for `dspy.Image`: {other}"))),
        }
    }
}

impl schemars::JsonSchema for Image {
    /// The serialized form is a string — the sentinel-wrapped block — so an output field carries a
    /// string's schema.
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Image".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        String::json_schema(generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::base::{CUSTOM_TYPE_END, CUSTOM_TYPE_START};

    /// `format` is one `image_url` block, and the serialized value wraps it in the sentinels so the
    /// render's string round trip can split it back into a content part.
    #[test]
    fn it_renders_as_a_sentinel_wrapped_image_block() {
        let image = Image::new("https://example.com/a.jpg");
        assert_eq!(
            image.format(),
            Formatted::Blocks(vec![
                json!({ "type": "image_url", "image_url": { "url": "https://example.com/a.jpg" } })
            ])
        );
        assert_eq!(
            serde_json::to_value(&image).expect("serializes"),
            json!(format!(
                "{CUSTOM_TYPE_START}{}{CUSTOM_TYPE_END}",
                r#"[{"type":"image_url","image_url":{"url":"https://example.com/a.jpg"}}]"#
            ))
        );
    }

    #[test]
    fn it_reads_a_bare_url_or_a_url_mapping() {
        let bare: Image = serde_json::from_value(json!("data:image/png;base64,AAAA")).expect("parses");
        assert_eq!(bare.url, "data:image/png;base64,AAAA");
        let mapped: Image = serde_json::from_value(json!({ "url": "u" })).expect("parses");
        assert_eq!(mapped.url, "u");
        assert!(serde_json::from_value::<Image>(json!({ "no_url": 1 })).is_err());
        assert!(serde_json::from_value::<Image>(json!(3)).is_err());
    }
}
