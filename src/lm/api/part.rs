//! The eleven content parts of dspy 3.3's `LMPart`, discriminated on `type`.

use std::path::PathBuf;

use serde_json::{Map, Value};

pub type Metadata = Map<String, Value>;

/// Where a provider-shaped block this crate does not model rides along verbatim, so it renders
/// back byte for byte instead of being guessed at.
pub const LEGACY_BLOCK: &str = "legacy_content_block";

/// Upstream spells this as four nullable fields plus a `validate_one_source` validator, because
/// Python cannot say "exactly one" in a type. Rust can, so that state is unrepresentable rather
/// than rejected at run time.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "SourceFields", into = "SourceFields")]
pub enum LmSource {
    Data(String),
    Url(String),
    FileId(String),
    Path(PathBuf),
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct SourceFields {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<PathBuf>,
}

impl TryFrom<SourceFields> for LmSource {
    type Error = String;

    fn try_from(fields: SourceFields) -> Result<Self, Self::Error> {
        let named: Vec<Self> = [
            fields.data.map(Self::Data),
            fields.url.map(Self::Url),
            fields.file_id.map(Self::FileId),
            fields.path.map(Self::Path),
        ]
        .into_iter()
        .flatten()
        .collect();
        let [only] = <[Self; 1]>::try_from(named).map_err(|named| {
            format!(
                "expected exactly one of data, url, file_id, or path, got {}",
                named.len()
            )
        })?;
        match only.is_empty() {
            true => Err("a source must not be empty".to_owned()),
            false => Ok(only),
        }
    }
}

impl From<LmSource> for SourceFields {
    fn from(source: LmSource) -> Self {
        match source {
            LmSource::Data(data) => Self {
                data: Some(data),
                ..Self::default()
            },
            LmSource::Url(url) => Self {
                url: Some(url),
                ..Self::default()
            },
            LmSource::FileId(file_id) => Self {
                file_id: Some(file_id),
                ..Self::default()
            },
            LmSource::Path(path) => Self {
                path: Some(path),
                ..Self::default()
            },
        }
    }
}

impl LmSource {
    fn is_empty(&self) -> bool {
        match self {
            Self::Data(value) | Self::Url(value) | Self::FileId(value) => value.is_empty(),
            Self::Path(path) => path.as_os_str().is_empty(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    Low,
    High,
    Auto,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LmPart {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    Image {
        #[serde(flatten)]
        source: LmSource,
        #[serde(default = "image_media_type")]
        media_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<Detail>,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    Audio {
        #[serde(flatten)]
        source: LmSource,
        #[serde(default = "audio_media_type")]
        media_type: String,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    Video {
        #[serde(flatten)]
        source: LmSource,
        #[serde(default = "video_media_type")]
        media_type: String,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    Binary {
        #[serde(flatten)]
        source: LmSource,
        #[serde(default = "binary_media_type")]
        media_type: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    /// Upstream declares this on `LMBasePart`, not `LMSourcePart`: it takes either one media
    /// source or a provider-shaped `source` dict, which is a weaker rule than its media
    /// siblings' exactly-one.
    Document {
        #[serde(flatten)]
        source: DocumentSource,
        #[serde(default = "document_media_type")]
        media_type: String,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        citations: Metadata,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    ToolCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        name: String,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        args: Metadata,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        provider_data: Metadata,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    ToolResult {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        content: Vec<LmPart>,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        provider_data: Metadata,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    Thinking {
        text: String,
        #[serde(default, skip_serializing_if = "is_false")]
        redacted: bool,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    Citation {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    Refusal {
        text: String,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentSource {
    Source(Metadata),
    #[serde(untagged)]
    Media(LmSource),
}

impl LmPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            metadata: Metadata::new(),
        }
    }

    pub fn legacy(block: Value) -> Self {
        let mut metadata = Metadata::new();
        metadata.insert(LEGACY_BLOCK.to_owned(), block);
        Self::Text {
            text: String::new(),
            metadata,
        }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: LmSource::Url(url.into()),
            media_type: image_media_type(),
            detail: None,
            metadata: Metadata::new(),
        }
    }

    /// A part carrying a block is spelled as text with an empty string, which is not prose.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text, metadata } if !metadata.contains_key(LEGACY_BLOCK) => Some(text),
            _ => None,
        }
    }

    pub fn legacy_block(&self) -> Option<&Value> {
        self.metadata().get(LEGACY_BLOCK)
    }

    pub fn metadata(&self) -> &Metadata {
        match self {
            Self::Text { metadata, .. }
            | Self::Image { metadata, .. }
            | Self::Audio { metadata, .. }
            | Self::Video { metadata, .. }
            | Self::Binary { metadata, .. }
            | Self::Document { metadata, .. }
            | Self::ToolCall { metadata, .. }
            | Self::ToolResult { metadata, .. }
            | Self::Thinking { metadata, .. }
            | Self::Citation { metadata, .. }
            | Self::Refusal { metadata, .. } => metadata,
        }
    }
}

fn image_media_type() -> String {
    "image/png".to_owned()
}

fn audio_media_type() -> String {
    "audio/wav".to_owned()
}

fn video_media_type() -> String {
    "video/mp4".to_owned()
}

fn binary_media_type() -> String {
    "application/octet-stream".to_owned()
}

fn document_media_type() -> String {
    "application/pdf".to_owned()
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_part_is_discriminated_by_its_type() {
        let part: LmPart = serde_json::from_value(json!({
            "type": "image",
            "url": "https://example.com/a.jpg",
            "media_type": "image/jpeg",
            "detail": "high",
        }))
        .expect("an image part");

        let LmPart::Image {
            source,
            media_type,
            detail,
            ..
        } = &part
        else {
            panic!("got {part:?}")
        };
        assert_eq!(*source, LmSource::Url("https://example.com/a.jpg".to_owned()));
        assert_eq!(media_type, "image/jpeg");
        assert_eq!(*detail, Some(Detail::High));
    }

    /// Serde's flatten takes the first key it matches, so without the counting deserializer a
    /// second source is dropped and a different image reaches the provider.
    #[test]
    fn two_sources_are_refused_rather_than_one_being_dropped() {
        let both = serde_json::from_value::<LmPart>(json!({
            "type": "image",
            "url": "https://example.com/a.jpg",
            "data": "aGk=",
        }));
        assert!(both.is_err(), "got {both:?}");
        assert!(
            serde_json::from_value::<LmPart>(json!({ "type": "image" })).is_err(),
            "no source at all is equally invalid"
        );
        assert!(
            serde_json::from_value::<LmPart>(json!({ "type": "image", "url": "" })).is_err(),
            "an empty source fails further from the mistake"
        );
    }

    #[test]
    fn an_unknown_part_type_is_refused_rather_than_flattened() {
        let unknown = serde_json::from_value::<LmPart>(json!({ "type": "hologram", "text": "hi" }));
        assert!(unknown.is_err(), "got {unknown:?}");
    }

    #[test]
    fn each_media_part_carries_the_default_media_type_upstream_gives_it() {
        let defaults = [
            (json!({ "type": "image", "url": "u" }), "image/png"),
            (json!({ "type": "audio", "url": "u" }), "audio/wav"),
            (json!({ "type": "video", "url": "u" }), "video/mp4"),
            (
                json!({ "type": "binary", "url": "u" }),
                "application/octet-stream",
            ),
        ];
        for (raw, expected) in defaults {
            let part: LmPart = serde_json::from_value(raw.clone()).expect("parses");
            let (LmPart::Image { media_type, .. }
            | LmPart::Audio { media_type, .. }
            | LmPart::Video { media_type, .. }
            | LmPart::Binary { media_type, .. }) = &part
            else {
                panic!("got {part:?}")
            };
            assert_eq!(media_type, expected, "for {raw}");
        }
    }

    #[test]
    fn a_document_accepts_the_provider_shaped_source_dict() {
        let part: LmPart = serde_json::from_value(json!({
            "type": "document",
            "source": { "type": "text", "media_type": "text/plain", "data": "the contract" },
            "citations": { "enabled": true },
            "title": "Contract",
        }))
        .expect("a document part");

        let LmPart::Document { source, title, .. } = &part else {
            panic!("got {part:?}")
        };
        assert_eq!(*title, Some("Contract".to_owned()));
        let DocumentSource::Source(dict) = source else {
            panic!("expected the dict form, got {source:?}")
        };
        assert_eq!(dict["data"], json!("the contract"));
    }

    #[test]
    fn a_tool_result_holds_parts_of_its_own() {
        let part = LmPart::ToolResult {
            call_id: Some("call_1".to_owned()),
            name: Some("search".to_owned()),
            content: vec![LmPart::text("42")],
            is_error: false,
            provider_data: Metadata::new(),
            metadata: Metadata::new(),
        };
        let written = serde_json::to_value(&part).expect("serializes");
        assert_eq!(written["content"][0]["text"], json!("42"));
        assert_eq!(
            serde_json::from_value::<LmPart>(written).expect("round-trips"),
            part
        );
    }

    #[test]
    fn a_carried_block_does_not_read_as_text() {
        let block = json!({ "type": "wildcard_v9", "payload": [1, 2] });
        let part = LmPart::legacy(block.clone());
        assert_eq!(part.as_text(), None);
        assert_eq!(part.legacy_block(), Some(&block));
        assert_eq!(LmPart::text("real prose").as_text(), Some("real prose"));
    }
}
