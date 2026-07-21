//! What a message is made of, once content stops being a string.
//!
//! dspy 3.3 replaces a message's `str | list[dict]` content with a typed hierarchy of eleven
//! parts discriminated on `type`. The gain is that a caller can read what a message holds — an
//! image's media type, a tool call's arguments, whether a reply was a refusal — where a `dict`
//! only ever offered `get("type")` and a guess.
//!
//! Nothing here reaches a provider. [`content_of`] converts a part back to the OpenAI-shaped block
//! that 3.2.1 already sent, which is what lets this type land without moving a single rendered
//! byte.

mod wire;

use std::path::PathBuf;

use serde_json::{Map, Value};

pub use wire::{Content, blocks_of, content_of};

/// A part's free-form annotations. Upstream's `metadata: dict[str, Any]`.
pub type Metadata = Map<String, Value>;

/// The key under which a provider-shaped block rides along verbatim.
///
/// A custom type writes JSON this crate has no type for, and upstream's answer is not to guess
/// at it: the block is parked here whole and handed back untouched at render time. It is why a
/// part tree can carry content nobody has modelled and still produce the exact bytes that
/// content arrived as.
pub const LEGACY_BLOCK: &str = "legacy_content_block";

/// Where a part's bytes come from — exactly one place.
///
/// Upstream spells this as four nullable fields and a `validate_one_source` validator that
/// raises when the count is not one. That validator exists because Python cannot say "exactly
/// one" in a type; Rust can, so the invalid state is unrepresentable rather than rejected at
/// run time, and the error upstream raises has no way to occur.
/// In memory that is the whole of it. Arriving as JSON it still has to be checked, because four
/// nullable keys can carry any number of sources — so deserializing counts them and refuses
/// anything but one, which is `_validate_one_source` doing its job at the only place the
/// invalid state can still appear.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "SourceFields", into = "SourceFields")]
pub enum LmSource {
    /// Base64, which is what a provider is handed for a local file.
    Data(String),
    Url(String),
    FileId(String),
    /// Read and encoded when the part is rendered, not when it is built.
    Path(PathBuf),
}

/// The four nullable source keys as they travel, which is the shape upstream validates.
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
        let [only] = <[Self; 1]>::try_from(named)
            .map_err(|named| format!("expected exactly one of data, url, file_id, or path, got {}", named.len()))?;
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
    /// Upstream rejects an empty source as well as a missing one, since a provider given `""`
    /// fails further away from the mistake.
    fn is_empty(&self) -> bool {
        match self {
            Self::Data(value) | Self::Url(value) | Self::FileId(value) => value.is_empty(),
            Self::Path(path) => path.as_os_str().is_empty(),
        }
    }
}

/// How much of an image a provider should look at. Upstream's
/// `Literal["low", "high", "auto"] | None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Detail {
    Low,
    High,
    Auto,
}

/// One item of a message's content.
///
/// Internally tagged on `type`, which is upstream's `Field(discriminator="type")` exactly: the
/// tag is the variant, so a part that arrives naming a type this crate does not know fails to
/// parse rather than becoming a silently empty one.
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
    /// Source material rather than an attachment: a report, a contract, a PDF to cite from.
    ///
    /// Deliberately not built on [`LmSource`] the way the media parts are. Upstream declares it
    /// on `LMBasePart`, not `LMSourcePart`, because it accepts *either* one media source or a
    /// provider-shaped `source` dict — a different rule than "exactly one of four", and making
    /// it a media sibling would impose a constraint upstream does not have.
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
    /// What a tool answered, which is itself content — hence the recursion.
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
    /// A model's reasoning, which some providers return and some redact.
    Thinking {
        text: String,
        #[serde(default, skip_serializing_if = "is_false")]
        redacted: bool,
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Metadata,
    },
    /// At least one of the three is always present — upstream raises when all three are absent,
    /// which is the one validator here that a type cannot express, since "not all empty" over
    /// three independent options has no shape to encode it in.
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

/// A document's origin: one media source, or the provider-shaped dict upstream also accepts.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentSource {
    /// Upstream's `source` dict, which it requires to be non-empty when given.
    Source(Metadata),
    #[serde(untagged)]
    Media(LmSource),
}

impl LmPart {
    /// Prose, which is what all but the multimodal fields ever produce.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            metadata: Metadata::new(),
        }
    }

    /// A provider-shaped block this crate does not model, kept whole so it renders back to
    /// itself. Upstream parks the same thing on an empty text part.
    pub fn legacy(block: Value) -> Self {
        let mut metadata = Metadata::new();
        metadata.insert(LEGACY_BLOCK.to_owned(), block);
        Self::Text {
            text: String::new(),
            metadata,
        }
    }

    /// An image at a URL, the one source that needs no encoding to send.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: LmSource::Url(url.into()),
            media_type: image_media_type(),
            detail: None,
            metadata: Metadata::new(),
        }
    }

    /// The prose of a text part, and `None` for everything else — including a text part standing
    /// in for a block it is carrying, whose text is empty and means nothing.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text, metadata } if !metadata.contains_key(LEGACY_BLOCK) => Some(text),
            _ => None,
        }
    }

    /// The block this part is carrying on behalf of a type nobody modelled.
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
    fn a_part_is_discriminated_by_its_type_the_way_upstream_discriminates_it() {
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

    /// Upstream's `validate_one_source` raises when two sources are set. Here there is nowhere
    /// to put the second one, so the state the validator exists to reject cannot be built — and
    /// a payload carrying both fails to parse rather than picking a winner.
    #[test]
    fn two_sources_are_not_a_thing_that_can_be_expressed() {
        let both = serde_json::from_value::<LmPart>(json!({
            "type": "image",
            "url": "https://example.com/a.jpg",
            "data": "aGk=",
        }));
        assert!(both.is_err(), "got {both:?}");
    }

    /// A part naming a type nobody knows is an error, not a part with its fields dropped — which
    /// is what `Field(discriminator="type")` buys upstream and what an untagged enum would lose.
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

    /// A document takes either a media source or a provider-shaped dict, which is why it is not
    /// built on `LmSource` the way its media siblings are.
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

    /// A tool's answer is content in its own right, so the type recurses where upstream's does.
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

    /// The empty-text carrier is not prose. Reading it as prose would put an empty string where
    /// a provider-shaped block belongs, which is a silent content drop.
    #[test]
    fn a_carried_block_does_not_read_as_text() {
        let block = json!({ "type": "wildcard_v9", "payload": [1, 2] });
        let part = LmPart::legacy(block.clone());
        assert_eq!(part.as_text(), None, "not prose");
        assert_eq!(part.legacy_block(), Some(&block));
        assert_eq!(LmPart::text("real prose").as_text(), Some("real prose"));
    }
}
