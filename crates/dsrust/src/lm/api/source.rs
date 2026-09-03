//! What a part names as its content: exactly one of a few places, and never two.
//!
//! Split from [`LmPart`](super::LmPart) because the question is different. A part is *what kind* of
//! content this is; a source is *where it came from*, and both of these enums exist to make "two at
//! once" unrepresentable — upstream needs `validate_one_source` and `validate_source` for the same
//! guarantee, because Python cannot say "exactly one" in a type.

use std::path::PathBuf;

use anyhow::Result;

use super::Metadata;

/// Upstream spells this as four nullable fields plus a `validate_one_source` validator, because
/// Python cannot say "exactly one" in a type. Rust can, so that state is unrepresentable rather
/// than rejected at run time.
///
/// ```
/// use dsrust::lm::api::LmSource;
///
/// // Exactly one, and the wire carries it under its own key — upstream needs a validator for the
/// // same guarantee because Python cannot say "exactly one" in a type.
/// let from_url = LmSource::Url("https://example.invalid/a.png".to_owned());
/// let sent = serde_json::to_value(&from_url).unwrap();
/// assert_eq!(sent["url"], "https://example.invalid/a.png");
/// assert!(sent.get("data").is_none(), "the other three are absent, not null");
///
/// // And a payload naming two sources is refused rather than silently preferring one.
/// let both = serde_json::json!({ "url": "https://example.invalid/a.png", "file_id": "f_1" });
/// assert!(serde_json::from_value::<LmSource>(both).is_err());
/// ```
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

/// Upstream's `validate_source`: `source` and a media source are mutually exclusive, but a payload
/// carrying `source` still spells the four media keys as nulls, so their presence is not what
/// decides — only a non-null one is.
///
/// ```
/// use dsrust::lm::api::{DocumentSource, LmSource};
///
/// // A payload carrying `source` still spells the four media keys as nulls, so their *presence*
/// // is not what decides — only a non-null one is, which is the trap in reading this shape.
/// let with_nulls = serde_json::json!({ "source": { "title": "France" }, "url": null });
/// assert!(matches!(
///     serde_json::from_value::<DocumentSource>(with_nulls),
///     Ok(DocumentSource::Source(_))
/// ));
///
/// let media = serde_json::json!({ "url": "https://example.invalid/a.pdf" });
/// assert!(matches!(
///     serde_json::from_value::<DocumentSource>(media),
///     Ok(DocumentSource::Media(LmSource::Url(_)))
/// ));
/// ```
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "DocumentFields", into = "DocumentFields")]
pub enum DocumentSource {
    Source(Metadata),
    Media(LmSource),
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct DocumentFields {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<Metadata>,
}

impl TryFrom<DocumentFields> for DocumentSource {
    type Error = String;

    fn try_from(fields: DocumentFields) -> Result<Self, Self::Error> {
        let media = SourceFields {
            data: fields.data,
            url: fields.url,
            file_id: fields.file_id,
            path: fields.path,
        };
        let has_media = LmSource::try_from(media.clone()).is_ok();
        match fields.source {
            Some(_) if has_media => {
                Err("a document takes either source or one media source, not both".to_owned())
            }
            Some(source) if source.is_empty() => {
                Err("a document's source must not be empty".to_owned())
            }
            Some(source) => Ok(Self::Source(source)),
            None => LmSource::try_from(media).map(Self::Media),
        }
    }
}

impl From<DocumentSource> for DocumentFields {
    fn from(source: DocumentSource) -> Self {
        match source {
            DocumentSource::Source(source) => Self {
                source: Some(source),
                ..Self::default()
            },
            DocumentSource::Media(media) => {
                let media = SourceFields::from(media);
                Self {
                    data: media.data,
                    url: media.url,
                    file_id: media.file_id,
                    path: media.path,
                    source: None,
                }
            }
        }
    }
}
