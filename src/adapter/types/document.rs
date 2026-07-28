//! dspy `adapters/types/document.py`: the `Document` type.

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value, json};

use crate::signature::TypeDescription;

use super::base::{Formatted, Type, serialized};

/// The content types a [`Document`] may carry. dspy spells it `Literal["text/plain",
/// "application/pdf"]`, so a value outside the pair is refused rather than passed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub enum MediaType {
    #[default]
    #[serde(rename = "text/plain")]
    PlainText,
    #[serde(rename = "application/pdf")]
    Pdf,
}

impl MediaType {
    pub fn as_str(self) -> &'static str {
        match self {
            MediaType::PlainText => "text/plain",
            MediaType::Pdf => "application/pdf",
        }
    }
}

/// dspy's `Document`: source material a model may quote and cite.
///
/// Rendered as the `document` content block a citation-enabled provider reads, with citations
/// switched on — which is what lets the reply come back carrying
/// [`Citations`](super::citation::Citations).
#[derive(Debug, Clone, PartialEq, Eq, Default, schemars::JsonSchema)]
pub struct Document {
    pub data: String,
    pub title: Option<String>,
    pub media_type: MediaType,
    pub context: Option<String>,
}

impl Document {
    /// Plain-text source material.
    pub fn new(data: impl Into<String>) -> Self {
        Self { data: data.into(), ..Self::default() }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_media_type(mut self, media_type: MediaType) -> Self {
        self.media_type = media_type;
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

impl Type for Document {
    /// dspy `Document.format`: one `document` block carrying the source and its citation switch,
    /// then the title and context where they are set — the order upstream builds the mapping in.
    fn format(&self) -> Formatted {
        let mut block = Map::new();
        block.insert("type".to_owned(), json!("document"));
        block.insert(
            "source".to_owned(),
            json!({ "type": "text", "media_type": self.media_type.as_str(), "data": self.data }),
        );
        block.insert("citations".to_owned(), json!({ "enabled": true }));
        // dspy tests each for Python truth, so an empty string is dropped as well as a missing one.
        if let Some(title) = self.title.as_deref().filter(|title| !title.is_empty()) {
            block.insert("title".to_owned(), json!(title));
        }
        if let Some(context) = self.context.as_deref().filter(|context| !context.is_empty()) {
            block.insert("context".to_owned(), json!(context));
        }
        Formatted::Blocks(vec![Value::Object(block)])
    }

    fn description() -> Option<TypeDescription> {
        Some(TypeDescription {
            name: "Document".to_owned(),
            text: "A document containing text content that can be referenced and cited. \
                   Include the full text content and optionally a title for proper referencing."
                .to_owned(),
            replaces_schema: false,
        })
    }
}

impl Serialize for Document {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&serialized(self))
    }
}

impl<'de> Deserialize<'de> for Document {
    /// dspy `Document.validate_input`: a bare string is the content, and a mapping is read as the
    /// document's fields.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let take = |fields: &mut Map<String, Value>, key: &str| match fields.remove(key) {
            Some(Value::String(value)) => Some(value),
            _ => None,
        };
        match Value::deserialize(deserializer)? {
            Value::String(data) => Ok(Self::new(data)),
            Value::Object(mut fields) => {
                let media_type = match fields.remove("media_type") {
                    Some(media_type) => {
                        serde_json::from_value(media_type).map_err(de::Error::custom)?
                    }
                    None => MediaType::default(),
                };
                let data = take(&mut fields, "data")
                    .ok_or_else(|| de::Error::custom("`data` field is required for `Document`"))?;
                Ok(Self {
                    data,
                    title: take(&mut fields, "title"),
                    media_type,
                    context: take(&mut fields, "context"),
                })
            }
            other => Err(de::Error::custom(format!(
                "Received invalid value for `Document`: {other}"
            ))),
        }
    }
}

impl std::fmt::Display for Document {
    /// dspy's `__str__`: the title, where there is one, and how much content the document holds.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let title = match self.title.as_deref() {
            Some(title) => format!("'{title}': "),
            None => String::new(),
        };
        // dspy counts Python characters, which are code points rather than bytes.
        write!(formatter, "Document({title}{} chars)", self.data.chars().count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_renders_a_document_block_with_citations_enabled() {
        let Formatted::Blocks(blocks) = Document::new("The sky is blue.").format() else {
            panic!("a document renders as a block");
        };
        assert_eq!(
            serde_json::to_string(&blocks[0]).expect("serializes"),
            r#"{"type":"document","source":{"type":"text","media_type":"text/plain","data":"The sky is blue."},"citations":{"enabled":true}}"#
        );
    }

    /// The title and context follow the required keys, and the media type reaches the source block.
    #[test]
    fn its_optional_fields_follow_the_required_ones() {
        let document = Document::new("A report.")
            .with_title("Weather")
            .with_context("From the archive")
            .with_media_type(MediaType::Pdf);
        let Formatted::Blocks(blocks) = document.format() else {
            panic!("a document renders as a block");
        };
        assert_eq!(
            serde_json::to_string(&blocks[0]).expect("serializes"),
            r#"{"type":"document","source":{"type":"text","media_type":"application/pdf","data":"A report."},"citations":{"enabled":true},"title":"Weather","context":"From the archive"}"#
        );
    }

    #[test]
    fn it_reads_a_bare_string_or_a_mapping_and_refuses_the_rest() {
        let bare: Document = serde_json::from_value(json!("Just text")).expect("parses");
        assert_eq!(bare, Document::new("Just text"));
        let mapped: Document =
            serde_json::from_value(json!({ "data": "Body", "title": "T", "media_type": "application/pdf" }))
                .expect("parses");
        assert_eq!(mapped, Document::new("Body").with_title("T").with_media_type(MediaType::Pdf));
        assert!(serde_json::from_value::<Document>(json!({ "title": "no data" })).is_err());
        // Outside the pair dspy's `Literal` allows.
        assert!(serde_json::from_value::<Document>(json!({ "data": "x", "media_type": "text/html" })).is_err());
        assert!(serde_json::from_value::<Document>(json!(3)).is_err());
    }

    #[test]
    fn its_string_form_names_the_title_and_counts_the_content() {
        assert_eq!(Document::new("abcd").to_string(), "Document(4 chars)");
        assert_eq!(Document::new("abcd").with_title("T").to_string(), "Document('T': 4 chars)");
    }
}
