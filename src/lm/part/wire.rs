//! Parts, back in the shape a provider actually reads.
//!
//! A typed part is this crate's own model; no provider has heard of it. dspy 3.3 keeps its wire
//! format unchanged through the same move — `clients/openai_format.py` turns each part back into
//! the OpenAI content block 3.2.1 was already sending — and that conversion is what lets the type
//! land without a single rendered byte moving.
//!
//! Two rules do all of it: a message that is one plain text part travels as a bare string rather
//! than a one-element list, and a part carrying a block it never modelled hands that block back
//! untouched.

use anyhow::{Result, bail};
use base64::Engine;
use serde_json::{Value, json};

use super::{Detail, DocumentSource, LmPart, LmSource};

/// What a turn says on the wire: prose, or the blocks a multimodal field turns it into.
///
/// dspy types a message's content as `str | list[dict]` for the same reason — an image cannot
/// reach a provider inside a string. Serializes as what it is, which is the shape every
/// OpenAI-compatible provider expects in a message's `content`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum Content {
    /// One string, which is every message a text-only signature produces.
    Text(String),
    /// Blocks in the order the provider reads them, each an OpenAI-shaped content part.
    Blocks(Vec<Value>),
}

impl Content {
    /// The prose of a text-only message, or `None` once it has been split into blocks.
    pub fn text(&self) -> Option<&str> {
        match self {
            Content::Text(text) => Some(text),
            Content::Blocks(_) => None,
        }
    }
}

impl<S: Into<String>> From<S> for Content {
    fn from(text: S) -> Self {
        Content::Text(text.into())
    }
}

/// A message's parts as its `content` field.
///
/// The bare-string case is not an optimisation — it is the shape a text-only prompt has always
/// had, and wrapping it in a one-element list would change the bytes of every ordinary call this
/// crate makes. Upstream draws the line in the same place and for the same reason, down to
/// excluding a text part that is only carrying a block.
pub fn content_of(parts: &[LmPart]) -> Result<Content> {
    if let [only] = parts
        && let Some(text) = only.as_text()
    {
        return Ok(Content::Text(text.to_owned()));
    }
    let mut blocks = Vec::new();
    for part in parts {
        blocks.extend(blocks_of(part)?);
    }
    Ok(Content::Blocks(blocks))
}

/// One part as the block or blocks a provider reads.
///
/// A carried block short-circuits everything else, which is what makes an unmodelled custom type
/// survive the round trip byte for byte.
pub fn blocks_of(part: &LmPart) -> Result<Vec<Value>> {
    if let Some(block) = part.legacy_block() {
        return Ok(vec![block.clone()]);
    }
    Ok(match part {
        LmPart::Text { text, .. } => vec![text_block(text)],
        LmPart::Image {
            source,
            media_type,
            detail,
            ..
        } => vec![image_block(source, media_type, *detail)?],
        LmPart::Audio {
            source, media_type, ..
        } => vec![audio_block(source, media_type)?],
        LmPart::Video {
            source, media_type, ..
        } => vec![json!({ "type": "video", "video": { "url": media_source(source, media_type)? } })],
        LmPart::Binary {
            source,
            media_type,
            filename,
            ..
        } => vec![binary_block(source, media_type, filename.as_deref())?],
        LmPart::Document { .. } => document_blocks(part)?,
        // A model's reasoning and a refusal are prose to every OpenAI-shaped provider; only the
        // typed side knows they were anything else.
        LmPart::Thinking { text, .. } | LmPart::Refusal { text, .. } => vec![text_block(text)],
        LmPart::Citation {
            text, title, url, ..
        } => vec![text_block(&citation_text(
            title.as_deref(),
            text.as_deref(),
            url.as_deref(),
        ))],
        LmPart::ToolResult { content, .. } => vec![text_block(&joined_text(content))],
        // A tool call is not content — it rides in the message's own `tool_calls`, so it
        // contributes no block here.
        LmPart::ToolCall { .. } => Vec::new(),
    })
}

fn text_block(text: &str) -> Value {
    json!({ "type": "text", "text": text })
}

fn image_block(source: &LmSource, media_type: &str, detail: Option<Detail>) -> Result<Value> {
    let mut image_url = json!({ "url": media_source(source, media_type)? });
    if let Some(detail) = detail {
        image_url["detail"] = json!(match detail {
            Detail::Low => "low",
            Detail::High => "high",
            Detail::Auto => "auto",
        });
    }
    Ok(json!({ "type": "image_url", "image_url": image_url }))
}

/// Audio travels as raw base64 beside a format name rather than as a data URI, which is the one
/// media block OpenAI shapes differently — and why a URL cannot carry it.
fn audio_block(source: &LmSource, media_type: &str) -> Result<Value> {
    let (data, media_type) = match source {
        LmSource::Data(data) => (data.clone(), media_type.to_owned()),
        LmSource::Path(path) => (read_base64(path)?, media_type_for(path, media_type)),
        _ => bail!("OpenAI-format audio input requires base64 `data` or a local `path`"),
    };
    Ok(json!({
        "type": "input_audio",
        "input_audio": { "data": data, "format": media_format(&media_type) },
    }))
}

fn binary_block(source: &LmSource, media_type: &str, filename: Option<&str>) -> Result<Value> {
    let mut file = serde_json::Map::new();
    match source {
        LmSource::FileId(id) => {
            file.insert("file_id".to_owned(), json!(id));
        }
        source => {
            file.insert("file_data".to_owned(), json!(media_source(source, media_type)?));
        }
    }
    if let Some(filename) = filename {
        file.insert("filename".to_owned(), json!(filename));
    }
    Ok(json!({ "type": "file", "file": file }))
}

fn document_blocks(part: &LmPart) -> Result<Vec<Value>> {
    let LmPart::Document {
        source,
        media_type,
        citations,
        title,
        context,
        ..
    } = part
    else {
        bail!("not a document part")
    };
    let mut block = json!({ "type": "document" });
    block["source"] = match source {
        DocumentSource::Source(dict) => Value::Object(dict.clone()),
        DocumentSource::Media(media) => {
            json!({ "type": "base64", "media_type": media_type, "data": media_source(media, media_type)? })
        }
    };
    if !citations.is_empty() {
        block["citations"] = Value::Object(citations.clone());
    }
    for (key, value) in [("title", title), ("context", context)] {
        if let Some(value) = value {
            block[key] = json!(value);
        }
    }
    Ok(vec![block])
}

/// Where a provider is told to find the bytes: a data URI for anything local, and the reference
/// itself for anything the provider can already address.
fn media_source(source: &LmSource, media_type: &str) -> Result<String> {
    Ok(match source {
        LmSource::Data(data) => data_uri(media_type, data),
        LmSource::Url(url) => url.clone(),
        LmSource::FileId(id) => id.clone(),
        LmSource::Path(path) => data_uri(&media_type_for(path, media_type), &read_base64(path)?),
    })
}

/// Data already spelled as a URI is left alone, so encoding one twice is not a thing that can
/// happen.
fn data_uri(media_type: &str, data: &str) -> String {
    match data.starts_with("data:") {
        true => data.to_owned(),
        false => format!("data:{media_type};base64,{data}"),
    }
}

fn read_base64(path: &std::path::Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

/// A path's own media type when its extension names one, and the part's otherwise.
fn media_type_for(path: &std::path::Path, fallback: &str) -> String {
    let Some(extension) = path.extension().and_then(|e| e.to_str()) else {
        return fallback.to_owned();
    };
    match extension.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "mp4" => "video/mp4",
        "pdf" => "application/pdf",
        _ => fallback,
    }
    .to_owned()
}

/// The bare format name an audio block carries, with the two spellings providers disagree on
/// folded the way upstream folds them.
fn media_format(media_type: &str) -> String {
    let format = media_type.split_once('/').map_or(media_type, |(_, rest)| rest);
    match format {
        "x-wav" => "wav",
        "mpeg" => "mp3",
        other => other,
    }
    .to_owned()
}

/// A citation reads as its title, its quote, then its link — whichever of the three it has.
fn citation_text(title: Option<&str>, text: Option<&str>, url: Option<&str>) -> String {
    [title, text, url]
        .into_iter()
        .flatten()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn joined_text(parts: &[LmPart]) -> String {
    parts.iter().filter_map(LmPart::as_text).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::part::Metadata;

    /// The load-bearing rule of the whole slice: an ordinary prompt is a string, not a list. Every
    /// text-only call this crate makes would change shape if this ever returned blocks.
    #[test]
    fn one_plain_text_part_travels_as_a_bare_string() {
        let content = content_of(&[LmPart::text("[[ ## question ## ]]\nWhy?")]).expect("renders");
        assert_eq!(content, Content::Text("[[ ## question ## ]]\nWhy?".to_owned()));
    }

    /// A block-carrying part is not prose even though it is spelled as a text part, so a message
    /// holding only one still renders as a list.
    #[test]
    fn a_lone_carried_block_still_renders_as_blocks() {
        let block = json!({ "type": "image_url", "image_url": { "url": "u" } });
        let content = content_of(&[LmPart::legacy(block.clone())]).expect("renders");
        assert_eq!(content, Content::Blocks(vec![block]));
    }

    /// The measured golden: what dspy 3.2.1's ChatAdapter renders for an `Image` input field,
    /// captured by running it. The parts in between are this crate's; the bytes either side are
    /// upstream's, and they have to match exactly.
    #[test]
    fn the_image_prompt_renders_the_blocks_python_dspy_renders() {
        let parts = [
            LmPart::text("[[ ## photo ## ]]\n"),
            LmPart::image_url("https://example.com/a.jpg"),
            LmPart::text("\n\nRespond with the corresponding output fields."),
        ];
        let Content::Blocks(blocks) = content_of(&parts).expect("renders") else {
            panic!("a multimodal message is blocks")
        };
        assert_eq!(
            blocks,
            vec![
                json!({ "type": "text", "text": "[[ ## photo ## ]]\n" }),
                json!({ "type": "image_url", "image_url": { "url": "https://example.com/a.jpg" } }),
                json!({ "type": "text", "text": "\n\nRespond with the corresponding output fields." }),
            ]
        );
    }

    /// An unmodelled block is handed back byte for byte. Verified against 3.3 itself, which
    /// round-trips the same wildcard through `_legacy_content_block_to_lm_part`.
    #[test]
    fn a_block_nobody_modelled_comes_back_exactly_as_it_went_in() {
        let block = json!({ "type": "wildcard_v9", "payload": { "k": [1, 2] } });
        assert_eq!(blocks_of(&LmPart::legacy(block.clone())).expect("renders"), vec![block]);
    }

    #[test]
    fn base64_data_becomes_a_data_uri_and_a_url_stays_a_url() {
        let from_data = image_block(&LmSource::Data("aGk=".to_owned()), "image/jpeg", None)
            .expect("renders");
        assert_eq!(from_data["image_url"]["url"], json!("data:image/jpeg;base64,aGk="));

        let already = image_block(
            &LmSource::Data("data:image/png;base64,aGk=".to_owned()),
            "image/png",
            None,
        )
        .expect("renders");
        assert_eq!(
            already["image_url"]["url"], json!("data:image/png;base64,aGk="),
            "an encoded URI is not encoded a second time"
        );
    }

    #[test]
    fn an_images_detail_reaches_the_block_only_when_it_was_asked_for() {
        let plain = image_block(&LmSource::Url("u".to_owned()), "image/png", None).expect("ok");
        assert_eq!(plain["image_url"].get("detail"), None);

        let detailed =
            image_block(&LmSource::Url("u".to_owned()), "image/png", Some(Detail::High)).expect("ok");
        assert_eq!(detailed["image_url"]["detail"], json!("high"));
    }

    /// Audio is the one media block that carries raw base64 and a bare format name, so a URL
    /// cannot express it — upstream raises there too rather than sending something a provider
    /// would reject.
    #[test]
    fn audio_carries_the_bare_format_name_and_refuses_a_url() {
        let block = audio_block(&LmSource::Data("YQ==".to_owned()), "audio/x-wav").expect("ok");
        assert_eq!(block["input_audio"]["format"], json!("wav"), "x-wav folds to wav");
        assert_eq!(block["input_audio"]["data"], json!("YQ=="), "not a data uri");

        assert!(audio_block(&LmSource::Url("u".to_owned()), "audio/wav").is_err());
    }

    #[test]
    fn a_citation_reads_as_title_then_quote_then_link() {
        let part = LmPart::Citation {
            text: Some("the quote".to_owned()),
            title: Some("The Paper".to_owned()),
            url: Some("https://example.com".to_owned()),
            metadata: Metadata::new(),
        };
        assert_eq!(
            blocks_of(&part).expect("renders")[0],
            json!({ "type": "text", "text": "The Paper the quote https://example.com" })
        );
    }

    /// A tool call is not content — it rides in the message's own `tool_calls` field, so putting
    /// a block here would send it twice.
    #[test]
    fn a_tool_call_contributes_no_content_block() {
        let part = LmPart::ToolCall {
            id: Some("call_1".to_owned()),
            name: "search".to_owned(),
            args: Metadata::new(),
            provider_data: Metadata::new(),
            metadata: Metadata::new(),
        };
        assert_eq!(blocks_of(&part).expect("renders"), Vec::<Value>::new());
    }
}
