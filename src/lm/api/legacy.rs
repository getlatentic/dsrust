//! Provider-shaped blocks read back as typed parts — dspy's `_legacy_content_block_to_lm_part`.
//!
//! The inverse of [`wire`](super::wire). A custom type still writes OpenAI-shaped JSON, so this
//! is what lets that reach the typed model without either side losing anything.

use serde_json::Value;

use super::part::{LmPart, LmSource, Metadata};

/// One block as the part it describes, or carried whole when it describes nothing known.
pub fn part_of_block(block: &Value) -> LmPart {
    let Some(object) = block.as_object() else {
        return LmPart::text(crate::adapter::python_json::format_value(block));
    };
    match object.get("type").and_then(Value::as_str) {
        Some("text") => LmPart::text(text_of(object)),
        Some("image_url") => image(object),
        Some("input_audio") => audio(object),
        Some("document") => document(object),
        // A file keeps its block: upstream re-emits that one verbatim rather than rebuilding it,
        // since `file_data` and `file_id` do not survive the trip through a media source.
        _ => LmPart::legacy(block.clone()),
    }
}

fn text_of(object: &serde_json::Map<String, Value>) -> String {
    object
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn image(object: &serde_json::Map<String, Value>) -> LmPart {
    let url = object
        .get("image_url")
        .and_then(|image| match image {
            Value::Object(image) => image.get("url").and_then(Value::as_str),
            other => other.as_str(),
        })
        .unwrap_or_default();
    let (media_type, source) = match data_uri(url) {
        Some((media_type, data)) => (media_type, LmSource::Data(data)),
        None => ("image/png".to_owned(), LmSource::Url(url.to_owned())),
    };
    LmPart::Image {
        source,
        media_type,
        detail: None,
        metadata: Metadata::new(),
    }
}

fn audio(object: &serde_json::Map<String, Value>) -> LmPart {
    let audio = object.get("input_audio").and_then(Value::as_object);
    let read = |key: &str| {
        audio
            .and_then(|audio| audio.get(key))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let format = match read("format").as_str() {
        "" => "wav".to_owned(),
        named => named.to_owned(),
    };
    LmPart::Audio {
        source: LmSource::Data(read("data")),
        media_type: format!("audio/{format}"),
        metadata: Metadata::new(),
    }
}

fn document(object: &serde_json::Map<String, Value>) -> LmPart {
    let source = match object.get("source") {
        Some(Value::Object(source)) => source.clone(),
        other => {
            let mut described = Metadata::new();
            described.insert("type".to_owned(), Value::String("text".to_owned()));
            let data = other.map(crate::adapter::python_json::format_value);
            described.insert("data".to_owned(), Value::String(data.unwrap_or_default()));
            described
        }
    };
    let citations = match object.get("citations").and_then(Value::as_object) {
        Some(citations) => citations.clone(),
        None => {
            let mut enabled = Metadata::new();
            enabled.insert("enabled".to_owned(), Value::Bool(true));
            enabled
        }
    };
    LmPart::Document {
        source: super::part::DocumentSource::Source(source),
        media_type: "application/pdf".to_owned(),
        citations,
        title: string_at(object, "title"),
        context: string_at(object, "context"),
        metadata: Metadata::new(),
    }
}

fn string_at(object: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    Some(object.get(key)?.as_str()?.to_owned())
}

/// `data:image/png;base64,AAAA` split into its media type and its payload.
fn data_uri(value: &str) -> Option<(String, String)> {
    let rest = value.strip_prefix("data:")?;
    let (media_type, data) = rest.split_once(',')?;
    let media_type = media_type.strip_suffix(";base64").unwrap_or(media_type);
    Some((media_type.to_owned(), data.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::super::wire::blocks_of;
    use super::*;
    use serde_json::json;

    /// The property that matters: whatever a custom type wrote comes back out unchanged. Every
    /// one of these is a shape dspy 3.2.1 actually emits.
    #[test]
    fn every_block_a_custom_type_writes_survives_the_round_trip() {
        let blocks = [
            json!({ "type": "text", "text": "plain" }),
            json!({ "type": "image_url", "image_url": { "url": "https://example.com/a.jpg" } }),
            json!({ "type": "input_audio", "input_audio": { "data": "YQ==", "format": "wav" } }),
            json!({ "type": "file", "file": { "file_id": "f_1", "filename": "a.pdf" } }),
            json!({ "type": "file", "file": { "file_data": "data:application/pdf;base64,YQ==" } }),
            json!({ "type": "wildcard_v9", "payload": { "k": [1, 2] } }),
        ];
        for block in blocks {
            let part = part_of_block(&block);
            assert_eq!(
                blocks_of(&part).expect("renders"),
                vec![block.clone()],
                "for {block}"
            );
        }
    }

    #[test]
    fn an_inline_image_keeps_its_media_type_rather_than_defaulting() {
        let block = json!({
            "type": "image_url",
            "image_url": { "url": "data:image/jpeg;base64,YQ==" },
        });
        let part = part_of_block(&block);
        let LmPart::Image {
            source, media_type, ..
        } = &part
        else {
            panic!("got {part:?}")
        };
        assert_eq!(*source, LmSource::Data("YQ==".to_owned()));
        assert_eq!(media_type, "image/jpeg", "read off the data uri");
        assert_eq!(blocks_of(&part).expect("renders"), vec![block]);
    }

    #[test]
    fn a_document_keeps_its_source_citations_and_title() {
        let block = json!({
            "type": "document",
            "source": { "type": "text", "media_type": "text/plain", "data": "the contract" },
            "citations": { "enabled": true },
            "title": "Contract",
        });
        assert_eq!(
            blocks_of(&part_of_block(&block)).expect("renders"),
            vec![block]
        );
    }

    #[test]
    fn a_block_that_is_not_even_an_object_becomes_its_text() {
        assert_eq!(part_of_block(&json!("bare")).as_text(), Some("bare"));
    }
}
