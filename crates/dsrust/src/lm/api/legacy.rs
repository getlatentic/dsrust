//! Provider-shaped blocks read back as typed parts — dspy's `_legacy_content_block_to_lm_part`.
//!
//! The inverse of [`wire`](super::wire). A custom type still writes OpenAI-shaped JSON, so this
//! is what lets that reach the typed model without either side losing anything.

use serde_json::Value;

use super::part::{DocumentSource, LmPart, LmSource, Metadata};

/// One block as the part it describes, or carried whole when it describes nothing known.
pub fn part_of_block(block: &Value) -> LmPart {
    let Some(object) = block.as_object() else {
        return LmPart::text(crate::adapter::python_json::format_value(block));
    };
    match object.get("type").and_then(Value::as_str) {
        Some("text") => LmPart::text(text_of(object)),
        Some("image_url") => image(object),
        Some("input_audio") => audio(object),
        Some("video") => video(object),
        Some("document") => document(object),
        Some("file") => binary(block, object),
        _ => LmPart::legacy(block.clone()),
    }
}

/// dspy `_binary_dict_to_part`: a file block as the binary part it names.
///
/// The block is kept *as well* — carried in the part's metadata, and re-emitted verbatim by
/// [`blocks_of`](super::wire::blocks_of) — whenever the typed part cannot spell it again: a part
/// holds one source, so a block naming both `file_data` and a `file_id` loses one on the way back,
/// and a `file_data` that is not a data URI would be re-wrapped under a media type nobody
/// declared. Upstream's own rule, and its reason.
///
/// **Diverges** on a file block naming no source at all, where upstream raises. `part_of_block` is
/// infallible — one of its callers deliberately keeps a malformed value visible rather than
/// dropping it — so that block is carried whole instead, which at least re-emits byte for byte.
fn binary(block: &Value, object: &serde_json::Map<String, Value>) -> LmPart {
    let file = object.get("file").and_then(Value::as_object);
    let read = |key: &str| file.and_then(|file| file.get(key)).and_then(Value::as_str);
    let filename = read("filename").map(str::to_owned);
    let part = |source, media_type: String, metadata| LmPart::Binary {
        source,
        media_type,
        filename: filename.clone(),
        metadata,
    };
    // Upstream's `_split_data_uri` is total where the crate's reader answers whether it was a data
    // URI at all: anything else is opaque bytes under the default media type.
    let split = |value: &str| {
        data_uri(value).unwrap_or_else(|| ("application/octet-stream".to_owned(), value.to_owned()))
    };

    if let Some(file_data) = read("file_data") {
        let lossy = read("file_id").is_some() || !file_data.starts_with("data:");
        let (media_type, data) = split(file_data);
        let mut metadata = Metadata::new();
        if lossy {
            metadata.insert(super::part::LEGACY_BLOCK.to_owned(), block.clone());
        }
        return part(LmSource::Data(data), media_type, metadata);
    }
    if let Some(data) = read("data") {
        let (media_type, data) = split(data);
        return part(LmSource::Data(data), media_type, Metadata::new());
    }
    if let Some(file_id) = read("file_id") {
        // No source to read a media type off, so the part keeps its own default.
        return part(
            LmSource::FileId(file_id.to_owned()),
            "application/octet-stream".to_owned(),
            Metadata::new(),
        );
    }
    LmPart::legacy(block.clone())
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

/// dspy `_audio_dict_to_part`: the format, then the first source the block actually carries.
///
/// The order matters and the fallthrough is not optional — a block naming a `url` and no `data`
/// must keep the url. Taking `data` unconditionally turned every URL-given audio into an empty
/// data source, losing the only thing that said where the audio was.
fn audio(object: &serde_json::Map<String, Value>) -> LmPart {
    let block = object.get("input_audio").and_then(Value::as_object);
    let read = |key: &str| {
        block
            .and_then(|block| block.get(key))
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
    };
    // A format already carrying a slash is the media type itself, as upstream reads it.
    let format = read("format").unwrap_or("wav");
    let media_type = match format.contains('/') {
        true => format.to_owned(),
        false => format!("audio/{format}"),
    };

    let (media_type, source) = match media_source(read("data"), media_type.clone()) {
        Some(found) => found,
        None => match (read("url"), read("file_id"), read("path")) {
            (Some(url), _, _) => (media_type, LmSource::Url(url.to_owned())),
            (_, Some(file_id), _) => (media_type, LmSource::FileId(file_id.to_owned())),
            (_, _, Some(path)) => (media_type, LmSource::Path(path.into())),
            // Upstream raises; a block with no source at all still has to become *some* part here,
            // and empty data is what the rest of the crate already reads as "nothing to send".
            _ => (media_type, LmSource::Data(String::new())),
        },
    };
    LmPart::Audio {
        source,
        media_type,
        metadata: Metadata::new(),
    }
}

/// dspy `_media_dict_to_video_part`, which reads the same four sources.
fn video(object: &serde_json::Map<String, Value>) -> LmPart {
    let block = object.get("video").and_then(Value::as_object);
    let read = |key: &str| {
        block
            .and_then(|block| block.get(key))
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
    };
    let media_type = read("media_type").unwrap_or("video/mp4").to_owned();
    let (media_type, source) = match media_source(read("data"), media_type.clone()) {
        Some(found) => found,
        None => match (read("url"), read("file_id"), read("path")) {
            (Some(url), _, _) => (media_type, LmSource::Url(url.to_owned())),
            (_, Some(file_id), _) => (media_type, LmSource::FileId(file_id.to_owned())),
            (_, _, Some(path)) => (media_type, LmSource::Path(path.into())),
            _ => (media_type, LmSource::Data(String::new())),
        },
    };
    LmPart::Video {
        source,
        media_type,
        metadata: Metadata::new(),
    }
}

/// Inline data as a source, its media type taken from a `data:` URI where the value is one.
fn media_source(data: Option<&str>, media_type: String) -> Option<(String, LmSource)> {
    let data = data?;
    Some(match data_uri(data) {
        Some((declared, decoded)) => (declared, LmSource::Data(decoded)),
        None => (media_type, LmSource::Data(data.to_owned())),
    })
}

/// dspy `_document_dict_to_part`: a source named directly on the block, else a `source` that is
/// either a mapping to keep or a *string* to classify.
///
/// The string case is the one that matters and is easy to get wrong: `"https://…/report.pdf"` is a
/// url, not the document's text. Reading it as inline text loses the only thing saying where the
/// document is.
fn document(object: &serde_json::Map<String, Value>) -> LmPart {
    let media_type =
        string_at(object, "media_type").unwrap_or_else(|| "application/pdf".to_owned());
    let described = |source, citations, media_type| LmPart::Document {
        source,
        media_type,
        citations,
        title: string_at(object, "title"),
        context: string_at(object, "context"),
        metadata: Metadata::new(),
    };

    // A source named on the block itself wins, in upstream's order.
    for (key, build) in named_sources() {
        if let Some(value) = string_at(object, key) {
            return described(
                DocumentSource::Media(build(value)),
                Metadata::new(),
                media_type,
            );
        }
    }

    match object.get("source") {
        // A mapping is kept as written, and only this path carries the block's own citations.
        Some(Value::Object(source)) => described(
            DocumentSource::Source(source.clone()),
            object
                .get("citations")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default(),
            media_type,
        ),
        Some(Value::String(source)) => {
            let (media_type, media) = classify(source, media_type);
            described(DocumentSource::Media(media), Metadata::new(), media_type)
        }
        // Upstream raises on a block with no source; an empty one is what the rest of the crate
        // already reads as nothing to send.
        _ => described(
            DocumentSource::Media(LmSource::Data(String::new())),
            Metadata::new(),
            media_type,
        ),
    }
}

/// The source keys a block may name directly, in the order upstream checks them.
fn named_sources() -> [(&'static str, fn(String) -> LmSource); 4] {
    [
        ("data", LmSource::Data),
        ("url", LmSource::Url),
        ("file_id", LmSource::FileId),
        ("path", |path| LmSource::Path(path.into())),
    ]
}

/// dspy `_media_source_kwargs`: a `data:` URI is inline data, an http(s) URL is a url whose media
/// type is guessed from its path, and anything else is a file id.
fn classify(source: &str, default_media_type: String) -> (String, LmSource) {
    if let Some((media_type, data)) = data_uri(source) {
        return (media_type, LmSource::Data(data));
    }
    match source.starts_with("http://") || source.starts_with("https://") {
        true => (
            guessed_media_type(source).unwrap_or(default_media_type),
            LmSource::Url(source.to_owned()),
        ),
        false => (default_media_type, LmSource::FileId(source.to_owned())),
    }
}

/// Python's `mimetypes.guess_type` for the handful of suffixes a document URL actually carries.
/// An unknown suffix falls back to the block's declared type, exactly as upstream's `or` does.
fn guessed_media_type(url: &str) -> Option<String> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let suffix = path.rsplit_once('.')?.1.to_ascii_lowercase();
    let media_type = match suffix.as_str() {
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "html" | "htm" => "text/html",
        "json" => "application/json",
        "csv" => "text/csv",
        "md" => "text/markdown",
        "xml" => "text/xml",
        _ => return None,
    };
    Some(media_type.to_owned())
}

fn string_at(object: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    Some(object.get(key)?.as_str()?.to_owned())
}

/// `data:image/png;base64,AAAA` split into its media type and its payload.
fn data_uri(value: &str) -> Option<(String, String)> {
    let rest = value.strip_prefix("data:")?;
    let (header, data) = rest.split_once(',')?;
    // Upstream's `_split_data_uri` keeps only what precedes the first `;`, which is the media type
    // however many parameters follow it. Stripping a `;base64` *suffix* handled the common header
    // and left `text/plain;charset=utf-8` reading as a media type with a charset glued on.
    let media_type = header.split_once(';').map_or(header, |(media, _)| media);
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

    /// A file block is read into a binary part, not carried as an opaque one.
    ///
    /// It used to fall to the carrier, which round-trips byte for byte and so looked right from
    /// the wire — but the part in between had no media type, no data and no filename, so anything
    /// reading the typed model rather than re-rendering it saw an empty text part.
    #[test]
    fn a_file_block_is_read_as_the_binary_part_it_names() {
        let part = part_of_block(&json!({
            "type": "file",
            "file": { "file_data": "data:application/pdf;base64,JVBERi0xLjQ=", "filename": "report.pdf" },
        }));
        let LmPart::Binary {
            source,
            media_type,
            filename,
            metadata,
        } = &part
        else {
            panic!("got {part:?}")
        };
        assert_eq!(*source, LmSource::Data("JVBERi0xLjQ=".to_owned()));
        assert_eq!(media_type, "application/pdf");
        assert_eq!(filename.as_deref(), Some("report.pdf"));
        // A data URI and no file id: the part spells the whole block, so nothing is carried.
        assert!(metadata.is_empty());
    }

    /// A block naming both a source and a file id keeps the block as well, because the part holds
    /// one source and would drop the other on the way back out.
    #[test]
    fn a_file_block_the_part_cannot_respell_is_carried_as_well_as_read() {
        let block = json!({
            "type": "file",
            "file": {
                "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
                "file_id": "file_123",
                "filename": "report.pdf",
            },
        });
        let part = part_of_block(&block);
        let LmPart::Binary {
            source, media_type, ..
        } = &part
        else {
            panic!("got {part:?}")
        };
        assert_eq!(*source, LmSource::Data("JVBERi0xLjQ=".to_owned()));
        assert_eq!(media_type, "application/pdf");
        assert_eq!(
            part.legacy_block(),
            Some(&block),
            "the file id is only here"
        );
        assert_eq!(blocks_of(&part).expect("renders"), vec![block]);
    }

    /// Only a file id: there is no source to read a media type off, so the default stands.
    #[test]
    fn a_file_block_of_only_an_id_keeps_the_default_media_type() {
        let part = part_of_block(&json!({
            "type": "file",
            "file": { "file_id": "f_1", "filename": "a.pdf" },
        }));
        let LmPart::Binary {
            source, media_type, ..
        } = &part
        else {
            panic!("got {part:?}")
        };
        assert_eq!(*source, LmSource::FileId("f_1".to_owned()));
        assert_eq!(media_type, "application/octet-stream");
    }

    /// The media type is what precedes the first `;`, however many parameters follow it.
    #[test]
    fn a_data_uri_parameter_does_not_become_part_of_the_media_type() {
        let part = part_of_block(&json!({
            "type": "file",
            "file": { "file_data": "data:text/plain;charset=utf-8;base64,YQ==" },
        }));
        let LmPart::Binary {
            source, media_type, ..
        } = &part
        else {
            panic!("got {part:?}")
        };
        assert_eq!(media_type, "text/plain");
        assert_eq!(*source, LmSource::Data("YQ==".to_owned()));
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
