//! Generated-media output items as typed parts — dspy's `output_{image,audio,file}_to_part`.
//!
//! A provider that can generate media (OpenAI's Responses API, Gemini) returns it as an output item;
//! this reads the source the way dspy does — base64 first (a `data:` uri split into its media type
//! and payload), then a url, then a file id — so a program that asked for an image reads it back as a
//! first-class [`Image`](api::LmPart::Image), [`Audio`](api::LmPart::Audio) or binary part.

use serde_json::Value;

use crate::lm::api::{self, Detail, LmSource, Metadata};

/// An image, audio or file output item as its typed part, or `None` for anything else — so the reply
/// parser can offer any output item here and keep only the media ones.
pub(super) fn part(item_type: &str, item: &Value) -> Option<api::LmPart> {
    match item_type {
        "image" | "output_image" | "image_generation_call" | "image_url" => image(item),
        "audio" | "output_audio" | "input_audio" => audio(item),
        "file" | "output_file" | "input_file" => file(item),
        _ => None,
    }
}

/// A media source: base64 data first — a `data:` uri overriding the media type — then a url, then a
/// file id. The default media type stands when none is split out; no source at all renders nothing.
fn source(
    b64: Option<&str>,
    url: Option<&str>,
    file_id: Option<&str>,
    default_media: &str,
) -> Option<(LmSource, String)> {
    if let Some(data) = b64 {
        let (media_type, data) = match data.starts_with("data:") {
            true => split_data_uri(data),
            false => (default_media.to_owned(), data.to_owned()),
        };
        return Some((LmSource::Data(data), media_type));
    }
    if let Some(url) = url {
        return Some((LmSource::Url(url.to_owned()), default_media.to_owned()));
    }
    file_id.map(|id| (LmSource::FileId(id.to_owned()), default_media.to_owned()))
}

/// dspy's `split_data_uri`: `data:<media_type>;base64,<data>` → its media type and payload; anything
/// that is not a data uri is left whole under a generic media type.
fn split_data_uri(value: &str) -> (String, String) {
    match value
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(','))
    {
        Some((header, data)) => (
            header.split(';').next().unwrap_or_default().to_owned(),
            data.to_owned(),
        ),
        None => ("application/octet-stream".to_owned(), value.to_owned()),
    }
}

fn image(item: &Value) -> Option<api::LmPart> {
    let url = item["image_url"]
        .as_str()
        .or_else(|| item["image_url"]["url"].as_str())
        .or_else(|| item["url"].as_str());
    // `result` is where OpenAI's real `image_generation_call` carries its base64 — the field litellm
    // reads and dspy 3.3's own parser misses (it reads only b64_json/data), so this follows the wire.
    let b64 = item["b64_json"]
        .as_str()
        .or_else(|| item["data"].as_str())
        .or_else(|| item["result"].as_str());
    let media = media_type_of(item, "image/png");
    let (source, media_type) = source(b64, url, item["file_id"].as_str(), &media)?;
    let detail = serde_json::from_value::<Option<Detail>>(item["detail"].clone()).unwrap_or(None);
    Some(api::LmPart::Image {
        source,
        media_type,
        detail,
        metadata: Metadata::new(),
    })
}

fn audio(item: &Value) -> Option<api::LmPart> {
    let audio = if item["audio"].is_object() {
        &item["audio"]
    } else {
        item
    };
    let b64 = audio["data"]
        .as_str()
        .or_else(|| audio["b64_json"].as_str());
    let media = media_type_of(audio, "audio/wav");
    let (source, media_type) = source(
        b64,
        audio["url"].as_str(),
        audio["file_id"].as_str(),
        &media,
    )?;
    Some(api::LmPart::Audio {
        source,
        media_type,
        metadata: Metadata::new(),
    })
}

fn file(item: &Value) -> Option<api::LmPart> {
    let file = if item["file"].is_object() {
        &item["file"]
    } else {
        item
    };
    let b64 = file["file_data"].as_str().or_else(|| file["data"].as_str());
    let file_id = file["file_id"].as_str().or_else(|| file["id"].as_str());
    let media = media_type_of(file, "application/octet-stream");
    let (source, media_type) = source(b64, file["url"].as_str(), file_id, &media)?;
    let filename = file["filename"].as_str().map(str::to_owned);
    Some(api::LmPart::Binary {
        source,
        media_type,
        filename,
        metadata: Metadata::new(),
    })
}

/// The media type a media item names under either spelling, or the type's own default.
fn media_type_of(item: &Value, default: &str) -> String {
    item["media_type"]
        .as_str()
        .or_else(|| item["mime_type"].as_str())
        .unwrap_or(default)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The real OpenAI Responses image-generation output item carries its base64 under `result` — the
    /// shape litellm reads and dspy 3.3's own parser misses (it reads b64_json/data/url/file_id and
    /// would raise on the real item). Grounded on litellm's `OutputImageGenerationCall`, not the dspy
    /// oracle, because dspy cannot produce this and so cannot be the reference for it.
    #[test]
    fn a_real_image_generation_call_reads_its_base64_from_result() {
        let item = json!({ "type": "image_generation_call", "status": "completed", "result": "iVBORw0KGgo=" });
        let Some(api::LmPart::Image {
            source: LmSource::Data(data),
            media_type,
            ..
        }) = part("image_generation_call", &item)
        else {
            panic!("expected an image part with base64 data")
        };
        assert_eq!(data, "iVBORw0KGgo=");
        assert_eq!(media_type, "image/png");
    }

    /// A data-url `result` splits into its media type and payload, as litellm's
    /// `_extract_base64_from_data_url` does.
    #[test]
    fn a_data_url_result_splits_into_media_type_and_payload() {
        let item =
            json!({ "type": "image_generation_call", "result": "data:image/webp;base64,UklGRg==" });
        let Some(api::LmPart::Image {
            source: LmSource::Data(data),
            media_type,
            ..
        }) = part("image_generation_call", &item)
        else {
            panic!("expected an image part")
        };
        assert_eq!(data, "UklGRg==");
        assert_eq!(media_type, "image/webp");
    }

    /// A type this crate does not model contributes no part — the reply stays valid as providers add
    /// output item types, the raw kept in `provider_output` for a caller that needs it. This is more
    /// forgiving than dspy, whose image parser raises on a shape it cannot read.
    #[test]
    fn an_unknown_output_item_type_contributes_no_part() {
        assert_eq!(
            part(
                "some_future_item",
                &json!({ "type": "some_future_item", "payload": 1 })
            ),
            None
        );
    }
}
