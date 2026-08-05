//! dspy `adapters/types/audio.py`: the `Audio` type.

use serde::{Deserialize, Serialize, Serializer};
use serde_json::json;

use super::base::{Formatted, Type, serialized};

/// dspy's `Audio`: base64 audio data and its format, rendered as an `input_audio` content block.
///
/// dspy also builds one from a URL, a file, or a numpy array — each read and base64-encoded. Those
/// are Python-side sources; here the value is the already-encoded `data` and its `audio_format`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct Audio {
    pub data: String,
    pub audio_format: String,
}

impl Audio {
    /// Already-encoded base64 and the format it is in. Touches nothing — see
    /// [`Image::new`](super::Image::new) for why a constructor must not dereference what it is
    /// handed, and [`from_path`](Self::from_path) for reading a local file.
    pub fn new(data: impl Into<String>, audio_format: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            audio_format: audio_format.into(),
        }
    }

    /// dspy `Audio.from_path`: read a local audio file and encode it as bare base64.
    ///
    /// Bare, not a `data:` URI — an `input_audio` block carries the payload and the format under
    /// separate keys, which makes this the one media type that does not travel as a URI.
    ///
    /// Refuses a suffix that is not audio, as upstream does: `format` reaches the provider, and
    /// guessing one for a `.txt` sends bytes no model can decode under a name saying it can.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let guessed = crate::mimetypes::guess(&path.to_string_lossy());
        let Some(media_type) = guessed.filter(|guess| guess.starts_with("audio/")) else {
            anyhow::bail!(
                "Unsupported MIME type for audio: {}",
                guessed.unwrap_or("None")
            );
        };
        Ok(Self::new(
            crate::resource::read_base64(path)?,
            normalized_format(media_type),
        ))
    }

    /// dspy `Audio.from_url`: download the audio and keep it as bare base64.
    ///
    /// See [`Image::from_url`](super::Image::from_url) for why downloading has its own name and for
    /// the SSRF position, which is upstream's and unchanged here.
    ///
    /// The format is what the server said, defaulting to `audio/wav` where it said nothing —
    /// upstream's default, and its refusal when what it said is not audio.
    pub async fn from_url(url: impl AsRef<str>) -> anyhow::Result<Self> {
        Self::downloaded(url.as_ref(), true).await
    }

    /// The same, without checking the TLS certificate — upstream's `verify=False`.
    pub async fn from_url_unverified(url: impl AsRef<str>) -> anyhow::Result<Self> {
        Self::downloaded(url.as_ref(), false).await
    }

    async fn downloaded(url: &str, verify: bool) -> anyhow::Result<Self> {
        if !crate::resource::is_http_url(url) {
            anyhow::bail!("Audio.from_url requires an HTTP(S) URL, received: {url}");
        }
        let (content_type, encoded) = crate::resource::fetch_base64(url, verify).await?;
        let media_type = content_type.unwrap_or_else(|| "audio/wav".to_owned());
        if !media_type.starts_with("audio/") {
            anyhow::bail!("Unsupported MIME type for audio: {media_type}");
        }
        Ok(Self::new(encoded, normalized_format(&media_type)))
    }
}

/// dspy `_normalize_audio_format`: the subtype, less the `x-` an unregistered spelling carries.
///
/// CPython's table calls a `.wav` file `audio/x-wav` and a provider expects `wav`, so the prefix is
/// upstream's to strip — and stripping it is what makes `from_path` on a `.wav` agree with a
/// hand-built `Audio::new(data, "wav")`.
fn normalized_format(media_type: &str) -> String {
    let subtype = media_type
        .split_once('/')
        .map_or(media_type, |(_, sub)| sub);
    subtype.strip_prefix("x-").unwrap_or(subtype).to_owned()
}

impl Type for Audio {
    /// dspy `Audio.format`: one `input_audio` block carrying the data and its format.
    fn format(&self) -> Formatted {
        Formatted::Blocks(vec![json!({
            "type": "input_audio",
            "input_audio": { "data": self.data, "format": self.audio_format },
        })])
    }
}

impl Serialize for Audio {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&serialized(self))
    }
}

impl schemars::JsonSchema for Audio {
    /// The serialized form is a string — the sentinel-wrapped block — so an output field carries a
    /// string's schema.
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Audio".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        String::json_schema(generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::base::{CUSTOM_TYPE_END, CUSTOM_TYPE_START};

    #[test]
    fn it_renders_as_a_sentinel_wrapped_input_audio_block() {
        let audio = Audio::new("QUJD", "wav");
        assert_eq!(
            audio.format(),
            Formatted::Blocks(vec![
                json!({ "type": "input_audio", "input_audio": { "data": "QUJD", "format": "wav" } })
            ])
        );
        assert_eq!(
            serde_json::to_value(&audio).expect("serializes"),
            json!(format!(
                "{CUSTOM_TYPE_START}{}{CUSTOM_TYPE_END}",
                r#"[{"type":"input_audio","input_audio":{"data":"QUJD","format":"wav"}}]"#
            ))
        );
    }

    #[test]
    fn it_reads_back_from_its_data_and_format() {
        let audio: Audio = serde_json::from_value(json!({ "data": "QUJD", "audio_format": "mp3" }))
            .expect("parses");
        assert_eq!(audio, Audio::new("QUJD", "mp3"));
    }

    /// Against dspy's own answer for the same bytes: `Audio.from_path` on a `.wav` holding
    /// `audio bytes` gives `data="YXVkaW8gYnl0ZXM="` and `audio_format="wav"` — bare base64, and
    /// the `x-` that CPython's `audio/x-wav` carries stripped back off.
    #[test]
    fn from_path_encodes_the_bytes_and_names_the_format_dspy_names() {
        let path = std::env::temp_dir().join("dsrs_audio_from_path.wav");
        std::fs::write(&path, b"audio bytes").expect("writes");
        let audio = Audio::from_path(&path).expect("reads");
        assert_eq!(audio.data, "YXVkaW8gYnl0ZXM=");
        assert_eq!(audio.audio_format, "wav");
        let _ = std::fs::remove_file(&path);
    }

    /// A suffix that is not audio is refused rather than guessed at, which is upstream's rule and
    /// its message. The alternative is a provider told `format: "plain"`.
    #[test]
    fn from_path_refuses_a_file_that_is_not_audio() {
        let path = std::env::temp_dir().join("dsrs_audio_from_path.txt");
        std::fs::write(&path, b"not audio").expect("writes");
        let why = Audio::from_path(&path).expect_err("refused").to_string();
        assert_eq!(why, "Unsupported MIME type for audio: text/plain");
        let _ = std::fs::remove_file(&path);

        let unknown = std::env::temp_dir().join("dsrs_audio_from_path.zzz");
        std::fs::write(&unknown, b"not audio").expect("writes");
        let why = Audio::from_path(&unknown).expect_err("refused").to_string();
        assert_eq!(why, "Unsupported MIME type for audio: None");
        let _ = std::fs::remove_file(&unknown);
    }

    /// The posture the whole resource-loading suite is about: a constructor is reachable from
    /// application input, so it must never dereference what it is handed. Nothing here can — the
    /// value is the encoded data — and this says so, so a later `Audio::new` that reads a path
    /// fails a test rather than a review.
    #[test]
    fn the_constructor_keeps_what_it_is_given_and_reads_nothing() {
        let locator = Audio::new("/etc/passwd", "wav");
        assert_eq!(locator.data, "/etc/passwd");
    }
}
