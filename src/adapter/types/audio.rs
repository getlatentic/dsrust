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
    pub fn new(data: impl Into<String>, audio_format: impl Into<String>) -> Self {
        Self { data: data.into(), audio_format: audio_format.into() }
    }
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
        let audio: Audio =
            serde_json::from_value(json!({ "data": "QUJD", "audio_format": "mp3" })).expect("parses");
        assert_eq!(audio, Audio::new("QUJD", "mp3"));
    }
}
