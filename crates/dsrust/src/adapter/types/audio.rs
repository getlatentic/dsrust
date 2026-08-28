//! dspy `adapters/types/audio.py`: the `Audio` type.

use serde::{Deserialize, Serialize, Serializer};
use serde_json::json;

use super::base::{Formatted, Type, serialized};

/// dspy's `Audio`: base64 audio data and its format, rendered as an `input_audio` content block.
///
/// dspy also builds one from a URL, a file, or a numpy array — each read and base64-encoded. Those
/// are Python-side sources; here the value is the already-encoded `data` and its `audio_format`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Audio {
    pub data: String,
    pub audio_format: String,
}

/// The map form, which is what upstream hands to pydantic once `encode_audio` has agreed to it.
#[derive(Deserialize)]
struct Encoded {
    data: String,
    audio_format: String,
}

impl<'de> Deserialize<'de> for Audio {
    /// dspy `encode_audio`, which `validate_input` runs before pydantic sees the value.
    ///
    /// A `mode="before"` validator runs on every construction, parsing a reply included, so the
    /// shapes it takes are part of the wire contract rather than a Python-side convenience. A
    /// derive read only the map, and a data URI — the one string form upstream accepts — came back
    /// as `invalid type: string, expected struct Audio`.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let value = serde_json::Value::deserialize(deserializer)?;
        match &value {
            serde_json::Value::String(source) => {
                Self::parse(source).map_err(|error| D::Error::custom(format!("{error}")))
            }
            // Presence of both keys is what upstream tests, leaving the field types to pydantic —
            // so a map carrying only one of them is not a malformed `Audio` but an unsupported
            // value, and says so.
            serde_json::Value::Object(fields)
                if fields.contains_key("data") && fields.contains_key("audio_format") =>
            {
                let encoded = Encoded::deserialize(&value).map_err(D::Error::custom)?;
                Ok(Self {
                    data: encoded.data,
                    audio_format: encoded.audio_format,
                })
            }
            other => Err(D::Error::custom(format!(
                "Unsupported type for encode_audio: {}",
                crate::python::type_of(other)
            ))),
        }
    }
}

impl Audio {
    /// Already-encoded base64 and the format it is in — upstream's `Audio(data=…, audio_format=…)`.
    ///
    /// Unchecked, and measured to be so: dspy's keyword form takes whatever it is handed as the
    /// payload, where its *positional* form validates. The distinction is real — this is the "I
    /// hold the encoded bytes" door, and there is nothing to validate against. A string of unknown
    /// shape goes through [`parse`](Self::parse), which is upstream's positional form and refuses
    /// a locator.
    pub fn new(data: impl Into<String>, audio_format: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            audio_format: audio_format.into(),
        }
    }

    /// dspy `encode_audio`, string branch: a `data:` URI, split into the payload and its format.
    ///
    /// **Refuses anything else.** Audio has no reference form — a provider is handed the bytes, not
    /// a URL — so unlike an image there is no string a caller could mean other than a data URI, and
    /// a path or a URL here is a caller reaching for the wrong door.
    pub fn parse(source: impl AsRef<str>) -> anyhow::Result<Self> {
        let source = source.as_ref();
        let Some(rest) = source.strip_prefix("data:audio/") else {
            anyhow::bail!(
                "String audio inputs must be data URIs. Load local files with Audio.from_path() \
                 and remote resources with Audio.from_url()."
            );
        };
        // `data:audio/<format>[;base64],<payload>` — the format is what precedes the first `;`,
        // and upstream calls a URI with neither separator malformed rather than guessing.
        let Some((header, payload)) = rest.split_once(',') else {
            anyhow::bail!("Malformed audio data URI");
        };
        let format = header.split_once(';').map_or(header, |(format, _)| format);
        if format.is_empty() {
            anyhow::bail!("Malformed audio data URI");
        }
        Ok(Self::new(payload, normalized_format(format)))
    }

    /// dspy `Audio(bytes, audio_format=…)`: raw bytes the caller already holds.
    ///
    /// No sniffing, because the caller names the format — audio containers are not as reliably
    /// self-describing as images, and upstream requires the name here for the same reason.
    pub fn from_bytes(bytes: impl AsRef<[u8]>, audio_format: impl Into<String>) -> Self {
        Self::new(crate::resource::encode(bytes.as_ref()), audio_format)
    }

    /// dspy `Audio.from_array`: samples, at a sampling rate, as a 16-bit PCM WAV.
    ///
    /// Upstream reaches `soundfile` for this, whose default is exactly a canonical mono RIFF with
    /// `PCM_16` — no vendor chunks, no padding — which is why it is reproducible here from
    /// primitives rather than by binding a codec. The bytes are pinned against `soundfile`'s own
    /// output in `tests/conformance/constants/wav_pcm16.json`.
    ///
    /// `f32` because that is what libsndfile converts from and what the measurement was taken in:
    /// a sample is scaled by 32768 and clamped, so `1.0` lands on `32767` rather than wrapping.
    pub fn from_samples(samples: &[f32], sampling_rate: u32) -> Self {
        Self::from_samples_as(samples, sampling_rate, Container::Wav)
            .expect("WAV is written from primitives and cannot fail")
    }

    /// The same, into a named container — dspy's `format=` on `from_array`.
    ///
    /// Both containers are given the *same* 16-bit samples, converted by libsndfile's own rule, so
    /// a FLAC and a WAV of one recording decode to identical values. That is the property worth
    /// having: which container a caller picks is a size and provider-support question, never a
    /// question about the audio.
    pub fn from_samples_as(
        samples: &[f32],
        sampling_rate: u32,
        container: Container,
    ) -> anyhow::Result<Self> {
        let pcm: Vec<i16> = samples.iter().map(|sample| pcm16(*sample)).collect();
        let encoded = match container {
            Container::Wav => wav_pcm16(&pcm, sampling_rate),
            Container::Flac => flac_pcm16(&pcm, sampling_rate)?,
            #[cfg(feature = "mp3")]
            Container::Mp3 => mp3_pcm16(&pcm, sampling_rate)?,
        };
        Ok(Self::new(
            crate::resource::encode(&encoded),
            container.name(),
        ))
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

/// What samples are written into — dspy's `format=`, which it hands to libsndfile.
///
/// Two rather than libsndfile's twenty-odd, and the cut is where the C toolchain starts. WAV is
/// written here from primitives and is byte-identical to libsndfile's. FLAC is `flacenc`, which is
/// pure Rust. Ogg Vorbis and MP3 are the two other formats a provider actually accepts and both
/// only exist in Rust as bindings — `vorbis_rs` over libvorbis, `mp3lame-encoder` over LAME — so
/// each would put a C build in front of everyone who depends on this crate. Named here so that
/// stays a decision rather than an omission.
///
/// The libsndfile bindings would have been the faithful route, since upstream's `soundfile` is
/// itself a binding to it and the bytes would agree exactly. The `sndfile` crate last shipped in
/// April 2022 and `sndfile-sys` in July 2021, so that route is closed on maintenance rather than
/// on principle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Container {
    /// 16-bit PCM in a RIFF container — libsndfile's default, and upstream's.
    Wav,
    /// Lossless, and roughly half of what the same samples cost as WAV.
    Flac,
    /// Lossy, and the only other container OpenAI accepts for audio input.
    ///
    /// Behind the **`mp3`** feature, off by default: the encoder is LAME through
    /// `mp3lame-encoder`, which is LGPL-3.0 where this crate is MIT OR Apache-2.0. Linking it
    /// brings LGPL obligations to whoever ships the binary, so the choice belongs to them and not
    /// to this crate. It also wants a C toolchain.
    #[cfg(feature = "mp3")]
    Mp3,
}

impl Container {
    /// The bare name an `input_audio` block carries, which is what a provider reads.
    fn name(self) -> &'static str {
        match self {
            Container::Wav => "wav",
            Container::Flac => "flac",
            #[cfg(feature = "mp3")]
            Container::Mp3 => "mp3",
        }
    }
}

/// libsndfile's float-to-PCM rule, derived from its own output: scale by 32768, floor, clamp.
///
/// Shared so every container gets the same samples. `as i16` truncates toward zero and `round` goes
/// to nearest, and each disagrees with libsndfile on three of the sixteen values in the golden's
/// sweep — -0.99998 floors to -32768 where truncation gives -32767. The clamp is only ever reached
/// by 1.0, which would otherwise scale to 32768 and wrap to -32768: the loudest sample becoming the
/// quietest.
fn pcm16(sample: f32) -> i16 {
    (sample * 32768.0).floor().clamp(-32768.0, 32767.0) as i16
}

/// One mono FLAC at 16 bits a sample.
fn flac_pcm16(pcm: &[i16], sampling_rate: u32) -> anyhow::Result<Vec<u8>> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let widened: Vec<i32> = pcm.iter().map(|sample| i32::from(*sample)).collect();
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|error| anyhow::anyhow!("flac encoder config: {error:?}"))?;
    let source = flacenc::source::MemSource::from_samples(&widened, 1, 16, sampling_rate as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|error| anyhow::anyhow!("flac encode: {error:?}"))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|error| anyhow::anyhow!("flac write: {error:?}"))?;
    Ok(sink.as_slice().to_vec())
}

/// One mono MP3 through LAME, at its default quality.
///
/// Lossy, so unlike WAV and FLAC the samples that come back are near the ones that went in rather
/// than equal to them. That is the format, not the binding — and it is why this is opt-in beside
/// its licence: a caller reaching for MP3 has a provider that demands it, not a preference.
#[cfg(feature = "mp3")]
fn mp3_pcm16(pcm: &[i16], sampling_rate: u32) -> anyhow::Result<Vec<u8>> {
    use mp3lame_encoder::{Builder, FlushNoGap, MonoPcm};

    let mut builder = Builder::new().ok_or_else(|| anyhow::anyhow!("lame would not start"))?;
    builder
        .set_num_channels(1)
        .map_err(|error| anyhow::anyhow!("lame channels: {error}"))?;
    builder
        .set_sample_rate(sampling_rate)
        .map_err(|error| anyhow::anyhow!("lame sample rate: {error}"))?;
    let mut encoder = builder
        .build()
        .map_err(|error| anyhow::anyhow!("lame build: {error}"))?;

    let mut out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(pcm.len()));
    let written = encoder
        .encode(MonoPcm(pcm), out.spare_capacity_mut())
        .map_err(|error| anyhow::anyhow!("lame encode: {error}"))?;
    // SAFETY: `encode` reports how many bytes of the spare capacity it initialised.
    unsafe { out.set_len(written) };
    let flushed = encoder
        .flush::<FlushNoGap>(out.spare_capacity_mut())
        .map_err(|error| anyhow::anyhow!("lame flush: {error}"))?;
    // SAFETY: as above, for the frames the flush emitted.
    unsafe { out.set_len(out.len() + flushed) };
    Ok(out)
}

/// One mono 16-bit PCM WAV: the 44-byte canonical RIFF header, then the samples.
///
/// libsndfile writes exactly this and nothing else — measured, not assumed. Every multi-byte field
/// is little-endian, which is what `RIFF` (as against `RIFX`) declares.
fn wav_pcm16(pcm: &[i16], sampling_rate: u32) -> Vec<u8> {
    const HEADER: u32 = 36;
    const CHANNELS: u16 = 1;
    const BITS: u16 = 16;
    let payload = (pcm.len() * 2) as u32;

    let mut wav = Vec::with_capacity(HEADER as usize + 8 + payload as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(HEADER + payload).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // the PCM fmt chunk is 16 bytes
    wav.extend_from_slice(&1u16.to_le_bytes()); // 1 = uncompressed PCM
    wav.extend_from_slice(&CHANNELS.to_le_bytes());
    wav.extend_from_slice(&sampling_rate.to_le_bytes());
    wav.extend_from_slice(
        &(sampling_rate * u32::from(CHANNELS) * u32::from(BITS / 8)).to_le_bytes(),
    );
    wav.extend_from_slice(&(CHANNELS * BITS / 8).to_le_bytes());
    wav.extend_from_slice(&BITS.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&payload.to_le_bytes());
    for sample in pcm {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
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

    /// The keyword door takes what it is handed, and dspy's does too — measured:
    /// `dspy.Audio(data="/etc/passwd", audio_format="wav")` keeps the string as the payload. There
    /// is nothing to validate against here; the *positional* form is [`Audio::parse`], and that one
    /// refuses.
    #[test]
    fn the_keyword_door_keeps_what_it_is_given_as_upstreams_does() {
        assert_eq!(Audio::new("/etc/passwd", "wav").data, "/etc/passwd");
    }

    /// dspy's positional string form: a data URI, split into payload and format, and nothing else
    /// accepted. Measured — `dspy.Audio("data:audio/x-wav;base64,AA==")` gives `AA==` and `wav`,
    /// the `x-` stripped by `_normalize_audio_format`.
    #[test]
    fn a_data_uri_splits_into_its_payload_and_format() {
        let audio = Audio::parse("data:audio/x-wav;base64,AA==").expect("parses");
        assert_eq!(audio, Audio::new("AA==", "wav"));
        // No `;base64` is still a data URI; the format is what precedes the first `;` or the comma.
        assert_eq!(
            Audio::parse("data:audio/mp3,QQ==")
                .expect("parses")
                .audio_format,
            "mp3"
        );
    }

    /// A locator is refused, in upstream's words. Audio has no reference form — the provider is
    /// handed bytes — so a path or a URL here is the wrong door rather than a value to keep.
    #[test]
    fn a_locator_is_refused_and_points_at_the_right_factory() {
        for locator in ["clip.wav", "https://example.com/a.wav", "/etc/passwd"] {
            let why = Audio::parse(locator).expect_err("refused").to_string();
            assert_eq!(
                why,
                "String audio inputs must be data URIs. Load local files with Audio.from_path() \
                 and remote resources with Audio.from_url().",
                "for {locator}"
            );
        }
        // A `data:audio/` prefix with no comma names no payload at all.
        assert_eq!(
            Audio::parse("data:audio/wav")
                .expect_err("refused")
                .to_string(),
            "Malformed audio data URI"
        );
    }

    /// Bytes the caller holds, base64'd under the format they named — measured against
    /// `dspy.Audio(b"audio bytes", audio_format="wav")`, which gives `YXVkaW8gYnl0ZXM=` and `wav`.
    #[test]
    fn bytes_are_encoded_under_the_format_the_caller_named() {
        assert_eq!(
            Audio::from_bytes(b"audio bytes", "wav"),
            Audio::new("YXVkaW8gYnl0ZXM=", "wav")
        );
    }

    /// Samples as a 16-bit PCM WAV, byte for byte against `soundfile`'s own output.
    ///
    /// This is the numpy branch of `encode_audio`, and it is reproducible from primitives because
    /// libsndfile's default WAV is a canonical RIFF header and nothing else. The edge cases are the
    /// point: `1.0` must land on 32767, where scaling by 32768 and casting would wrap it to
    /// -32768 — the loudest possible sample becoming the quietest, which is a click no test of the
    /// header would catch.
    #[test]
    fn samples_are_written_as_the_wav_soundfile_writes() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/conformance/constants/wav_pcm16.json");
        let text = std::fs::read_to_string(&path).expect("the wav golden is committed");
        let golden: serde_json::Value = serde_json::from_str(&text).expect("the golden parses");
        let cases = golden["cases"].as_object().expect("cases");
        assert!(!cases.is_empty(), "the golden records no cases");

        for (name, case) in cases {
            let samples: Vec<f32> = case["samples"]
                .as_array()
                .expect("samples")
                .iter()
                .map(|sample| sample.as_f64().expect("a number") as f32)
                .collect();
            let rate = case["sampling_rate"].as_u64().expect("a rate") as u32;
            let audio = Audio::from_samples(&samples, rate);
            assert_eq!(
                audio.data,
                case["base64"].as_str().expect("base64"),
                "for {name}"
            );
            assert_eq!(audio.audio_format, "wav", "for {name}");
        }
    }

    /// FLAC is lossless, so the samples it decodes to must be *exactly* the ones WAV carries.
    ///
    /// Decoded by a second implementation — `claxon`, a dev-dependency — rather than by the encoder
    /// that wrote it, so this asserts what a provider will get. The comparison is against the WAV
    /// path's own PCM, which is the property that matters: which container a caller picks is a size
    /// question, never a question about the audio.
    #[test]
    fn flac_carries_exactly_the_samples_wav_does() {
        let samples: Vec<f32> = (0..2048)
            .map(|step| (step as f32 * 0.05).sin() * 0.8)
            .chain([1.0, -1.0, 0.0, -1e-5])
            .collect();

        let flac = Audio::from_samples_as(&samples, 44100, Container::Flac).expect("encodes");
        assert_eq!(flac.audio_format, "flac");

        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&flac.data)
            .expect("base64");
        let mut reader = claxon::FlacReader::new(std::io::Cursor::new(bytes)).expect("a flac");
        assert_eq!(reader.streaminfo().channels, 1);
        assert_eq!(reader.streaminfo().bits_per_sample, 16);
        assert_eq!(reader.streaminfo().sample_rate, 44100);
        let decoded: Vec<i32> = reader.samples().map(|s| s.expect("a sample")).collect();

        let expected: Vec<i32> = samples.iter().map(|s| i32::from(pcm16(*s))).collect();
        assert_eq!(decoded, expected, "FLAC is lossless or it is not FLAC");

        // And smaller than the WAV of the same audio, which is the only reason to pick it.
        let wav = Audio::from_samples(&samples, 44100);
        assert!(
            flac.data.len() < wav.data.len(),
            "flac {} vs wav {}",
            flac.data.len(),
            wav.data.len()
        );
    }

    /// MP3 is lossy, so the claim is bounded rather than exact: real frames, the right duration,
    /// and audio that still resembles what went in.
    ///
    /// Decoded by symphonia — a second implementation again — because the failure this guards
    /// against is silence. A LAME call that is never flushed returns bytes that look like an MP3,
    /// decode without error, and contain nothing; only listening to the samples catches it.
    #[cfg(feature = "mp3")]
    #[test]
    fn mp3_decodes_back_to_audio_that_resembles_what_went_in() {
        use symphonia::core::codecs::audio::AudioDecoderOptions;
        use symphonia::core::formats::probe::Hint;
        use symphonia::core::formats::{FormatOptions, TrackType};
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;

        // A second of a steady tone, which is easy to compare against and impossible to fake.
        let rate = 44100u32;
        let samples: Vec<f32> = (0..rate)
            .map(|step| (step as f32 / rate as f32 * 440.0 * std::f32::consts::TAU).sin() * 0.5)
            .collect();

        let audio = Audio::from_samples_as(&samples, rate, Container::Mp3).expect("encodes");
        assert_eq!(audio.audio_format, "mp3");

        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&audio.data)
            .expect("base64");
        assert!(!bytes.is_empty(), "an unflushed encoder returns nothing");

        let stream =
            MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
        let mut hint = Hint::new();
        hint.with_extension("mp3");
        let mut format = symphonia::default::get_probe()
            .probe(
                &hint,
                stream,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .expect("what was written probes as an mp3");
        let track = format
            .default_track(TrackType::Audio)
            .expect("an audio track");
        let track_id = track.id;
        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(
                track
                    .codec_params
                    .as_ref()
                    .expect("codec parameters")
                    .audio()
                    .expect("audio parameters"),
                &AudioDecoderOptions::default(),
            )
            .expect("a decoder");

        let mut decoded: Vec<f32> = Vec::new();
        while let Ok(Some(packet)) = format.next_packet() {
            if packet.track_id != track_id {
                continue;
            }
            if let Ok(buffer) = decoder.decode(&packet) {
                let mut interleaved = Vec::new();
                buffer.copy_to_vec_interleaved(&mut interleaved);
                decoded.extend_from_slice(&interleaved);
            }
        }

        // LAME pads the start, so the length is near rather than equal — but it must be *audio*,
        // and roughly a second of it.
        assert!(
            decoded.len() as f32 > rate as f32 * 0.9,
            "decoded {} samples of an expected {rate}",
            decoded.len()
        );
        // The tone survived: a silent or garbage decode fails this and a correct one passes easily.
        let peak = decoded.iter().fold(0f32, |peak, s| peak.max(s.abs()));
        assert!(
            (0.3..=0.7).contains(&peak),
            "peak {peak} — the 0.5 tone did not survive"
        );
    }
}
