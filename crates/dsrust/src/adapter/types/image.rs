//! dspy `adapters/types/image.py`: the `Image` type.

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Value, json};

use super::base::{Formatted, Type, serialized};

/// dspy's `Image`: an image by URL or base64 data URI, rendered as an `image_url` content block.
///
/// The value is always a `url` — a reference a provider resolves, or a `data:` URI carrying the
/// bytes — and every way of arriving at one is a named constructor rather than a constructor that
/// guesses. [`new`](Self::new) takes a reference, [`from_bytes`](Self::from_bytes) an encoded
/// image, [`from_rgb`](Self::from_rgb) and [`from_rgba`](Self::from_rgba) decoded pixels,
/// [`from_path`](Self::from_path) a local file and [`from_url`](Self::from_url) a remote one. Only
/// the last two touch anything outside the process, which is upstream's 3.3.0 split and the reason
/// a value that merely looks like a path is refused rather than read.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Image {
    pub url: String,
}

impl Image {
    /// dspy `encode_image`, string branch: a remote reference, or an already-encoded data URI.
    ///
    /// **Refuses anything else**, including a local path, and says which factory to use instead.
    /// That is stronger than "does not fetch it", and the strength is the point: a constructor is
    /// reachable from application input, so one that quietly kept `/etc/passwd` would put a local
    /// path in an `image_url` block and send it to a provider. Upstream refuses; so does this.
    ///
    /// A URL is kept as a reference for the provider to resolve and a `data:` URI as itself.
    /// Neither is dereferenced — reading is [`from_path`](Self::from_path), downloading is
    /// [`from_url`](Self::from_url), and both are asked for by name.
    pub fn new(source: impl AsRef<str>) -> anyhow::Result<Self> {
        let source = source.as_ref();
        if source.starts_with("data:") || is_url(source) {
            return Ok(Self::reference(source));
        }
        anyhow::bail!(
            "Unrecognized image string: {source}. Local files must be loaded with Image.from_path()."
        )
    }

    /// dspy `encode_image`, bytes branch: raw image bytes as a `data:` URI.
    ///
    /// The format is read off the bytes themselves rather than from a filename, because there is no
    /// filename — upstream hands them to PIL for the same reason.
    ///
    /// **The payload diverges, deliberately.** Upstream reaches PIL to *identify* these bytes and
    /// then, because it already has a PIL object, re-encodes through it — so what reaches the model
    /// is what PIL wrote rather than what the caller passed. That is not free, and the cost is
    /// measured rather than argued:
    ///
    ///   - a JPEG saved at quality 95 comes back re-encoded at PIL's default 75 — 5446 bytes to
    ///     2837, with a maximum channel difference of 69 out of 255. Visible degradation of an
    ///     image the caller had already encoded the way they wanted it;
    ///   - a three-frame animated GIF comes back as a one-frame still, 219 bytes to 95. `Image.save`
    ///     writes one frame unless asked for `save_all`, and upstream does not ask.
    ///
    /// So matching upstream byte for byte would mean reproducing a data loss. These are the
    /// caller's bytes, unmodified, under the media type upstream would have named — which is the
    /// part a provider reads as meaning. Identification is the only job PIL is really doing on this
    /// branch, and `image::guess_format` does it without decoding.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> anyhow::Result<Self> {
        let bytes = bytes.as_ref();
        let Some(media_type) = sniffed(bytes) else {
            anyhow::bail!(
                "Bytes could not be identified as an image: {} bytes",
                bytes.len()
            );
        };
        Ok(Self::reference(&crate::resource::data_uri(
            media_type,
            &crate::resource::encode(bytes),
        )))
    }

    /// dspy `Image(pil_image)`: decoded pixels, encoded as a PNG and embedded.
    ///
    /// This is the branch PIL is genuinely load-bearing for, and it is **not** deprecated —
    /// `Image(pil_image)` is what 3.3.0 points `Image.from_PIL` at. A caller holding pixels rather
    /// than a file (a render, a chart, a captured frame) has upstream's exact need, and the
    /// counterpart to a PIL object is a decoded buffer.
    ///
    /// Three bytes a pixel, row-major and tightly packed. PNG because that is what
    /// `_encode_pil_image` falls back to for an image with no format of its own, which an in-memory
    /// one never has.
    ///
    /// The bytes are the `image` crate's, not PIL's, and no two PNG encoders agree — filter choice
    /// and deflate settings are an encoder's own. What holds is that the pixels survive, which is
    /// the round trip asserted in the tests.
    pub fn from_rgb(width: u32, height: u32, pixels: &[u8]) -> anyhow::Result<Self> {
        Self::encoded(
            image::RgbImage::from_raw(width, height, pixels.to_vec()),
            3,
            width,
            height,
            pixels.len(),
        )
    }

    /// The same with an alpha channel: four bytes a pixel.
    pub fn from_rgba(width: u32, height: u32, pixels: &[u8]) -> anyhow::Result<Self> {
        Self::encoded(
            image::RgbaImage::from_raw(width, height, pixels.to_vec()),
            4,
            width,
            height,
            pixels.len(),
        )
    }

    /// One decoded buffer as a PNG data URI, or why the buffer was not one.
    ///
    /// `from_raw` answers `None` for a length that does not match the dimensions, and that is the
    /// whole of the validation — a buffer half the size it claims would otherwise be encoded as an
    /// image of whatever it happened to contain.
    fn encoded<P, C>(
        buffer: Option<image::ImageBuffer<P, C>>,
        samples: usize,
        width: u32,
        height: u32,
        given: usize,
    ) -> anyhow::Result<Self>
    where
        P: image::Pixel<Subpixel = u8> + image::PixelWithColorType,
        C: std::ops::Deref<Target = [u8]>,
    {
        let wanted = width as usize * height as usize * samples;
        let Some(buffer) = buffer else {
            anyhow::bail!(
                "{width}x{height} at {samples} bytes a pixel needs {wanted} bytes, given {given}"
            );
        };
        let mut png = Vec::new();
        buffer.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
        Ok(Self::reference(crate::resource::data_uri(
            "image/png",
            &crate::resource::encode(&png),
        )))
    }

    /// A value already known to be a reference or a data URI — the factories' own way back in,
    /// which must not re-run a check the value has already passed.
    fn reference(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }

    /// dspy `Image.from_path`: read a local file and encode it as a `data:` URI.
    ///
    /// The media type is guessed from the suffix and defaults to `image/png`, as upstream's does
    /// when `mimetypes` knows nothing. The guess is CPython's *shipped* table rather than the one
    /// `mimetypes.guess_type` builds, which merges `/etc/mime.types` over it — so dspy's own answer
    /// depends on the host for a handful of suffixes and this crate's does not.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let media_type = crate::resource::media_type_for(path, "image/png");
        let encoded = crate::resource::read_base64(path)?;
        Ok(Self::reference(crate::resource::data_uri(
            &media_type,
            &encoded,
        )))
    }

    /// dspy `Image.from_url`: download the image and embed it as a `data:` URI.
    ///
    /// The name is the whole API. 3.3.0 made this the *only* way to fetch — `Image(url)` keeps a
    /// reference for the provider to resolve, and nothing reachable from parsing a model's reply
    /// can reach here. A caller asking for a download is asking on purpose.
    ///
    /// **No SSRF protection**, which is upstream's position stated in upstream's words: it follows
    /// redirects and will reach loopback, private and cloud-metadata hosts. A URL derived from
    /// untrusted input is the caller's to allowlist before calling this.
    ///
    /// The media type is what the server said; where it said nothing, the URL's suffix, and a URL
    /// whose suffix names nothing is an error rather than a guess.
    pub async fn from_url(url: impl AsRef<str>) -> anyhow::Result<Self> {
        Self::downloaded(url.as_ref(), true).await
    }

    /// The same, without checking the TLS certificate — upstream's `verify=False`, for a host with
    /// a self-signed one. Named rather than a `bool`, because a `false` at a call site says nothing
    /// about what it switches off.
    pub async fn from_url_unverified(url: impl AsRef<str>) -> anyhow::Result<Self> {
        Self::downloaded(url.as_ref(), false).await
    }

    async fn downloaded(url: &str, verify: bool) -> anyhow::Result<Self> {
        if !crate::resource::is_http_url(url) {
            anyhow::bail!("Image.from_url requires an HTTP(S) URL, received: {url}");
        }
        let (content_type, encoded) = crate::resource::fetch_base64(url, verify).await?;
        let media_type = content_type
            .or_else(|| crate::mimetypes::guess(url).map(str::to_owned))
            .ok_or_else(|| anyhow::anyhow!("Could not determine MIME type for URL: {url}"))?;
        Ok(Self::reference(crate::resource::data_uri(
            &media_type,
            &encoded,
        )))
    }
}

/// dspy `is_url`: a scheme a provider can resolve, and a host to resolve it against.
///
/// Three schemes, not two — `gs://` is Google Cloud Storage, which Gemini reads directly. Wider
/// than [`is_http_url`](crate::resource::is_http_url), which guards *fetching* and so must not
/// admit a scheme this process would have to interpret itself.
fn is_url(source: &str) -> bool {
    let Some((scheme, rest)) = source.split_once("://") else {
        return false;
    };
    matches!(scheme, "http" | "https" | "gs")
        && !rest.split(['/', '?', '#']).next().unwrap_or("").is_empty()
}

/// The media type raw bytes announce about themselves — what PIL's `Image.open` reads before it
/// decodes anything, here `image::guess_format`.
///
/// The `image` crate rather than a hand-written signature table: it knows every format it can name
/// rather than the six worth typing out, and identification is exactly the job PIL is doing on this
/// path. Its answers are `image/{format}`, which is the shape `_encode_pil_image` builds too.
fn sniffed(bytes: &[u8]) -> Option<&'static str> {
    image::guess_format(bytes)
        .ok()
        .map(|format| format.to_mime_type())
}

impl Type for Image {
    /// dspy `Image.format`: one `image_url` block carrying the URL.
    fn format(&self) -> Formatted {
        Formatted::Blocks(vec![
            json!({ "type": "image_url", "image_url": { "url": self.url } }),
        ])
    }
}

impl Serialize for Image {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&serialized(self))
    }
}

impl<'de> Deserialize<'de> for Image {
    /// dspy accepts a bare URL string or the legacy `{"url": ...}` mapping.
    ///
    /// A mapping carrying `download` or `verify` is **refused**, which is upstream's 3.3.0 rule and
    /// the reason this path exists at all. Those two are a deprecated *direct-construction* shim —
    /// `Image(url, download=True)` — and honouring them here would mean a value like
    /// `{"url": "http://169.254.169.254/…", "download": true}` fetching a host's cloud-metadata
    /// endpoint while an LM's output was being parsed. Ignoring them silently is not the answer
    /// either: a caller who wrote `download` believes the image was embedded and is handed a bare
    /// reference instead, and the day this crate grows a `download` field the silence becomes a
    /// fetch. So it is an error, in upstream's words.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::String(url) => Self::new(url).map_err(de::Error::custom),
            Value::Object(mut map) => match map.remove("url") {
                _ if map.contains_key("download") || map.contains_key("verify") => {
                    Err(de::Error::custom(
                        "`download` and `verify` are only valid with a positional image source; \
                         use Image.from_url(url, verify=...) to download a remote image.",
                    ))
                }
                Some(Value::String(url)) => Self::new(url).map_err(de::Error::custom),
                _ => Err(de::Error::custom(
                    "`url` field is required for `dspy.Image`",
                )),
            },
            other => Err(de::Error::custom(format!(
                "Received invalid value for `dspy.Image`: {other}"
            ))),
        }
    }
}

impl schemars::JsonSchema for Image {
    /// The serialized form is a string — the sentinel-wrapped block — so an output field carries a
    /// string's schema.
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Image".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        String::json_schema(generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::types::base::{CUSTOM_TYPE_END, CUSTOM_TYPE_START};

    /// `format` is one `image_url` block, and the serialized value wraps it in the sentinels so the
    /// render's string round trip can split it back into a content part.
    #[test]
    fn it_renders_as_a_sentinel_wrapped_image_block() {
        let image = Image::new("https://example.com/a.jpg").expect("a reference");
        assert_eq!(
            image.format(),
            Formatted::Blocks(vec![
                json!({ "type": "image_url", "image_url": { "url": "https://example.com/a.jpg" } })
            ])
        );
        assert_eq!(
            serde_json::to_value(&image).expect("serializes"),
            json!(format!(
                "{CUSTOM_TYPE_START}{}{CUSTOM_TYPE_END}",
                r#"[{"type":"image_url","image_url":{"url":"https://example.com/a.jpg"}}]"#
            ))
        );
    }

    /// Against dspy's own answer for the same bytes: `Image.from_path` on a `.png` holding
    /// `image bytes` gives `data:image/png;base64,aW1hZ2UgYnl0ZXM=`.
    #[test]
    fn from_path_encodes_the_bytes_under_the_media_type_dspy_names() {
        let path = std::env::temp_dir().join("dsrs_image_from_path.png");
        std::fs::write(&path, b"image bytes").expect("writes");
        assert_eq!(
            Image::from_path(&path).expect("reads").url,
            "data:image/png;base64,aW1hZ2UgYnl0ZXM=",
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The rule is stronger than "does not fetch it": a local path is **refused**.
    ///
    /// Measured — `dspy.Image("/etc/passwd")` raises `ValueError`, and so does
    /// `dspy.Image(url="/etc/passwd")`. An earlier version of this test asserted the path was
    /// *kept*, which was a divergence written down as the expected answer: keeping it puts a local
    /// path inside an `image_url` block and sends it to a provider.
    #[test]
    fn a_local_path_is_refused_rather_than_kept() {
        for locator in ["/etc/passwd", "clip.png", "file:///etc/passwd"] {
            let why = Image::new(locator).expect_err("refused").to_string();
            assert_eq!(
                why,
                format!(
                    "Unrecognized image string: {locator}. \
                     Local files must be loaded with Image.from_path()."
                ),
                "for {locator}"
            );
        }
    }

    /// A reference the provider resolves is kept and not fetched — including `gs://`, which dspy's
    /// `is_url` admits alongside http(s) because Gemini reads Cloud Storage directly.
    #[test]
    fn a_provider_resolvable_reference_is_kept_untouched() {
        for locator in [
            "https://evil.example/secret.png",
            "http://example.com/a.png",
            "gs://bucket/a.png",
            "data:image/png;base64,QQ==",
        ] {
            assert_eq!(Image::new(locator).expect("kept").url, locator);
        }
    }

    /// Raw bytes name their own format — upstream reaches PIL, this reads the signature.
    ///
    /// **The payload diverges on purpose**: upstream re-encodes through PIL, which costs a JPEG its
    /// quality (measured: 5446 bytes to 2837, max channel delta 69) and an animated GIF every frame
    /// but the first (three frames to one). The byte-for-byte assertion below is the whole claim —
    /// these bytes were not touched. Upstream's own test asserts no more than the media-type prefix.
    #[test]
    fn bytes_are_identified_by_signature_and_kept_as_given() {
        let png = b"\x89PNG\r\n\x1a\n and then some";
        let image = Image::from_bytes(png).expect("identified");
        assert!(
            image.url.starts_with("data:image/png;base64,"),
            "{}",
            image.url
        );
        assert_eq!(
            image.url,
            format!("data:image/png;base64,{}", crate::resource::encode(png)),
            "the caller's bytes, not a re-encoding"
        );

        for (bytes, expected) in [
            (b"\xff\xd8\xff\xe0rest".to_vec(), "image/jpeg"),
            (b"GIF89a rest".to_vec(), "image/gif"),
            (b"RIFF\x00\x00\x00\x00WEBPrest".to_vec(), "image/webp"),
            (b"BM rest".to_vec(), "image/bmp"),
            (b"II\x2a\x00rest".to_vec(), "image/tiff"),
        ] {
            let image = Image::from_bytes(&bytes).expect("identified");
            assert!(
                image.url.starts_with(&format!("data:{expected};base64,")),
                "{expected}: {}",
                image.url
            );
        }
    }

    /// Decoded pixels survive the encode — the whole claim, since the bytes cannot match PIL's.
    ///
    /// Closed in Rust rather than against Python: the same library that encodes will decode, so
    /// this asserts what a provider will get rather than what the encoder believed it wrote.
    #[test]
    fn pixels_round_trip_through_the_png_they_are_encoded_as() {
        let rgb: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9];
        let image = Image::from_rgb(2, 2, &rgb).expect("encodes");
        let payload = image
            .url
            .strip_prefix("data:image/png;base64,")
            .expect("a png data uri");
        let read = decoded(payload);
        assert_eq!((read.width(), read.height()), (2, 2));
        assert_eq!(read.to_rgb8().into_raw(), rgb);

        // Alpha has to survive too, which is the reason the two constructors are separate: RGB
        // would silently drop it, and a transparent image would arrive opaque.
        let rgba: Vec<u8> = vec![255, 0, 0, 128, 0, 255, 0, 0, 0, 0, 255, 255, 1, 2, 3, 4];
        let image = Image::from_rgba(2, 2, &rgba).expect("encodes");
        let payload = image
            .url
            .strip_prefix("data:image/png;base64,")
            .expect("a png data uri");
        assert_eq!(decoded(payload).to_rgba8().into_raw(), rgba);
    }

    fn decoded(payload: &str) -> image::DynamicImage {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(payload)
            .expect("the payload is base64");
        image::load_from_memory(&bytes).expect("what was written decodes")
    }

    /// A buffer that does not match the dimensions is refused rather than encoded as whatever it
    /// happened to hold — the one thing `from_raw` checks, and the one a caller gets wrong.
    #[test]
    fn a_buffer_that_does_not_match_its_dimensions_is_refused() {
        let why = Image::from_rgb(4, 4, &[0; 12])
            .expect_err("refused")
            .to_string();
        assert_eq!(why, "4x4 at 3 bytes a pixel needs 48 bytes, given 12");
        assert!(
            Image::from_rgba(2, 2, &[0; 12]).is_err(),
            "RGBA needs four a pixel"
        );
    }

    /// Bytes that are not an image are refused, where upstream's PIL raises
    /// `UnidentifiedImageError` and it is re-raised as a `ValueError`. A RIFF container that is not
    /// a WEBP is the interesting one — it shares four opening bytes with one.
    #[test]
    fn bytes_that_are_not_an_image_are_refused() {
        for bytes in [
            b"not an image".to_vec(),
            b"RIFF\x00\x00\x00\x00WAVEfmt ".to_vec(),
            Vec::new(),
        ] {
            let why = Image::from_bytes(&bytes).expect_err("refused").to_string();
            assert!(
                why.starts_with("Bytes could not be identified as an image:"),
                "{why}"
            );
        }
    }

    /// 3.3.0's breaking change, from the side an attacker reaches: parsing a model's output must
    /// not be able to ask the host to fetch something. dspy raises for a mapping carrying either
    /// key, in these words, and the payload in its own test is the AWS metadata endpoint.
    ///
    /// The crate accepted both and ignored them, which is safer than fetching and still wrong: a
    /// caller who wrote `download` is handed a bare reference and told nothing.
    #[test]
    fn a_mapping_asking_to_download_is_refused_rather_than_ignored() {
        for payload in [
            json!({ "url": "http://169.254.169.254/latest/meta-data", "download": true }),
            json!({ "url": "https://example.com/a.png", "verify": false }),
        ] {
            let why = serde_json::from_value::<Image>(payload.clone())
                .expect_err("refused")
                .to_string();
            assert!(
                why.starts_with(
                    "`download` and `verify` are only valid with a positional image source;"
                ),
                "for {payload}: {why}"
            );
        }
        // The reference form is still exactly what it was: no fetch, no complaint.
        let plain: Image =
            serde_json::from_value(json!({ "url": "https://example.com/a.png" })).expect("parses");
        assert_eq!(plain.url, "https://example.com/a.png");
    }

    #[test]
    fn it_reads_a_bare_url_or_a_url_mapping() {
        let bare: Image =
            serde_json::from_value(json!("data:image/png;base64,AAAA")).expect("parses");
        assert_eq!(bare.url, "data:image/png;base64,AAAA");
        let mapped: Image =
            serde_json::from_value(json!({ "url": "https://example.com/a.png" })).expect("parses");
        assert_eq!(mapped.url, "https://example.com/a.png");
        // The mapping form is validated too, which is where an untrusted payload arrives:
        // `TypeAdapter(Image).validate_python({"url": "/etc/passwd"})` raises upstream.
        assert!(serde_json::from_value::<Image>(json!({ "url": "/etc/passwd" })).is_err());
        assert!(serde_json::from_value::<Image>(json!({ "no_url": 1 })).is_err());
        assert!(serde_json::from_value::<Image>(json!(3)).is_err());
    }
}
