//! dspy `adapters/types/image.py`: the `Image` type.

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Value, json};

use super::base::{Formatted, Type, serialized};

/// dspy's `Image`: an image by URL or base64 data URI, rendered as an `image_url` content block.
///
/// dspy's constructor also accepts raw bytes, a PIL image, or a remote URL to download — each
/// encoded to a data URI first. Those are Python objects a Rust caller does not hold; here the
/// value is the `url` (an `http(s)`/`gs` URL, a local path, or a `data:` URI) as given.
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
    /// **Diverges in the payload, and this one cannot be closed.** `_encode_pil_image` *re-encodes*
    /// through PIL and emits what PIL wrote, so upstream's base64 is not the caller's bytes: for
    /// the one-pixel PNG in upstream's own test the two are the same length and different content.
    /// Matching that would mean reproducing PIL's zlib settings, which is a worse implementation of
    /// a worse idea — re-encoding is lossy for a JPEG and pointless for the rest. These are the
    /// caller's bytes, under the media type upstream would have named.
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

/// The media type raw bytes announce about themselves, by the signature every one of these formats
/// opens with — what PIL's `Image.open` does before it decodes anything.
///
/// Named as PIL names them: `image/{format.lower()}`, which is what `_encode_pil_image` builds.
fn sniffed(bytes: &[u8]) -> Option<&'static str> {
    let starts = |signature: &[u8]| bytes.starts_with(signature);
    match () {
        _ if starts(b"\x89PNG\r\n\x1a\n") => Some("image/png"),
        _ if starts(b"\xff\xd8\xff") => Some("image/jpeg"),
        _ if starts(b"GIF87a") || starts(b"GIF89a") => Some("image/gif"),
        // RIFF containers name their payload at byte 8; only the WEBP one is an image.
        _ if starts(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") => Some("image/webp"),
        _ if starts(b"BM") => Some("image/bmp"),
        _ if starts(b"II\x2a\x00") || starts(b"MM\x00\x2a") => Some("image/tiff"),
        _ => None,
    }
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
    /// **The payload diverges and cannot be made not to**: `_encode_pil_image` re-encodes, so
    /// dspy's base64 for upstream's own one-pixel PNG is a different 68 bytes from the input's.
    /// These are the caller's bytes. What is held is the part that reaches the provider as meaning
    /// — the media type — and upstream's own test asserts no more than that prefix either.
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
