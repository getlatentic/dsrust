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
    /// The image at this locator, **without touching it**.
    ///
    /// A URL is kept as a reference and a `data:` URI as itself; neither is dereferenced. That is
    /// upstream's rule and it is a security posture rather than a performance one — a constructor
    /// is reachable from application input, so one that fetched what it was handed would turn a
    /// user-supplied string into a request the host makes. Reading a local file is
    /// [`from_path`](Self::from_path), which the caller asks for by name.
    pub fn new(url: impl Into<String>) -> Self {
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
        Ok(Self::new(crate::resource::data_uri(&media_type, &encoded)))
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
        Ok(Self::new(crate::resource::data_uri(&media_type, &encoded)))
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
            Value::String(url) => Ok(Self::new(url)),
            Value::Object(mut map) => match map.remove("url") {
                _ if map.contains_key("download") || map.contains_key("verify") => {
                    Err(de::Error::custom(
                        "`download` and `verify` are only valid with a positional image source; \
                         use Image.from_url(url, verify=...) to download a remote image.",
                    ))
                }
                Some(Value::String(url)) => Ok(Self::new(url)),
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
        let image = Image::new("https://example.com/a.jpg");
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

    /// The posture the resource-loading suite is about: a locator handed to the constructor stays
    /// a locator. `Image("/etc/passwd")` must not read `/etc/passwd`, and
    /// `Image("https://evil.example/x.png")` must not fetch it — a constructor is reachable from
    /// application input, and reading it is a request the *host* makes on a stranger's behalf.
    #[test]
    fn the_constructor_keeps_a_locator_and_dereferences_nothing() {
        for locator in ["/etc/passwd", "https://evil.example/secret.png"] {
            assert_eq!(Image::new(locator).url, locator);
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
        let mapped: Image = serde_json::from_value(json!({ "url": "u" })).expect("parses");
        assert_eq!(mapped.url, "u");
        assert!(serde_json::from_value::<Image>(json!({ "no_url": 1 })).is_err());
        assert!(serde_json::from_value::<Image>(json!(3)).is_err());
    }
}
