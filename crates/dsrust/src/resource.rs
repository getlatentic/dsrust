//! A local resource as the wire spells it: base64 under a `data:` URI.
//!
//! dspy's `data_uri` / `read_path_base64` / `data_uri_from_path`, which every media type reaches
//! through — an `Image` built from a file, a `File`'s `file_data`, a `Video` rendered from a path.
//! Its own home because both sides of the crate need it and neither should own it: `adapter::types`
//! builds these when a caller constructs a value, and `lm::api::wire` builds them again when a part
//! reaches a provider.

use base64::Engine;

/// `data:<media type>;base64,<data>`, leaving alone data that is already spelled that way.
///
/// The second half is not defensive tidiness — a source may arrive already encoded, and wrapping
/// it twice produces a URI whose payload is itself a URI, which decodes to text rather than to the
/// image the caller meant.
pub(crate) fn data_uri(media_type: &str, data: &str) -> String {
    match data.starts_with("data:") {
        true => data.to_owned(),
        false => format!("data:{media_type};base64,{data}"),
    }
}

/// A local file's bytes, base64-encoded.
///
/// Refused by name where the path is not a file, which is upstream's `File not found: …` — the
/// message a caller sees for a typo, and the one place these factories touch the host at all.
pub(crate) fn read_base64(path: &std::path::Path) -> anyhow::Result<String> {
    if !path.is_file() {
        anyhow::bail!("File not found: {}", path.display());
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(std::fs::read(path)?))
}

/// Whether this is a URL a resource may be fetched from — dspy's `_is_http_url`.
///
/// An HTTP(S) scheme *and* a host. It is the only check upstream's factories make and it is not an
/// SSRF defence: `http://169.254.169.254/…` passes it, as upstream's own docstring says at length.
/// The caller allowlists what it derived from untrusted input; this only refuses `file://` and
/// friends, which would otherwise make a "download" read the local disk.
pub(crate) fn is_http_url(url: &str) -> bool {
    let Some((scheme, rest)) = url.split_once("://") else {
        return false;
    };
    matches!(scheme, "http" | "https")
        && !rest.split(['/', '?', '#']).next().unwrap_or("").is_empty()
}

/// Fetch a resource and hand back what the server called it and its bytes, base64-encoded.
///
/// **A caller-initiated request with no SSRF protection**, which is upstream's design and its
/// warning: it follows redirects and will reach loopback, private and cloud-metadata hosts. It is
/// reachable only from a factory named for downloading — never from a constructor and never from
/// parsing a model's output, which is the whole point of dspy 3.3.0's split.
///
/// `verify` is upstream's TLS switch, for a self-signed certificate.
pub(crate) async fn fetch_base64(
    url: &str,
    verify: bool,
) -> anyhow::Result<(Option<String>, String)> {
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(!verify)
        .build()?;
    let response = client.get(url).send().await?.error_for_status()?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_owned);
    let bytes = response.bytes().await?;
    Ok((
        content_type,
        base64::engine::general_purpose::STANDARD.encode(bytes),
    ))
}

/// The media type a filename implies, or the caller's fallback — dspy's `media_type_for_path`,
/// which is `mimetypes.guess_type(path)[0] or fallback`.
pub(crate) fn media_type_for(path: &std::path::Path, fallback: &str) -> String {
    crate::mimetypes::guess(&path.to_string_lossy())
        .unwrap_or(fallback)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_already_spelled_as_a_uri_is_not_wrapped_again() {
        assert_eq!(
            data_uri("image/png", "data:image/jpeg;base64,QQ=="),
            "data:image/jpeg;base64,QQ==",
        );
        assert_eq!(data_uri("image/png", "QQ=="), "data:image/png;base64,QQ==");
    }

    /// The fallback is only for a suffix the table does not know — a known one wins even when the
    /// caller names a different default, because dspy's `or` reads the same way.
    #[test]
    fn a_known_suffix_beats_the_fallback() {
        let known = std::path::Path::new("/tmp/clip.wav");
        assert_eq!(media_type_for(known, "audio/mpeg"), "audio/x-wav");
        let unknown = std::path::Path::new("/tmp/clip.zzz");
        assert_eq!(media_type_for(unknown, "audio/mpeg"), "audio/mpeg");
    }

    #[test]
    fn a_path_that_is_not_there_is_refused_by_name() {
        let why = read_base64(std::path::Path::new("/tmp/dsrs-no-such-resource.png"))
            .expect_err("not a file");
        assert!(why.to_string().starts_with("File not found: "), "{why}");
    }
}
