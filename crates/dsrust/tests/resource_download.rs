//! `Image::from_url` and `Audio::from_url` against a real server on loopback.
//!
//! dspy 3.3.0 split resource loading in two: a constructor keeps a locator and never dereferences
//! it, and a factory named for downloading does the fetch. The constructor half is unit-tested
//! beside each type — a string stays a string, which needs no server. This half does need one,
//! because the thing under test is what comes back off the wire: which media type is used when the
//! server names one and when it does not, and that the bytes arrive base64 rather than as prose.
//!
//! A real socket rather than a mocked client. What this crate does with a `Content-Type` header is
//! the whole behaviour, and a double that hands one over has already done the part that could be
//! wrong.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::thread::JoinHandle;

use dsrust::adapter::types::{Audio, Image};

/// A server that answers one request with a canned body, and says so or does not.
struct Serving {
    url: String,
    served: JoinHandle<()>,
}

impl Serving {
    /// `content_type` is `None` for a server that sends no `Content-Type` at all, which is the
    /// branch that falls back to the URL's own suffix.
    fn once(path: &str, content_type: Option<&'static str>, body: &'static [u8]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let url = format!(
            "http://{}{path}",
            listener.local_addr().expect("a bound address")
        );
        let served = std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            drain_request(&mut stream);
            let mut head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n", body.len());
            if let Some(content_type) = content_type {
                head.push_str(&format!("Content-Type: {content_type}\r\n"));
            }
            head.push_str("\r\n");
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body);
            let _ = stream.flush();
        });
        Self { url, served }
    }
}

impl Drop for Serving {
    fn drop(&mut self) {
        // Joining would hang a test that never connected; the thread ends on its own either way.
        if self.served.is_finished() {
            // `take` is not available on a `&mut` field without an Option, and there is nothing to
            // recover from a finished thread, so the handle is simply dropped.
        }
    }
}

fn drain_request(stream: &mut std::net::TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().expect("clones"));
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
            break;
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0; length];
    let _ = reader.read_exact(&mut body);
}

/// The server named the type, so that is the type — and the bytes come back base64 inside a
/// `data:` URI, which is dspy's `f"data:{mime_type};base64,{encoded_data}"`.
#[tokio::test]
async fn an_image_download_uses_the_media_type_the_server_named() {
    let server = Serving::once("/photo.bin", Some("image/jpeg"), b"image bytes");
    let image = Image::from_url(&server.url).await.expect("downloads");
    assert_eq!(image.url, "data:image/jpeg;base64,aW1hZ2UgYnl0ZXM=");
}

/// No `Content-Type`, so the URL's own suffix decides — upstream's `mimetypes.guess_type(url)`.
#[tokio::test]
async fn an_image_download_without_a_content_type_falls_back_to_the_suffix() {
    let server = Serving::once("/photo.png", None, b"image bytes");
    let image = Image::from_url(&server.url).await.expect("downloads");
    assert_eq!(image.url, "data:image/png;base64,aW1hZ2UgYnl0ZXM=");
}

/// Neither a `Content-Type` nor a suffix worth guessing from: upstream raises rather than picking
/// something, because the media type is what tells a provider how to read the bytes.
#[tokio::test]
async fn an_image_download_with_nothing_naming_its_type_is_refused() {
    let server = Serving::once("/photo", None, b"image bytes");
    let why = Image::from_url(&server.url)
        .await
        .expect_err("nothing names the type")
        .to_string();
    assert!(
        why.starts_with("Could not determine MIME type for URL: "),
        "{why}"
    );
}

/// Audio keeps bare base64 beside a bare format name, so a `.wav` served as `audio/x-wav` arrives
/// as `wav` — the `x-` stripped, which is upstream's `_normalize_audio_format`.
#[tokio::test]
async fn an_audio_download_keeps_bare_base64_and_a_bare_format() {
    let server = Serving::once("/clip.wav", Some("audio/x-wav"), b"audio bytes");
    let audio = Audio::from_url(&server.url).await.expect("downloads");
    assert_eq!(audio.data, "YXVkaW8gYnl0ZXM=");
    assert_eq!(audio.audio_format, "wav");
}

/// A server that answers an audio request with something that is not audio is refused rather than
/// sent on: `format` reaches the provider, and `html` is not a format any model decodes.
#[tokio::test]
async fn an_audio_download_that_is_not_audio_is_refused() {
    let server = Serving::once("/clip.wav", Some("text/html"), b"<html>nope</html>");
    let why = Audio::from_url(&server.url)
        .await
        .expect_err("not audio")
        .to_string();
    assert_eq!(why, "Unsupported MIME type for audio: text/html");
}

/// The one check upstream makes before fetching: an HTTP(S) scheme and a host. It is not an SSRF
/// defence and does not pretend to be — it refuses `file://`, which would otherwise turn a
/// "download" into a local read, and that is all it is for.
#[tokio::test]
async fn only_an_http_url_is_fetched_at_all() {
    for locator in [
        "file:///etc/passwd",
        "/etc/passwd",
        "ftp://example.com/a.png",
        "https://",
    ] {
        let why = Image::from_url(locator)
            .await
            .expect_err("not an http url")
            .to_string();
        assert_eq!(
            why,
            format!("Image.from_url requires an HTTP(S) URL, received: {locator}"),
            "for {locator}"
        );
        let why = Audio::from_url(locator)
            .await
            .expect_err("not an http url")
            .to_string();
        assert_eq!(
            why,
            format!("Audio.from_url requires an HTTP(S) URL, received: {locator}"),
            "for {locator}"
        );
    }
}
