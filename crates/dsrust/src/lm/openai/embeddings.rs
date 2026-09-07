//! The OpenAI-compatible `/embeddings` call litellm makes for `dspy.Embedder("openai/...")`.
//!
//! The body is what litellm sends — `model`, `input`, and whatever keyword arguments the caller
//! added — held to `tests/conformance/lm_api/embedding_wire.json`, recorded from litellm at the
//! HTTP layer. The reply's `data` carries one `embedding` per input, in input order.

use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde_json::{Map, Value};

use super::OpenAiConfig;

/// The request body litellm posts: `input`, then `model`, then the caller's keyword arguments,
/// then the `encoding_format` of `float` the OpenAI SDK adds unless the caller chose one.
pub fn request_body(model_id: &str, inputs: &[String], kwargs: &Map<String, Value>) -> Value {
    let mut body = Map::new();
    body.insert(
        "input".to_owned(),
        Value::Array(
            inputs
                .iter()
                .map(|text| Value::String(text.clone()))
                .collect(),
        ),
    );
    body.insert("model".to_owned(), Value::String(model_id.to_owned()));
    for (key, value) in kwargs {
        body.insert(key.clone(), value.clone());
    }
    body.entry("encoding_format".to_owned())
        .or_insert_with(|| Value::String("float".to_owned()));
    Value::Object(body)
}

/// One embedding per input, read off the reply's `data`.
pub fn embeddings_of(reply: &Value) -> Result<Vec<Vec<f32>>> {
    let data = reply
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("the embeddings reply carries no `data` list: {reply}"))?;
    data.iter()
        .map(|entry| {
            let embedding = entry
                .get("embedding")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("an embeddings entry carries no `embedding`: {entry}"))?;
            embedding
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .map(|float| float as f32)
                        .ok_or_else(|| anyhow!("an embedding value is not a number: {value}"))
                })
                .collect()
        })
        .collect()
}

/// `POST {base_url}/embeddings`, bearer-authenticated as the chat call is.
pub(crate) async fn embed(
    http: &reqwest::Client,
    config: &OpenAiConfig,
    model_id: &str,
    inputs: &[String],
    kwargs: &Map<String, Value>,
    timeout: Duration,
) -> Result<Vec<Vec<f32>>> {
    let url = format!("{}/embeddings", config.base_url.trim_end_matches('/'));
    let mut request = http
        .post(&url)
        .timeout(timeout)
        .json(&request_body(model_id, inputs, kwargs));
    if let Some(key) = &config.api_key {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("embeddings request to {url} failed"))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .context("reading the embeddings reply")?;
    if !status.is_success() {
        return Err(anyhow!(
            "embeddings request to {url} answered {status}: {text}"
        ));
    }
    let reply: Value = serde_json::from_str(&text).context("the embeddings reply is not JSON")?;
    embeddings_of(&reply)
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::time::Instant;

    use serde_json::json;

    use super::*;

    const TIMEOUT: Duration = Duration::from_secs(5);
    /// How long the server waits to be asked, and the test waits to hear about it. A mutant that
    /// answers without calling out never connects, so both must give up: a test that blocks on a
    /// request which is never made reports a timed-out job rather than the reason.
    const PATIENCE: Duration = Duration::from_secs(5);

    /// What the server was asked, so a test can hold the route and the credentials.
    struct Asked {
        target: String,
        authorization: Option<String>,
        body: Value,
    }

    /// Answers one request with `status` and `body`, reporting what it was asked.
    fn serving(
        status: u16,
        body: &'static str,
    ) -> (String, mpsc::Receiver<Asked>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let address = format!("http://{}", listener.local_addr().expect("a bound address"));
        let (seen, heard) = mpsc::channel();
        listener
            .set_nonblocking(true)
            .expect("a listener that can be polled");
        let served = std::thread::spawn(move || {
            let Some(mut stream) = accept_before(&listener, Instant::now() + PATIENCE) else {
                return;
            };
            if let Some(asked) = read_request(&mut stream) {
                let _ = seen.send(asked);
            }
            let _ = write_response(&mut stream, status, body);
        });
        (address, heard, served)
    }

    /// The one connection this server expects, or nothing once `deadline` has passed.
    fn accept_before(listener: &TcpListener, deadline: Instant) -> Option<TcpStream> {
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).ok()?;
                    return Some(stream);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return None;
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => return None,
            }
        }
    }

    fn read_request(stream: &mut TcpStream) -> Option<Asked> {
        let mut reader = BufReader::new(stream);
        let mut start = String::new();
        reader.read_line(&mut start).ok()?;
        let target = start.split_whitespace().nth(1)?.to_owned();

        let mut authorization = None;
        let mut length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).ok()? == 0 || line.trim_end().is_empty() {
                break;
            }
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            if name.eq_ignore_ascii_case("authorization") {
                authorization = Some(value.trim().to_owned());
            } else if name.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap_or(0);
            }
        }

        let mut raw = vec![0u8; length];
        reader.read_exact(&mut raw).ok()?;
        Some(Asked {
            target,
            authorization,
            body: serde_json::from_slice(&raw).unwrap_or(Value::Null),
        })
    }

    fn write_response(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
        let response = format!(
            "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes())?;
        stream.flush()
    }

    fn configured(base_url: &str, api_key: Option<&str>) -> OpenAiConfig {
        OpenAiConfig {
            base_url: base_url.to_owned(),
            api_key: api_key.map(str::to_owned),
            ..OpenAiConfig::default()
        }
    }

    /// `request_body` is held against a litellm recording and `embeddings_of` decodes in
    /// isolation, but the call joining them had no test: its whole body answering `Ok(vec![])`
    /// survived mutation, as did inverting its status check.
    #[tokio::test]
    async fn an_embedding_call_posts_to_the_route_and_answers_in_input_order() {
        let reply = r#"{"data":[{"embedding":[0.5,-0.25]},{"embedding":[1.0,0.0]}]}"#;
        let (address, heard, served) = serving(200, reply);

        let vectors = embed(
            &reqwest::Client::new(),
            // A trailing slash, because a self-hosted base URL routinely carries one.
            &configured(&format!("{address}/v1/"), Some("sk-test")),
            "text-embedding-3-small",
            &["first".to_owned(), "second".to_owned()],
            &Map::new(),
            TIMEOUT,
        )
        .await
        .expect("the endpoint answered");

        assert_eq!(vectors, vec![vec![0.5, -0.25], vec![1.0, 0.0]]);

        let asked = heard
            .recv_timeout(PATIENCE)
            .expect("the endpoint was actually called");
        let _ = served.join();
        assert_eq!(asked.target, "/v1/embeddings", "the route it posts to");
        assert_eq!(asked.authorization.as_deref(), Some("Bearer sk-test"));
        assert_eq!(asked.body["input"], json!(["first", "second"]));
        assert_eq!(asked.body["model"], json!("text-embedding-3-small"));
    }

    #[tokio::test]
    async fn a_failing_status_is_an_error_and_not_an_embedding_of_nothing() {
        let (address, _heard, served) = serving(500, r#"{"error":"overloaded"}"#);

        let refused = embed(
            &reqwest::Client::new(),
            &configured(&format!("{address}/v1"), None),
            "text-embedding-3-small",
            &["first".to_owned()],
            &Map::new(),
            TIMEOUT,
        )
        .await
        .expect_err("a 500 is not an embedding");

        let _ = served.join();
        let rendered = format!("{refused:#}");
        assert!(rendered.contains("500"), "names the status: {rendered}");
        assert!(rendered.contains("overloaded"), "and the body: {rendered}");
    }

    #[tokio::test]
    async fn an_unauthenticated_config_sends_no_authorization() {
        let (address, heard, served) = serving(200, r#"{"data":[{"embedding":[1.0]}]}"#);

        embed(
            &reqwest::Client::new(),
            &configured(&format!("{address}/v1"), None),
            "text-embedding-3-small",
            &["only".to_owned()],
            &Map::new(),
            TIMEOUT,
        )
        .await
        .expect("the endpoint answered");

        let asked = heard
            .recv_timeout(PATIENCE)
            .expect("the endpoint was actually called");
        let _ = served.join();
        assert_eq!(asked.authorization, None, "no key, no bearer header");
    }
}
