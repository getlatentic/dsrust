//! The OpenAI-compatible provider over a real socket.
//!
//! A one-shot server on the loopback interface stands in for OpenAI, Groq, vLLM or LM Studio,
//! so the route, the credential header and the request body are asserted as they leave the
//! process rather than as the builder imagined them. The stub is std's `TcpListener`: no
//! network, and no mocking crate to add.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::JoinHandle;

use dsrs::lm::{ChatModel, ChatTurn, JsonFormat, LM, OutputMode};
use serde_json::{Value, json};

const REPLY: &str = r#"{"choices":[{"message":{"content":"the reply"}}]}"#;

/// What the provider actually received.
struct Request {
    path: String,
    headers: Vec<(String, String)>,
    body: Value,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        find_header(&self.headers, name)
    }
}

/// A server that answers exactly one request and hands it back for inspection.
struct Stub {
    base_url: String,
    served: JoinHandle<Request>,
}

impl Stub {
    fn answering(status: u16, body: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let base_url = format!(
            "http://{}/v1",
            listener.local_addr().expect("a bound address")
        );
        let body = body.to_owned();
        let served = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("one connection");
            let request = read_request(&mut stream);
            write_response(&mut stream, status, &body);
            request
        });
        Self { base_url, served }
    }

    /// Blocks until the exchange is over, which it is once the call under test has returned.
    fn received(self) -> Request {
        self.served.join().expect("the stub thread finished")
    }
}

fn read_request(stream: &mut TcpStream) -> Request {
    let mut reader = BufReader::new(stream);
    let head = read_head(&mut reader);
    let path = head[0]
        .split_whitespace()
        .nth(1)
        .expect("a request target")
        .to_owned();
    let headers: Vec<(String, String)> = head[1..]
        .iter()
        .filter_map(|line| split_header(line))
        .collect();
    let mut body = vec![0; content_length(&headers)];
    reader.read_exact(&mut body).expect("the whole body");
    Request {
        path,
        headers,
        body: serde_json::from_slice(&body).expect("a JSON body"),
    }
}

/// The request line and its headers, up to the blank line that ends the head.
fn read_head(reader: &mut impl BufRead) -> Vec<String> {
    let mut lines = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).expect("a header line");
        let line = line.trim_end().to_owned();
        if line.is_empty() {
            return lines;
        }
        lines.push(line);
    }
}

fn split_header(line: &str) -> Option<(String, String)> {
    let (name, value) = line.split_once(':')?;
    Some((name.to_ascii_lowercase(), value.trim().to_owned()))
}

fn find_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

fn content_length(headers: &[(String, String)]) -> usize {
    find_header(headers, "content-length")
        .and_then(|value| value.parse().ok())
        .unwrap_or_default()
}

fn write_response(stream: &mut TcpStream, status: u16, body: &str) {
    let reason = match status {
        200..=299 => "OK",
        _ => "Error",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .expect("the reply goes out");
    stream.flush().expect("the reply is flushed");
}

fn probe_lm(stub: &Stub) -> LM {
    LM::new("openai/gpt-4o-mini")
        .expect("valid model ref")
        .with_openai_key("sk-test")
        .with_openai_base_url(stub.base_url.as_str())
}

async fn ask(lm: &LM, mode: &OutputMode<'_>) -> anyhow::Result<String> {
    lm.chat(
        &reqwest::Client::new(),
        "be helpful",
        &[ChatTurn::user("hi")],
        mode,
    )
    .await
}

fn probe_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false,
    })
}

#[tokio::test]
async fn a_call_reaches_chat_completions_under_the_configured_base_url() {
    let stub = Stub::answering(200, REPLY);
    let reply = ask(&probe_lm(&stub), &OutputMode::Text)
        .await
        .expect("the stub answers");
    assert_eq!(reply, "the reply");

    let request = stub.received();
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(request.header("authorization"), Some("Bearer sk-test"));
    assert_eq!(request.body["model"], "gpt-4o-mini");
    assert_eq!(request.body["messages"][0]["role"], "system");
    assert_eq!(request.body["messages"][0]["content"], "be helpful");
    assert_eq!(request.body["messages"][1]["content"], "hi");
    assert_eq!(request.body.get("response_format"), None);
}

#[tokio::test]
async fn a_base_url_with_a_trailing_slash_reaches_the_same_route() {
    let stub = Stub::answering(200, REPLY);
    let lm = LM::new("openai/gpt-4o-mini")
        .expect("valid model ref")
        .with_openai_key("sk-test")
        .with_openai_base_url(format!("{}/", stub.base_url));
    ask(&lm, &OutputMode::Text).await.expect("the stub answers");
    assert_eq!(stub.received().path, "/v1/chat/completions");
}

#[tokio::test]
async fn json_mode_asks_for_an_object_by_default() {
    let stub = Stub::answering(200, REPLY);
    let schema = probe_schema();
    ask(&probe_lm(&stub), &OutputMode::Json { schema: &schema })
        .await
        .expect("the stub answers");
    assert_eq!(
        stub.received().body["response_format"],
        json!({ "type": "json_object" })
    );
}

#[tokio::test]
async fn the_schema_envelope_is_opt_in_and_carries_the_schema() {
    let stub = Stub::answering(200, REPLY);
    let lm = probe_lm(&stub).with_openai_json_format(JsonFormat::Schema);
    let schema = probe_schema();
    ask(&lm, &OutputMode::Json { schema: &schema })
        .await
        .expect("the stub answers");

    let request = stub.received();
    let format = &request.body["response_format"];
    assert_eq!(format["type"], "json_schema");
    assert_eq!(format["json_schema"]["strict"], true);
    assert_eq!(format["json_schema"]["schema"], schema);
}

#[tokio::test]
async fn a_refused_call_carries_the_status_and_the_services_own_message() {
    let stub = Stub::answering(401, r#"{"error":{"message":"Incorrect API key provided"}}"#);
    let error = ask(&probe_lm(&stub), &OutputMode::Text)
        .await
        .expect_err("401 is a failure");
    let rendered = format!("{error:#}");
    assert!(rendered.contains("openai 401"), "got: {rendered}");
    assert!(
        rendered.contains("Incorrect API key provided"),
        "got: {rendered}"
    );
}

/// No stub: the call must fail on the missing credential before anything is sent.
#[tokio::test]
async fn a_missing_key_names_the_variable_the_endpoint_was_told_to_read() {
    let lm = LM::new("openai/llama-3.3-70b")
        .expect("valid model ref")
        .with_openai_key_env("DSRS_TEST_KEY_THAT_IS_NOT_SET");
    let error = ask(&lm, &OutputMode::Text)
        .await
        .expect_err("no credential is configured");
    assert!(
        error
            .to_string()
            .contains("DSRS_TEST_KEY_THAT_IS_NOT_SET is not set"),
        "got: {error}"
    );
}
