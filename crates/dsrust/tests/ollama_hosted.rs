//! An ollama server that is not on this machine.
//!
//! `OLLAMA_HOST` has always pointed anywhere, but nothing carried a credential, so a hosted server
//! behind auth could be configured and not reached. The capability probe makes that sharper: it is
//! a second endpoint on the same server, and one that skipped the credential would report every
//! model incapable — a hosted ollama would silently never get native tool calls. The probe is the
//! chat route's (`ollama_chat/`); the `/api/generate` route carries no native tools to probe for.
//!
//! A one-shot server on the loopback interface stands in for the host, so what is asserted is what
//! left the process. The stub is std's `TcpListener`, as in `openai_compatible.rs`: no network, and
//! no mocking crate to add.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::JoinHandle;

use dsrust::lm::{ChatModel, LM};
use serde_json::{Value, json};

/// What the server actually received.
struct Request {
    path: String,
    authorization: Option<String>,
    body: Value,
}

/// A server that answers one request and hands it back.
struct Stub {
    host: String,
    served: JoinHandle<Request>,
}

impl Stub {
    fn answering(body: Value) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let host = format!("http://{}", listener.local_addr().expect("a bound address"));
        let body = body.to_string();
        let served = std::thread::spawn(move || {
            // Bounded, because a blocking accept turns "the code under test never called the
            // server" into a hang at the later join — three capability mutants were detected only
            // as suite timeouts for exactly this reason. Ten seconds is far past any real call.
            listener.set_nonblocking(true).expect("nonblocking");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(
                            std::time::Instant::now() < deadline,
                            "the code under test never called the stub"
                        );
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    Err(error) => panic!("accept failed: {error}"),
                }
            };
            stream.set_nonblocking(false).expect("blocking stream");
            let request = read_request(&mut stream);
            write_response(&mut stream, &body);
            request
        });
        Self { host, served }
    }

    fn received(self) -> Request {
        self.served.join().expect("the stub thread finished")
    }
}

fn read_request(stream: &mut TcpStream) -> Request {
    let mut reader = BufReader::new(stream);
    let mut start = String::new();
    reader.read_line(&mut start).expect("a request line");
    let path = start
        .split_whitespace()
        .nth(1)
        .unwrap_or_default()
        .to_owned();

    let mut authorization = None;
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).expect("a header line");
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match name.to_ascii_lowercase().as_str() {
            "authorization" => authorization = Some(value.to_owned()),
            "content-length" => length = value.parse().unwrap_or_default(),
            _ => {}
        }
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).expect("the whole body");
    Request {
        path,
        authorization,
        body: serde_json::from_slice(&body).unwrap_or(Value::Null),
    }
}

fn write_response(stream: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .expect("a written response");
    stream.flush().expect("a flushed response");
}

/// A model name no other test asks about, so the process-wide probe cache cannot answer for it.
fn unlisted(tag: &str) -> String {
    format!("ollama_chat/a-model-litellm-never-heard-of:{tag}")
}

/// litellm asks `POST /api/show` and reads the template; a hosted server needs the credential on
/// that call as much as on a chat.
#[tokio::test]
async fn the_probe_asks_the_configured_host_and_carries_its_credential() {
    let stub = Stub::answering(json!({ "template": "{{ if .Tools }}{{ end }}" }));
    let lm = LM::new(unlisted("carries"))
        .expect("a valid reference")
        .ollama_host(&stub.host)
        .ollama_api_key("hosted-secret");

    let found = lm.capabilities().await;
    assert!(found.function_calling, "the template offers tools");

    let asked = stub.received();
    assert_eq!(asked.path, "/api/show");
    assert_eq!(asked.authorization.as_deref(), Some("Bearer hosted-secret"));
    // The name is sent without the provider prefix, as litellm strips it.
    assert_eq!(
        asked.body["name"],
        json!(unlisted("carries").trim_start_matches("ollama_chat/"))
    );
}

/// A local server wants no credential, and sending one to a server that did not ask is its own
/// kind of wrong.
#[tokio::test]
async fn an_unauthenticated_host_is_asked_without_a_credential() {
    let stub = Stub::answering(json!({ "template": "{{ .Prompt }}" }));
    let lm = LM::new(unlisted("bare"))
        .expect("a valid reference")
        .ollama_host(&stub.host);

    let found = lm.capabilities().await;
    assert!(!found.function_calling, "this template offers no tools");
    assert_eq!(stub.received().authorization, None);
}

/// litellm's reading of a probe it could not make: nothing. A host that is down must not stall a
/// program or claim capabilities it never confirmed.
#[tokio::test]
async fn a_host_that_cannot_be_reached_grants_nothing() {
    let lm = LM::new(unlisted("unreachable"))
        .expect("a valid reference")
        // Port 9 is discard: nothing listens, and the connection is refused rather than hanging.
        .ollama_host("http://127.0.0.1:9");

    let found = lm.capabilities().await;
    assert!(!found.function_calling);
}

/// A model litellm's own registry lists is answered from there, so no probe is made at all —
/// which is what keeps a listed model working against a host that is not up yet.
#[tokio::test]
async fn a_model_the_registry_lists_is_never_asked_about() {
    let stub = Stub::answering(json!({ "template": "{{ if .Tools }}{{ end }}" }));
    let lm = LM::new("ollama_chat/llama2")
        .expect("a valid reference")
        .ollama_host(&stub.host);

    // `llama2` is in litellm's registry crediting nothing; the stub would say otherwise if asked.
    assert!(!lm.capabilities().await.function_calling);
    drop(stub);
}

/// Against a real ollama, which the stubs above deliberately are not. Ignored by default, as the
/// other live-provider tests here are: it needs a daemon with these models pulled.
///
///     cargo test -p dsrust --test ollama_hosted -- --ignored
#[tokio::test]
#[ignore = "needs a live ollama with qwen2.5:7b-instruct and gemma3:4b pulled"]
async fn a_live_daemon_answers_the_way_litellm_reads_it() {
    for (model, tools) in [
        ("ollama_chat/qwen2.5:7b-instruct", true),
        ("ollama_chat/gemma3:4b", false),
    ] {
        let lm = LM::new(model).expect("a valid reference");
        assert_eq!(lm.capabilities().await.function_calling, tools, "{model}");
    }
}
