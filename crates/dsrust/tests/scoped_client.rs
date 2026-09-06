//! The client a scope names is the client its calls go out on.
//!
//! `lm::client()` prefers a scope's own client over the process-wide one, and nothing proved it:
//! replacing the whole function with `reqwest::Client::default()` left every test green.
//!
//! A timeout cannot tell them apart — every provider sets its own per-request bound, which
//! overrides whatever the client carries — so this asks the server which client called. A
//! `User-Agent` is set on the client and nowhere else, so the header arriving is the scope's
//! client having made the request.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;

use dsrust::lm::LM;
use dsrust::{Example, Module, Predict};

const SCOPED_AGENT: &str = "dsrust-scoped-client";
const PROCESS_AGENT: &str = "dsrust-process-client";

/// Answers once, reporting the `User-Agent` it was asked with.
fn listening() -> (String, mpsc::Receiver<String>, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
    let address = format!("http://{}", listener.local_addr().expect("a bound address"));
    let (seen, heard) = mpsc::channel();
    let served = std::thread::spawn(move || {
        let Ok((mut stream, _)) = listener.accept() else {
            return;
        };
        let agent = read_user_agent(&mut stream);
        let _ = seen.send(agent);
        let body = r#"{"message":{"role":"assistant","content":"[[ ## answer ## ]]\nhere"}}"#;
        let _ = write_response(&mut stream, body);
    });
    (address, heard, served)
}

fn read_user_agent(stream: &mut TcpStream) -> String {
    let mut reader = BufReader::new(stream);
    let mut agent = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim_end().is_empty() {
            return agent;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("user-agent")
        {
            agent = value.trim().to_owned();
        }
    }
}

fn write_response(stream: &mut TcpStream, body: &str) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

fn named(agent: &str) -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(agent)
        .build()
        .expect("client builds")
}

fn asking(host: &str) -> LM {
    LM::new("ollama/probe")
        .expect("valid model ref")
        .ollama_host(host)
}

#[tokio::test]
async fn a_scope_calls_on_its_own_client_and_not_the_process_one() {
    let (address, heard, served) = listening();
    dsrust::lm::configure_with_client(named(PROCESS_AGENT), asking(&address));

    let _ = dsrust::lm::context_with_client(named(SCOPED_AGENT), asking(&address))
        .run(async {
            Predict::from_signature("question -> answer".parse().expect("parses"))
                .forward(Example::new([(
                    "question".to_owned(),
                    serde_json::json!("where?"),
                )]))
                .await
        })
        .await;

    let agent = heard.recv().expect("the server was asked");
    let _ = served.join();
    assert_eq!(
        agent, SCOPED_AGENT,
        "the scope's client made the call, not the process-wide one"
    );
}
