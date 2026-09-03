//! The JSON-RPC 2.0 conversation with the sandbox, over the child's stdin and stdout.
//!
//! One line per message, which is what upstream's `runner.js` reads and writes. Pyodide prints
//! package-loading chatter on the same stream, so a line that does not begin `{` is skipped rather
//! than treated as an answer — up to a bound, since skipping forever is how a dead child looks like
//! a slow one. dspy 3.3.1 also skips what the sandbox says *out of band* — a notification such as
//! `unhandled_error`, or an error naming no request — keeping the last of them to explain a later
//! silence, and treats a protocol error naming no request as the end of the session.

use crate::interpreter::InterpreterFailure;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{ChildStdin, ChildStdout};

use anyhow::Result;
use serde_json::{Value, json};

/// dspy's `_MAX_SKIP_LINES`: how much non-JSON the sandbox may print before a read gives up.
const MAX_SKIPPED: usize = 100;

/// dspy's `JSONRPC_PROTOCOL_ERRORS`: the codes that mean the conversation itself broke.
const PROTOCOL_ERRORS: [i64; 3] = [-32700, -32600, -32601];

/// The sandbox's own pipes.
pub(super) type Rpc = Conversation<ChildStdin, ChildStdout>;

/// The two sides of one JSON-RPC conversation, and what the other end last said out of band.
///
/// Generic over the pipes rather than over `Child`'s: nothing here is about a subprocess, and a
/// conversation held to its protocol in a test should not have to spawn one.
pub(super) struct Conversation<W: Write, R: Read> {
    writer: W,
    reader: BufReader<R>,
    last_diagnostic: Option<String>,
}

impl<W: Write, R: Read> Conversation<W, R> {
    pub(super) fn new(writer: W, reader: R) -> Self {
        Self {
            writer,
            reader: BufReader::new(reader),
            last_diagnostic: None,
        }
    }

    /// Send a request and answer with the id it went out under, so the caller can match the reply.
    ///
    /// dspy 3.3.1 draws the id at random — `secrets.token_hex(16)` — rather than counting, so
    /// sandboxed code that prints a line shaped like a reply cannot guess which request it answers.
    pub(super) fn request(&mut self, method: &str, params: Value) -> Result<String> {
        let id = request_id();
        self.last_diagnostic = None;
        self.send(json!({ "jsonrpc": "2.0", "method": method, "params": params, "id": id }))?;
        Ok(id)
    }

    /// Answer a request the *sandbox* made — a tool call it wants run on this side.
    pub(super) fn reply(&mut self, id: &Value, result: Value) -> Result<()> {
        self.send(json!({ "jsonrpc": "2.0", "result": result, "id": id }))
    }

    /// Answer a sandbox request that failed, in the shape `runner.js` reads back as an exception.
    pub(super) fn reply_error(&mut self, id: &Value, code: i64, message: &str) -> Result<()> {
        self.send(json!({
            "jsonrpc": "2.0",
            "error": { "code": code, "message": message },
            "id": id,
        }))
    }

    /// Tell the sandbox something with no reply expected — a JSON-RPC notification, which carries
    /// no id. Upstream sends `sync_file` this way, and reading for an answer would block.
    ///
    /// `params` is omitted entirely when null, as upstream omits it: `shutdown` takes none.
    pub(super) fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        let mut message = json!({ "jsonrpc": "2.0", "method": method });
        if !params.is_null() {
            message["params"] = params;
        }
        self.send(message)
    }

    fn send(&mut self, message: Value) -> Result<()> {
        writeln!(self.writer, "{message}")?;
        self.writer.flush()?;
        Ok(())
    }

    /// The next message the sandbox sends *to this side*, skipping whatever else it printed and
    /// whatever it said out of band.
    pub(super) fn receive(&mut self, context: &str) -> Result<Value> {
        // Bounded by the range rather than by a counter the body maintains: a `+= 1` a mutation
        // could drop is a read that never ends, which is the one failure no assertion reports.
        for _ in 0..=MAX_SKIPPED {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                return Err(self.session_failure(format!(
                    "the sandbox closed its output {context}{}",
                    self.diagnostic()
                )));
            }
            let line = line.trim();
            let message = match line.starts_with('{') {
                true => serde_json::from_str::<Value>(line).ok(),
                false => None,
            };
            // Malformed JSON is Pyodide's chatter that happened to start with a brace, not a
            // message; upstream skips it on the same reasoning.
            let Some(message) = message else {
                continue;
            };
            if self.out_of_band(&message, context)? {
                continue;
            }
            return Ok(message);
        }
        // dspy's own count: its `while skipped <= _MAX_SKIP_LINES` leaves the loop having skipped
        // one more than the bound.
        Err(self.session_failure(format!(
            "Too many skipped lines ({}) {context}",
            MAX_SKIPPED + 1
        )))
    }

    /// dspy 3.3.1's `_handle_out_of_band_message`: a notification, or an error naming no request,
    /// is consumed and remembered; a *protocol* error naming no request ends the session.
    fn out_of_band(&mut self, message: &Value, context: &str) -> Result<bool> {
        let payload = match (
            message.get("method"),
            message.get("id"),
            message.get("error"),
        ) {
            (Some(_), None, _) => message.get("params").cloned().unwrap_or(Value::Null),
            (_, id, Some(error)) if id.is_none_or(Value::is_null) => {
                let code = error.get("code").and_then(Value::as_i64);
                if code.is_some_and(|code| PROTOCOL_ERRORS.contains(&code)) {
                    let said = error
                        .get("message")
                        .and_then(Value::as_str)
                        .map_or_else(|| message.to_string(), str::to_owned);
                    return Err(self.session_failure(format!("Protocol error {context}: {said}")));
                }
                error.clone()
            }
            _ => return Ok(false),
        };
        let said = payload
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| message.to_string(), str::to_owned);
        tracing::debug!("Skipping out-of-band sandbox message {context}: {said}");
        self.last_diagnostic = Some(said);
        Ok(true)
    }

    /// dspy 3.3.1's ` (last sandbox diagnostic: …)` suffix, where there is one.
    fn diagnostic(&self) -> String {
        self.last_diagnostic
            .as_ref()
            .map(|said| format!(" (last sandbox diagnostic: {said})"))
            .unwrap_or_default()
    }

    fn session_failure(&self, why: String) -> anyhow::Error {
        anyhow::Error::new(InterpreterFailure::Session(why))
    }
}

/// 32 hex characters drawn from the process's own randomness — the shape of `secrets.token_hex(16)`.
fn request_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut id = String::with_capacity(32);
    for salt in 0..2u64 {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u64(salt);
        id.push_str(&format!("{:016x}", hasher.finish()));
    }
    id
}

/// dspy's `JSONRPC_APP_ERRORS["SyntaxError"]`, the one code it reads differently from the rest.
const SYNTAX_ERROR: i64 = -32000;

/// dspy's `JSONRPC_APP_ERRORS`: the codes the sandbox uses to report *the submitted code's* own
/// failure. Anything else on an error reply is the protocol going wrong, which ends the session.
///
/// Upstream branches on exactly this set — `if error_code in JSONRPC_APP_ERRORS.values()` — and the
/// branch decides whether a module rewrites the code or stops. Reading the text instead is how a
/// dead sandbox gets handed to the model as a syntax error to fix.
const APP_ERRORS: [i64; 10] = [
    -32000, // SyntaxError
    -32001, // NameError
    -32002, // TypeError
    -32003, // ValueError
    -32004, // AttributeError
    -32005, // IndexError
    -32006, // KeyError
    -32007, // RuntimeError
    -32008, // CodeInterpreterError
    -32099, // Unknown
];

/// What the code's own failure says, in dspy's wording.
///
/// The text matters more than it looks: a module hands it straight back to the model as the thing
/// to correct, so `NameError: ["name 'x' is not defined"]` is the prompt and "something failed" is
/// not. Pyodide leaves `message` blank and puts the exception's type and args under `data`, which
/// is why reading `message` alone answers with nothing at all.
fn said(error: &Value) -> String {
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let data = error.get("data").cloned().unwrap_or(Value::Null);
    let kind = data.get("type").and_then(Value::as_str).unwrap_or("Error");
    if error.get("code").and_then(Value::as_i64) == Some(SYNTAX_ERROR) {
        return format!("Invalid Python syntax. message: {message}");
    }
    match data.get("args") {
        Some(args) if !args.is_null() => format!("{kind}: {args}"),
        _ => format!("{kind}: {message}"),
    }
}

/// One reply's result, checked against the request it answers.
///
/// An error naming another request is not this request's failure: dspy 3.3.1 ends the session on
/// it, where before it would read an error carrying no id as the current request's.
pub(super) fn answered(message: &Value, id: &str, context: &str) -> Result<Value> {
    let answers = message.get("id").and_then(Value::as_str) == Some(id);
    if let Some(error) = message.get("error") {
        if !answers {
            return Err(anyhow::Error::new(InterpreterFailure::Session(format!(
                "Response ID mismatch: expected {id}, got {}",
                message.get("id").cloned().unwrap_or(Value::Null)
            ))));
        }
        // dspy's split: an application code is the code's failure and a module feeds it back to the
        // model; anything else is the protocol's, and upstream makes that terminal.
        let code = error.get("code").and_then(Value::as_i64);
        let failure = match code.is_some_and(|code| APP_ERRORS.contains(&code)) {
            true => InterpreterFailure::Execution(said(error)),
            false => InterpreterFailure::Session(format!("Error {context}: {}", said(error))),
        };
        return Err(anyhow::Error::new(failure));
    }
    match answers {
        true => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
        // A reply that answers a different request means the stream is out of step, which no
        // rewrite of the submitted code repairs — upstream's `_raise_terminal_error`.
        false => Err(anyhow::Error::new(InterpreterFailure::Session(format!(
            "Unexpected response {context}: {message}"
        )))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reply naming another request is not this request's answer, and taking it would hand a
    /// caller the result of something else entirely.
    #[test]
    fn a_reply_to_another_request_is_refused() {
        let refused = answered(&json!({ "result": 1, "id": "7" }), "8", "while testing")
            .expect_err("refused");
        assert!(
            refused
                .to_string()
                .starts_with("Unexpected response while testing:"),
            "{refused}"
        );
        let mismatched = answered(
            &json!({ "error": { "code": -32001 }, "id": "7" }),
            "8",
            "while testing",
        )
        .expect_err("refused");
        assert_eq!(
            mismatched.to_string(),
            "Response ID mismatch: expected 8, got \"7\""
        );
    }

    /// Pyodide leaves `message` blank and puts the exception under `data`, so reading `message`
    /// alone answers with an empty string — which is what a module would then show the model.
    #[test]
    fn an_error_reply_reads_the_exception_out_of_data() {
        let refused = answered(
            &json!({
                "error": {
                    "code": -32001,
                    "message": "",
                    "data": { "type": "NameError", "args": ["name 'x' is not defined"] },
                },
                "id": "1",
            }),
            "1",
            "while testing",
        )
        .expect_err("refused");
        assert_eq!(
            refused.to_string(),
            r#"NameError: ["name 'x' is not defined"]"#
        );
    }

    /// A syntax error is the one dspy words differently, because there is no exception object to
    /// read args from — the code never ran.
    #[test]
    fn a_syntax_error_takes_dspys_own_wording() {
        let refused = answered(
            &json!({ "error": { "code": -32000, "message": "bad token" }, "id": "1" }),
            "1",
            "while testing",
        )
        .expect_err("refused");
        assert_eq!(
            refused.to_string(),
            "Invalid Python syntax. message: bad token"
        );
    }

    #[test]
    fn a_request_id_is_thirty_two_hex_characters_and_fresh_each_time() {
        let first = request_id();
        let second = request_id();
        assert_eq!(first.len(), 32);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(first, second);
    }

    /// The out-of-band rules, over a pipe scripted with each kind of line.
    #[test]
    fn out_of_band_messages_are_skipped_and_remembered_and_a_protocol_error_ends_the_session() {
        let mut rpc = scripted(concat!(
            "not json\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"unhandled_error\",\"params\":{\"message\":\"boom\"}}\n",
            "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32099,\"message\":\"late\"},\"id\":null}\n",
            "{\"jsonrpc\":\"2.0\",\"result\":1,\"id\":\"a\"}\n",
            "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32700,\"message\":\"parse\"}}\n",
        ));
        let answer = rpc.receive("during a test").expect("the real reply");
        assert_eq!(answer["id"], "a");
        assert_eq!(
            rpc.last_diagnostic.as_deref(),
            Some("late"),
            "the last out-of-band message is kept"
        );
        let ended = rpc
            .receive("during a test")
            .expect_err("a protocol error is terminal");
        assert_eq!(ended.to_string(), "Protocol error during a test: parse");
        let closed = rpc.receive("during a test").expect_err("the pipe is spent");
        assert_eq!(
            closed.to_string(),
            "the sandbox closed its output during a test (last sandbox diagnostic: late)"
        );
    }

    /// A conversation over a scripted transcript, which needs no process at all: the reader is the
    /// bytes the sandbox would have written and the writer is a sink.
    fn scripted(lines: &str) -> Conversation<Vec<u8>, std::io::Cursor<Vec<u8>>> {
        Conversation::new(Vec::new(), std::io::Cursor::new(lines.as_bytes().to_vec()))
    }
}
