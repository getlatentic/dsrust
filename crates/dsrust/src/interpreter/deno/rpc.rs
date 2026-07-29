//! The JSON-RPC 2.0 conversation with the sandbox, over the child's stdin and stdout.
//!
//! One line per message, which is what upstream's `runner.js` reads and writes. Pyodide prints
//! package-loading chatter on the same stream, so a line that does not begin `{` is skipped rather
//! than treated as an answer — up to a bound, since skipping forever is how a dead child looks like
//! a slow one.

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout};

use anyhow::{Result, bail};
use serde_json::{Value, json};

/// dspy's `_MAX_SKIP_LINES`: how much non-JSON the sandbox may print before a read gives up.
const MAX_SKIPPED: usize = 100;

/// The two sides of one sandbox process, and the request counter they share.
pub(super) struct Rpc {
    writer: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: u64,
}

impl Rpc {
    pub(super) fn new(writer: ChildStdin, reader: ChildStdout) -> Self {
        Self {
            writer,
            reader: BufReader::new(reader),
            next_id: 0,
        }
    }

    /// Send a request and answer with the id it went out under, so the caller can match the reply.
    pub(super) fn request(&mut self, method: &str, params: Value) -> Result<u64> {
        self.next_id += 1;
        let id = self.next_id;
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

    fn send(&mut self, message: Value) -> Result<()> {
        writeln!(self.writer, "{message}")?;
        self.writer.flush()?;
        Ok(())
    }

    /// The next JSON message the sandbox sends, skipping whatever else it printed.
    pub(super) fn receive(&mut self, context: &str) -> Result<Value> {
        for _ in 0..=MAX_SKIPPED {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                bail!("the sandbox closed its output {context}");
            }
            let line = line.trim();
            if !line.starts_with('{') {
                continue;
            }
            match serde_json::from_str(line) {
                Ok(message) => return Ok(message),
                // Malformed JSON is Pyodide's chatter that happened to start with a brace, not a
                // message; upstream skips it on the same reasoning.
                Err(_) => continue,
            }
        }
        bail!("the sandbox printed {MAX_SKIPPED} lines of non-JSON {context}")
    }
}

/// dspy's `JSONRPC_APP_ERRORS["SyntaxError"]`, the one code it reads differently from the rest.
const SYNTAX_ERROR: i64 = -32000;

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
pub(super) fn answered(message: &Value, id: u64, context: &str) -> Result<Value> {
    if let Some(error) = message.get("error") {
        bail!("{}", said(error));
    }
    match message.get("id").and_then(Value::as_u64) {
        Some(answered) if answered == id => {
            Ok(message.get("result").cloned().unwrap_or(Value::Null))
        }
        other => bail!("the sandbox answered {other:?} where {id} was asked, {context}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reply naming another request is not this request's answer, and taking it would hand a
    /// caller the result of something else entirely.
    #[test]
    fn a_reply_to_another_request_is_refused() {
        let refused =
            answered(&json!({ "result": 1, "id": 7 }), 8, "while testing").expect_err("refused");
        assert!(
            refused
                .to_string()
                .contains("answered Some(7) where 8 was asked"),
            "{refused}"
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
                "id": 1,
            }),
            1,
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
            &json!({ "error": { "code": -32000, "message": "bad token" }, "id": 1 }),
            1,
            "while testing",
        )
        .expect_err("refused");
        assert_eq!(
            refused.to_string(),
            "Invalid Python syntax. message: bad token"
        );
    }
}
