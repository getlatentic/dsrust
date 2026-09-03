//! The tools the sandboxed code calls back into: which one, run it, and hand the answer across in
//! the shape `runner.js` decodes.
//!
//! Its own file because it is the one direction that runs *this* side's code — everything else in
//! `deno.rs` asks the sandbox something and reads the reply.
use anyhow::{Result, bail};
use serde_json::{Value, json};

use super::{DenoInterpreter, Session, UNKNOWN_ERROR};

impl DenoInterpreter {
    /// Run one tool the sandboxed code called, and hand the answer back through the pipe.
    pub(super) fn answer_tool_call(&self, session: &mut Session, request: &Value) -> Result<()> {
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let params = request.get("params").cloned().unwrap_or(Value::Null);
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let arguments = params.get("kwargs").cloned().unwrap_or_else(|| json!({}));

        match self.run_tool(name, &arguments) {
            Ok(value) => session.rpc.reply(&id, value),
            Err(error) => session
                .rpc
                .reply_error(&id, UNKNOWN_ERROR, &format!("{error:#}")),
        }
    }

    /// What one tool answered, in the shape `runner.js` reads: a JSON value carries its type so the
    /// sandbox can decode it, and anything else crosses as a string.
    fn run_tool(&self, name: &str, arguments: &Value) -> Result<Value> {
        self.handling_tool_call
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let answered = crate::observe::interpreter_tool_call(name, arguments, || {
            self.invoke_tool(name, arguments)
        });
        self.handling_tool_call
            .store(false, std::sync::atomic::Ordering::SeqCst);
        answered
    }

    /// dspy 3.3.1's `invoke_tool`, inside its callbacks: the named tool, run on the arguments.
    fn invoke_tool(&self, name: &str, arguments: &Value) -> Result<Value> {
        let tools = self.tools.lock().expect("the tool list");
        let Some(tool) = tools.iter().find(|tool| tool.name() == name) else {
            bail!("Unknown tool: {name}");
        };
        let answered = crate::observe::tool_call(tool.as_ref(), arguments)?;
        // dspy's rule, and the whole of it: `None` and `str` cross as `"string"`; *everything
        // else* crosses as `"json"`, so the sandbox decodes it back to its own type. Only a value
        // JSON cannot hold falls back to its string form — upstream reaches that through
        // `json.dumps(..., allow_nan=False)` raising on a non-finite float.
        //
        // This kept `"json"` for arrays and objects alone, so a tool returning `4` arrived in the
        // sandbox as the string `"4"` and `n + 1` failed with "can only concatenate str".
        Ok(match &answered {
            Value::Null => json!({ "value": "", "type": "string" }),
            Value::String(text) => json!({ "value": text, "type": "string" }),
            Value::Number(number) if number.as_f64().is_some_and(|n| !n.is_finite()) => {
                json!({ "value": answered.to_string(), "type": "string" })
            }
            _ => json!({ "value": answered.to_string(), "type": "json" }),
        })
    }
}
