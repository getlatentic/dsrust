//! dspy `primitives/python_interpreter.py`: Python in a WASM sandbox, run by Deno and Pyodide.
//!
//! The sandbox is not reimplemented here. Upstream's `runner.js` is vendored beside this file and
//! executed as-is, so the environment a model's code lands in is the one dspy's own tests exercise
//! — same Pyodide, same `SUBMIT`, same captured stdout. What this module is, is the other half of
//! that conversation: build the `deno run` argv, speak JSON-RPC 2.0 over the child's pipes, answer
//! the tool calls the sandboxed code makes, and read one execution's result out.
//!
//! `deno` is a prerequisite rather than a dependency, exactly as it is for dspy: upstream shells
//! out to the same binary and tells a caller to install it. [`DenoInterpreter::available`] answers
//! whether it is there, so a program can say so itself rather than failing at the first ask.

mod command;
mod register;
mod rpc;

use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

pub use command::Permissions;
pub use register::OutputField;

use super::{CodeInterpreter, Executed};
use crate::react::Tool;
use rpc::Rpc;

/// dspy's `JSONRPC_APP_ERRORS["Unknown"]`, which a tool failure answers under.
const UNKNOWN_ERROR: i64 = -32099;

/// dspy `PythonInterpreter`: a Deno/Pyodide sandbox, one child process per interpreter.
///
/// State persists between calls, as upstream's does — a name bound by one `execute` is in scope for
/// the next. The child starts on the first execution rather than on construction, so building a
/// module that may never run code costs nothing.
pub struct DenoInterpreter {
    permissions: Permissions,
    /// Behind a lock because [`CodeInterpreter::execute`] takes `&self` while a conversation with
    /// the child is inherently sequential — the same reason a `Tool` holding a subprocess does.
    session: Mutex<Option<Session>>,
    tools: Mutex<Vec<Arc<dyn Tool>>>,
    /// dspy's `output_fields`: the names a typed `SUBMIT` takes. Empty leaves the sandbox on its
    /// default single-argument one.
    outputs: Mutex<Vec<OutputField>>,
}

/// One live child and the pipes to it.
struct Session {
    child: Child,
    rpc: Rpc,
    /// Whether this child has been told about the tools and outputs. A restart clears it with the
    /// rest of the session, which is what replays the registration upstream replays.
    registered: bool,
}

impl Default for DenoInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl DenoInterpreter {
    /// A sandbox that may read only what it must: the runner and Pyodide's cache. No network, no
    /// writes, no environment — upstream's defaults, which are the point of a sandbox.
    pub fn new() -> Self {
        Self::with_permissions(Permissions::default())
    }

    /// The same, with what the sandboxed code is allowed to reach spelled out.
    pub fn with_permissions(permissions: Permissions) -> Self {
        Self {
            permissions,
            session: Mutex::new(None),
            tools: Mutex::new(Vec::new()),
            outputs: Mutex::new(Vec::new()),
        }
    }

    /// The output fields a typed `SUBMIT` takes, in order — dspy's `output_fields`.
    ///
    /// `CodeAct` and `RLM` set this so the model calls `SUBMIT(answer=…, confidence=…)` rather
    /// than passing one positional value; without it the sandbox keeps its single-argument default.
    pub fn with_output_fields(self, outputs: impl IntoIterator<Item = OutputField>) -> Self {
        *self.outputs.lock().expect("the output fields") = outputs.into_iter().collect();
        self
    }

    /// Whether `deno` is on the path, so a program can refuse early and say why.
    ///
    /// dspy answers the same question by failing the first `execute` with install instructions.
    /// Asking first is what lets a caller pick a different interpreter instead.
    pub fn available() -> bool {
        Command::new("deno")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    /// Start the child if there is not a live one, and answer with the session to talk to.
    fn started(&self, session: &mut Option<Session>) -> Result<()> {
        if let Some(live) = session
            && live.child.try_wait().is_ok_and(|exited| exited.is_none())
        {
            return Ok(());
        }
        let runner = command::runner_path()?;
        let mut child = Command::new("deno")
            .args(command::argv(&runner, &self.permissions))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context(
                "deno is not installed. The sandbox is Deno and Pyodide, as dspy's is: \
                 `curl -fsSL https://deno.land/install.sh | sh`, or `brew install deno`",
            )?;
        let writer = child.stdin.take().expect("stdin was piped");
        let reader = child.stdout.take().expect("stdout was piped");
        *session = Some(Session {
            child,
            rpc: Rpc::new(writer, reader),
            registered: false,
        });
        Ok(())
    }

    /// Tell the sandbox about the tools and the shape of `SUBMIT`, once per child.
    ///
    /// `runner.js` writes a Python `def` per entry, so this has to happen before any code runs and
    /// again after a restart — a new child knows nothing about either.
    fn register(&self, session: &mut Session) -> Result<()> {
        if session.registered {
            return Ok(());
        }
        let tools = self.tools.lock().expect("the tool list").clone();
        let outputs = self.outputs.lock().expect("the output fields").clone();
        if let Some(params) = register::params(&tools, &outputs) {
            let id = session.rpc.request("register", params)?;
            let answer = session.rpc.receive("while registering tools and outputs")?;
            rpc::answered(&answer, id, "while registering tools and outputs")?;
        }
        session.registered = true;
        Ok(())
    }

    /// Run the code and read the conversation to its end, answering tool calls on the way.
    fn ask(&self, session: &mut Session, code: &str) -> Result<Executed> {
        let id = session.rpc.request("execute", json!({ "code": code }))?;
        loop {
            let message = session.rpc.receive("during execution")?;
            // A `method` is the sandbox asking *us* something, which is only ever a tool call.
            if message.get("method").and_then(Value::as_str) == Some("tool_call") {
                self.answer_tool_call(session, &message)?;
                continue;
            }
            let result = rpc::answered(&message, id, "during execution")?;
            // dspy encodes `SUBMIT(...)` as a success carrying `final`; anything else is whatever
            // the code printed, and `null` where it printed nothing.
            return Ok(match result.get("final") {
                Some(final_output) => Executed::Submitted(final_output.clone()),
                None => Executed::Printed(result.get("output").cloned().unwrap_or(Value::Null)),
            });
        }
    }

    /// Run one tool the sandboxed code called, and hand the answer back through the pipe.
    fn answer_tool_call(&self, session: &mut Session, request: &Value) -> Result<()> {
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
        let tools = self.tools.lock().expect("the tool list");
        let Some(tool) = tools.iter().find(|tool| tool.name() == name) else {
            bail!("Unknown tool: {name}");
        };
        let answered = tool.call_value(arguments)?;
        Ok(match answered.is_array() || answered.is_object() {
            true => json!({ "value": answered.to_string(), "type": "json" }),
            false => json!({
                "value": answered.as_str().map(str::to_owned)
                    .unwrap_or_else(|| match answered.is_null() {
                        true => String::new(),
                        false => answered.to_string(),
                    }),
                "type": "string",
            }),
        })
    }
}

impl CodeInterpreter for DenoInterpreter {
    fn execute(&self, code: &str, variables: &Map<String, Value>) -> Result<Executed> {
        let prepared = super::variables::prepended(code, variables)?;
        let mut session = self.session.lock().expect("the sandbox session");
        self.started(&mut session)?;
        let live = session.as_mut().expect("a session was started");
        self.register(live)?;
        self.ask(live, &prepared)
    }

    /// Upstream's interpreter dispatches host functions the sandboxed code calls back into. Holding
    /// them is all this needs: `runner.js` asks by name, and [`Self::answer_tool_call`] answers.
    fn define_tools(&self, tools: &[Arc<dyn Tool>]) -> Result<()> {
        *self.tools.lock().expect("the tool list") = tools.to_vec();
        Ok(())
    }

    fn shutdown(&self) {
        if let Some(mut session) = self.session.lock().expect("the sandbox session").take() {
            let _ = session.child.kill();
            let _ = session.child.wait();
        }
    }
}

impl Drop for DenoInterpreter {
    /// A child outliving its interpreter is a leaked process, and a module that ends by dropping
    /// rather than by calling `shutdown` is an ordinary way to end.
    fn drop(&mut self) {
        self.shutdown();
    }
}
