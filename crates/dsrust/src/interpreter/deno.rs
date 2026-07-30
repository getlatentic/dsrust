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
mod files;
mod register;
mod rpc;

use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

pub use command::Permissions;

use super::{CodeInterpreter, Executed, OutputField};
use crate::react::Tool;
use rpc::Rpc;

/// dspy's `JSONRPC_APP_ERRORS["Unknown"]`, which a tool failure answers under.
const UNKNOWN_ERROR: i64 = -32099;

/// How long to let a child finish writing files back before killing it. Upstream waits without a
/// bound; a bound is here so a wedged sandbox cannot hang the program that is done with it.
const SHUTDOWN_POLL: std::time::Duration = std::time::Duration::from_millis(20);
const SHUTDOWN_POLLS: u32 = 250;

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
    /// dspy's `sync_files`, and its default: a writable file's sandbox copy is written back to the
    /// host after each run. Off means the sandbox may write and the host never sees it.
    write_back: bool,
}

/// One live child and the pipes to it.
struct Session {
    child: Child,
    rpc: Rpc,
    /// Whether this child has been told about the tools and outputs. A restart clears it with the
    /// rest of the session, which is what replays the registration upstream replays.
    registered: bool,
    /// Whether the host's files are in this child's filesystem. Pyodide's is in memory, so a new
    /// child starts empty and the mounts replay with the registration.
    mounted: bool,
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
            write_back: true,
        }
    }

    /// Leave the host's files alone: the sandbox may still write, and nothing is copied back.
    ///
    /// dspy's `sync_files=False`. Worth reaching for when the sandbox writes scratch output that
    /// would otherwise overwrite the file a caller handed in.
    pub fn without_write_back(mut self) -> Self {
        self.write_back = false;
        self
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
            mounted: false,
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

    /// Copy the host's readable and writable files into the sandbox's own filesystem, once per
    /// child. Pyodide's filesystem is in memory, so a granted path is still not an openable one.
    fn mount(&self, session: &mut Session) -> Result<()> {
        if session.mounted {
            return Ok(());
        }
        for (host, virtual_at) in files::to_mount(&self.permissions.read, &self.permissions.write)?
        {
            let id = session
                .rpc
                .request("mount_file", files::mount_request(&host, &virtual_at))?;
            let answer = session.rpc.receive("while mounting files")?;
            rpc::answered(&answer, id, "while mounting files")?;
        }
        session.mounted = true;
        Ok(())
    }

    /// Write each writable file's sandbox copy back to the host.
    ///
    /// A notification rather than a request, as upstream sends it: there is no reply to wait for,
    /// and waiting for one would hang on the next execution's output.
    fn sync(&self, session: &mut Session) -> Result<()> {
        if !self.write_back {
            return Ok(());
        }
        for params in files::to_sync(&self.permissions.write) {
            session.rpc.notify("sync_file", params)?;
        }
        Ok(())
    }

    /// Write each oversized value into the sandbox's filesystem, where the code reads it back.
    ///
    /// Every execution, not once per child: the values belong to this call's variables, and the
    /// previous call's are neither wanted nor still named by the code about to run.
    fn inject(&self, session: &mut Session, large: &[(String, String)]) -> Result<()> {
        for (name, payload) in large {
            let id = session
                .rpc
                .request("inject_var", json!({ "name": name, "value": payload }))?;
            let answer = session.rpc.receive("while injecting a large variable")?;
            rpc::answered(&answer, id, "while injecting a large variable")?;
        }
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
            // Upstream syncs before reading `final`, so a run that submitted still writes its
            // files back — the answer and the side effects are not either/or.
            self.sync(session)?;
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
        let answered = crate::observe::tool_call(tool.as_ref(), arguments)?;
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
        let prepared = super::variables::prepared(code, variables)?;
        let mut session = self.session.lock().expect("the sandbox session");
        self.started(&mut session)?;
        let live = session.as_mut().expect("a session was started");
        self.register(live)?;
        self.mount(live)?;
        self.inject(live, &prepared.large)?;
        self.ask(live, &prepared.code)
    }

    /// Upstream's interpreter dispatches host functions the sandboxed code calls back into. Holding
    /// them is all this needs: `runner.js` asks by name, and this answers.
    fn define_tools(&self, tools: &[Arc<dyn Tool>]) -> Result<()> {
        *self.tools.lock().expect("the tool list") = tools.to_vec();
        Ok(())
    }

    /// Ask the sandbox to stop, and let it finish before it does.
    ///
    /// Killing the child here loses whatever it had not read yet, and `sync_file` is exactly that:
    /// a notification already written to the pipe. Upstream sends `shutdown`, closes stdin and
    /// waits, which is what makes a file written in the sandbox appear on the host. A child that
    /// will not exit is killed, because a caller dropping an interpreter should not hang.
    /// Hold the fields, and clear the registration so a live child is told about them.
    fn define_outputs(&self, outputs: &[OutputField]) -> Result<()> {
        *self.outputs.lock().expect("the output fields") = outputs.to_vec();
        if let Some(live) = self.session.lock().expect("the sandbox session").as_mut() {
            live.registered = false;
        }
        Ok(())
    }

    fn shutdown(&self) {
        let Some(session) = self.session.lock().expect("the sandbox session").take() else {
            return;
        };
        let Session {
            mut child, mut rpc, ..
        } = session;
        let _ = rpc.notify("shutdown", Value::Null);
        drop(rpc);
        for _ in 0..SHUTDOWN_POLLS {
            if matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
                return;
            }
            std::thread::sleep(SHUTDOWN_POLL);
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

impl Drop for DenoInterpreter {
    /// A child outliving its interpreter is a leaked process, and a module that ends by dropping
    /// rather than by calling `shutdown` is an ordinary way to end.
    fn drop(&mut self) {
        self.shutdown();
    }
}
