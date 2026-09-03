//! dspy 3.3.1's four interpreter points: one execution, one tool the sandboxed code called back
//! into, and the two ends of a sandbox's life.
//!
//! Public for the reason the rest of [`super`] is: a caller running their own interpreter puts it
//! on the same callback tree this crate's own sandbox is on, rather than beside it.
use anyhow::Result;
use serde_json::Value;
use tracing::field;

use super::{TARGET, Watch, opening};
use crate::callback;

/// dspy 3.3.1 `on_interpreter_execute_start`/`_end`: one execution in a sandbox.
pub fn executing(
    interpreter: &'static str,
    code: &str,
    running: impl FnOnce() -> Result<crate::interpreter::Executed>,
) -> Result<crate::interpreter::Executed> {
    let watch = interpreter_point("execute", interpreter);
    let _entered = watch.span.enter();
    watch.shown(|| code.to_owned());
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_interpreter_execute_start(&watch.call, interpreter, code)
        });
    }
    let _under = callback::entered(&watch.call);
    let answered = running();
    watch.finished(answered.as_ref(), |executed| format!("{executed:?}"));
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_interpreter_execute_end(&watch.call, answered.as_ref())
        });
    }
    answered
}

/// dspy 3.3.1 `on_interpreter_tool_call_start`/`_end`: sandboxed code calling back into one of the
/// interpreter's tools. The tool's own point ([`super::tool_call`]) opens inside this one, as upstream's
/// `invoke_tool` wraps `Tool.__call__`.
pub fn interpreter_tool_call(
    tool: &str,
    args: &Value,
    running: impl FnOnce() -> Result<Value>,
) -> Result<Value> {
    let watch = interpreter_point("tool_call", "DenoInterpreter");
    let _entered = watch.span.enter();
    watch.shown(|| args.to_string());
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_interpreter_tool_call_start(&watch.call, tool, args)
        });
    }
    let _under = callback::entered(&watch.call);
    let answered = running();
    watch.finished(answered.as_ref(), Value::to_string);
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_interpreter_tool_call_end(&watch.call, answered.as_ref())
        });
    }
    answered
}

/// dspy 3.3.1 `on_interpreter_startup_start`/`_end` and the shutdown pair: the sandbox coming up
/// — which upstream reports on every execution, since each one makes sure it is running — or
/// going down.
pub fn interpreter_lifecycle(
    point: InterpreterLifecycle,
    interpreter: &'static str,
    running: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let watch = interpreter_point(point.name(), interpreter);
    let _entered = watch.span.enter();
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| match point {
            InterpreterLifecycle::Startup => {
                callback.on_interpreter_startup_start(&watch.call, interpreter)
            }
            InterpreterLifecycle::Shutdown => {
                callback.on_interpreter_shutdown_start(&watch.call, interpreter)
            }
        });
    }
    let _under = callback::entered(&watch.call);
    let answered = running();
    watch.finished(answered.as_ref(), |()| String::new());
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| match point {
            InterpreterLifecycle::Startup => {
                callback.on_interpreter_startup_end(&watch.call, answered.as_ref().copied())
            }
            InterpreterLifecycle::Shutdown => {
                callback.on_interpreter_shutdown_end(&watch.call, answered.as_ref().copied())
            }
        });
    }
    answered
}

/// The two ends of a sandbox's life that dspy 3.3.1 reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpreterLifecycle {
    Startup,
    Shutdown,
}

impl InterpreterLifecycle {
    fn name(self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::Shutdown => "shutdown",
        }
    }
}

fn interpreter_point(point: &'static str, interpreter: &'static str) -> Watch {
    opening(tracing::info_span!(
        target: TARGET,
        "interpreter",
        point = point,
        interpreter = interpreter,
        inputs = field::Empty,
        outputs = field::Empty,
        error = field::Empty,
    ))
}
