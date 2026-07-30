//! dspy `primitives/code_interpreter.py`: the seam a code-running module executes through.
//!
//! Upstream states this as a `Protocol` with `PythonInterpreter` — a Deno/Pyodide sandbox — as one
//! implementation and a scriptable mock as another. The same split lives here: the modules that
//! *write* code ([`ProgramOfThought`](crate::predict::ProgramOfThought)) are the crate's, and what
//! *runs* it is the caller's, supplied as a [`CodeInterpreter`].
//!
//! [`DenoInterpreter`] is that sandbox, and it is upstream's: dspy's own `runner.js` is vendored
//! and executed unchanged, so generated code lands in the same Pyodide with the same `SUBMIT` and
//! the same captured stdout. `deno` is a prerequisite rather than a dependency, which is the
//! arrangement dspy already asks of its users.
//!
//! The seam stays a trait — the same shape as [`ChatModel`](crate::lm::ChatModel) and
//! [`Tool`] — because what runs a model's code reaches a prompt: float
//! formatting, exception text and dict ordering all land in the next ask. A second interpreter is
//! a documented divergence a caller opts into, not a swap.

use std::sync::Arc;

use anyhow::Result;
use serde_json::{Map, Value};

use crate::react::Tool;

pub mod deno;
pub mod repl;
pub mod sandbox;
mod variables;

pub use deno::{DenoInterpreter, Permissions};
pub use repl::{ReplEntry, ReplHistory, ReplVariable};
pub use sandbox::{SandboxSerializable, build_repl_variable, with_constraints};

/// What one execution produced.
///
/// dspy's `execute` returns `FinalOutput` when the code called the preloaded `SUBMIT()`, and the
/// captured stdout otherwise; the two mean different things to the loop, so they are distinct here
/// rather than a value plus a flag.
#[derive(Debug, Clone, PartialEq)]
pub enum Executed {
    /// The code called `SUBMIT(...)`: this is the answer, and the loop is done.
    Submitted(Value),
    /// Whatever the code printed. `Value::Null` where it printed nothing.
    Printed(Value),
}

impl Executed {
    /// The value either way — what a module records as the code's output.
    pub fn value(&self) -> &Value {
        match self {
            Executed::Submitted(value) | Executed::Printed(value) => value,
        }
    }

    pub fn is_submitted(&self) -> bool {
        matches!(self, Executed::Submitted(_))
    }
}

/// One field a typed `SUBMIT` takes.
///
/// dspy sends `{"name": …, "type": …}` per output, and `runner.js` writes the `def` from it — so a
/// type is optional here for the same reason it is optional on a tool's argument: an annotation the
/// generated signature cannot carry is left off rather than guessed at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputField {
    pub name: String,
    pub python_type: Option<String>,
}

impl OutputField {
    /// A field with no annotation, which is what a signature's own output names give.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            python_type: None,
        }
    }
}

/// An environment that runs generated code.
///
/// State persists across calls within one session, as upstream's does: a name bound by one
/// `execute` is in scope for the next, until [`shutdown`](Self::shutdown).
///
/// `&self` rather than `&mut self` because a module holds its interpreter behind a shared
/// reference and [`Module::forward`](crate::module::Module::forward) takes `&self`; an
/// implementation that owns a subprocess keeps it behind its own lock, exactly as
/// [`Tool`] does.
pub trait CodeInterpreter: Send + Sync {
    /// Run the code and answer with what it produced.
    ///
    /// `variables` are bound in the namespace before the code runs, which is how a module puts its
    /// caller's inputs where generated code can reach them — `RLM` passes its input fields every
    /// turn, so `SUBMIT(sum(numbers))` can see `numbers`. The other two modules pass none.
    ///
    /// An error is the code's own failure — an undefined name, a raised exception — and a module
    /// feeds it back to the model as the error to correct, so it reaches a prompt and should read
    /// the way upstream's does.
    fn execute(&self, code: &str, variables: &Map<String, Value>) -> Result<Executed>;

    /// Make these tools callable from generated code, by the names they carry.
    ///
    /// Upstream's Protocol carries a `tools` property for exactly this — host functions the
    /// sandboxed code calls back into — and its own interpreters dispatch to them. dspy's `CodeAct`
    /// takes the other route open to Python and executes each tool function's *source* in the
    /// sandbox (`inspect.getsource`); a Rust tool has no source to inject, so the tools are handed
    /// over as values and the interpreter arranges the callback. Same effect, through the seam
    /// upstream already provides.
    ///
    /// Doing nothing is a valid answer for an interpreter that was built knowing its tools.
    fn define_tools(&self, tools: &[Arc<dyn Tool>]) -> Result<()> {
        let _ = tools;
        Ok(())
    }

    /// The fields a typed `SUBMIT` takes, in order — dspy's `output_fields`.
    ///
    /// A multi-output signature cannot submit without this: the sandbox keeps a single-argument
    /// `SUBMIT(value)` whose result arrives under `output`, and every field the signature asked for
    /// is then missing. Upstream builds its own interpreter to pass them, which is the same wiring
    /// through the seam this crate has.
    ///
    /// Doing nothing is a valid answer for an interpreter that was built knowing its signature.
    fn define_outputs(&self, outputs: &[OutputField]) -> Result<()> {
        let _ = outputs;
        Ok(())
    }

    /// Allocate whatever the environment needs, ahead of the first [`execute`](Self::execute).
    /// Doing nothing is a valid answer, and calling it twice must be safe.
    fn start(&self) -> Result<()> {
        Ok(())
    }

    /// Release the environment. A module calls this when its episode ends, and upstream's modules
    /// do so on the way out of `forward` — including the way out that raises.
    fn shutdown(&self) {}
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Mutex;

    /// dspy's `MockInterpreter`: canned answers in order, so a test drives the loop without a
    /// sandbox. The shutdown count is recorded because upstream's modules promise to call it.
    pub(crate) struct Scripted {
        answers: Mutex<std::collections::VecDeque<Result<Executed, String>>>,
        pub(crate) ran: Mutex<Vec<String>>,
        /// What each `execute` was asked to bind, so a module's variable passing is checkable.
        pub(crate) bound: Mutex<Vec<Map<String, Value>>>,
        pub(crate) shutdowns: Mutex<usize>,
        /// What a module registered as its typed `SUBMIT`, so a test can see it was told.
        pub(crate) outputs: Mutex<Vec<OutputField>>,
    }

    impl Scripted {
        pub(crate) fn new(answers: impl IntoIterator<Item = Result<Executed, String>>) -> Self {
            Self {
                answers: Mutex::new(answers.into_iter().collect()),
                ran: Mutex::new(Vec::new()),
                bound: Mutex::new(Vec::new()),
                shutdowns: Mutex::new(0),
                outputs: Mutex::new(Vec::new()),
            }
        }
    }

    impl CodeInterpreter for Scripted {
        fn define_outputs(&self, outputs: &[OutputField]) -> Result<()> {
            *self.outputs.lock().expect("the output fields") = outputs.to_vec();
            Ok(())
        }

        fn execute(&self, code: &str, variables: &Map<String, Value>) -> Result<Executed> {
            self.ran.lock().expect("ran").push(code.to_owned());
            self.bound.lock().expect("bound").push(variables.clone());
            match self.answers.lock().expect("answers").pop_front() {
                Some(Ok(executed)) => Ok(executed),
                Some(Err(error)) => Err(anyhow::anyhow!(error)),
                None => Err(anyhow::anyhow!("the script ran out of answers")),
            }
        }

        fn shutdown(&self) {
            *self.shutdowns.lock().expect("shutdowns") += 1;
        }
    }

    #[test]
    fn a_submitted_value_is_distinguishable_from_printed_output() {
        let submitted = Executed::Submitted(serde_json::json!({ "answer": 2 }));
        assert!(submitted.is_submitted());
        assert_eq!(submitted.value(), &serde_json::json!({ "answer": 2 }));
        assert!(!Executed::Printed(serde_json::json!("2")).is_submitted());
    }
}
