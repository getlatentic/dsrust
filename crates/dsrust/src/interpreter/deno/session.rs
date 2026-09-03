//! The child process, and what ending it means.
//!
//! Separated from running code because the two answer different questions. `deno.rs` asks "what did
//! this code do"; this asks "is there still a sandbox to ask, and if not, why not" — and the second
//! is where the failures that are worth explaining live.

use std::process::Child;

use anyhow::Result;

use super::{DenoInterpreter, InterpreterFailure, Rpc};

/// One live child and the pipes to it.
pub(super) struct Session {
    pub(super) child: Child,
    pub(super) rpc: Rpc,
    /// Deno's own stderr, read only when the session fails.
    ///
    /// A child that dies at startup — the commonest reason being that it could not resolve
    /// `npm:pyodide` — writes the reason here and nothing to stdout, so the RPC layer sees a closed
    /// pipe and can say only that. The explanation is sitting in this handle, and reporting "the
    /// sandbox closed its output" without it leaves a caller with nothing to act on where dspy
    /// hands them the missing package by name.
    pub(super) errors: Option<std::process::ChildStderr>,
    /// Whether this child has been told about the tools and outputs. A restart clears it with the
    /// rest of the session, which is what replays the registration upstream replays.
    pub(super) registered: bool,
    /// Whether the host's files are in this child's filesystem. Pyodide's is in memory, so a new
    /// child starts empty and the mounts replay with the registration.
    pub(super) mounted: bool,
}

impl DenoInterpreter {
    /// dspy's `_check_session_active`: refuse once the session has ended.
    pub(super) fn check_active(&self) -> Result<()> {
        match self.ended.load(std::sync::atomic::Ordering::SeqCst) {
            true => Err(anyhow::Error::new(InterpreterFailure::Session(
                "PythonInterpreter session has ended; create a new interpreter for a fresh session."
                    .to_owned(),
            ))),
            false => Ok(()),
        }
    }

    /// A session failure, with whatever deno wrote to explain itself.
    ///
    /// The RPC layer sees only a closed pipe and can say only that. Deno's reason — an unresolvable
    /// `npm:pyodide`, a permission it was not granted — went to stderr, and a caller told "the
    /// sandbox closed its output" has nothing to act on. dspy names the missing package; this reads
    /// the handle so it can too.
    pub(super) fn explained(session: &mut Session, failure: anyhow::Error) -> anyhow::Error {
        if failure.downcast_ref::<InterpreterFailure>().is_none() {
            return failure;
        }
        let Some(errors) = session.errors.as_mut() else {
            return failure;
        };
        // The child is gone or going, so this read ends rather than blocking on a live pipe.
        let mut said = String::new();
        let _ = std::io::Read::read_to_string(errors, &mut said);
        let said = said.trim();
        match said.is_empty() {
            true => failure,
            false => anyhow::Error::new(InterpreterFailure::Session(format!(
                "{failure}. Deno said: {said}"
            ))),
        }
    }

    /// End the session when the failure was the interpreter's, and pass the result through.
    ///
    /// The two error kinds are built where they happen — down in the RPC layer, which knows the
    /// JSON-RPC code — and this is where the consequence is applied. Upstream's
    /// `_raise_terminal_error` does both at once; here the sandbox conversation cannot reach the
    /// flag, so the flag reaches out to it.
    pub(super) fn note_terminal<T>(&self, outcome: Result<T>) -> Result<T> {
        if let Err(error) = &outcome
            && matches!(
                error.downcast_ref::<InterpreterFailure>(),
                Some(InterpreterFailure::Session(_))
            )
        {
            self.ended.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        outcome
    }

    /// dspy's `_raise_terminal_error`: end the session and report why.
    ///
    /// Returns the error rather than raising, so a caller writes `return Err(self.end_session(…))`
    /// and the compiler keeps the two halves together.
    pub(super) fn end_session(&self, why: String) -> anyhow::Error {
        self.ended.store(true, std::sync::atomic::Ordering::SeqCst);
        anyhow::Error::new(InterpreterFailure::Session(why))
    }
}
