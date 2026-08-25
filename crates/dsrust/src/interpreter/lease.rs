//! Who owns the interpreter a forward pass runs in, and who shuts it down.
//!
//! dspy 3.3.0 moved the sandbox from the module to the call: `__init__` takes an
//! `interpreter_factory`, `forward` builds one, and a `finally` shuts it down. Before that a module
//! held one for its lifetime, which is what this crate did — and still would, except that a
//! long-lived sandbox is a long-lived child process, so "one per module" and "one per call" are
//! different programs rather than different spellings.
//!
//! The rule is upstream's `_interpreter_context`, and it is entirely about ownership:
//!
//!   - a caller who hands one in keeps it. The module uses it and does **not** shut it down, so the
//!     same interpreter can carry state across several calls, which is what a caller passing one
//!     wants;
//!   - otherwise the factory makes one, and the module shuts it down whatever happens — upstream's
//!     `finally`, so a failed pass releases the process as surely as a successful one does.
//!
//! Getting that backwards either leaks a process per call or shuts down an interpreter the caller
//! is still using, and neither shows up as a wrong answer.

use std::sync::Arc;

use anyhow::Result;

use super::CodeInterpreter;

/// dspy's `interpreter_factory`: builds an interpreter for one forward pass.
///
/// Fallible, where upstream's is not, because building one here starts a process and that can fail
/// — Python raises out of the same place. Upstream's `_create_interpreter` also checks the factory
/// returned a `CodeInterpreter` at all, which the type does here.
///
/// `Send + Sync` because upstream says the callable "may be invoked concurrently", and a module is
/// shared across a `Parallel`'s threads.
pub type InterpreterFactory = Arc<dyn Fn() -> Result<Arc<dyn CodeInterpreter>> + Send + Sync>;

/// A factory that builds one of these each time it is called.
pub fn factory<I, F>(build: F) -> InterpreterFactory
where
    I: CodeInterpreter + 'static,
    F: Fn() -> I + Send + Sync + 'static,
{
    Arc::new(move || Ok(Arc::new(build()) as Arc<dyn CodeInterpreter>))
}

/// A factory that hands back the same interpreter every time.
///
/// For a scripted double a test wants to read afterwards, and for a caller who genuinely has one
/// environment to run everything in. **It is still module-owned**, so the first pass shuts it down
/// and a second gets an interpreter that has already stopped — which is the trap the per-pass change
/// exists to remove. A caller who wants one interpreter across several passes hands it to the
/// module's `ask_in` instead, where the lease is borrowed and nothing shuts it down.
pub fn handing_back(interpreter: Arc<dyn CodeInterpreter>) -> InterpreterFactory {
    Arc::new(move || Ok(interpreter.clone()))
}

/// An interpreter for the duration of one forward pass, and whether this pass has to close it.
///
/// Named for what it is: a borrow with a rule attached. `Drop` does the shutting down, so an early
/// `?` releases the process exactly as a clean return does — which is the whole of upstream's
/// ```
/// use dsrust::interpreter::{DenoInterpreter, Lease};
/// use std::sync::Arc;
///
/// // A caller's own interpreter survives the pass — dspy's `if interpreter is not None: yield;
/// // return`, with no `finally`, because the caller is still holding it.
/// let mine: Arc<dyn dsrust::interpreter::CodeInterpreter> = Arc::new(DenoInterpreter::new());
/// let borrowed = Lease::borrowed(Arc::clone(&mine));
/// drop(borrowed);
/// // `mine` is still usable here: dropping a borrowed lease shuts nothing down.
/// assert_eq!(Arc::strong_count(&mine), 1);
/// ```
/// `try/finally` and the half a hand-written call at the end of `forward` would miss.
pub struct Lease {
    interpreter: Arc<dyn CodeInterpreter>,
    owned: bool,
}

impl Lease {
    /// The caller's own interpreter, which this pass must not shut down.
    ///
    /// dspy's `if interpreter is not None: yield interpreter; return` — no `finally`, because the
    /// caller is still holding it.
    pub fn borrowed(interpreter: Arc<dyn CodeInterpreter>) -> Self {
        Self {
            interpreter,
            owned: false,
        }
    }

    /// One built for this pass, which this pass shuts down.
    ///
    /// Deliberately does **not** call `start`. Upstream's `_interpreter_context` creates, injects
    /// and yields — nothing starts there — and `RLM` starts later, inside
    /// `_prepare_serializable_vars`, *after* its tools and output fields are injected. Starting
    /// here reordered that, and the ordering is real: `define_outputs` clears the registration flag
    /// on a *live* session so the running child is told about the new fields, which is a different
    /// path from setting them before any child exists.
    ///
    /// So each module keeps its own `start`, where upstream keeps it.
    pub fn created(factory: &InterpreterFactory) -> Result<Self> {
        Ok(Self {
            interpreter: factory()?,
            owned: true,
        })
    }

    /// Whichever of the two this pass asked for: the caller's, or a fresh one.
    pub fn open(
        factory: &InterpreterFactory,
        caller: Option<Arc<dyn CodeInterpreter>>,
    ) -> Result<Self> {
        match caller {
            Some(interpreter) => Ok(Self::borrowed(interpreter)),
            None => Self::created(factory),
        }
    }

    /// The interpreter itself, for the duration.
    pub fn get(&self) -> &Arc<dyn CodeInterpreter> {
        &self.interpreter
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.owned {
            self.interpreter.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::Executed;
    use serde_json::{Map, Value};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts what was asked of it, so ownership is observable rather than argued.
    #[derive(Default)]
    struct Counting {
        started: AtomicUsize,
        stopped: AtomicUsize,
    }

    impl CodeInterpreter for Counting {
        fn execute(&self, _code: &str, _variables: &Map<String, Value>) -> Result<Executed> {
            Ok(Executed::Printed(Value::Null))
        }
        fn start(&self) -> Result<()> {
            self.started.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn shutdown(&self) {
            self.stopped.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn counting() -> (InterpreterFactory, Arc<Counting>) {
        let built = Arc::new(Counting::default());
        let handed = built.clone();
        let make: InterpreterFactory =
            Arc::new(move || Ok(handed.clone() as Arc<dyn CodeInterpreter>));
        (make, built)
    }

    /// A factory-built interpreter is shut down when the pass ends — and is *not* started here.
    ///
    /// Upstream's context manager creates, injects and yields; `RLM` starts afterwards, once its
    /// tools and output fields are in. Starting here would put the start before the injection.
    #[test]
    fn a_created_lease_shuts_down_but_does_not_start() {
        let (make, seen) = counting();
        {
            let lease = Lease::open(&make, None).expect("opens");
            assert_eq!(
                seen.started.load(Ordering::SeqCst),
                0,
                "the module starts it"
            );
            assert_eq!(seen.stopped.load(Ordering::SeqCst), 0, "not while in use");
            let _ = lease.get();
        }
        assert_eq!(seen.stopped.load(Ordering::SeqCst), 1);
    }

    /// A caller's own interpreter is neither started nor shut down: they own it.
    ///
    /// Upstream returns before its `try`, so its `finally` never runs. Shutting one down here would
    /// close a process the caller is still holding, and nothing about the answer would look wrong.
    #[test]
    fn a_borrowed_lease_is_left_alone() {
        let (make, seen) = counting();
        let theirs: Arc<dyn CodeInterpreter> = seen.clone();
        {
            let lease = Lease::open(&make, Some(theirs)).expect("opens");
            let _ = lease.get();
        }
        assert_eq!(
            seen.started.load(Ordering::SeqCst),
            0,
            "the caller started it"
        );
        assert_eq!(
            seen.stopped.load(Ordering::SeqCst),
            0,
            "the caller closes it"
        );
    }

    /// The shutdown is `Drop`, so a pass that fails releases the process too — upstream's `finally`.
    #[test]
    fn a_failed_pass_still_shuts_its_interpreter_down() {
        let (make, seen) = counting();
        let ran: Result<()> = (|| {
            let _lease = Lease::open(&make, None)?;
            anyhow::bail!("the model never answered")
        })();
        assert!(ran.is_err());
        assert_eq!(seen.stopped.load(Ordering::SeqCst), 1);
    }

    /// Each pass builds its own, which is the change: two passes are two interpreters.
    #[test]
    fn two_passes_do_not_share_one_interpreter() {
        let built = Arc::new(AtomicUsize::new(0));
        let counted = built.clone();
        let make: InterpreterFactory = Arc::new(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(Counting::default()) as Arc<dyn CodeInterpreter>)
        });
        for _ in 0..2 {
            let _lease = Lease::open(&make, None).expect("opens");
        }
        assert_eq!(built.load(Ordering::SeqCst), 2);
    }
}
