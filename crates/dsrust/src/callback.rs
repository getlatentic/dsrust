//! dspy's `BaseCallback`: the twelve handlers a caller implements to watch a run.
//!
//! Upstream is a base class whose methods are no-ops, subclassed and registered through
//! `dspy.configure(callbacks=[…])`. A base class of no-ops is a Rust trait with defaulted methods,
//! so this is close to a transcription — [`Callback`] has upstream's twelve handlers, each
//! defaulted, and [`configure_callbacks`] is upstream's registration.
//!
//! Each handler is typed to the value its point carries rather than to `Any`, which is the only
//! difference that changes a signature: upstream's `inputs` is a dict assembled by
//! `inspect.getcallargs` because Python has no other way to name a call's arguments.
//!
//! [`crate::observe`] fires these, at the same six points it opens a span. Both, not either: a span
//! is the identifier, the parent linkage and the start/end at once — where upstream needs a uuid,
//! a context variable and paired handlers — and a subscriber can neither mutate what it is shown
//! nor break the run. A caller who wants those properties keeps using spans and registers nothing.
//!
//! ```no_run
//! # use std::sync::Arc;
//! use dsrust::{Callback, CallId, Example, configure_callbacks};
//!
//! struct Logging;
//!
//! impl Callback for Logging {
//!     fn on_module_start(&self, _call: &CallId, module: &str, inputs: &Example) {
//!         println!("{module} was asked {}", dsrust::observe::as_json(inputs));
//!     }
//! }
//!
//! configure_callbacks([Arc::new(Logging) as Arc<dyn Callback>]);
//! ```

use std::sync::{Arc, RwLock};

use anyhow::Error;
use serde_json::Value;

use crate::adapter::Input;
use crate::evaluate::{Evaluation, Pass};
use crate::example::{Example, Prediction};
use crate::lm::api;
use crate::signature::Signature;

mod context;

pub use context::CallId;
pub(crate) use context::{Under, entered};

/// What an adapter rendered, for [`Callback::on_adapter_format_end`]: the message list, which is
/// what dspy's `format` answers with and the value the adapter actually produced.
pub struct Rendered<'a> {
    pub messages: &'a [api::LmMessage],
}

/// dspy's `BaseCallback`: implement the handlers you want, ignore the rest.
///
/// Every method is defaulted to doing nothing, so a callback that only watches models is the two
/// lm methods and no others — which is what upstream's no-op base class gives a Python subclass.
///
/// `call` identifies one call and is the same value at its start and its end. The call that
/// enclosed it is [`CallId::parent`], so a handler can rebuild the tree a program ran as.
///
/// A handler runs on the thread that reached the point, in the order the list was registered, and
/// it is not passed anything it could mutate. A handler that panics is caught and logged rather
/// than allowed to end the run, which is what upstream's `try/except` around every handler is for.
pub trait Callback: Send + Sync {
    /// A module's `forward` was entered. `module` is the type — `Predict`, `ReAct` — which is what
    /// upstream reads its `instance` for at this point.
    fn on_module_start(&self, call: &CallId, module: &str, inputs: &Example) {
        let _ = (call, module, inputs);
    }

    /// A module's `forward` returned, with what it answered or why it could not.
    fn on_module_end(&self, call: &CallId, answered: Result<&Prediction, &Error>) {
        let _ = (call, answered);
    }

    /// A model is about to be asked. The request names the model, so upstream's `instance` and its
    /// `inputs` are both here.
    fn on_lm_start(&self, call: &CallId, request: &api::LmRequest) {
        let _ = (call, request);
    }

    /// A model answered, or the call to it failed.
    fn on_lm_end(&self, call: &CallId, answered: Result<&api::LmResponse, &Error>) {
        let _ = (call, answered);
    }

    /// An adapter is about to render a prompt, with everything it renders from — dspy's `inputs`
    /// dict for this point, minus the `lm` and `lm_kwargs` an adapter here is not handed.
    fn on_adapter_format_start(
        &self,
        call: &CallId,
        adapter: &str,
        signature: &Signature,
        demos: &[Example],
        inputs: &[Input<'_>],
    ) {
        let _ = (call, adapter, signature, demos, inputs);
    }

    /// An adapter finished rendering.
    fn on_adapter_format_end(&self, call: &CallId, answered: Result<&Rendered<'_>, &Error>) {
        let _ = (call, answered);
    }

    /// An adapter is about to read a reply back into fields. `raw` is what the model said, which is
    /// the value a reader opens a trace for when a parse fails.
    fn on_adapter_parse_start(&self, call: &CallId, adapter: &str, raw: &str) {
        let _ = (call, adapter, raw);
    }

    /// An adapter finished reading a reply.
    fn on_adapter_parse_end(&self, call: &CallId, answered: Result<&Value, &Error>) {
        let _ = (call, answered);
    }

    /// A tool is about to run, with the arguments the model wrote.
    fn on_tool_start(&self, call: &CallId, tool: &str, args: &Value) {
        let _ = (call, tool, args);
    }

    /// A tool returned, or refused.
    fn on_tool_end(&self, call: &CallId, answered: Result<&Value, &Error>) {
        let _ = (call, answered);
    }

    /// A run over a devset began.
    ///
    /// The rows are handed over whole, as upstream's `inputs["devset"]` is — a handler that only
    /// wants the count takes it. `pass` is dspy's `callback_metadata`: which pass of a search this
    /// is, and `None` for a caller scoring directly.
    fn on_evaluate_start(
        &self,
        call: &CallId,
        devset: &[Example],
        threads: usize,
        pass: Option<Pass>,
    ) {
        let _ = (call, devset, threads, pass);
    }

    /// A run over a devset finished, with its score or why it gave up.
    ///
    /// The error is never a single row's: a failing row scores `failure_score` and the run carries
    /// on. It is the run abandoning the devset once `max_errors` rows have failed, which upstream
    /// raises out of `Evaluate.__call__` and reports here with `outputs=None`.
    fn on_evaluate_end(&self, call: &CallId, evaluated: Result<&Evaluation, &Error>) {
        let _ = (call, evaluated);
    }
}

static REGISTERED: RwLock<Vec<Arc<dyn Callback>>> = RwLock::new(Vec::new());

/// Watch every run in this process with these — dspy's `dspy.configure(callbacks=[…])`.
///
/// Replaces whatever was registered before, as upstream's does. Registering an empty list is how a
/// caller stops watching.
pub fn configure_callbacks(callbacks: impl IntoIterator<Item = Arc<dyn Callback>>) {
    *REGISTERED.write().expect("lock not poisoned") = callbacks.into_iter().collect();
}

/// The registered callbacks, cloned out so a handler can register more without deadlocking and so
/// nothing holds the lock across a point's own work.
///
/// Anything [`watched_by`] scoped comes after the process-wide ones,
/// because upstream appends to the list it inherited rather than replacing it.
pub(crate) fn registered() -> Vec<Arc<dyn Callback>> {
    let mut all = REGISTERED.read().expect("lock not poisoned").clone();
    all.extend(SCOPED.with(|scoped| scoped.borrow().clone()));
    all
}

thread_local! {
    /// Watchers added for one piece of work — dspy's
    /// `settings.context(callbacks=[*settings.callbacks, extra])`.
    static SCOPED: std::cell::RefCell<Vec<Arc<dyn Callback>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Watch this piece of work with `extra`, *beside* whatever is registered process-wide, then stop.
///
/// dspy's `streamify` does exactly this — `callbacks = list(settings.callbacks)`, append, then
/// `settings.context(callbacks=callbacks)` — and the reason is the same: a run that wants to be
/// narrated should not silence a caller's own watchers, and should not leave its own behind.
///
/// **It scopes a future, not a block**, for the reason [`crate::lm::context`] does: entered on each
/// poll and left when the poll returns, so two pieces of work interleaved in one task each see
/// their own watchers.
/// ```
/// use dsrust::callback::{CallId, Callback, watched_by};
/// use std::sync::{Arc, Mutex};
///
/// #[derive(Default)]
/// struct Counting(Mutex<usize>);
///
/// impl Callback for Counting {
///     fn on_module_start(&self, _call: &CallId, _module: &str, _inputs: &dsrust::Example) {
///         *self.0.lock().expect("not poisoned") += 1;
///     }
/// }
///
/// # async fn wrapper(program: dsrust::Predict) -> anyhow::Result<()> {
/// let counting = Arc::new(Counting::default());
/// // Additive and scoped: a caller's own process-wide watchers still hear this run, and this
/// // watcher hears nothing outside it.
/// watched_by(counting.clone()).run(program.call("a question")).await?;
/// println!("{} module calls", counting.0.lock().expect("not poisoned"));
/// # Ok(()) }
/// ```
pub fn watched_by(extra: Arc<dyn Callback>) -> Watching {
    Watching { extra }
}

/// A scope of extra watchers, from [`watched_by`].
pub struct Watching {
    extra: Arc<dyn Callback>,
}

impl Watching {
    /// Run `work` with the extra watcher listening.
    pub async fn run<T>(self, work: impl Future<Output = T>) -> T {
        Watched {
            extra: self.extra,
            inner: Box::pin(work),
        }
        .await
    }
}

struct Watched<F> {
    extra: Arc<dyn Callback>,
    inner: std::pin::Pin<Box<F>>,
}

impl<F: Future> Future for Watched<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<F::Output> {
        let watched = self.get_mut();
        SCOPED.with(|scoped| scoped.borrow_mut().push(Arc::clone(&watched.extra)));
        let answered = watched.inner.as_mut().poll(context);
        SCOPED.with(|scoped| {
            scoped.borrow_mut().pop();
        });
        answered
    }
}

/// Whether anything is watching this point. Every point asks this first, so a program that
/// registered nothing pays for no rendering — the same reason [`crate::observe`] checks a span for
/// interest.
pub(crate) fn watching(instance: &[Arc<dyn Callback>]) -> bool {
    !instance.is_empty()
        || !REGISTERED.read().expect("lock not poisoned").is_empty()
        || SCOPED.with(|scoped| !scoped.borrow().is_empty())
}

/// Tell every callback about one point, with a handler that cannot end the run.
///
/// The process-wide ones first, then the ones the instance carries — upstream's
/// `settings.get("callbacks", []) + getattr(instance, "callbacks", [])`, in that order.
///
/// dspy wraps each handler in `try/except` and logs. Rust's equivalent of an escaping exception is
/// an unwinding panic, so that is what is caught: a watcher is not part of the program's answer, and
/// one that is broken should not decide whether the answer is delivered.
pub(crate) fn tell(instance: &[Arc<dyn Callback>], each: impl Fn(&dyn Callback)) {
    let global = registered();
    for callback in global.iter().chain(instance) {
        let told =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| each(callback.as_ref())));
        if told.is_err() {
            tracing::warn!("a callback panicked; the run continues without it");
        }
    }
}

/// The value a watched point ends with, and which handler is told about it.
///
/// Implemented for the two things an asynchronous point answers with, so
/// [`observe::watching`](crate::observe::watching) is one function over both rather than a `describe`
/// argument and an `ended` argument that every call site has to get right together.
///
/// ```
/// use dsrust::callback::Ends;
///
/// // Implemented for the two things an asynchronous watched point answers with, so `watching` is
/// // one function over both rather than taking a `describe` argument and an `ended` argument that
/// // every call site has to keep in step.
/// fn describes<T: Ends>(answered: &T) -> String {
///     answered.describe()
/// }
/// # fn used<T: Ends>(a: &T) { let _ = describes(a); }
/// ```
pub trait Ends {
    /// What the span records — one line, not the whole value.
    fn describe(&self) -> String;

    /// The `on_*_end` handler for this point.
    fn ended(call: &CallId, instance: &[Arc<dyn Callback>], answered: Result<&Self, &Error>);
}

impl Ends for Prediction {
    fn describe(&self) -> String {
        crate::observe::as_json(&self.example)
    }

    fn ended(call: &CallId, instance: &[Arc<dyn Callback>], answered: Result<&Self, &Error>) {
        tell(instance, |callback| callback.on_module_end(call, answered));
    }
}

impl Ends for api::LmResponse {
    /// The text, whether it was replayed, and what it cost — not the whole response.
    ///
    /// dspy's `on_lm_end` is handed the outputs, and a span field is a line in a log: a reply's every
    /// part, its provider envelope and its logprobs would bury the values a reader is looking for.
    /// A callback is handed the response itself, unchanged, because it asked for it in Rust.
    fn describe(&self) -> String {
        let usage = self
            .usage
            .as_ref()
            .and_then(|usage| usage.total_tokens)
            .map_or_else(|| "null".to_owned(), |tokens| tokens.to_string());
        format!(
            "{{\"text\":{},\"cache_hit\":{},\"total_tokens\":{usage}}}",
            serde_json::json!(self.first_text()),
            self.cache_hit,
        )
    }

    fn ended(call: &CallId, instance: &[Arc<dyn Callback>], answered: Result<&Self, &Error>) {
        tell(instance, |callback| callback.on_lm_end(call, answered));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Registration is process-wide, as dspy's is, so a test that registers has to keep every other
    /// such test out meanwhile — the same reason `lm::global::install_for_test` exists.
    ///
    /// The lock is not enough on its own and cannot be: it excludes the other tests *in this
    /// module* and nothing about the dozens elsewhere in this binary that run a module and fire
    /// these same points. Two things close that gap — [`Counting::seen`] keys on the call, and the
    /// tests that can use [`watched_by`] do, since a scoped watcher lives in a thread-local and is
    /// therefore invisible to work on any other thread.
    /// Registers `callbacks` process-wide and returns the token that keeps other tests in this
    /// file out until it drops. Held across awaits on purpose, for the reason
    /// [`crate::lm::global::install_for_test`] gives: `SERIAL` is taken by nothing else.
    fn install(callbacks: Vec<Arc<dyn Callback>>) -> std::sync::MutexGuard<'static, ()> {
        static SERIAL: Mutex<()> = Mutex::new(());
        let guard = SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        configure_callbacks(callbacks);
        guard
    }

    /// Records which call each point belonged to, not merely that one happened.
    ///
    /// Registration is process-wide, so while these are installed *every* test in this binary that
    /// runs a module is heard too — `install`'s lock keeps other `callback.rs` tests out and can do
    /// nothing about the rest. Counting bare names made this depend on what else happened to be
    /// running; keying on the call id makes each assertion about its own work.
    #[derive(Default)]
    struct Counting(Mutex<Vec<(u64, String)>>);

    impl Counting {
        /// The modules seen for one call, in order.
        fn seen(&self, call: &CallId) -> Vec<String> {
            self.0
                .lock()
                .expect("not poisoned")
                .iter()
                .filter(|(id, _)| *id == call.id())
                .map(|(_, module)| module.clone())
                .collect()
        }
    }

    impl Callback for Counting {
        fn on_module_start(&self, call: &CallId, module: &str, _inputs: &Example) {
            self.0
                .lock()
                .expect("not poisoned")
                .push((call.id(), module.to_owned()));
        }
    }

    struct Panicking;

    impl Callback for Panicking {
        fn on_module_start(&self, _call: &CallId, _module: &str, _inputs: &Example) {
            panic!("a broken watcher");
        }
    }

    /// A handler that panics is not allowed to end the run, and the ones after it still hear about
    /// the point — upstream's `try/except` around each handler, in the shape Rust has for it.
    ///
    /// Scoped rather than registered process-wide, which is what makes it deterministic: a scoped
    /// watcher lives in a thread-local, so no other test in this binary can be heard by this
    /// recorder however they interleave. Registering these globally is what made an earlier
    /// version of this test fail once in a gate run.
    #[tokio::test]
    async fn a_panicking_callback_does_not_break_the_run() {
        let counting = Arc::new(Counting::default());
        let call = CallId::next();

        // The default hook prints the panic and its backtrace, which is noise for a panic the test
        // is asserting gets caught.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        watched_by(Arc::new(Panicking) as Arc<dyn Callback>)
            .run(
                watched_by(counting.clone() as Arc<dyn Callback>).run(async {
                    tell(&[], |callback| {
                        callback.on_module_start(&call, "Predict", &Example::default());
                    });
                }),
            )
            .await;
        std::panic::set_hook(hook);

        assert_eq!(
            counting.seen(&call),
            ["Predict"],
            "the watcher after the panicking one still heard the point"
        );
    }

    /// Two calls, one recorder: each sees its own and not the other's.
    ///
    /// This is the mechanism the test above depends on, asserted directly. Registration is
    /// process-wide, so while a recorder is installed every module run anywhere in this binary is
    /// heard — and the assertion that used to sit above counted bare names, which made it a claim
    /// about what else happened to be running. It failed once in a gate run for exactly that
    /// reason. The race itself was not reproducible on demand; what is reproducible, and what the
    /// fix rests on, is this.
    #[test]
    fn a_recorder_tells_one_calls_points_from_anothers() {
        let counting = Counting::default();
        let ours = CallId::next();
        let theirs = CallId::next();

        counting.on_module_start(&ours, "Predict", &Example::default());
        counting.on_module_start(&theirs, "ChainOfThought", &Example::default());
        counting.on_module_start(&ours, "Predict", &Example::default());

        assert_eq!(counting.seen(&ours), ["Predict", "Predict"]);
        assert_eq!(counting.seen(&theirs), ["ChainOfThought"]);
    }

    /// A scoped watcher is found by the interest check, not only by the telling.
    ///
    /// Every point asks [`watching`] first and does no rendering when nothing is listening. That
    /// check consulted only the process-wide registry, so a scoped watcher was registered, never
    /// asked, and silent — which is exactly how it failed: `streamify`'s status messages came out
    /// empty with the callback correctly installed. A short-circuit has to know about everything
    /// the thing it short-circuits would have told.
    #[allow(clippy::await_holding_lock)] // the installer's own note: `SERIAL` is a test token, taken by nothing under test
    #[tokio::test]
    async fn a_scoped_watcher_is_seen_by_the_interest_check() {
        let _installed = install(Vec::new());
        assert!(!watching(&[]), "nothing is registered to begin with");

        let counting = Arc::new(Counting::default());
        let call = CallId::next();
        watched_by(counting.clone() as Arc<dyn Callback>)
            .run(async {
                assert!(watching(&[]), "the scoped watcher is interest enough");
                tell(&[], |callback| {
                    callback.on_module_start(&call, "Predict", &Example::default());
                });
            })
            .await;

        assert_eq!(counting.seen(&call), ["Predict"]);
        assert!(!watching(&[]), "and it is gone once the work is done");
    }

    /// Nothing registered is nothing to tell, which is what every point checks before it renders
    /// anything for a handler to read.
    #[test]
    fn nothing_is_watching_by_default() {
        let _installed = install(Vec::new());
        assert!(!watching(&[]));
    }
}
