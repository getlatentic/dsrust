//! The process-wide default LM, DSPy-style: the server configures once at startup and the
//! modules in [`mod@crate::predict`] resolve it at call time, so call sites stop threading an
//! HTTP client and model through every layer. Reconfigurable, so a later configure wins.

use std::sync::{Arc, RwLock};

use anyhow::{Result, anyhow};

use super::{DynChatModel, LM};

/// The configured pair travels together: provider calls must go out on the client the
/// configurer chose (the server passes its pooled one).
struct Configured {
    http: reqwest::Client,
    lm: Arc<dyn DynChatModel>,
}

static GLOBAL: RwLock<Option<Configured>> = RwLock::new(None);

thread_local! {
    /// The model a [`context`] scope installed, read in preference to [`GLOBAL`].
    ///
    /// dspy's `thread_local_overrides`, a `ContextVar`: `context` layers over `configure` rather
    /// than replacing it, and a scope inside a scope inherits the outer one's overrides before
    /// applying its own.
    static SCOPED: std::cell::RefCell<Option<Configured>> = const { std::cell::RefCell::new(None) };
}

/// Make `lm` the process-wide default, with a client of its own.
pub fn configure(lm: LM) {
    configure_with_client(reqwest::Client::new(), lm);
}

/// Make `lm` the process-wide default, sending its provider calls on `http`.
pub fn configure_with_client(http: reqwest::Client, lm: LM) {
    configure_model(http, Arc::new(lm));
}

/// Install any model as the process-wide default, including a scripted one. dspy's `DummyLM`
/// exists for the same reason: a module reaches its model through the global, so without this
/// nothing built on `Module` could be tested without a provider.
pub fn configure_model(http: reqwest::Client, lm: Arc<dyn DynChatModel>) {
    *GLOBAL.write().expect("lock not poisoned") = Some(Configured { http, lm });
}

/// Install a model for the duration of one test, keeping every other such test out meanwhile.
///
/// [`GLOBAL`] is one static for the whole test binary and cargo runs tests on parallel threads,
/// so two tests that each configure it race: the later install wins, and the earlier test's
/// module then reads a script written for another test. That surfaces as a scripted answer
/// appearing in the wrong assertion, or as `No more responses` — intermittently, and nowhere near
/// the test that caused it.
///
/// Hold the returned guard for as long as the test uses the model, which means binding it:
/// `let _configured = install_for_test(...)` and not `let _ = ...`, which drops it at once.
#[cfg(test)]
pub(crate) fn install_for_test(lm: Arc<dyn DynChatModel>) -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A test that panicked while holding this poisoned it. Its failure is already reported, and
    // the next test still needs the lock, so the poison is stepped over rather than propagated.
    let guard = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    configure_model(reqwest::Client::new(), lm);
    guard
}

/// The current default model, cloned out so in-flight calls never hold the lock across await
/// points and a concurrent reconfigure only affects later calls.
///
/// The client it goes out on is [`client`], asked for separately. They were one call returning a
/// pair, which coupled every caller needing a model to also naming a client — and every caller of
/// this wants the model. See the `lm-shared-client` story: `ChatModel::forward` should name no
/// client at all, and this split is the step that makes the rest tractable.
pub(crate) fn current() -> Result<Arc<dyn DynChatModel>> {
    // A scope wins over the process-wide default, which is upstream's
    // `{**main_thread_config, **thread_local_overrides}`.
    if let Some(scoped) = SCOPED.with(|scoped| {
        scoped
            .borrow()
            .as_ref()
            .map(|configured| Arc::clone(&configured.lm))
    }) {
        return Ok(scoped);
    }
    GLOBAL
        .read()
        .expect("lock not poisoned")
        .as_ref()
        .map(|configured| Arc::clone(&configured.lm))
        .ok_or_else(|| anyhow!("no global LM; call lm::configure(...) first"))
}

/// The configured HTTP client, or a fresh one.
///
/// A module carrying its own model still needs a client to make the call, but not the global
/// model behind it — so this never errors, where [`current`] does when nothing is configured.
pub(crate) fn client() -> reqwest::Client {
    // Scoped alongside the model: a block that overrides which model answers should send on that
    // model's client, not on the one the process was configured with.
    if let Some(scoped) = SCOPED.with(|scoped| {
        scoped
            .borrow()
            .as_ref()
            .map(|configured| configured.http.clone())
    }) {
        return scoped;
    }
    GLOBAL
        .read()
        .expect("lock not poisoned")
        .as_ref()
        .map(|configured| configured.http.clone())
        .unwrap_or_default()
}

/// dspy `dspy.context(lm=...)`: ask this model for the duration of one piece of work, then go back.
///
/// The scoped model wins over [`configure`]'s process-wide one for every module the work reaches,
/// however deeply nested and without any of them being rebuilt — which is the difference from
/// [`Predict::set_lm`](crate::predict::Predict::set_lm), a construction-time choice on one module.
/// A five-module pipeline handed to you already built can be pointed at another model this way and
/// no other.
///
/// ```no_run
/// # use dsrust::{Module, lm};
/// # async fn ask(program: impl Module, inputs: dsrust::Example, other: dsrust::LM) -> anyhow::Result<()> {
/// // Everything this program calls asks `other`, and anything after the scope asks the configured
/// // model again.
/// let answered = lm::context(other).run(program.forward(inputs)).await?;
/// # let _ = answered;
/// # Ok(())
/// # }
/// ```
///
/// **It scopes a future, not a block.** dspy's is a `with` statement because a `ContextVar` in
/// asyncio is per-Task; a Rust guard held across an `.await` would instead be read by whatever the
/// runtime polled next. [`Scope::run`] enters on each poll and leaves when the poll returns, so two
/// pieces of work interleaved in one task each see their own model — the same mechanism, and the
/// same reason, as [`CallId`](crate::CallId)'s parent linkage.
pub fn context(lm: LM) -> Scope {
    context_with_client(reqwest::Client::new(), lm)
}

/// As [`context`], sending the scope's provider calls on `http` rather than a client of its own.
///
/// Worth reaching for when the process already has a client whose pool, proxy or timeouts are
/// configured — `context` builds one per scope, and a program that opens many scopes would build
/// many:
///
/// ```
/// # async fn wrapper(program: dsrust::Predict) -> anyhow::Result<()> {
/// let shared = reqwest::Client::builder()
///     .timeout(std::time::Duration::from_secs(60))
///     .build()?;
/// let lm = dsrust::LM::builder("openai/gpt-4o-mini").build()?;
/// dsrust::lm::context_with_client(shared, lm)
///     .run(program.call("a question"))
///     .await?;
/// # Ok(()) }
/// ```
pub fn context_with_client(http: reqwest::Client, lm: LM) -> Scope {
    context_model(http, Arc::new(lm))
}

/// Scope any model, including a scripted one — the counterpart to [`configure_model`], and the
/// shape dspy's own tests use: `with dspy.context(lm=DummyLM(...))`.
///
/// `context` takes an [`LM`] because that is what a program does; this takes anything implementing
/// the trait, which is what a test does. Without it a scripted model could be made the process-wide
/// default but never scoped, and the two paths would not be testable the same way.
pub fn context_model(http: reqwest::Client, lm: Arc<dyn DynChatModel>) -> Scope {
    Scope {
        configured: Configured { http, lm },
    }
}

/// A model installed for the duration of one piece of work. See [`context`].
pub struct Scope {
    configured: Configured,
}

impl Scope {
    /// Run `work` with this scope's model in force.
    pub async fn run<T>(self, work: impl Future<Output = T>) -> T {
        Scoped {
            configured: self.configured,
            inner: Box::pin(work),
        }
        .await
    }
}

/// A future that runs under a scoped model: installed for the duration of each poll, and the
/// enclosing scope put back when the poll returns.
///
/// Per poll rather than for the whole future, for the reason
/// [`callback::context::Under`](crate::callback) is: `Evaluate` interleaves its rows inside one task,
/// so a value set once and left would be read by whichever row was polled next.
struct Scoped<F> {
    configured: Configured,
    inner: std::pin::Pin<Box<F>>,
}

impl<F: Future> Future for Scoped<F> {
    type Output = F::Output;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<F::Output> {
        let scoped = self.get_mut();
        // dspy layers: `{**main_thread_config, **original_overrides, **kwargs}`. Replacing rather
        // than merging is the same thing here, because this crate scopes exactly one setting — the
        // model — so an inner scope overriding it leaves nothing of the outer one to inherit.
        let restore = SCOPED.with(|slot| {
            slot.borrow_mut().replace(Configured {
                http: scoped.configured.http.clone(),
                lm: Arc::clone(&scoped.configured.lm),
            })
        });
        let polled = scoped.inner.as_mut().poll(context);
        SCOPED.with(|slot| *slot.borrow_mut() = restore);
        polled
    }
}
