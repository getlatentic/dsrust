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
/// appearing in the wrong assertion, or as "no answer left" — intermittently, and nowhere near
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

/// The current default, cloned out so in-flight calls never hold the lock across await
/// points and a concurrent reconfigure only affects later calls.
pub(crate) fn current() -> Result<(reqwest::Client, Arc<dyn DynChatModel>)> {
    GLOBAL
        .read()
        .expect("lock not poisoned")
        .as_ref()
        .map(|configured| (configured.http.clone(), Arc::clone(&configured.lm)))
        .ok_or_else(|| anyhow!("no global LM; call lm::configure(...) first"))
}
