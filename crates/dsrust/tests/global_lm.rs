//! The process-wide LM global, exercised in its own test process: the global stores the
//! concrete [`LM`], so a scripted model cannot stand in for it here. Instead the test drives
//! Every LM here is built with `.cache(false)`. The reply cache lives in `~/.dsrs_cache` and
//! outlives the run, so a cached answer for one of these model refs would be replayed instead of
//! the connection being attempted — and a test asserting a *transport* failure would read a
//! successful empty reply. It passed on a machine that had never cached one and failed on a
//! machine that had.
//!
//! a typed `call` before any configure (the unconfigured error) and after pointing the
//! global at an unroutable host (a provider error), proving resolution goes through the
//! global. Both assertions live in one test fn because a sibling test configuring first
//! would race the unconfigured half.

use std::time::Duration;

use dsrust::lm::{self, LM};
use dsrust::signature::{Predict, Signature};

/// Answer the question.
// The derive is declaration data; the struct itself is never built.
#[allow(dead_code)]
#[derive(Signature)]
struct ProbeTask {
    #[input]
    question: String,
    #[output]
    answer: String,
}

const UNROUTABLE_OLLAMA: &str = "http://127.0.0.1:9";

#[tokio::test]
async fn typed_calls_resolve_the_global_and_name_the_fix_when_it_is_missing() {
    let inputs = ProbeTaskInputs {
        question: "anything".into(),
    };

    let unconfigured = ProbeTask::predict()
        .call_inputs(&inputs)
        .await
        .expect_err("nothing configured yet");
    assert!(
        unconfigured
            .to_string()
            .contains("no global LM; call lm::configure(...) first"),
        "got: {unconfigured:#}"
    );

    // A tiny client timeout keeps the failure fast even if the port swallows the connect.
    let http = reqwest::Client::builder()
        .timeout(Duration::from_millis(250))
        .build()
        .expect("client builds");
    lm::configure_with_client(
        http,
        LM::new("ollama/whatever")
            .expect("valid model ref")
            .ollama_host(UNROUTABLE_OLLAMA)
            .cache(false),
    );
    let provider_error = ProbeTask::predict()
        .call_inputs(&inputs)
        .await
        .expect_err("host is unroutable");
    let rendered = format!("{provider_error:#}");
    assert!(!rendered.contains("no global LM"), "got: {rendered}");
    // A configured LM that cannot be reached is a transport failure, not a missing one — and the
    // type says so rather than the prose, so a caller can retry it without matching a string.
    let failed = provider_error
        .downcast_ref::<dsrust::lm::LmFailure>()
        .unwrap_or_else(|| panic!("a typed LM failure, got: {rendered}"));
    assert_eq!(failed.kind, dsrust::lm::LmErrorKind::Transport);
    assert_eq!(failed.provider.as_deref(), Some("ollama"));

    // Reconfiguring must win over the previous configure. Named differently on purpose: the
    // first LM is also ollama at the same unroutable host, so asserting the provider cannot tell
    // a reconfigure from a no-op — `configure` deleted entirely used to pass this.
    lm::configure(
        LM::new("ollama/reconfigured")
            .expect("valid model ref")
            .ollama_host(UNROUTABLE_OLLAMA)
            .cache(false),
    );
    let reconfigured = ProbeTask::chain_of_thought()
        .call_inputs(&inputs)
        .await
        .expect_err("host is still unroutable");
    let failed = reconfigured
        .downcast_ref::<dsrust::lm::LmFailure>()
        .unwrap_or_else(|| panic!("a typed LM failure, got: {reconfigured:#}"));
    assert_eq!(failed.provider.as_deref(), Some("ollama"));
    assert_eq!(
        failed.model.as_deref(),
        Some("reconfigured"),
        "the model this configure installed is the one that was reached"
    );

    // The call macro's expansion resolves the same global: it must reach the provider, not
    // the unconfigured error.
    let via_macro = Predict!(ProbeTask {
        question: "anything"
    })
    .await
    .expect_err("host is still unroutable");
    let rendered = format!("{via_macro:#}");
    assert!(!rendered.contains("no global LM"), "got: {rendered}");
    assert!(
        via_macro
            .downcast_ref::<dsrust::lm::LmFailure>()
            .is_some_and(|failed| { failed.provider.as_deref() == Some("ollama") }),
        "the macro's expansion reached the configured provider: {rendered}"
    );
}
