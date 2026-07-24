//! What a model can be asked for natively, against litellm's own answers.
//!
//! dspy reads these three off litellm, so the fixture is litellm answering for each model string
//! exactly as dspy asks — `litellm.supports_function_calling(model="openai/gpt-4o")`. The
//! vendored registry beside the crate is the same data, but reaching a row in it means resolving
//! a reference the way litellm does, and that resolution is what these cases pin.

use dsrust::lm::{Capabilities, ChatModel, LM};
use serde_json::Value;

#[test]
fn every_model_resolves_to_what_litellm_says_it_can_do() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/lm_api/capabilities.json");
    let raw = std::fs::read_to_string(&path).expect("fixture is readable");
    let fixture: Value = serde_json::from_str(&raw).expect("fixture is valid json");
    let cases = fixture["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "no cases to check");

    for case in cases {
        let model = case["model"].as_str().expect("a model name");
        let flags = case["capabilities"].as_array().expect("three flags");
        let upstream = Capabilities {
            function_calling: flags[0].as_bool().expect("a bool"),
            reasoning: flags[1].as_bool().expect("a bool"),
            response_schema: flags[2].as_bool().expect("a bool"),
        };
        assert_eq!(Capabilities::of(model), upstream, "{model}");
    }
}

/// The registry is only reachable through a configured model, so the lookup has to survive the
/// trip through `ModelRef` — which takes the reference apart and this puts back together.
#[tokio::test]
async fn a_configured_model_answers_for_the_reference_it_was_built_from() {
    for (reference, function_calling) in
        [("openai/gpt-4o", true), ("anthropic/claude-2.1", false), ("ollama/llama3.2", false)]
    {
        let lm = LM::new(reference).expect("a valid reference");
        assert_eq!(lm.model.reference(), reference);
        let found = lm.capabilities(&reqwest::Client::new()).await;
        assert_eq!(found.function_calling, function_calling, "{reference}");
    }
}

/// dspy's `BaseLM` defaults every one of these to `False`, so a model that says nothing about
/// itself grants nothing — the crate must not decide on a provider's behalf.
#[tokio::test]
async fn a_model_that_says_nothing_grants_nothing() {
    struct Silent;
    impl ChatModel for Silent {
        fn forward<'a>(
            &'a self,
            _http: &'a reqwest::Client,
            _request: &'a dsrust::lm::api::LmRequest,
        ) -> impl std::future::Future<Output = anyhow::Result<dsrust::lm::api::LmResponse>> + Send + 'a
        {
            async { unreachable!("not called") }
        }
    }
    assert_eq!(Silent.capabilities(&reqwest::Client::new()).await, Capabilities::default());
}
