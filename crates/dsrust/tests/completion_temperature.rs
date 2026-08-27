//! Asking for several completions raises the temperature dspy raises it to.
//!
//! `predict.py::_forward_preprocess` sends **0.7** when more than one completion is asked for at a
//! near-zero temperature. This crate sent whatever the caller set, so any `BestOfN`-shaped call, or
//! any `Predict` with a completion count, went out at a temperature dspy would not have used — a
//! different request and a different reply.
//!
//! The rule reads both fields *through* the model, and `or` is Python's, so a temperature of 0.0
//! reads as unset and falls through. The golden is 110 pairs of model kwargs and module config
//! recorded by calling `_forward_preprocess` itself, which is the only way to get the truthiness
//! right without transcribing it.

use dsrust::lm::api::{LmConfig, LmRequest, LmResponse};
use dsrust::lm::{ChatModel, Sampling};
use dsrust::{Example, Module, Predict};
use serde_json::Value;

/// A model that answers one field and reports whatever kwargs its case named.
struct Scripted {
    defaults: LmConfig,
    asked: std::sync::Mutex<Vec<LmConfig>>,
}

impl ChatModel for Scripted {
    async fn forward(&self, request: &LmRequest) -> anyhow::Result<LmResponse> {
        self.asked
            .lock()
            .expect("no other thread holds it")
            .push(request.config.clone());
        Ok(LmResponse::text("[[ ## answer ## ]]\nfine"))
    }

    fn defaults(&self) -> LmConfig {
        self.defaults.clone()
    }
}

/// A case's `model` block as this crate's config — dspy's `lm.kwargs`.
fn model_kwargs(named: &Value) -> LmConfig {
    let mut config = LmConfig::default();
    if let Some(temperature) = named["temperature"].as_f64() {
        config.temperature = Some(temperature);
    }
    if let Some(count) = named["n"].as_u64() {
        config.n = Some(count as u32);
    }
    // No field of its own upstream either: `dspy.LM(num_generations=3)` keeps it as a plain
    // kwarg, which is what `extensions` is.
    if let Some(count) = named["num_generations"].as_u64() {
        config
            .extensions
            .insert("num_generations".to_owned(), Value::from(count));
    }
    config
}

/// A case's `config` block as this crate's module sampling — dspy's `self.config`.
fn module_config(named: &Value) -> Sampling {
    Sampling {
        temperature: named["temperature"].as_f64(),
        completions: named["n"].as_u64().map(|count| count as u32),
        ..Sampling::default()
    }
}

#[tokio::test]
async fn the_temperature_sent_is_the_one_dspy_resolves() {
    let fixture: Value = serde_json::from_str(include_str!(
        "conformance/predict/completion_temperature.json"
    ))
    .expect("the golden parses");

    let mut fired = 0;
    for case in fixture["cases"].as_array().expect("cases") {
        let model = std::sync::Arc::new(Scripted {
            defaults: model_kwargs(&case["model"]),
            asked: std::sync::Mutex::new(Vec::new()),
        });
        let predict = Predict::parse("question -> answer")
            .expect("parses")
            .set_lm(model.clone())
            .config(module_config(&case["config"]));
        predict
            .forward(Example::new([("question", Value::from("hi"))]))
            .await
            .expect("the scripted model answers");

        let sent = model.asked.lock().expect("no other thread holds it")[0].clone();
        let theirs = case["resolved"]["temperature"].as_f64();
        let where_at = format!("model {} config {}", case["model"], case["config"]);
        assert_eq!(sent.temperature, theirs, "{where_at}: temperature");
        fired += usize::from(theirs == Some(0.7));
    }
    assert_eq!(
        fired, 45,
        "the cases where the rule fires are what make this a test"
    );
}
