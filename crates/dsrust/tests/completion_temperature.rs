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
use dsrust::predict::Steering;
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

/// A per-call `config=` laid over the module's, and the rule running after the merge.
///
/// dspy merges `{**self.config, **config}` in `_forward_preprocess` and *then* asks whether to
/// raise the temperature, so a call asking for three completions is three however the module was
/// configured. 180 recorded (model, module config, call config) triples. `rollout_id` is in the
/// call set because it is the one field that never reaches a provider — it varies the cache key,
/// which is what makes a second attempt a second call rather than a replay.
#[tokio::test]
async fn a_per_call_config_lies_over_the_modules() {
    let fixture: Value = serde_json::from_str(include_str!(
        "conformance/predict/completion_temperature.json"
    ))
    .expect("the golden parses");

    for case in fixture["merges"].as_array().expect("merges") {
        let model = std::sync::Arc::new(Scripted {
            defaults: model_kwargs(&case["model"]),
            asked: std::sync::Mutex::new(Vec::new()),
        });
        let predict = Predict::parse("question -> answer")
            .expect("parses")
            .set_lm(model.clone())
            .config(module_config(&case["config"]));
        let steering = Steering {
            config: Some(Sampling {
                rollout_id: case["call"]["rollout_id"].as_u64(),
                ..module_config(&case["call"])
            }),
            ..Steering::default()
        };
        predict
            .forward_with_steering(Example::new([("question", Value::from("hi"))]), &steering)
            .await
            .expect("the scripted model answers");

        let sent = model.asked.lock().expect("no other thread holds it")[0].clone();
        let where_at = format!(
            "model {} config {} call {}",
            case["model"], case["config"], case["call"]
        );
        let resolved = &case["resolved"];
        assert_eq!(
            sent.temperature,
            resolved["temperature"].as_f64(),
            "{where_at}: temperature"
        );
        assert_eq!(
            sent.n,
            resolved["n"].as_u64().map(|count| count as u32),
            "{where_at}: n"
        );
    }
}
