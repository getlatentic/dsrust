//! Which model names are reasoning models, held to dspy's *two* answers.
//!
//! dspy defines `_is_openai_reasoning_model` twice — once in `clients/openai_format.py`, which
//! decides the chat body's token key and whether a reasoning temperature is refused, and once in
//! `clients/lm.py`, which decides what `LM.dump_state` writes. The two disagree on five of the
//! thirty names in the golden, in both directions, so a port answering with one predicate is wrong
//! somewhere whichever rule it picks.
//!
//! The golden records consequences rather than predicates: the token key that appears in the body
//! `to_openai_chat_request` built, and the ordered keys of a `dump_state` block, whose `max_tokens`
//! moves to the end when the state rule fires. Both are bytes dspy produced.
//!
//! This is the check `lm_api/openai_chat.json` could not be: that fixture names `gpt-4o-mini` and
//! `o3-mini`, and both sit in the region where the two rules agree.

use dsrust::LM;
use dsrust::lm::{TokenLimitField, TokenLimitRule};
use serde_json::Value;

/// The saved block dspy writes for a reasoning model puts `max_tokens` last, after the three
/// finetuning keys, because `dump_state` pops and re-inserts it. Anything else leaves it in place.
fn state_rule_fired(keys: &[Value]) -> bool {
    let at = keys
        .iter()
        .position(|key| key == "max_tokens")
        .expect("every block states a cap");
    at + 1 == keys.len()
}

fn golden() -> Value {
    serde_json::from_str(include_str!("conformance/lm_api/reasoning_models.json"))
        .expect("the golden parses")
}

#[test]
fn the_chat_token_key_is_the_one_dspy_sends() {
    for case in golden()["cases"].as_array().expect("cases") {
        let model = case["model"].as_str().expect("a model");
        let theirs = case["chat_token_key"].as_str().expect("a token key");
        let ours = TokenLimitRule::ByOpenAiModelFamily
            .field_for(model)
            .wire_name();
        assert_eq!(ours, theirs, "model {model}: chat token key");
    }
}

/// dspy refuses a reasoning effort at a temperature that is neither unset nor 1, and the models it
/// refuses for are the *wire* rule's — the same predicate as the token key, which is why one
/// assertion over the same golden covers both.
#[test]
fn a_reasoning_temperature_is_refused_for_the_models_dspy_refuses_it_for() {
    for case in golden()["cases"].as_array().expect("cases") {
        let model = case["model"].as_str().expect("a model");
        let refused = case["temperature_refused"].as_bool().expect("a verdict");
        let reasons = TokenLimitRule::ByOpenAiModelFamily.field_for(model)
            == TokenLimitField::MaxCompletionTokens;
        assert_eq!(reasons, refused, "model {model}: reasoning temperature");
    }
}

/// The state rule, read off the block this crate writes: `max_tokens` sits last for exactly the
/// names dspy puts it last for.
///
/// A name this crate cannot build is skipped and named. Two shapes are: a bare model id, where
/// dspy takes `o3` and this crate requires `provider/model-id`, and an `azure/` prefix, which is a
/// provider it does not have. Both are in the golden for the wire rule, which reads a name rather
/// than resolving it. The skip is asserted so it cannot widen quietly, and so are the names it
/// leaves behind — four of the five the two rules disagree about, which is what makes this a test
/// of the state rule rather than of the wire one.
#[test]
fn the_saved_block_places_the_cap_where_dspy_places_it() {
    let fixture = golden();
    let mut unbuildable = Vec::new();
    let mut covered = Vec::new();
    for case in fixture["cases"].as_array().expect("cases") {
        let model = case["model"].as_str().expect("a model");
        let theirs = case["state_keys"].as_array().expect("dspy's keys");
        let Ok(mut lm) = LM::new(model) else {
            unbuildable.push(model);
            continue;
        };
        lm.config.temperature = Some(1.0);
        lm.config.max_tokens = Some(16_000);
        let ours: Vec<Value> = dsrust::lm::saved::dump(&lm)
            .keys()
            .map(|key| Value::String(key.clone()))
            .collect();
        assert_eq!(
            state_rule_fired(&ours),
            state_rule_fired(theirs),
            "model {model}: where the saved block puts `max_tokens`"
        );
        assert_eq!(ours, *theirs, "model {model}: the saved block's key order");
        if (case["chat_token_key"] == "max_completion_tokens") != state_rule_fired(theirs) {
            covered.push(model);
        }
    }
    assert_eq!(
        unbuildable,
        ["o3", "gpt-5", "azure/o3"],
        "the names this crate does not route"
    );
    assert_eq!(
        covered,
        [
            "openai/o1-preview",
            "openai/gpt-5.1",
            "openai/o5",
            "openrouter/openai/gpt-5",
        ],
        "the disagreeing names this assertion actually reaches"
    );
}

/// The two rules are not the same rule. Without a name they disagree on, every assertion above
/// would still pass with one predicate answering both — which is how this went unnoticed.
#[test]
fn the_golden_holds_names_the_two_rules_disagree_about() {
    let fixture = golden();
    let disagreeing: Vec<&str> = fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .filter(|case| {
            let wire = case["chat_token_key"] == "max_completion_tokens";
            let state = state_rule_fired(case["state_keys"].as_array().expect("keys"));
            wire != state
        })
        .map(|case| case["model"].as_str().expect("a model"))
        .collect();
    assert_eq!(
        disagreeing,
        [
            "openai/o1-preview",
            "openai/gpt-5.1",
            "openai/o5",
            "azure/o3",
            "openrouter/openai/gpt-5",
        ],
        "the names that make this test a test"
    );
}
