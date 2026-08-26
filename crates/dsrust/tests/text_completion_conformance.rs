//! The prompt the legacy completions wire sends is the one dspy sends.
//!
//! `model_type="text"` is the one place upstream turns a rendered message list back into a single
//! string, and the rule is entirely its own: each message's content, then `BEGIN RESPONSE:`, joined
//! by blank lines. Nothing about it is derivable from the endpoint, so it is held against a fixture
//! recorded by calling `litellm_text_completion` with the litellm module replaced by a recorder —
//! what lands in the golden is what dspy passed, not a reading of its source.
//!
//! The `model` upstream sends is `text-completion-openai/<id>`, which is litellm's routing token
//! rather than part of the wire. This crate talks to the endpoint directly and sends the bare id,
//! so that field is asserted as a *difference* rather than skipped.

use dsrust::lm::api::{LmMessage, LmPart, LmRequest};
use serde_json::Value;

/// The messages a fixture case names, as this crate's typed request carries them.
fn messages(case: &Value) -> Vec<LmMessage> {
    case["messages"]
        .as_array()
        .expect("a list of messages")
        .iter()
        .map(|message| {
            let role = message["role"].as_str().expect("a role");
            let text = message["content"].as_str().expect("content");
            LmMessage::new(role, [LmPart::text(text)])
        })
        .collect()
}

#[test]
fn the_prompt_is_the_one_dspy_builds() {
    let fixture: Value = serde_json::from_str(include_str!("conformance/lm/text_completion.json"))
        .expect("the golden parses");

    for case in fixture["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("a name");
        let request = LmRequest::new("gpt-3.5-turbo-instruct", messages(case));
        let ours = dsrust::lm::openai::text::prompt(&request.messages);
        let theirs = case["sent"]["prompt"].as_str().expect("dspy's prompt");
        assert_eq!(ours, theirs, "case {name}: prompt");

        // Recorded so the difference stays visible: upstream re-prefixes the model for litellm's
        // router, and this crate posts the id the endpoint expects.
        assert_eq!(
            case["sent"]["model"].as_str().expect("dspy's model"),
            "text-completion-openai/gpt-3.5-turbo-instruct",
            "case {name}: dspy still routes through litellm's prefix"
        );
    }
}
