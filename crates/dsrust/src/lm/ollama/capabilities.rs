//! What an ollama server says one of its models can do.
//!
//! Every other provider answers this from litellm's bundled registry, which travels with this
//! crate. Ollama cannot: what a server can do is whatever has been pulled onto it, which is a
//! property of that server and not of any table. So litellm asks it — `POST /api/show`, then
//! whether the model's prompt template mentions tools — and this asks the same question of
//! whichever host is configured, local or remote.
//!
//! A server that cannot be reached, or does not know the model, grants nothing. That is litellm's
//! own reading of a failed probe, and the safe one: asking a model for tool calls it cannot make
//! wastes a request, while rendering tools into the prompt of a model that could have called them
//! natively still works.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use serde_json::{Value, json};

use crate::lm::Capabilities;

/// What `host` says `model` can do, asked once per host and model.
pub(crate) async fn capabilities(
    http: &reqwest::Client,
    host: &str,
    api_key: Option<&str>,
    timeout: Duration,
    model: &str,
) -> Capabilities {
    let asked = (host.to_owned(), model.to_owned());
    if let Some(known) = remembered(&asked) {
        return known;
    }
    let found = ask(http, host, api_key, timeout, model).await;
    remember(asked, found);
    found
}

/// The server's answer for one model, or nothing where it did not give one.
async fn ask(
    http: &reqwest::Client,
    host: &str,
    api_key: Option<&str>,
    timeout: Duration,
    model: &str,
) -> Capabilities {
    let request = http
        .post(format!("{host}/api/show"))
        .timeout(timeout)
        .json(&json!({ "name": model }));
    let Ok(response) = super::authorized(request, api_key).send().await else {
        return Capabilities::default();
    };
    if !response.status().is_success() {
        return Capabilities::default();
    }
    let Ok(shown) = response.json::<Value>().await else {
        return Capabilities::default();
    };
    Capabilities {
        function_calling: supports_tools(&shown),
        // ollama exposes no counterpart to either of these, and litellm reads neither from it.
        ..Capabilities::default()
    }
}

/// litellm's `OllamaModelInfo._supports_function_calling`: the model's own prompt template
/// mentions tools.
///
/// A roundabout test for something ollama now states directly — its `capabilities` array carries
/// `"tools"` — but litellm reads the template, and matching what dspy would answer is the point.
/// The two agree on every model checked; where they ever disagree, litellm's is the answer dspy
/// gives and so the one this has to give.
fn supports_tools(shown: &Value) -> bool {
    shown["template"]
        .as_str()
        .is_some_and(|template| template.to_lowercase().contains("tools"))
}

/// Answers already given, keyed by the server asked and the model asked about.
///
/// A pulled model does not change what it can do while a process runs, and the probe is a round
/// trip to a server that may not be local — so paying for it on every request would be a latency
/// cost with nothing bought.
fn known() -> &'static Mutex<HashMap<(String, String), Capabilities>> {
    static KNOWN: OnceLock<Mutex<HashMap<(String, String), Capabilities>>> = OnceLock::new();
    KNOWN.get_or_init(Default::default)
}

fn remembered(asked: &(String, String)) -> Option<Capabilities> {
    known().lock().ok()?.get(asked).copied()
}

fn remember(asked: (String, String), found: Capabilities) {
    if let Ok(mut known) = known().lock() {
        known.insert(asked, found);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The memo is one map for the process, keyed by host and model, and a written answer reads
    /// back. Five mutants replaced `known()` with a fresh map per call — remember would write into
    /// one map and remembered read from another — and nothing noticed, because no test ever went
    /// round the trip.
    #[test]
    fn a_remembered_answer_reads_back_under_its_own_key() {
        let mine = (
            "http://memo-test-host:11434".to_owned(),
            "memo-test-model".to_owned(),
        );
        assert_eq!(remembered(&mine), None, "not asked yet");
        remember(
            mine.clone(),
            Capabilities {
                function_calling: true,
                ..Capabilities::default()
            },
        );
        let read = remembered(&mine).expect("written answers read back");
        assert!(read.function_calling);
        let other = (mine.0.clone(), "some-other-model".to_owned());
        assert_eq!(remembered(&other), None, "keys do not bleed across models");
    }

    #[test]
    fn a_template_that_mentions_tools_is_a_model_that_can_call_them() {
        // Both spellings ollama's own templates use, and the case litellm folds away.
        assert!(supports_tools(
            &json!({ "template": "{{ if .Tools }}...{{ end }}" })
        ));
        assert!(supports_tools(&json!({ "template": "you may use tools" })));
        assert!(!supports_tools(&json!({ "template": "{{ .Prompt }}" })));
        // Nothing said is nothing granted, however the server phrased its silence.
        assert!(!supports_tools(&json!({})));
        assert!(!supports_tools(&json!({ "template": null })));
    }
}
