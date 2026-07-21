//! A model that answers from a script, for testing programs without a provider.
//!
//! dspy's `DummyLM` takes field *values* rather than raw completions and formats them the way
//! the adapter would. That is the difference between a test that says what the model means and
//! one that hand-writes `[[ ## answer ## ]]\nred\n\n[[ ## completed ## ]]` and gets a marker
//! subtly wrong. Two modes, both from upstream: answers in order, or answers keyed by
//! something appearing in the request.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use serde_json::Value;

use crate::adapter::python_json::format_value;
use crate::example::Example;
use crate::lm::{ChatModel, ChatTurn, LmRequest, OutputMode, Sampling};

/// What the model was asked, kept so a test can assert on the prompt it produced.
#[derive(Debug, Clone)]
pub struct Asked {
    pub system: String,
    pub turns: Vec<ChatTurn>,
    pub json_mode: bool,
    /// How the caller asked for the reply to be sampled. A scripted model answers from its
    /// script regardless; recording it is what lets a test assert that a re-ask differed.
    pub sampling: Sampling,
}

impl Asked {
    /// The last thing the model was told, which is the request it is answering.
    pub fn last_message(&self) -> &str {
        self.turns
            .last()
            .and_then(|turn| turn.content.text())
            .unwrap_or_default()
    }
}

/// A scripted model.
///
/// ```
/// use dsrs::{DummyLM, example};
/// let lm = DummyLM::new([example! { answer: "red" }, example! { answer: "blue" }]);
/// assert_eq!(lm.remaining(), 2);
/// ```
pub struct DummyLM {
    answers: Mutex<VecDeque<Example>>,
    keyed: BTreeMap<String, Example>,
    asked: Mutex<Vec<Asked>>,
    /// Returned when the script runs dry and nothing is keyed. `None` errors instead, which
    /// is usually what a test wants: a call it did not plan for should be loud.
    fallback: Option<Example>,
}

impl DummyLM {
    /// Answer with each example in turn. dspy's mode 1.
    pub fn new(answers: impl IntoIterator<Item = Example>) -> Self {
        Self {
            answers: Mutex::new(answers.into_iter().collect()),
            keyed: BTreeMap::new(),
            asked: Mutex::new(Vec::new()),
            fallback: None,
        }
    }

    /// Answer with whichever example's key appears in the request. dspy's mode 2, which suits
    /// a program whose call order is not fixed — an agent loop, say.
    pub fn keyed(pairs: impl IntoIterator<Item = (impl Into<String>, Example)>) -> Self {
        Self {
            answers: Mutex::new(VecDeque::new()),
            keyed: pairs
                .into_iter()
                .map(|(key, example)| (key.into(), example))
                .collect(),
            asked: Mutex::new(Vec::new()),
            fallback: None,
        }
    }

    /// Answer with this when nothing else matches, instead of erroring.
    pub fn with_fallback(mut self, answer: Example) -> Self {
        self.fallback = Some(answer);
        self
    }

    /// Every request this model received, in order.
    pub fn asked(&self) -> Vec<Asked> {
        self.asked.lock().expect("not poisoned").clone()
    }

    pub fn call_count(&self) -> usize {
        self.asked.lock().expect("not poisoned").len()
    }

    pub fn remaining(&self) -> usize {
        self.answers.lock().expect("not poisoned").len()
    }

    /// The next answer: a keyed match first, then the queue, then the fallback.
    fn choose(&self, request: &str) -> Result<Example> {
        for (key, answer) in &self.keyed {
            if request.contains(key.as_str()) {
                return Ok(answer.clone());
            }
        }
        if let Some(answer) = self.answers.lock().expect("not poisoned").pop_front() {
            return Ok(answer);
        }
        self.fallback
            .clone()
            .ok_or_else(|| anyhow!("DummyLM has no answer left for this request: {request:.120}"))
    }
}

/// Render field values the way the chat adapter would, so a scripted answer parses through
/// the same path a real reply does.
fn as_marker_reply(answer: &Example) -> String {
    let mut blocks: Vec<String> = answer
        .fields()
        .map(|(name, value)| format!("[[ ## {name} ## ]]\n{}", format_value(value)))
        .collect();
    blocks.push("[[ ## completed ## ]]".to_owned());
    blocks.join("\n\n")
}

/// In JSON mode a provider returns an object, not marker blocks.
fn as_json_reply(answer: &Example) -> String {
    let object: serde_json::Map<String, Value> = answer
        .fields()
        .map(|(name, value)| (name.to_owned(), value.clone()))
        .collect();
    Value::Object(object).to_string()
}

impl ChatModel for DummyLM {
    fn chat<'a>(
        &'a self,
        _http: &'a reqwest::Client,
        request: &'a LmRequest<'a>,
    ) -> impl Future<Output = Result<String>> + Send + 'a {
        let (system, turns, mode) = (request.system, request.turns, &request.mode);
        let json_mode = matches!(mode, OutputMode::Json { .. });
        let asked = Asked {
            system: system.to_owned(),
            turns: turns.to_vec(),
            json_mode,
            sampling: request.sampling.clone(),
        };
        let request = asked.last_message().to_owned();
        self.asked.lock().expect("not poisoned").push(asked);
        async move {
            let answer = self.choose(&request)?;
            Ok(match json_mode {
                true => as_json_reply(&answer),
                false => as_marker_reply(&answer),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;

    fn ask(lm: &DummyLM, message: &str) -> Result<String> {
        futures_lite_block_on(lm.chat(
            &reqwest::Client::new(),
            &LmRequest::new("system", &[ChatTurn::user(message)], OutputMode::Text),
        ))
    }

    /// The dummy never awaits anything real, so a trivial executor keeps these tests
    /// synchronous rather than pulling a runtime in for four assertions.
    fn futures_lite_block_on<F: Future>(future: F) -> F::Output {
        use std::task::{Context, Poll, Wake, Waker};
        struct Noop;
        impl Wake for Noop {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = Waker::from(std::sync::Arc::new(Noop));
        let mut context = Context::from_waker(&waker);
        let mut future = Box::pin(future);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
        }
    }

    #[test]
    fn answers_come_back_in_order_as_marker_blocks() {
        let lm = DummyLM::new([example! { answer: "red" }, example! { answer: "blue" }]);
        assert_eq!(
            ask(&lm, "what colour?").unwrap(),
            "[[ ## answer ## ]]\nred\n\n[[ ## completed ## ]]"
        );
        assert!(ask(&lm, "again?").unwrap().contains("blue"));
        assert_eq!(lm.call_count(), 2);
    }

    #[test]
    fn multiple_fields_each_get_their_own_block() {
        let lm = DummyLM::new([example! { colour: "blue", why: "It reads as still." }]);
        let reply = ask(&lm, "pick one").unwrap();
        assert!(reply.contains("[[ ## colour ## ]]\nblue"));
        assert!(reply.contains("[[ ## why ## ]]\nIt reads as still."));
    }

    #[test]
    fn a_keyed_model_answers_by_what_the_request_contains() {
        // Suits an agent loop, where call order is decided by the model rather than the test.
        let lm = DummyLM::keyed([
            ("France", example! { answer: "Paris" }),
            ("Germany", example! { answer: "Berlin" }),
        ]);
        assert!(ask(&lm, "capital of Germany?").unwrap().contains("Berlin"));
        assert!(ask(&lm, "capital of France?").unwrap().contains("Paris"));
    }

    #[test]
    fn an_unplanned_call_is_loud_rather_than_silent() {
        let lm = DummyLM::new([example! { answer: "only one" }]);
        ask(&lm, "first").unwrap();
        let error = ask(&lm, "second").expect_err("the script is exhausted");
        assert!(error.to_string().contains("no answer left"));
    }

    #[test]
    fn a_fallback_answers_anything_the_script_did_not_plan() {
        let lm = DummyLM::new([]).with_fallback(example! { answer: "whatever" });
        assert!(ask(&lm, "anything").unwrap().contains("whatever"));
    }

    #[test]
    fn json_mode_returns_an_object_the_way_a_provider_would() {
        let lm = DummyLM::new([example! { answer: "red" }]);
        let schema = serde_json::json!({});
        let reply = futures_lite_block_on(lm.chat(
            &reqwest::Client::new(),
            &LmRequest::new(
                "system",
                &[ChatTurn::user("ask")],
                OutputMode::Json { schema: &schema },
            ),
        ))
        .unwrap();
        assert_eq!(reply, r#"{"answer":"red"}"#);
    }

    #[test]
    fn the_prompt_is_kept_for_inspection() {
        let lm = DummyLM::new([example! { answer: "red" }]);
        ask(&lm, "what colour is the sky?").unwrap();
        let asked = lm.asked();
        assert_eq!(asked[0].system, "system");
        assert_eq!(asked[0].last_message(), "what colour is the sky?");
        assert!(!asked[0].json_mode);
    }
}
