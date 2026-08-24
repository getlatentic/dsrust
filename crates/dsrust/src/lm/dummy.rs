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
use crate::lm::api::{self, RolloutId};
use crate::lm::{ChatModel, Sampling};

/// What the model was asked, kept so a test can assert on the prompt it produced.
///
/// The messages as the request carried them. dspy's `DummyLM.forward` reads `messages` end to end
/// — `messages[-1]["content"]`, `messages[:-1]` — and never splits the system prompt out; the
/// split this held instead re-derived a pair the adapter had stopped producing, and mapped every
/// role that was not `assistant` to `user` on the way, so a tool result recorded as something the
/// user said.
#[derive(Debug, Clone)]
pub struct Asked {
    pub messages: Vec<api::LmMessage>,
    pub json_mode: bool,
    /// How the caller asked for the reply to be sampled. A scripted model answers from its
    /// script regardless; recording it is what lets a test assert that a re-ask differed.
    pub config: Sampling,
}

impl Asked {
    /// The system prompt, or `""` when there is none — the same read [`LmRequest::system`]
    /// makes, since it is the same question about the same list.
    ///
    /// [`LmRequest::system`]: crate::lm::api::LmRequest::system
    pub fn system(&self) -> &str {
        api::system_of(&self.messages)
    }

    /// The conversation without the system prompt: what a test means when it says "the second
    /// turn" and counts from the first thing anyone said.
    pub fn turns(&self) -> &[api::LmMessage] {
        api::after_system(&self.messages)
    }

    /// The last thing the model was told, which is the request it is answering.
    pub fn last_message(&self) -> String {
        self.messages
            .last()
            .and_then(|message| message.text())
            .unwrap_or_default()
    }
}

/// A scripted model.
///
/// ```
/// use dsrust::{DummyLM, example};
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
    pub fn fallback(mut self, answer: Example) -> Self {
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
    fn forward<'a>(
        &'a self,
        request: &'a api::LmRequest,
    ) -> impl Future<Output = Result<api::LmResponse>> + Send + 'a {
        let json_mode = request.output_schema().is_some();
        let asked = Asked {
            messages: request.messages.clone(),
            json_mode,
            config: recorded_config(&request.config),
        };
        let message = asked.last_message();
        // dspy's `DummyLM.forward` loops `for _ in range(n)` and *chooses again* each time, so a
        // queue pops a different answer per completion and a keyed model matches the same key n
        // times over. This choosing once and repeating it, which is what stood here, was right for
        // the keyed mode and wrong for the queue — and a test reading several completions is
        // exactly the caller that would notice.
        let completions = request.config.n.unwrap_or(1).max(1) as usize;
        self.asked.lock().expect("not poisoned").push(asked);
        async move {
            let mut replies = Vec::with_capacity(completions);
            for _ in 0..completions {
                let answer = self.choose(&message)?;
                replies.push(match json_mode {
                    true => as_json_reply(&answer),
                    false => as_marker_reply(&answer),
                });
            }
            // No usage: a scripted answer had no cost, and reporting zero would let a test
            // assert a total that no provider produced.
            Ok(api::LmResponse::completions(replies))
        }
    }
}

/// The four sampling fields a scripted model records for inspection, read back from the typed
/// config the module handed it.
fn recorded_config(config: &api::LmConfig) -> Sampling {
    Sampling {
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        completions: config.n,
        rollout_id: config
            .cache
            .as_ref()
            .and_then(|cache| cache.rollout_id.as_ref())
            .and_then(|rollout| match rollout {
                RolloutId::Number(id) => u64::try_from(*id).ok(),
                RolloutId::Text(_) => None,
            }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use crate::lm::OutputMode;
    use crate::lm::api::{LmMessage, request_of};

    fn ask(lm: &DummyLM, message: &str) -> Result<String> {
        let request = request_of(
            vec![LmMessage::system(["system"]), LmMessage::user([message])],
            OutputMode::Text,
            &Sampling::default(),
        );
        futures_lite_block_on(lm.forward(&request)).map(|answered| answered.first_text())
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

    /// The typed request the module hands the model is recorded as the system prompt and turns a
    /// test asserts on, the multi-part messages collapsed back to prose.
    #[test]
    fn the_typed_forward_records_the_system_prompt_and_turns() {
        use crate::lm::api::{LmMessage, LmPart, LmRequest as ApiRequest};

        let lm = DummyLM::new([example! { answer: "Paris" }]);
        let typed = ApiRequest::new(
            "openai/gpt-4o",
            vec![
                LmMessage::system(vec![LmPart::text("Be concise.")]),
                LmMessage::user(vec![LmPart::text("capital of France?")]),
            ],
        );

        let answered =
            futures_lite_block_on(lm.forward(&typed)).expect("the dummy answers a typed request");
        assert_eq!(
            answered.first_text(),
            "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"
        );

        let seen = lm.asked();
        let seen = seen.last().expect("one call was recorded");
        assert_eq!(seen.system(), "Be concise.");
        assert_eq!(seen.turns().len(), 1);
        assert_eq!(
            seen.turns()[0].text().as_deref(),
            Some("capital of France?")
        );
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
        let lm = DummyLM::new([]).fallback(example! { answer: "whatever" });
        assert!(ask(&lm, "anything").unwrap().contains("whatever"));
    }

    #[test]
    fn json_mode_returns_an_object_the_way_a_provider_would() {
        let lm = DummyLM::new([example! { answer: "red" }]);
        let schema = serde_json::json!({});
        let request = request_of(
            vec![LmMessage::system(["system"]), LmMessage::user(["ask"])],
            OutputMode::Json { schema: &schema },
            &Sampling::default(),
        );
        let reply = futures_lite_block_on(lm.forward(&request)).unwrap();
        assert_eq!(reply.first_text(), r#"{"answer":"red"}"#);
        assert_eq!(reply.usage, None, "a scripted answer cost nothing to make");
    }

    #[test]
    fn the_prompt_is_kept_for_inspection() {
        let lm = DummyLM::new([example! { answer: "red" }]);
        ask(&lm, "what colour is the sky?").unwrap();
        let asked = lm.asked();
        assert_eq!(asked[0].system(), "system");
        assert_eq!(asked[0].last_message(), "what colour is the sky?");
        assert!(!asked[0].json_mode);
    }
}
