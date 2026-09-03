//! A model that answers from a script, for testing programs without a provider.
//!
//! dspy's `DummyLM` takes field *values* rather than raw completions and formats them the way
//! the adapter would. That is the difference between a test that says what the model means and
//! one that hand-writes `[[ ## answer ## ]]\nred\n\n[[ ## completed ## ]]` and gets a marker
//! subtly wrong. Two modes, both from upstream: answers in order, or answers keyed by
//! something appearing in the request.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use anyhow::Result;
use serde_json::Value;

use crate::adapter::python_json::format_value;
use crate::example::Example;
use crate::lm::LmUsage;
use crate::lm::api::{self, RolloutId};
use crate::lm::{ChatModel, Sampling};

/// What the model was asked, kept so a test can assert on the prompt it produced.
///
/// The messages as the request carried them. dspy's `DummyLM.forward` reads `messages` end to end
/// — `messages[-1]["content"]`, `messages[:-1]` — and never splits the system prompt out; the
/// split this held instead re-derived a pair the adapter had stopped producing, and mapped every
/// role that was not `assistant` to `user` on the way, so a tool result recorded as something the
/// user said.
///
/// ```
/// use dsrust::DummyLM;
/// use dsrust::lm::dummy::Asked;
///
/// # async fn wrapper() -> anyhow::Result<()> {
/// let lm = std::sync::Arc::new(DummyLM::new([dsrust::example! { answer: "Paris" }]));
/// // ... after a program has run under it ...
/// let asked: Vec<Asked> = lm.asked();
/// if let Some(first) = asked.first() {
///     // The system message leads, as every adapter here renders it.
///     assert_eq!(first.messages[0].role, "system");
/// }
/// # Ok(()) }
/// ```
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
    /// Which of dspy's two modes this is. Upstream reads `isinstance(self.answers, dict)` and
    /// answers an unscripted request differently on each side of it, so an empty script has to
    /// remember which door it came through.
    keyed_mode: bool,
    asked: Mutex<Vec<Asked>>,
    /// Answered with instead of upstream's "No more responses" when the script runs dry. Rust-only,
    /// and additive: a test that plans every call never reaches either.
    fallback: Option<Example>,
}

/// dspy's answer to a request its script does not cover.
const NO_MORE: &str = "No more responses";

impl DummyLM {
    /// Answer with each example in turn. dspy's mode 1.
    pub fn new(answers: impl IntoIterator<Item = Example>) -> Self {
        Self {
            answers: Mutex::new(answers.into_iter().collect()),
            keyed: BTreeMap::new(),
            keyed_mode: false,
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
            keyed_mode: true,
            asked: Mutex::new(Vec::new()),
            fallback: None,
        }
    }

    /// Answer with this when nothing else matches, in place of upstream's "No more responses".
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

    /// The next answer: a keyed match first, then the queue, then the fallback, then dspy's own
    /// words for a request it cannot answer.
    fn choose(&self, request: &str) -> Chosen {
        for (key, answer) in &self.keyed {
            if request.contains(key.as_str()) {
                return Chosen::Answer(answer.clone());
            }
        }
        if let Some(answer) = self.answers.lock().expect("not poisoned").pop_front() {
            return Chosen::Answer(answer);
        }
        if let Some(answer) = self.fallback.clone() {
            return Chosen::Answer(answer);
        }
        // The asymmetry is upstream's, and it is observable: the keyed miss returns the bare
        // words, so the reply does not parse and the caller retries through the JSON adapter,
        // while the exhausted queue returns a rendered `answer` field that parses cleanly.
        match self.keyed_mode {
            true => Chosen::Unscripted,
            false => Chosen::Answer(crate::example! { answer: NO_MORE }),
        }
    }
}

/// Render field values the way the chat adapter would, so a scripted answer parses through
/// the same path a real reply does.
fn as_marker_reply(answer: &Example) -> String {
    // No trailing `[[ ## completed ## ]]`, because dspy's `_format_answer_fields` writes none —
    // measured: `DummyLM([{"answer": "x"}])` answers `[[ ## answer ## ]]\nx`. The marker is
    // something the *adapter* asks a real model for, and a scripted reply carrying one anyway made
    // every test that reads `Prediction::raw` compare against a string upstream never produces.
    answer
        .fields()
        .map(|(name, value)| format!("[[ ## {name} ## ]]\n{}", format_value(value)))
        .collect::<Vec<String>>()
        .join("\n\n")
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
                replies.push(match self.choose(&message) {
                    Chosen::Unscripted => NO_MORE.to_owned(),
                    Chosen::Answer(answer) => match json_mode {
                        true => as_json_reply(&answer),
                        false => as_marker_reply(&answer),
                    },
                });
            }
            // dspy's `DummyLM` reports a usage of zero tokens, and a program that tracks usage
            // sees it; so does this one.
            Ok(api::LmResponse::completions(replies).usage(Some(LmUsage::counted(0, 0))))
        }
    }
}

/// What the script had for one request.
enum Chosen {
    Answer(Example),
    /// dspy's keyed miss: the bare words, never run through the field renderer.
    Unscripted,
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
        use std::task::{Context, Poll, Waker};
        let mut context = Context::from_waker(Waker::noop());
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
        assert_eq!(ask(&lm, "what colour?").unwrap(), "[[ ## answer ## ]]\nred");
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
        assert_eq!(answered.first_text(), "[[ ## answer ## ]]\nParis");

        let seen = lm.asked();
        let seen = seen.last().expect("one call was recorded");
        assert_eq!(seen.system(), "Be concise.");
        assert_eq!(seen.turns().len(), 1);
        assert_eq!(
            seen.turns()[0].text().as_deref(),
            Some("capital of France?")
        );
    }

    /// An unplanned call is answered, not refused — and the two modes answer differently.
    ///
    /// Measured from the pinned dspy: `DummyLM([])` asked anything gives
    /// `[[ ## answer ## ]]\nNo more responses`, and `DummyLM({})` gives the bare
    /// `No more responses`. Upstream renders the exhausted queue's fallback through
    /// `_format_answer_fields` and returns the keyed miss as it stands, so only one of the two
    /// parses. This refused both, on a ledger reason that said upstream raises.
    #[test]
    fn an_unplanned_call_is_answered_the_way_dspy_answers_it() {
        let queued = DummyLM::new([example! { answer: "only one" }]);
        assert!(ask(&queued, "first").unwrap().contains("only one"));
        assert_eq!(
            ask(&queued, "second").expect("the queue answers past its script"),
            "[[ ## answer ## ]]\nNo more responses"
        );

        let keyed = DummyLM::keyed([("Tokyo", example! { answer: "a" })]);
        assert_eq!(
            ask(&keyed, "somewhere else").expect("a keyed miss answers too"),
            "No more responses",
            "the bare words, so the reply does not parse — which is what upstream's caller sees"
        );
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
        assert_eq!(
            reply.usage,
            Some(LmUsage::counted(0, 0)),
            "a scripted answer reports zero tokens, as dspy's does"
        );
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
