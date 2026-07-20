use std::marker::PhantomData;

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::adapter::{Adapter, ChatAdapter, Feedback, JsonAdapter, turns_for};
use crate::example::Example;
use crate::lm::{ChatModel, global};
use crate::signature::{FieldKind, OutField, Signature, SignatureSpec};

/// dspy.Predict: ask through the configured adapter, demand the signature's fields back,
/// and recover the way DSPy does — a reply that parses but fails the signature gets
/// one ask carrying the previous output and the precise error. Three provider calls at most
/// on the value-level paths; the typed task paths add one more possible ask when the
/// validated reply does not deserialize into the task's outputs, so four at most there.
/// Every call is already bounded by the provider timeout.
pub struct Predict {
    pub signature: Signature,
    pub adapter: Box<dyn Adapter>,
    /// Solved examples shown before the request. An optimizer's output is a chosen set of
    /// these, so a compiled program is this field plus the signature's instructions.
    pub demos: Vec<Example>,
}

/// One accepted reply: the value that passed coercion and validation, the raw text it was
/// parsed from, and the adapter that produced it — enough for a typed caller to push a
/// deeper failure back through the same feedback path.
struct Validated {
    raw: String,
    value: Value,
}

impl Predict {
    pub fn new(signature: Signature) -> Self {
        Self {
            signature,
            adapter: Box::new(ChatAdapter::default()),
            demos: Vec::new(),
        }
    }

    /// Show the model these solved examples before the request.
    pub fn with_demos(mut self, demos: impl IntoIterator<Item = Example>) -> Self {
        self.demos = demos.into_iter().collect();
        self
    }

    /// Send this module's prompts through a different wire format. Any [`Adapter`] works,
    /// including one a caller writes: dspy chooses its adapter the same way.
    pub fn with_adapter(mut self, adapter: impl Adapter + 'static) -> Self {
        self.adapter = Box::new(adapter);
        self
    }

    /// The module for a derived signature; its caller speaks the signature's own types.
    pub fn task<S: SignatureSpec>() -> TypedPredict<S> {
        TypedPredict {
            predict: Predict::new(S::signature()),
            spec: PhantomData,
        }
    }

    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call(&self, input: &str) -> Result<Value> {
        let (http, lm) = global::current()?;
        self.call_with(&http, lm.as_ref(), input).await
    }

    /// Ask through an explicit client and model: the per-call override, and the seam tests
    /// script with a canned [`ChatModel`].
    pub async fn call_with(
        &self,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        input: &str,
    ) -> Result<Value> {
        let name = self.signature.inputs.first().map_or("request", |f| f.name);
        Ok(self
            .call_with_inputs(http, lm, &[(name, input.to_owned())])
            .await?
            .value)
    }

    /// One attempt: render through the adapter, ask the model, hand back the raw reply.
    /// dspy's `Adapter.__call__` in the module rather than the adapter, because a Rust trait
    /// carrying an async model call could not also be object-safe.
    async fn ask_through(
        &self,
        adapter: &dyn Adapter,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        inputs: &[(&str, String)],
        feedback: Option<&Feedback>,
    ) -> Result<String> {
        let schema = self.signature.schema();
        let (system, opening) = adapter.format(&self.signature, &self.demos, inputs);
        let mode = adapter.output_mode(&schema);
        lm.chat(http, &system, &turns_for(opening, feedback), &mode)
            .await
    }

    async fn ask(
        &self,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        inputs: &[(&str, String)],
        feedback: Option<&Feedback>,
    ) -> Result<String> {
        self.ask_through(self.adapter.as_ref(), http, lm, inputs, feedback)
            .await
    }

    async fn call_with_inputs(
        &self,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        inputs: &[(&str, String)],
    ) -> Result<Validated> {
        // dspy's ChatAdapter catches a parse failure and re-asks the whole exchange through
        // the JSON adapter; `use_json_adapter_fallback` turns that off. The adapter states the
        // policy, this module carries it out, because only the module can call the model.
        let raw = self.ask(http, lm, inputs, None).await?;
        let (raw, mut value) = match self.adapter.parse(&self.signature, &raw) {
            Ok(value) => (raw, value),
            Err(error) => match self.adapter.json_fallback() {
                None => return Err(error),
                Some(fallback) => {
                    tracing::warn!(%error, "reply did not parse; re-asking through the fallback");
                    let raw = self
                        .ask_through(fallback.as_ref(), http, lm, inputs, None)
                        .await?;
                    let value = fallback.parse(&self.signature, &raw)?;
                    (raw, value)
                }
            },
        };
        // Coercion failures ride the same feedback retry as validation failures: the reply
        // spoke the adapter's format, only a value was off, so the model gets the precise
        // error rather than a different wire format.
        match self
            .signature
            .coerce(&mut value)
            .and_then(|()| self.signature.ensure(&value))
        {
            Ok(()) => Ok(Validated { raw, value }),
            Err(error) => {
                tracing::warn!(%error, "retrying with feedback");
                let feedback = Feedback {
                    previous: raw,
                    error: error.to_string(),
                };
                let (raw, value) = self.feedback_ask(http, lm, inputs, &feedback).await?;
                Ok(Validated { raw, value })
            }
        }
    }

    /// One more ask on the same adapter carrying the rejected reply and its error; every
    /// failure past this point is final.
    async fn feedback_ask(
        &self,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        inputs: &[(&str, String)],
        feedback: &Feedback,
    ) -> Result<(String, Value)> {
        let raw = self.ask(http, lm, inputs, Some(feedback)).await?;
        let mut value = self.adapter.parse(&self.signature, &raw)?;
        self.signature.coerce(&mut value)?;
        self.signature.ensure(&value)?;
        Ok((raw, value))
    }

    /// The validated reply as a caller-owned struct instead of loose JSON.
    pub async fn call_typed<T: DeserializeOwned>(&self, input: &str) -> Result<T> {
        typed(self.call(input).await?)
    }

    /// [`Self::call_typed`] through an explicit client and model.
    pub async fn call_typed_with<T: DeserializeOwned>(
        &self,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        input: &str,
    ) -> Result<T> {
        typed(self.call_with(http, lm, input).await?)
    }
}

/// A [`Predict`] bound to a derived signature: the inputs struct in, the outputs struct back,
/// through the same adapter-fallback and feedback-retry path.
pub struct TypedPredict<S: SignatureSpec> {
    predict: Predict,
    spec: PhantomData<S>,
}

impl<S: SignatureSpec> TypedPredict<S> {
    /// Send this module's prompts through a different wire format; see
    /// [`Predict::with_adapter`].
    pub fn with_adapter(mut self, adapter: impl Adapter + 'static) -> Self {
        self.predict = self.predict.with_adapter(adapter);
        self
    }

    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call(&self, inputs: &S::Inputs) -> Result<S::Outputs> {
        let (http, lm) = global::current()?;
        self.call_with(&http, lm.as_ref(), inputs).await
    }

    /// Ask through an explicit client and model: the per-call override, and the seam tests
    /// script with a canned [`ChatModel`].
    pub async fn call_with(
        &self,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        inputs: &S::Inputs,
    ) -> Result<S::Outputs> {
        typed_task::<S>(&self.predict, http, lm, inputs, std::convert::identity).await
    }
}

/// The typed tail shared by [`TypedPredict`] and [`TypedChainOfThought`]: deserialize the
/// validated reply into the task's outputs, giving a shape mismatch one feedback retry that
/// carries the serde error — the typed paths' fourth possible provider call. A second
/// failure of any kind is final. `shape` trims module-owned fields (chain-of-thought's
/// `reasoning`) before deserializing.
async fn typed_task<S: SignatureSpec>(
    predict: &Predict,
    http: &reqwest::Client,
    lm: &impl ChatModel,
    inputs: &S::Inputs,
    shape: fn(Value) -> Value,
) -> Result<S::Outputs> {
    let pairs = S::input_pairs(inputs);
    let Validated { raw, value } = predict.call_with_inputs(http, lm, &pairs).await?;
    let error = match typed::<S::Outputs>(shape(value)) {
        Ok(outputs) => return Ok(outputs),
        Err(error) => error,
    };
    tracing::warn!(%error, "retrying with shape feedback");
    let feedback = Feedback {
        previous: raw,
        error: format!("{error:#}"),
    };
    let (_, value) = predict.feedback_ask(http, lm, &pairs, &feedback).await?;
    typed(shape(value))
}

fn typed<T: DeserializeOwned>(value: Value) -> Result<T> {
    serde_json::from_value(value).context("validated reply did not fit the requested type")
}

/// dspy.ChainOfThought: the same signature with a leading `reasoning` field. The model puts
/// its thinking there; the caller receives only the signature's own fields.
pub struct ChainOfThought {
    predict: Predict,
}

impl ChainOfThought {
    pub fn new(mut signature: Signature) -> Self {
        signature.outputs.insert(
            0,
            OutField {
                name: "reasoning",
                desc: "think step by step about the request before the other fields".into(),
                kind: FieldKind::Str,
                values: None,
                schema: None,
            },
        );
        Self {
            predict: Predict::new(signature),
        }
    }

    /// The module for a derived signature; its caller speaks the signature's own types.
    pub fn task<S: SignatureSpec>() -> TypedChainOfThought<S> {
        TypedChainOfThought {
            cot: ChainOfThought::new(S::signature()),
            spec: PhantomData,
        }
    }

    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call(&self, input: &str) -> Result<Value> {
        let (http, lm) = global::current()?;
        self.call_with(&http, lm.as_ref(), input).await
    }

    /// Ask through an explicit client and model: the per-call override, and the seam tests
    /// script with a canned [`ChatModel`].
    pub async fn call_with(
        &self,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        input: &str,
    ) -> Result<Value> {
        Ok(without_reasoning(
            self.predict.call_with(http, lm, input).await?,
        ))
    }

    /// The validated reply as a caller-owned struct instead of loose JSON.
    pub async fn call_typed<T: DeserializeOwned>(&self, input: &str) -> Result<T> {
        typed(self.call(input).await?)
    }

    /// [`Self::call_typed`] through an explicit client and model.
    pub async fn call_typed_with<T: DeserializeOwned>(
        &self,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        input: &str,
    ) -> Result<T> {
        typed(self.call_with(http, lm, input).await?)
    }
}

/// A [`ChainOfThought`] bound to a derived signature, mirroring [`TypedPredict`].
pub struct TypedChainOfThought<S: SignatureSpec> {
    cot: ChainOfThought,
    spec: PhantomData<S>,
}

impl<S: SignatureSpec> TypedChainOfThought<S> {
    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call(&self, inputs: &S::Inputs) -> Result<S::Outputs> {
        let (http, lm) = global::current()?;
        self.call_with(&http, lm.as_ref(), inputs).await
    }

    /// Ask through an explicit client and model: the per-call override, and the seam tests
    /// script with a canned [`ChatModel`].
    pub async fn call_with(
        &self,
        http: &reqwest::Client,
        lm: &impl ChatModel,
        inputs: &S::Inputs,
    ) -> Result<S::Outputs> {
        typed_task::<S>(&self.cot.predict, http, lm, inputs, without_reasoning).await
    }
}

fn without_reasoning(mut value: Value) -> Value {
    if let Some(map) = value.as_object_mut() {
        map.remove("reasoning");
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{ChatTurn, OutputMode, Role};
    use anyhow::anyhow;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    fn signature() -> Signature {
        Signature::single_input(
            "Pick a color.",
            vec![
                OutField {
                    name: "color",
                    desc: "the chosen color".into(),
                    kind: FieldKind::Str,
                    values: Some(vec!["red", "blue"]),
                    schema: None,
                },
                OutField {
                    name: "why",
                    desc: "one short sentence".into(),
                    kind: FieldKind::Str,
                    values: None,
                    schema: None,
                },
            ],
        )
    }

    fn typed_signature() -> Signature {
        let typed = |name: &'static str, kind: FieldKind| OutField {
            name,
            desc: name.into(),
            kind,
            values: None,
            schema: None,
        };
        Signature::single_input(
            "Size the gift.",
            vec![
                typed("amount", FieldKind::Float),
                typed("double", FieldKind::Bool),
                typed("count", FieldKind::Int),
            ],
        )
    }

    /// Scripted stand-in for a provider: pops one canned reply per call and records what
    /// each call asked, so tests can assert on the retry conversation.
    struct Scripted {
        replies: Mutex<VecDeque<&'static str>>,
        calls: Mutex<Vec<Call>>,
    }

    #[derive(Clone)]
    struct Call {
        system: String,
        turns: Vec<ChatTurn>,
        json_mode: bool,
    }

    impl Scripted {
        fn new(replies: &[&'static str]) -> Self {
            Self {
                replies: Mutex::new(replies.iter().copied().collect()),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<Call> {
            self.calls.lock().expect("not poisoned").clone()
        }
    }

    impl ChatModel for Scripted {
        async fn chat(
            &self,
            _http: &reqwest::Client,
            system: &str,
            turns: &[ChatTurn],
            mode: &OutputMode<'_>,
        ) -> Result<String> {
            self.calls.lock().expect("not poisoned").push(Call {
                system: system.to_owned(),
                turns: turns.to_vec(),
                json_mode: matches!(mode, OutputMode::Json { .. }),
            });
            self.replies
                .lock()
                .expect("not poisoned")
                .pop_front()
                .map(str::to_owned)
                .ok_or_else(|| anyhow!("script exhausted"))
        }
    }

    const MARKER_REPLY: &str =
        "[[ ## color ## ]]\nred\n\n[[ ## why ## ]]\ncalm\n\n[[ ## completed ## ]]";

    #[test]
    fn chain_of_thought_leads_with_reasoning_and_strips_it() {
        let cot = ChainOfThought::new(signature());
        let sig = &cot.predict.signature;
        assert_eq!(sig.outputs[0].name, "reasoning");
        assert_eq!(sig.schema()["required"][0], json!("reasoning"));

        let value = json!({ "reasoning": "…", "color": "red", "why": "calm" });
        assert_eq!(
            without_reasoning(value),
            json!({ "color": "red", "why": "calm" })
        );
    }

    #[tokio::test]
    async fn a_marker_reply_flows_through_the_chat_adapter() {
        let lm = Scripted::new(&[MARKER_REPLY]);
        let value = Predict::new(signature())
            .call_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("valid reply");
        assert_eq!(value, json!({ "color": "red", "why": "calm" }));
        let calls = lm.calls();
        assert_eq!(calls.len(), 1);
        assert!(!calls[0].json_mode);
        assert!(calls[0].system.contains("[[ ## color ## ]]"));
    }

    #[tokio::test]
    async fn a_validation_failure_retries_once_with_the_error_and_previous_output() {
        let bad = "[[ ## color ## ]]\ngreen\n\n[[ ## why ## ]]\ncalm";
        let lm = Scripted::new(&[bad, MARKER_REPLY]);
        let value = Predict::new(signature())
            .call_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("second reply is valid");
        assert_eq!(value["color"], "red");

        let calls = lm.calls();
        assert_eq!(calls.len(), 2);
        let retry = &calls[1].turns;
        assert_eq!(retry.len(), 3);
        assert_eq!(retry[1].role, Role::Assistant);
        assert_eq!(retry[1].content, bad);
        assert!(retry[2].content.contains("color must be one of red, blue"));
    }

    #[tokio::test]
    async fn an_unparseable_reply_re_asks_through_the_json_adapter() {
        // dspy `test_chat_adapter_fallback_to_json_adapter_on_exception`: a reply the marker
        // parser rejects sends the whole exchange again through the JSON adapter.
        let lm = Scripted::new(&[
            "red because it is calm",
            r#"{ "color": "red", "why": "calm" }"#,
        ]);
        let value = Predict::new(signature())
            .call_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("the fallback parses");
        assert_eq!(value["color"], "red");

        let calls = lm.calls();
        assert_eq!(calls.len(), 2);
        assert!(!calls[0].json_mode);
        assert!(calls[1].json_mode, "the second ask engages structured output");
    }

    #[tokio::test]
    async fn the_fallback_can_be_turned_off() {
        // dspy `test_chat_adapter_respects_use_json_adapter_fallback_flag`: with the flag
        // cleared the parse failure is final and the JSON adapter is never reached.
        let lm = Scripted::new(&["red because it is calm", r#"{ "color": "red" }"#]);
        let predict =
            Predict::new(signature()).with_adapter(ChatAdapter::without_json_fallback());
        assert!(
            predict
                .call_with(&reqwest::Client::new(), &lm, "draft it")
                .await
                .is_err()
        );
        assert_eq!(lm.calls().len(), 1, "no second ask when the fallback is off");
    }

    #[tokio::test]
    async fn attempts_stay_bounded_at_one_feedback_retry() {
        // A reply that parses but fails validation earns exactly one more ask, carrying the
        // rejected text and its error.
        let lm = Scripted::new(&[
            "[[ ## color ## ]]\ngreen\n\n[[ ## why ## ]]\ncalm",
            "[[ ## color ## ]]\nblue\n\n[[ ## why ## ]]\ncalm",
        ]);
        let value = Predict::new(signature())
            .call_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("second reply is valid");
        assert_eq!(value["color"], "blue");
        assert_eq!(lm.calls().len(), 2);

        let lm = Scripted::new(&[
            "[[ ## color ## ]]\ngreen\n\n[[ ## why ## ]]\ncalm",
            "[[ ## color ## ]]\nmauve\n\n[[ ## why ## ]]\ncalm",
        ]);
        assert!(
            Predict::new(signature())
                .call_with(&reqwest::Client::new(), &lm, "draft it")
                .await
                .is_err()
        );
        assert_eq!(lm.calls().len(), 2, "one ask plus one feedback retry, then stop");
    }

    #[derive(Debug, serde::Deserialize)]
    struct Pick {
        color: String,
        why: String,
    }

    #[tokio::test]
    async fn call_typed_hands_back_a_struct_or_a_shape_error() {
        let lm = Scripted::new(&[MARKER_REPLY]);
        let pick: Pick = Predict::new(signature())
            .call_typed_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("deserializes");
        assert_eq!(pick.color, "red");
        assert_eq!(pick.why, "calm");

        #[derive(Debug, serde::Deserialize)]
        struct Wrong {
            #[allow(dead_code)]
            color: u32,
        }
        let lm = Scripted::new(&[MARKER_REPLY]);
        let wrong: Result<Wrong> = Predict::new(signature())
            .call_typed_with(&reqwest::Client::new(), &lm, "draft it")
            .await;
        assert!(wrong.is_err());
    }

    #[tokio::test]
    async fn chain_of_thought_strips_reasoning_from_the_typed_path() {
        let reply = "[[ ## reasoning ## ]]\nthinking\n\n[[ ## color ## ]]\nred\n\n[[ ## why ## ]]\ncalm\n\n[[ ## completed ## ]]";
        let lm = Scripted::new(&[reply]);
        let cot = ChainOfThought::new(signature());
        let value = cot
            .call_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("valid reply");
        assert_eq!(value, json!({ "color": "red", "why": "calm" }));

        let lm = Scripted::new(&[reply]);
        let pick: Pick = cot
            .call_typed_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("deserializes");
        assert_eq!(pick.color, "red");
    }

    /// The derive is declaration data; the struct itself is never built.
    #[allow(dead_code)]
    #[derive(crate::signature::Signature)]
    #[signature(instructions = "Pick a color for the room.")]
    struct RoomTask {
        #[input(desc = "the room being painted")]
        room: String,
        #[input(desc = "the mood to set")]
        mood: String,
        #[output(desc = "the chosen color", values("red", "blue"))]
        color: String,
        #[output(desc = "one short sentence")]
        why: String,
    }

    fn room_inputs() -> RoomTaskInputs {
        RoomTaskInputs {
            room: "the study".into(),
            mood: "calm focus".into(),
        }
    }

    #[tokio::test]
    async fn a_typed_task_renders_every_input_and_returns_the_outputs_struct() {
        let lm = Scripted::new(&[MARKER_REPLY]);
        let outputs = Predict::task::<RoomTask>()
            .call_with(&reqwest::Client::new(), &lm, &room_inputs())
            .await
            .expect("valid reply");
        assert_eq!(outputs.color, "red");
        assert_eq!(outputs.why, "calm");

        let calls = lm.calls();
        assert!(
            calls[0]
                .system
                .contains("1. `room` (str): the room being painted")
        );
        assert!(calls[0].system.contains("2. `mood` (str): the mood to set"));
        let opening = &calls[0].turns[0].content;
        assert!(opening.contains("[[ ## room ## ]]\nthe study"));
        assert!(opening.contains("[[ ## mood ## ]]\ncalm focus"));
    }

    #[tokio::test]
    async fn a_typed_task_keeps_the_feedback_retry_with_every_input_in_place() {
        let bad = "[[ ## color ## ]]\ngreen\n\n[[ ## why ## ]]\ncalm";
        let lm = Scripted::new(&[bad, MARKER_REPLY]);
        let outputs = RoomTask::predict()
            .call_with(&reqwest::Client::new(), &lm, &room_inputs())
            .await
            .expect("second reply is valid");
        assert_eq!(outputs.color, "red");

        let calls = lm.calls();
        assert_eq!(calls.len(), 2);
        let retry = &calls[1].turns;
        assert_eq!(retry.len(), 3);
        assert!(retry[0].content.contains("[[ ## mood ## ]]\ncalm focus"));
        assert_eq!(retry[1].content, bad);
        assert!(retry[2].content.contains("color must be one of red, blue"));
    }

    /// Pins down what a call macro evaluates to: the module call's future, yielding the
    /// task's outputs. Constructing an async-fn future runs nothing, so the expansions
    /// typecheck and drop here without a configured global.
    fn expands_to_a_call_future<F>(_: F)
    where
        F: std::future::Future<Output = Result<RoomTaskOutputs>>,
    {
    }

    #[test]
    fn call_macros_take_literal_borrowed_and_owned_values() {
        use crate::signature::{chain_of_thought, predict};

        let mood: &str = "calm focus";
        let room: String = "the study".into();
        expands_to_a_call_future(predict!(RoomTask {
            room: "the study",
            mood: mood
        }));
        expands_to_a_call_future(predict!(RoomTask {
            room: room.clone(),
            mood: "calm focus",
        }));
        expands_to_a_call_future(chain_of_thought!(RoomTask {
            room: room,
            mood: mood.to_owned(),
        }));
    }

    #[tokio::test]
    async fn a_typed_chain_of_thought_strips_reasoning_before_deserializing() {
        let reply = "[[ ## reasoning ## ]]\nthinking\n\n[[ ## color ## ]]\nblue\n\n[[ ## why ## ]]\nfresh\n\n[[ ## completed ## ]]";
        let lm = Scripted::new(&[reply]);
        let outputs = RoomTask::chain_of_thought()
            .call_with(&reqwest::Client::new(), &lm, &room_inputs())
            .await
            .expect("valid reply");
        assert_eq!(outputs.color, "blue");
        assert_eq!(outputs.why, "fresh");
    }

    /// The derive is declaration data; the struct itself is never built.
    #[allow(dead_code)]
    #[derive(crate::signature::Signature)]
    #[signature(instructions = "Size the gift.")]
    struct SizeTask {
        #[input(desc = "the age turned")]
        age: u32,
        #[input(desc = "a lifelong fan")]
        fan: bool,
        #[input(desc = "the budget in MON")]
        budget: f64,
        #[input(desc = "years known")]
        years: i64,
        #[output(desc = "amount in MON")]
        amount: f64,
        #[output(desc = "double it")]
        double: bool,
        #[output(desc = "how many gifts")]
        count: u32,
    }

    const SIZE_REPLY: &str = "[[ ## amount ## ]]\n0.04\n\n[[ ## double ## ]]\ntrue\n\n[[ ## count ## ]]\n3\n\n[[ ## completed ## ]]";

    #[tokio::test]
    async fn a_typed_task_renders_inputs_and_coerces_marker_outputs() {
        let lm = Scripted::new(&[SIZE_REPLY]);
        let inputs = SizeTaskInputs {
            age: 61,
            fan: true,
            budget: 0.5,
            years: 30,
        };
        let outputs = SizeTask::predict()
            .call_with(&reqwest::Client::new(), &lm, &inputs)
            .await
            .expect("valid reply");
        assert_eq!(outputs.amount, 0.04);
        assert!(outputs.double);
        assert_eq!(outputs.count, 3);

        let calls = lm.calls();
        assert!(
            calls[0]
                .system
                .contains("1. `age` (int): the age turned")
        );
        assert!(
            calls[0]
                .system
                .contains("2. `fan` (bool): a lifelong fan")
        );
        assert!(
            calls[0]
                .system
                .contains("1. `amount` (float): amount in MON")
        );
        let opening = &calls[0].turns[0].content;
        assert!(opening.contains("[[ ## age ## ]]\n61"));
        assert!(opening.contains("[[ ## fan ## ]]\ntrue"));
        assert!(opening.contains("[[ ## budget ## ]]\n0.5"));
    }

    #[tokio::test]
    async fn a_type_error_rides_the_feedback_retry_not_the_adapter_fallback() {
        let bad = "[[ ## amount ## ]]\nabc\n\n[[ ## double ## ]]\ntrue\n\n[[ ## count ## ]]\n3";
        let good = "[[ ## amount ## ]]\n0.02\n\n[[ ## double ## ]]\nfalse\n\n[[ ## count ## ]]\n1";
        let lm = Scripted::new(&[bad, good]);
        let value = Predict::new(typed_signature())
            .call_with(&reqwest::Client::new(), &lm, "size it")
            .await
            .expect("second reply is valid");
        assert_eq!(
            value,
            json!({ "amount": 0.02, "double": false, "count": 1 })
        );

        let calls = lm.calls();
        assert_eq!(calls.len(), 2);
        assert!(!calls[0].json_mode);
        assert!(!calls[1].json_mode, "a type error must not switch adapters");
        let retry = &calls[1].turns;
        assert_eq!(retry.len(), 3);
        assert_eq!(retry[1].content, bad);
        assert!(
            retry[2]
                .content
                .contains("amount must be a number, got \"abc\"")
        );
    }

    #[tokio::test]
    async fn the_json_adapter_accepts_native_and_string_typed_values() {
        for reply in [
            r#"{ "amount": 0.04, "double": true, "count": 3 }"#,
            r#"{ "amount": "0.04", "double": "true", "count": "3" }"#,
        ] {
            let lm = Scripted::new(&[reply]);
            let predict = Predict {
                signature: typed_signature(),
                adapter: Box::new(JsonAdapter),
                demos: Vec::new(),
            };
            let value = predict
                .call_with(&reqwest::Client::new(), &lm, "size it")
                .await
                .expect("valid reply");
            assert_eq!(value, json!({ "amount": 0.04, "double": true, "count": 3 }));
            assert!(lm.calls()[0].json_mode);
        }
    }

    fn expands_to_a_size_future<F>(_: F)
    where
        F: std::future::Future<Output = Result<SizeTaskOutputs>>,
    {
    }

    #[test]
    fn call_macros_take_typed_literals_and_bindings() {
        use crate::signature::{chain_of_thought, predict};

        // Unsuffixed integer literals fall back to i32, which converts into i64 but not
        // into u32; an unsigned field takes a suffixed literal or a typed binding.
        expands_to_a_size_future(predict!(SizeTask {
            age: 61u32,
            fan: true,
            budget: 0.5,
            years: 30,
        }));
        let age: u32 = 61;
        let budget: f64 = 0.5;
        expands_to_a_size_future(chain_of_thought!(SizeTask {
            age: age,
            fan: false,
            budget: budget,
            years: 30i64,
        }));
    }
}
