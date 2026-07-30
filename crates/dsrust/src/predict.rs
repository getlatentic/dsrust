use std::marker::PhantomData;

use anyhow::Result;
use serde_json::{Map, Value};

use crate::adapter::native_reasoning::{self, ReasoningEffort};
use crate::adapter::parse::FieldMismatch;
use crate::adapter::{Adapter, Feedback, Input, native_tools, turns_for};
use crate::example::{Example, Prediction};
use crate::lm::{Capabilities, DynChatModel, LmUsage, Sampling, api};
use crate::module::{Module, NamedPredictor, TraceStep};
use crate::signature::Signature;

/// dspy's per-call `config=`: the fields one ask steers that the module's own config does not carry.
/// A reasoning budget, and a tool the provider must call — ReActV2's forced submit sets both, to
/// pin the last ask to `submit` and turn native reasoning off for it.
#[derive(Debug, Clone, Default)]
pub struct Steering {
    pub reasoning_effort: ReasoningEffort,
    /// A tool the provider must call, sent only under native function calling, since dspy drops
    /// `tool_choice` for an adapter that renders the tools instead.
    pub forced_tool: Option<String>,
    /// OpenAI's Predicted Outputs: text the reply is expected to mostly repeat, which the provider
    /// bills and generates against rather than writing again. See `predicted_output` for the
    /// other way in — an input field of the same shape, which takes precedence over this.
    pub predicted_output: Option<Value>,
}

/// dspy `_forward_preprocess`: an input named `prediction` shaped like OpenAI's Predicted Outputs
/// is not an input at all. It comes off the render and goes to the provider as a call parameter.
///
/// The *shape* is what decides, not the signature. A signature declaring `prediction` as an input
/// still loses it to the model when a caller passes this exact object, and a `prediction` input
/// holding anything else — a string, a differently-tagged object — stays an ordinary input. Both
/// halves are upstream's, quirk included.
fn predicted_output(inputs: &Example) -> Option<Value> {
    let offered = inputs.get("prediction")?;
    let fields = offered.as_object()?;
    let shaped = fields.get("type").and_then(Value::as_str) == Some("content")
        && fields.contains_key("content");
    shaped.then(|| offered.clone())
}

/// What one call renders, and the predicted output taken out of it.
///
/// Both ways in need this — one answer or several — because upstream runs `_forward_preprocess`
/// ahead of either, and a rule applied on only one path is a rule that depends on `n`.
pub(crate) fn rendered_inputs(inputs: &Example) -> (Vec<Input<'_>>, Option<Value>) {
    let lifted = predicted_output(inputs);
    let pairs = inputs
        .fields()
        .filter(|(name, _)| lifted.is_none() || *name != "prediction")
        .map(|(name, value)| Input::new(name, value.clone()))
        .collect();
    (pairs, lifted)
}

/// A reply and the signature that was actually rendered to get it.
///
/// dspy's `_call_preprocess` returns the render signature and `_call_postprocess` reads the reply
/// against it: fields taken off the render for a native feature — the tool calls, the reasoning —
/// are absent from what the reply spoke, and are filled from their own channels instead.
struct Reply {
    response: api::LmResponse,
    rendered: Signature,
}

mod aggregation;
mod best_of_n;
mod building;
mod chain_of_thought;
pub mod code_act;
mod completions;
mod derived;
mod hint;
mod multi_chain_comparison;
mod native;
mod parallel;
pub mod program_of_thought;
mod recovery;
pub mod refine;
pub mod rlm;
mod shorthand;
pub use aggregation::{Normalize, majority, normalize_text};
pub use best_of_n::BestOfN;
pub use chain_of_thought::{ChainOfThought, TypedChainOfThought};
pub use code_act::CodeAct;
pub use derived::TypedPredict;
use derived::typed;
pub use multi_chain_comparison::MultiChainComparison;
use native::{ask_for_parallel_calls, force_tool};
pub use parallel::{Answered, Parallel};
pub use program_of_thought::ProgramOfThought;
pub use refine::Refine;
pub use rlm::Rlm;

#[cfg(test)]
mod scripted;

/// dspy.Predict: ask through the configured adapter, demand the signature's fields back,
/// and recover the way DSPy does — a reply that parses but fails the signature gets
/// one ask carrying the previous output and the precise error. Three provider calls at most
/// on the value-level paths; the typed task paths add one more possible ask when the
/// validated reply does not deserialize into the task's outputs, so four at most there.
/// Every call is already bounded by the provider timeout.
pub struct Predict<S = Dynamic> {
    pub signature: Signature,
    pub adapter: Box<dyn Adapter>,
    /// Solved examples shown before the request. An optimizer's output is a chosen set of
    /// these, so a compiled program is this field plus the signature's instructions.
    pub demos: Vec<Example>,
    /// The model this module asks, when it is not the configured one.
    ///
    /// dspy's `set_lm`, and the seam an optimizer needs: `BestOfN` runs a module several times
    /// against models that differ only in their config, and `BootstrapFewShot` gives each
    /// round after the first a model that will not answer from a cache. Neither can reach a
    /// process-wide default to do it.
    lm: Option<std::sync::Arc<dyn DynChatModel>>,
    /// How this module asks for its reply to be sampled.
    ///
    /// The other half of the same seam: `BestOfN` and a bootstrap round after the first vary
    /// this rather than the model, since what they need to differ is one call's config and
    /// not which provider answers.
    config: Sampling,
    /// What an earlier attempt was told to do differently. See [`NamedPredictor::hint`].
    hint: Option<String>,
    /// Whether a reply that parsed but did not validate is re-asked with the error attached.
    ///
    /// Off, because dspy has no such ask. Upstream raises `AdapterParseError` for a missing or
    /// unusable field and `ChatAdapter.__call__` re-asks through `JSONAdapter` — one extra call
    /// either way, but the prompt is the JSON adapter's rather than a sentence dspy never sends.
    /// Turned on with [`Self::with_feedback_retry`], for a caller who wants the recovery and
    /// accepts that the second ask is this crate's own.
    feedback_retry: bool,
    spec: PhantomData<S>,
}

/// A signature carried as field names rather than as a type.
///
/// dspy has one `Predict` class because a signature there is always a value it holds. This is
/// the same shape: one type, and what a call answers with follows from whether that type knows
/// its outputs. `Predict<Dynamic>` knows only the names, so it answers with the fields it parsed.
pub struct Dynamic;

/// One accepted reply: the value that passed coercion and validation, the raw text it was
/// parsed from, and the adapter that produced it — enough for a typed caller to push a
/// deeper failure back through the same feedback path.
struct Validated {
    raw: String,
    value: Value,
    /// What every call behind this one cost together — a fallback and a feedback retry each add
    /// their own, so the accepted reply carries the whole exchange rather than its last ask.
    usage: Option<LmUsage>,
}

impl<S> Predict<S> {
    /// One attempt: render through the adapter, ask the model, hand back the raw reply.
    /// dspy's `Adapter.__call__` in the module rather than the adapter, because a Rust trait
    /// carrying an async model call could not also be object-safe.
    async fn ask_through(
        &self,
        adapter: &dyn Adapter,
        lm: &dyn DynChatModel,
        inputs: &[Input<'_>],
        feedback: Option<&Feedback>,
        steering: &Steering,
    ) -> Result<Reply> {
        let hint = self.hint.as_deref();
        let asked = hint::signature_with(&self.signature, hint);
        let hinted = hint::inputs_with(inputs, hint);
        // dspy's `_call_preprocess`: a native feature moves its field off the render — the provider
        // calls the tools itself, the reasoning model thinks on its own channel — so what gets
        // rendered never mentions it. Both adaptations ask the model's capabilities, so they are
        // fetched once, and only for a signature that could use one.
        let asking = adapter.native_function_calling();
        let reasons = native_reasoning::reasoning_output_field(&asked).is_some();
        let capabilities = match asking.enabled || reasons {
            true => lm.capabilities_dyn().await,
            false => Capabilities::default(),
        };
        let native = match asking.enabled {
            true => native_tools::plan(&asked, &hinted, capabilities)?,
            false => None,
        };
        let asked = native.as_ref().map_or(asked, |plan| plan.signature.clone());
        let reasoning = native_reasoning::plan(
            &asked,
            capabilities,
            &steering.reasoning_effort,
            lm.native_reasoning_usable_dyn(),
        );
        let asked = reasoning
            .as_ref()
            .map_or(asked, |plan| plan.signature.clone());
        let schema = asked.schema();
        let (system, opening) =
            crate::observe::formatting(adapter, &asked, &self.demos, &hinted, || {
                adapter.format(&asked, &self.demos, &hinted)
            })?;
        let mode = adapter.output_mode(&schema);
        let turns = turns_for(opening, feedback);
        // The typed 3.3 boundary: predict hands the model an `LMRequest`. Behind it the request
        // is lowered to the shape the providers still speak, so no wire byte moves.
        let mut request = api::interop::raise_request(&system, &turns, mode, &self.config);
        if let Some(plan) = native {
            request.tools = plan.tools;
            ask_for_parallel_calls(&mut request, asking.parallel);
        }
        if let Some(reasoning) = reasoning {
            request.config.reasoning = Some(api::LmReasoningConfig {
                effort: Some(reasoning.effort),
                ..Default::default()
            });
        }
        // dspy drops `tool_choice` unless the adapter asks natively, so a forced tool is sent only
        // then; a rendered-tools exchange has no provider-side choice to steer.
        if asking.enabled
            && let Some(tool) = &steering.forced_tool
        {
            force_tool(&mut request, tool);
        }
        // Predicted Outputs is not a field the normalized config models, upstream's included — it
        // rides in `extensions`, which every provider flattens back onto the call it makes.
        if let Some(predicted) = &steering.predicted_output {
            request
                .config
                .extensions
                .insert("prediction".to_owned(), predicted.clone());
        }
        let response = lm.forward_dyn(&request).await?;
        Ok(Reply {
            response,
            rendered: asked,
        })
    }

    async fn ask(
        &self,
        lm: &dyn DynChatModel,
        inputs: &[Input<'_>],
        feedback: Option<&Feedback>,
        steering: &Steering,
    ) -> Result<Reply> {
        self.ask_through(self.adapter.as_ref(), lm, inputs, feedback, steering)
            .await
    }

    async fn call_with_inputs(
        &self,
        lm: &dyn DynChatModel,
        inputs: &[Input<'_>],
        steering: &Steering,
    ) -> Result<Validated> {
        // dspy's ChatAdapter catches a parse failure and re-asks the whole exchange through
        // the JSON adapter; `use_json_adapter_fallback` turns that off. The adapter states the
        // policy, this module carries it out, because only the module can call the model.
        let answered = self.ask(lm, inputs, None, steering).await?;
        // A provider that returned no completions at all did not return an empty one. Upstream's
        // `Adapter.__call__` loops over the outputs, so an empty list never reaches its parser and
        // `Predict` hands back a prediction with no fields; parsing `""` instead would report a
        // malformed reply for a call that produced none. `forward_completions` already answers with
        // an empty list here, and the two paths cannot disagree about the same response.
        if answered.response.outputs.is_empty() {
            return Ok(Validated {
                raw: String::new(),
                value: Value::Object(Map::new()),
                usage: answered.response.spend(),
            });
        }
        // dspy `_call_postprocess`: a reply whose fields left the render for a native feature is
        // filled from those channels and skips the whole text path — parse, coercion, feedback —
        // exactly as upstream returns the native reply without re-validating it.
        if let Some(value) = self.native_value(&answered)? {
            return Ok(Validated {
                raw: answered.response.first_text(),
                value,
                usage: answered.response.spend(),
            });
        }
        let usage = answered.response.spend();
        let raw = answered.response.first_text();
        // An adapter that answers in prose has a second model read the fields out of it. The
        // adapter says what to ask and who to ask; only this module can do the asking.
        if let Some(extraction) = self.adapter.extraction(&self.signature) {
            return self.extract(extraction, raw, usage).await;
        }
        let parsed = crate::observe::parsing(self.adapter.as_ref(), &raw, || {
            self.adapter.parse(&self.signature, &raw)
        });
        let (raw, mut value, usage) = match parsed {
            Ok(value) => (raw, value, usage),
            // A reply that spoke the format but left a field out. Upstream raises
            // `AdapterParseError` here and `ChatAdapter.__call__` re-asks through `JSONAdapter`,
            // which is the arm below — so this one only runs where a caller asked for the
            // feedback ask instead, and it carries the partial forward for `ensure` to name.
            Err(error) if self.feedback_retry && error.is::<FieldMismatch>() => {
                let partial = error
                    .downcast::<FieldMismatch>()
                    .map(|mismatch| mismatch.parsed)
                    .unwrap_or(Value::Null);
                (raw, partial, usage)
            }
            Err(error) => match self.adapter.json_fallback() {
                None => return Err(error),
                Some(fallback) => {
                    tracing::warn!(%error, "reply did not parse; re-asking through the fallback");
                    let answered = self
                        .ask_through(fallback.as_ref(), lm, inputs, None, steering)
                        .await?;
                    let answered_text = answered.response.first_text();
                    let value = crate::observe::parsing(fallback.as_ref(), &answered_text, || {
                        fallback.parse(&self.signature, &answered_text)
                    })?;
                    let merged = LmUsage::merge(usage, answered.response.spend());
                    (answered_text, value, merged)
                }
            },
        };
        // A value that will not coerce is upstream's `AdapterParseError` too — `parse_value`
        // raises inside `parse`, so the JSON fallback is what answers it there. Without the
        // feedback ask there is nothing left to try, and the error is the caller's.
        match self
            .signature
            .coerce(&mut value)
            .and_then(|()| self.signature.ensure(&value))
        {
            Ok(()) => Ok(Validated { raw, value, usage }),
            Err(error) if !self.feedback_retry => Err(error),
            Err(error) => {
                tracing::warn!(%error, "retrying with feedback");
                let feedback = Feedback {
                    previous: raw,
                    error: error.to_string(),
                };
                let (raw, value, retried) =
                    self.feedback_ask(lm, inputs, &feedback, steering).await?;
                Ok(Validated {
                    raw,
                    value,
                    usage: LmUsage::merge(usage, retried),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{ChatAdapter, JsonAdapter};
    use crate::lm::Role;
    use crate::predict::scripted::{
        Pick, RoomTask, RoomTaskInputs, RoomTaskOutputs, Scripted, room_inputs, signature,
    };
    use crate::signature::{FieldKind, OutField};
    use crate::{input, module::Module};
    use serde_json::json;

    fn typed_signature() -> Signature {
        let typed = |name: &str, kind: FieldKind| OutField {
            name: name.into(),
            desc: name.into(),
            kind,
            ..Default::default()
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

    const MARKER_REPLY: &str =
        "[[ ## color ## ]]\nred\n\n[[ ## why ## ]]\ncalm\n\n[[ ## completed ## ]]";

    /// A provider that answers one request with several candidates at once, which is what asking
    /// for `n` completions returns — the shape `forward_completions` exists to read.
    struct ManyCompletions(Vec<&'static str>);

    impl crate::lm::ChatModel for ManyCompletions {
        async fn forward(&self, _request: &api::LmRequest) -> Result<api::LmResponse> {
            Ok(api::LmResponse::completions(
                self.0.iter().map(|reply| reply.to_string()),
            ))
        }
    }

    /// A provider that keeps the request it was handed, so a test can read what the crate decided
    /// to *send* and not only what it decided to render.
    #[derive(Default)]
    struct Captured(std::sync::Mutex<Option<api::LmRequest>>);

    impl crate::lm::ChatModel for Captured {
        async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
            *self.0.lock().expect("the captured request") = Some(request.clone());
            Ok(api::LmResponse::completions([MARKER_REPLY.to_owned()]))
        }
    }

    /// What one call sent, for a predictor answering through [`Captured`].
    async fn sent_by(spec: &str, inputs: Example) -> api::LmRequest {
        let model = std::sync::Arc::new(Captured::default());
        let predict = Predict::parse(spec).expect("parses").with_lm(model.clone());
        predict.forward(inputs).await.expect("answers");
        model
            .0
            .lock()
            .expect("captured")
            .clone()
            .expect("a request reached the model")
    }

    fn rendered(request: &api::LmRequest) -> String {
        serde_json::to_string(&request.messages).expect("the messages serialize")
    }

    /// OpenAI's Predicted Outputs is a call parameter, not an input. dspy moves it off the render
    /// into the per-call config, and the normalized config models no such field on either side —
    /// so it travels in `extensions`, which is what a provider flattens onto its call.
    #[tokio::test]
    async fn a_predicted_output_reaches_the_provider_and_not_the_prompt() {
        let offered = json!({ "type": "content", "content": "a room that is red" });
        let sent = sent_by(
            "request -> color, why",
            Example::new([
                ("request", json!("pick a colour")),
                ("prediction", offered.clone()),
            ]),
        )
        .await;

        assert_eq!(sent.config.extensions.get("prediction"), Some(&offered));
        assert!(
            !rendered(&sent).contains("a room that is red"),
            "and never reached the prompt"
        );
    }

    /// The quirk, reproduced on purpose: upstream tests the *value*, never the field list, so a
    /// signature declaring `prediction` as an input still loses it when the value is that shape.
    /// Declaring the field is no protection, and this is the case that proves the filter runs.
    #[tokio::test]
    async fn a_declared_prediction_input_is_taken_too_when_it_is_that_shape() {
        let offered = json!({ "type": "content", "content": "a room that is red" });
        let sent = sent_by(
            "request, prediction -> color, why",
            Example::new([
                ("request", json!("pick a colour")),
                ("prediction", offered.clone()),
            ]),
        )
        .await;

        assert_eq!(sent.config.extensions.get("prediction"), Some(&offered));
        assert!(
            !rendered(&sent).contains("a room that is red"),
            "declared, and still not rendered"
        );
    }

    /// The other half of upstream's test: a `prediction` input holding anything else is an
    /// ordinary input. It renders, and no call parameter is set for it.
    #[tokio::test]
    async fn a_prediction_input_of_another_shape_stays_an_input() {
        let sent = sent_by(
            "request, prediction -> color, why",
            Example::new([
                ("request", json!("pick a colour")),
                ("prediction", json!("to get to the other side")),
            ]),
        )
        .await;

        assert!(
            sent.config.extensions.is_empty(),
            "nothing was lifted out of the inputs"
        );
        assert!(
            rendered(&sent).contains("to get to the other side"),
            "it rendered as an input"
        );
    }

    /// An instruction optimizer proposes `n` candidates in one call; `forward_completions` reads
    /// every one, not just the first, and parses each into its own prediction.
    #[tokio::test]
    async fn forward_completions_reads_every_candidate_not_just_the_first() {
        let replies = vec![
            "[[ ## color ## ]]\nred\n\n[[ ## why ## ]]\nwarm\n\n[[ ## completed ## ]]",
            "[[ ## color ## ]]\nblue\n\n[[ ## why ## ]]\ncool\n\n[[ ## completed ## ]]",
            "[[ ## color ## ]]\ngreen\n\n[[ ## why ## ]]\nfresh\n\n[[ ## completed ## ]]",
        ];
        let predict = Predict::parse("request -> color, why")
            .expect("parses")
            .with_lm(std::sync::Arc::new(ManyCompletions(replies)))
            .with_config(Sampling {
                completions: Some(3),
                ..Sampling::default()
            });

        let candidates = predict
            .forward_completions(input! { request: "pick a colour" })
            .await
            .expect("candidates");

        assert_eq!(candidates.len(), 3, "every completion is read");
        let colors: Vec<_> = candidates
            .iter()
            .filter_map(|c| c.get("color").cloned())
            .collect();
        assert_eq!(colors, [json!("red"), json!("blue"), json!("green")]);
    }

    /// `Refine` writes advice onto a predictor between attempts, and the model has to actually
    /// see it — as one more input field, which is what upstream appends.
    #[tokio::test]
    async fn a_hint_reaches_the_prompt_as_one_more_input_field() {
        let lm = Scripted::new(&[MARKER_REPLY, MARKER_REPLY]);

        let mut plain = Predict::from_signature(signature());
        plain.call_with(&lm, "draft it").await.expect("valid reply");

        for predictor in plain.named_predictors() {
            *predictor.hint = Some("name a warm colour".to_owned());
        }
        plain.call_with(&lm, "draft it").await.expect("valid reply");

        let calls = lm.calls();
        assert!(
            !calls[0].system.contains("hint_"),
            "a module with no advice renders as though the field did not exist"
        );
        assert!(
            calls[1]
                .system
                .contains("`hint_` (str): A hint to the module from an earlier run"),
            "got: {}",
            calls[1].system
        );
        assert!(
            calls[1].turns[0]
                .content
                .text()
                .expect("a request")
                .contains("name a warm colour"),
            "the advice itself travels, not only the field describing it"
        );
    }

    #[tokio::test]
    async fn a_marker_reply_flows_through_the_chat_adapter() {
        let lm = Scripted::new(&[MARKER_REPLY]);
        let value = Predict::from_signature(signature())
            .call_with(&lm, "draft it")
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
        let value = Predict::from_signature(signature())
            .with_feedback_retry()
            .call_with(&lm, "draft it")
            .await
            .expect("second reply is valid");
        assert_eq!(value["color"], "red");

        let calls = lm.calls();
        assert_eq!(calls.len(), 2);
        let retry = &calls[1].turns;
        assert_eq!(retry.len(), 3);
        assert_eq!(retry[1].role, Role::Assistant);
        assert_eq!(retry[1].content.text().unwrap(), bad);
        assert!(
            retry[2]
                .content
                .text()
                .unwrap()
                .contains("color must be one of red, blue")
        );
    }

    /// A `forward` can make up to four provider calls, and every one of them is billed. Reporting
    /// only the last would understate a retry by exactly the calls the recovery cost — which is
    /// the number a caller is trying to see when they go looking.
    #[tokio::test]
    async fn what_a_forward_cost_is_every_call_behind_it_summed() {
        let bad = "[[ ## color ## ]]\ngreen\n\n[[ ## why ## ]]\ncalm";
        let lm = Scripted::new(&[bad, MARKER_REPLY]).costing(30, 12);
        let answered = Predict::from_signature(signature())
            .with_feedback_retry()
            .with_lm(std::sync::Arc::new(lm))
            .forward(input! { request: "draft it" })
            .await
            .expect("the retry is valid");

        let usage = answered.usage.expect("both calls reported a cost");
        assert_eq!(usage.input_tokens, Some(60), "the first ask and the retry");
        assert_eq!(usage.output_tokens, Some(24));
    }

    /// A model reporting nothing must not report zero: a caller adding a scripted run to a real
    /// one would otherwise get a total that looks whole and is not.
    #[tokio::test]
    async fn a_model_that_reports_no_cost_answers_with_none_rather_than_zero() {
        let answered = Predict::from_signature(signature())
            .with_lm(std::sync::Arc::new(Scripted::new(&[MARKER_REPLY])))
            .forward(input! { request: "draft it" })
            .await
            .expect("a valid reply");
        assert_eq!(answered.usage, None);
    }

    #[tokio::test]
    async fn an_unparseable_reply_re_asks_through_the_json_adapter() {
        // dspy `test_chat_adapter_fallback_to_json_adapter_on_exception`: a reply the marker
        // parser rejects sends the whole exchange again through the JSON adapter.
        let lm = Scripted::new(&[
            "red because it is calm",
            r#"{ "color": "red", "why": "calm" }"#,
        ]);
        let value = Predict::from_signature(signature())
            .call_with(&lm, "draft it")
            .await
            .expect("the fallback parses");
        assert_eq!(value["color"], "red");

        let calls = lm.calls();
        assert_eq!(calls.len(), 2);
        assert!(!calls[0].json_mode);
        assert!(
            calls[1].json_mode,
            "the second ask engages structured output"
        );
    }

    #[tokio::test]
    async fn the_fallback_can_be_turned_off() {
        // dspy `test_chat_adapter_respects_use_json_adapter_fallback_flag`: with the flag
        // cleared the parse failure is final and the JSON adapter is never reached.
        let lm = Scripted::new(&["red because it is calm", r#"{ "color": "red" }"#]);
        let predict =
            Predict::from_signature(signature()).with_adapter(ChatAdapter::without_json_fallback());
        assert!(predict.call_with(&lm, "draft it").await.is_err());
        assert_eq!(
            lm.calls().len(),
            1,
            "no second ask when the fallback is off"
        );
    }

    /// A missing field takes upstream's route by default: dspy raises `AdapterParseError` and
    /// `ChatAdapter.__call__` re-asks through `JSONAdapter`. The second ask therefore carries the
    /// JSON adapter's prompt, not a sentence naming the rejection — which is text dspy never sends.
    #[tokio::test]
    async fn a_missing_field_re_asks_through_the_json_adapter_as_dspy_does() {
        let lm = Scripted::new(&[
            "[[ ## color ## ]]\nred",
            r#"{"color": "blue", "why": "calm"}"#,
        ]);
        let value = Predict::from_signature(signature())
            .call_with(&lm, "draft it")
            .await
            .expect("the fallback answers");
        assert_eq!(value["color"], "blue");

        let calls = lm.calls();
        assert_eq!(calls.len(), 2, "one ask, then the JSON fallback");
        let second: String = calls[1]
            .turns
            .iter()
            .map(|turn| format!("{:?}", turn.content))
            .collect();
        assert!(
            !second.contains("previous reply was rejected"),
            "the second ask is the JSON adapter's, not a feedback sentence: {second}"
        );
        assert!(calls[1].json_mode, "and it asked in JSON mode");
    }

    #[tokio::test]
    async fn attempts_stay_bounded_at_one_feedback_retry() {
        // A reply that parses but fails validation earns exactly one more ask, carrying the
        // rejected text and its error.
        let lm = Scripted::new(&[
            "[[ ## color ## ]]\ngreen\n\n[[ ## why ## ]]\ncalm",
            "[[ ## color ## ]]\nblue\n\n[[ ## why ## ]]\ncalm",
        ]);
        let value = Predict::from_signature(signature())
            .with_feedback_retry()
            .call_with(&lm, "draft it")
            .await
            .expect("second reply is valid");
        assert_eq!(value["color"], "blue");
        assert_eq!(lm.calls().len(), 2);

        let lm = Scripted::new(&[
            "[[ ## color ## ]]\ngreen\n\n[[ ## why ## ]]\ncalm",
            "[[ ## color ## ]]\nmauve\n\n[[ ## why ## ]]\ncalm",
        ]);
        assert!(
            Predict::from_signature(signature())
                .with_feedback_retry()
                .call_with(&lm, "draft it")
                .await
                .is_err()
        );
        assert_eq!(
            lm.calls().len(),
            2,
            "one ask plus one feedback retry, then stop"
        );
    }

    #[tokio::test]
    async fn call_typed_hands_back_a_struct_or_a_shape_error() {
        let lm = Scripted::new(&[MARKER_REPLY]);
        let pick: Pick = Predict::from_signature(signature())
            .call_typed_with(&lm, "draft it")
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
        let wrong: Result<Wrong> = Predict::from_signature(signature())
            .call_typed_with(&lm, "draft it")
            .await;
        assert!(wrong.is_err());
    }

    #[tokio::test]
    async fn a_typed_task_renders_every_input_and_returns_the_outputs_struct() {
        let lm = Scripted::new(&[MARKER_REPLY]);
        let outputs = Predict::task::<RoomTask>()
            .call_inputs_with(&lm, &room_inputs())
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
        let opening = calls[0].turns[0].content.text().unwrap();
        assert!(opening.contains("[[ ## room ## ]]\nthe study"));
        assert!(opening.contains("[[ ## mood ## ]]\ncalm focus"));
    }

    #[tokio::test]
    async fn a_typed_task_keeps_the_feedback_retry_with_every_input_in_place() {
        let bad = "[[ ## color ## ]]\ngreen\n\n[[ ## why ## ]]\ncalm";
        let lm = Scripted::new(&[bad, MARKER_REPLY]);
        let outputs = RoomTask::predict()
            .with_feedback_retry()
            .call_inputs_with(&lm, &room_inputs())
            .await
            .expect("second reply is valid");
        assert_eq!(outputs.color, "red");

        let calls = lm.calls();
        assert_eq!(calls.len(), 2);
        let retry = &calls[1].turns;
        assert_eq!(retry.len(), 3);
        assert!(
            retry[0]
                .content
                .text()
                .unwrap()
                .contains("[[ ## mood ## ]]\ncalm focus")
        );
        assert_eq!(retry[1].content.text().unwrap(), bad);
        assert!(
            retry[2]
                .content
                .text()
                .unwrap()
                .contains("color must be one of red, blue")
        );
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
        use crate::signature::{ChainOfThought, Predict};

        let mood: &str = "calm focus";
        let room: String = "the study".into();
        expands_to_a_call_future(Predict!(RoomTask {
            room: "the study",
            mood: mood
        }));
        expands_to_a_call_future(Predict!(RoomTask {
            room: room.clone(),
            mood: "calm focus",
        }));
        expands_to_a_call_future(ChainOfThought!(RoomTask {
            room: room,
            mood: mood.to_owned(),
        }));
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
            .call_inputs_with(&lm, &inputs)
            .await
            .expect("valid reply");
        assert_eq!(outputs.amount, 0.04);
        assert!(outputs.double);
        assert_eq!(outputs.count, 3);

        let calls = lm.calls();
        assert!(calls[0].system.contains("1. `age` (int): the age turned"));
        assert!(calls[0].system.contains("2. `fan` (bool): a lifelong fan"));
        assert!(
            calls[0]
                .system
                .contains("1. `amount` (float): amount in MON")
        );
        let opening = calls[0].turns[0].content.text().unwrap();
        assert!(opening.contains("[[ ## age ## ]]\n61"));
        // Python's spelling: dspy 3.2.1 hands a bare bool to `str`, so the model reads `True`.
        assert!(opening.contains("[[ ## fan ## ]]\nTrue"));
        assert!(opening.contains("[[ ## budget ## ]]\n0.5"));
    }

    #[tokio::test]
    async fn a_type_error_rides_the_feedback_retry_not_the_adapter_fallback() {
        let bad = "[[ ## amount ## ]]\nabc\n\n[[ ## double ## ]]\ntrue\n\n[[ ## count ## ]]\n3";
        let good = "[[ ## amount ## ]]\n0.02\n\n[[ ## double ## ]]\nfalse\n\n[[ ## count ## ]]\n1";
        let lm = Scripted::new(&[bad, good]);
        let value = Predict::from_signature(typed_signature())
            .with_feedback_retry()
            .call_with(&lm, "size it")
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
        assert_eq!(retry[1].content.text().unwrap(), bad);
        assert!(
            retry[2]
                .content
                .text()
                .unwrap()
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
                spec: PhantomData,
                lm: None,
                config: Sampling::default(),
                hint: None,
                feedback_retry: false,
                signature: typed_signature(),
                adapter: Box::new(JsonAdapter::default()),
                demos: Vec::new(),
            };
            let value = predict
                .call_with(&lm, "size it")
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
        use crate::signature::{ChainOfThought, Predict};

        // Unsuffixed integer literals fall back to i32, which converts into i64 but not
        // into u32; an unsigned field takes a suffixed literal or a typed binding.
        expands_to_a_size_future(Predict!(SizeTask {
            age: 61u32,
            fan: true,
            budget: 0.5,
            years: 30,
        }));
        let age: u32 = 61;
        let budget: f64 = 0.5;
        expands_to_a_size_future(ChainOfThought!(SizeTask {
            age: age,
            fan: false,
            budget: budget,
            years: 30i64,
        }));
    }
}

/// dspy's `Predict` is a `Module`, and so is this one. Without this an optimizer could not
/// walk a program to reach the demos it exists to rewrite, and an evaluator could not take a
/// built-in module and a caller's own module through the same door.
impl<S: Send + Sync> Predict<S> {
    /// One call steered as dspy's per-call `config=` steers it — a reasoning budget cleared or set,
    /// a tool the provider must choose. [`Module::forward`] is this with nothing steered; ReActV2's
    /// forced submit is this with the last ask pinned to `submit` and native reasoning turned off.
    pub async fn forward_with_steering(
        &self,
        inputs: Example,
        steering: &Steering,
    ) -> Result<Prediction> {
        let lm = self.asking()?;
        let (pairs, lifted) = rendered_inputs(&inputs);
        let mut steering = steering.clone();
        // Upstream assigns `config["prediction"]` after merging the caller's `config=`, so an input
        // of that shape wins over one steered directly. Only over one: a call that steers a
        // predicted output and passes no such input keeps what it steered.
        if lifted.is_some() {
            steering.predicted_output = lifted;
        }
        let validated = self
            .call_with_inputs(lm.as_ref(), &pairs, &steering)
            .await?;
        Ok(
            Prediction::new(prediction_example(&validated.value), validated.raw)
                .with_usage(validated.usage),
        )
    }
}

impl<S: Send + Sync> Module for Predict<S> {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let span = crate::observe::module_shown("Predict", &inputs);
            crate::observe::watching(
                span,
                self.forward_with_steering(inputs, &Steering::default()),
            )
            .await
        })
    }

    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        vec![NamedPredictor {
            name: "self".to_owned(),
            signature: &mut self.signature,
            demos: &mut self.demos,
            config: &mut self.config,
            hint: &mut self.hint,
        }]
    }

    /// One call, so one step, under the name [`Self::named_predictors`] answers with.
    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move {
            let prediction = self.forward(inputs.clone()).await?;
            trace.push(TraceStep {
                predictor: "self".to_owned(),
                inputs,
                outputs: prediction.example.clone(),
            });
            Ok(prediction)
        })
    }
}

/// The parsed reply as an [`Example`], so a metric can read its fields by name.
fn prediction_example(value: &Value) -> Example {
    match value.as_object() {
        Some(fields) => Example::new(
            fields
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        ),
        None => Example::default(),
    }
}

crate::asks_with_a_prediction!(Predict);
#[cfg(test)]
mod one_api {
    use crate::predict::scripted::{RoomTask, Scripted};
    use crate::{Predict, call};

    /// One constructor and one call across both ways of declaring a task, which is what dspy
    /// gives and what a caller should not have to think about.
    ///
    /// The answer differs on purpose: a task declared by its field names has only fields to
    /// hand back, while a derived one hands back its own outputs, so `.color` keeps meaning the
    /// field rather than a lookup that might miss.
    #[tokio::test]
    async fn both_signature_forms_are_built_and_asked_the_same_way() {
        let reply = "[[ ## color ## ]]\nred\n\n[[ ## why ## ]]\ncalm\n\n[[ ## completed ## ]]";

        let _configured =
            crate::lm::global::install_for_test(std::sync::Arc::new(Scripted::new(&[
                reply, reply,
            ])));

        let declared = Predict!("request -> color, why");
        let derived = Predict!(RoomTask);

        let from_declared = call!(declared, request = "something calm")
            .await
            .expect("asks");
        let from_derived = call!(derived, request = "something calm")
            .await
            .expect("asks");

        assert_eq!(from_declared.get("color").unwrap(), "red");
        assert_eq!(from_derived.color, "red");
    }
}

#[cfg(test)]
mod per_call_model {
    use std::sync::Arc;

    use super::*;
    use crate::example;
    use crate::lm::dummy::DummyLM;
    use crate::module::Module;
    use crate::predict::scripted::Scripted;
    use crate::{Predict, input};

    /// A module asks its own model when it has one, and the configured one otherwise.
    ///
    /// This is the seam an optimizer needs: `BestOfN` runs a module several times against models
    /// that differ only in their config, and a bootstrap round after the first needs one that
    /// will not answer from a cache. Neither can reach a process-wide default to arrange it.
    #[tokio::test]
    async fn a_module_asks_its_own_model_over_the_configured_one() {
        let _configured = crate::lm::global::install_for_test(Arc::new(DummyLM::new([
            example! { answer: "from the default" },
        ])));

        let asking = Predict!("question -> answer");
        assert!(asking.lm().is_none(), "it defers until given one");
        let default = asking
            .forward(input! { question: "q" })
            .await
            .expect("asks");
        assert_eq!(default.get("answer").unwrap(), "from the default");

        let mine = Predict!("question -> answer").with_lm(Arc::new(DummyLM::new([
            example! { answer: "from its own" },
        ])));
        assert!(mine.lm().is_some());
        let own = mine.forward(input! { question: "q" }).await.expect("asks");
        assert_eq!(own.get("answer").unwrap(), "from its own");
    }

    /// The override survives being told which task it asks, so a typed module can carry one too.
    #[test]
    fn the_override_survives_being_given_a_task() {
        let _ = Scripted::new(&[]);
        let carried = Predict::from_signature("q -> a".parse().expect("parses"))
            .with_lm(Arc::new(DummyLM::new([])));
        assert!(carried.into_task::<()>().lm().is_some());
    }

    /// The gap this closes: config chosen on a module had no way through `forward` to the
    /// request the model is handed, so `max_rounds` could re-ask but nothing could make the
    /// answer differ. Asserting it arrives is the whole point — a module that quietly dropped
    /// it would still return a reply and still look like it worked.
    #[tokio::test]
    async fn the_sampling_a_module_carries_reaches_the_request_the_model_is_handed() {
        let lm = Arc::new(DummyLM::new([
            example! { answer: "first" },
            example! { answer: "second" },
        ]));

        let defaults = Predict!("question -> answer").with_lm(lm.clone());
        defaults
            .forward(input! { question: "q" })
            .await
            .expect("asks");

        let varied = Predict!("question -> answer")
            .with_lm(lm.clone())
            .with_config(Sampling {
                temperature: Some(1.0),
                max_tokens: Some(64),
                ..Sampling::default()
            });
        varied
            .forward(input! { question: "q" })
            .await
            .expect("asks");

        let asked = lm.asked();
        assert_eq!(asked[0].config, Sampling::default());
        assert_eq!(asked[1].config.temperature, Some(1.0));
        assert_eq!(asked[1].config.max_tokens, Some(64));
    }

    /// Sampling travels with the module the same way the model override does.
    #[test]
    fn the_sampling_survives_being_given_a_task() {
        let carried =
            Predict::from_signature("q -> a".parse().expect("parses")).with_config(Sampling {
                temperature: Some(0.5),
                ..Sampling::default()
            });
        assert_eq!(carried.into_task::<()>().config().temperature, Some(0.5));
    }
}
