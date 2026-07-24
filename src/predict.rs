use std::marker::PhantomData;

use anyhow::{Context, Result, anyhow};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::adapter::native_reasoning::{self, ReasoningEffort};
use crate::adapter::parse::FieldMismatch;
use crate::adapter::{
    Adapter, ChatAdapter, Extraction, Feedback, Input, ToolCall, ToolCalls, native_tools, turns_for,
};
use crate::example::{Example, Prediction};
use crate::lm::{Capabilities, DynChatModel, LmConfig, LmUsage, api, global};
use crate::module::{Module, NamedPredictor, TraceStep};
use crate::signature::{FieldKind, Signature, SignatureSpec};

/// dspy's per-call `config=`: the fields one ask steers that the module's own config does not carry.
/// A reasoning budget, and a tool the provider must call — ReActV2's forced submit sets both, to
/// pin the last ask to `submit` and turn native reasoning off for it.
#[derive(Debug, Clone, Default)]
pub struct Steering {
    pub reasoning_effort: ReasoningEffort,
    /// A tool the provider must call, sent only under native function calling, since dspy drops
    /// `tool_choice` for an adapter that renders the tools instead.
    pub forced_tool: Option<String>,
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
mod chain_of_thought;
mod completions;
mod derived;
mod hint;
mod multi_chain_comparison;
mod parallel;
pub mod refine;
pub use aggregation::{Normalize, majority, normalize_text};
pub use best_of_n::BestOfN;
pub use chain_of_thought::{ChainOfThought, TypedChainOfThought};
pub use refine::Refine;
pub use derived::TypedPredict;
use derived::typed;
pub use multi_chain_comparison::MultiChainComparison;
pub use parallel::{Answered, Parallel};

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
    config: LmConfig,
    /// What an earlier attempt was told to do differently. See [`NamedPredictor::hint`].
    hint: Option<String>,
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
    /// The same module, told which task it asks. The signature and demos are unchanged; only
    /// what a call answers with follows from the type.
    pub(crate) fn into_task<T>(self) -> Predict<T> {
        Predict {
            spec: PhantomData,
            signature: self.signature,
            adapter: self.adapter,
            demos: self.demos,
            lm: self.lm,
            config: self.config,
            hint: self.hint,
        }
    }

    /// Ask this model rather than the configured one. dspy's `set_lm`.
    pub fn with_lm(mut self, lm: std::sync::Arc<dyn DynChatModel>) -> Self {
        self.lm = Some(lm);
        self
    }

    /// Ask for the reply to be sampled this way rather than at the provider's defaults.
    ///
    /// dspy reaches the same setting through `lm.copy(temperature=...)`, which needs a model to
    /// copy; this is per call, so a module that defers to the configured model can still vary
    /// how it is asked.
    pub fn with_config(mut self, config: LmConfig) -> Self {
        self.config = config;
        self
    }

    /// The config this module asks for. An optimizer reads it to vary one field and leave
    /// the rest alone.
    pub fn config(&self) -> &LmConfig {
        &self.config
    }

    /// dspy's `get_lm`: the model this module asks, or nothing if it defers to the configured
    /// one. An optimizer reads it to copy the settings it is about to vary.
    pub fn lm(&self) -> Option<&std::sync::Arc<dyn DynChatModel>> {
        self.lm.as_ref()
    }

    /// The model and client one call should use: this module's own, or the configured default.
    ///
    /// A module with its own model needs only a client, not the global model behind it, so it
    /// runs where none is configured — which is what `BestOfN` and `Refine` rely on when they
    /// hand a predictor a scripted model.
    fn asking(&self) -> Result<(reqwest::Client, std::sync::Arc<dyn DynChatModel>)> {
        match &self.lm {
            Some(lm) => Ok((global::client(), lm.clone())),
            None => global::current(),
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
}

impl Predict<Dynamic> {
    /// A module for a signature held as field names. `predict!("question -> answer")` is the
    /// spelling to reach for; this is what it expands to.
    pub fn from_signature(signature: Signature) -> Self {
        Self {
            spec: PhantomData,
            lm: None,
            config: LmConfig::default(),
            hint: None,
            signature,
            adapter: Box::new(ChatAdapter::default()),
            demos: Vec::new(),
        }
    }

    /// dspy `Predict("email -> sentiment")`: declare the task by naming its fields.
    ///
    /// The shortest way to a working module. A field with no type is a string, which is what
    /// makes the untyped spelling useful for a first pass; `Predict::task` takes a derived
    /// signature when the types matter.
    pub fn parse(signature: &str) -> Result<Self> {
        Ok(Self::from_signature(signature.parse()?))
    }

    /// The module for a derived signature, which is `Predict::<Task>::new()` reached from the
    /// untyped name.
    pub fn task<S: SignatureSpec + Send + Sync>() -> Predict<S> {
        Predict::<S>::new()
    }

    /// Ask through the globally configured LM; see [`crate::lm::configure`].
    pub async fn call(&self, input: &str) -> Result<Value> {
        let (http, lm) = self.asking()?;
        self.call_with(&http, lm.as_ref(), input).await
    }

    /// Ask through an explicit client and model: the per-call override, and the seam tests
    /// script with a canned [`ChatModel`](crate::lm::ChatModel).
    pub async fn call_with(
        &self,
        http: &reqwest::Client,
        lm: &dyn DynChatModel,
        input: &str,
    ) -> Result<Value> {
        let name = self
            .signature
            .inputs
            .first()
            .map_or("request", |f| f.name.as_str());
        Ok(self
            .call_with_inputs(
                http,
                lm,
                &[Input::new(name, Value::String(input.to_owned()))],
                &Steering::default(),
            )
            .await?
            .value)
    }
}

/// dspy sets `parallel_tool_calls` only when the adapter states one and the request has not
/// already asked, so an explicit `false` from the caller is never overwritten by the default.
///
/// The normalized config spells it `tool_choice.parallel`, which is where upstream's own
/// `LMToolChoice.from_value` puts a `parallel_tool_calls` kwarg.
fn ask_for_parallel_calls(request: &mut api::LmRequest, parallel: Option<bool>) {
    let Some(parallel) = parallel else {
        return;
    };
    let choice = request.config.tool_choice.get_or_insert_with(Default::default);
    if choice.parallel.is_none() {
        choice.parallel = Some(parallel);
    }
}

/// dspy `_provider_tool_call_to_tool_call_dict`: a provider's native call as a [`ToolCall`].
///
/// The reply parser already structured the arguments and settled the id from `id`/`call_id`, so a
/// part carries what upstream reads out of a litellm tool call — no repair or renaming is left to
/// do here. A part that is not a tool call is not one of these and is dropped.
fn as_tool_call(part: &api::LmPart) -> Option<ToolCall> {
    match part {
        api::LmPart::ToolCall { id, name, args, .. } => {
            Some(ToolCall { id: id.clone(), name: name.clone(), args: args.clone() })
        }
        _ => None,
    }
}

/// dspy's forced `tool_choice`: pin the provider to one tool by name. The normalized config states
/// it as the sole allowed tool under `required`, which every provider lowers to
/// `{"type":"function","function":{"name": tool}}`.
fn force_tool(request: &mut api::LmRequest, tool: &str) {
    let choice = request.config.tool_choice.get_or_insert_with(Default::default);
    choice.mode = api::ToolChoiceMode::Required;
    choice.allowed = Some(vec![tool.to_owned()]);
}

/// dspy `Reasoning.parse_lm_response` reads `reasoning_content` off a reply; here that is the
/// thinking part the providers already lift out of it.
fn thinking_text(output: &api::LmOutput) -> Option<String> {
    output.parts.iter().find_map(|part| match part {
        api::LmPart::Thinking { text, .. } => Some(text.clone()),
        _ => None,
    })
}

impl<S> Predict<S> {
    /// One attempt: render through the adapter, ask the model, hand back the raw reply.
    /// dspy's `Adapter.__call__` in the module rather than the adapter, because a Rust trait
    /// carrying an async model call could not also be object-safe.
    async fn ask_through(
        &self,
        adapter: &dyn Adapter,
        http: &reqwest::Client,
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
            true => lm.capabilities_dyn(http).await,
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
        let asked = reasoning.as_ref().map_or(asked, |plan| plan.signature.clone());
        let schema = asked.schema();
        let (system, opening) = adapter.format(&asked, &self.demos, &hinted)?;
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
            request.config.reasoning =
                Some(api::LmReasoningConfig { effort: Some(reasoning.effort), ..Default::default() });
        }
        // dspy drops `tool_choice` unless the adapter asks natively, so a forced tool is sent only
        // then; a rendered-tools exchange has no provider-side choice to steer.
        if asking.enabled && let Some(tool) = &steering.forced_tool {
            force_tool(&mut request, tool);
        }
        let response = lm.forward_dyn(http, &request).await?;
        Ok(Reply { response, rendered: asked })
    }

    async fn ask(
        &self,
        http: &reqwest::Client,
        lm: &dyn DynChatModel,
        inputs: &[Input<'_>],
        feedback: Option<&Feedback>,
        steering: &Steering,
    ) -> Result<Reply> {
        self.ask_through(self.adapter.as_ref(), http, lm, inputs, feedback, steering)
            .await
    }

    /// dspy `_call_postprocess`: the value of a reply whose fields left the render for a native
    /// feature.
    ///
    /// A field removed by native function calling or native reasoning is absent from what the reply
    /// spoke, so it is read from its own channel instead — the provider's tool calls fill the
    /// tool-call output, the model's own thinking fills the reasoning output — while whatever content
    /// did come back is parsed against what remained. Returns `None` when the render kept every
    /// output field, which is the marker text path the caller handles itself.
    fn native_value(&self, reply: &Reply) -> Result<Option<Value>> {
        let removed: Vec<&crate::signature::OutField> = self
            .signature
            .outputs
            .iter()
            .filter(|field| !reply.rendered.outputs.iter().any(|kept| kept.name == field.name))
            .collect();
        if removed.is_empty() {
            return Ok(None);
        }

        let text = reply.response.first_text();
        let mut value = match !text.is_empty() && !reply.rendered.outputs.is_empty() {
            true => self
                .adapter
                .parse(&reply.rendered, &text)
                .unwrap_or_else(|_| Value::Object(Map::new())),
            false => Value::Object(Map::new()),
        };
        let object = value.as_object_mut().ok_or_else(|| anyhow!("a parsed reply is an object"))?;
        for field in &self.signature.outputs {
            object.entry(field.name.clone()).or_insert(Value::Null);
        }

        // The provider's own tool calls fill the tool-call output field.
        if let Some(tool_field) = native_tools::tool_call_output_field(&self.signature) {
            let calls: Vec<ToolCall> = reply
                .response
                .outputs
                .first()
                .into_iter()
                .flat_map(api::LmOutput::tool_calls)
                .filter_map(as_tool_call)
                .collect();
            if !calls.is_empty() {
                object.insert(tool_field.to_owned(), ToolCalls::new(calls).to_value_with_ids());
            }
        }

        // dspy `Reasoning.parse_lm_response`: a reasoning model's thinking fills the reasoning
        // output that left the render for it.
        if let Some(thinking) = reply.response.outputs.first().and_then(thinking_text) {
            for field in &removed {
                if matches!(field.kind, FieldKind::Reasoning) {
                    object.insert(field.name.clone(), Value::String(thinking.clone()));
                }
            }
        }
        Ok(Some(value))
    }

    async fn call_with_inputs(
        &self,
        http: &reqwest::Client,
        lm: &dyn DynChatModel,
        inputs: &[Input<'_>],
        steering: &Steering,
    ) -> Result<Validated> {
        // dspy's ChatAdapter catches a parse failure and re-asks the whole exchange through
        // the JSON adapter; `use_json_adapter_fallback` turns that off. The adapter states the
        // policy, this module carries it out, because only the module can call the model.
        let answered = self.ask(http, lm, inputs, None, steering).await?;
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
            return self.extract(http, extraction, raw, usage).await;
        }
        let (raw, mut value, usage) = match self.adapter.parse(&self.signature, &raw) {
            Ok(value) => (raw, value, usage),
            // A reply that spoke the format but left a field out is the case the feedback ask
            // exists for, so it carries on with whatever the reply did say and lets `ensure`
            // name the gap. Upstream rejects it at parse because it has no such second ask.
            Err(error) if error.is::<FieldMismatch>() => {
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
                        .ask_through(fallback.as_ref(), http, lm, inputs, None, steering)
                        .await?;
                    let answered_text = answered.response.first_text();
                    let value = fallback.parse(&self.signature, &answered_text)?;
                    let merged = LmUsage::merge(usage, answered.response.spend());
                    (answered_text, value, merged)
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
            Ok(()) => Ok(Validated { raw, value, usage }),
            Err(error) => {
                tracing::warn!(%error, "retrying with feedback");
                let feedback = Feedback {
                    previous: raw,
                    error: error.to_string(),
                };
                let (raw, value, retried) =
                    self.feedback_ask(http, lm, inputs, &feedback, steering).await?;
                Ok(Validated {
                    raw,
                    value,
                    usage: LmUsage::merge(usage, retried),
                })
            }
        }
    }

    /// The second ask of a two-step adapter: hand the first reply to the extraction model and
    /// read the signature's fields out of what it answers.
    ///
    /// The extraction speaks its own adapter over its own signature — `text` in, the task's
    /// outputs out — so nothing here knows it is reading prose rather than a fresh request.
    async fn extract(
        &self,
        http: &reqwest::Client,
        extraction: Extraction<'_>,
        raw: String,
        asking: Option<LmUsage>,
    ) -> Result<Validated> {
        let text = [Input::new("text", Value::String(raw.clone()))];
        let (system, turns) = extraction
            .adapter
            .format(&extraction.signature, &[], &text)?;
        let schema = extraction.signature.schema();
        let mode = extraction.adapter.output_mode(&schema);
        // Left at the provider's defaults rather than given the module's config: this call
        // rewrites prose the model already produced into fields, so a temperature chosen to
        // vary the *answer* would only vary the transcription of one.
        let request = api::interop::raise_request(&system, &turns, mode, &LmConfig::default());
        let extracted = extraction
            .model
            .forward_dyn(http, &request)
            .await
            .context("the extraction model did not answer")?;
        let extracted_text = extracted.first_text();
        let mut value = extraction
            .adapter
            .parse(&extraction.signature, &extracted_text)
            // dspy names the *first* reply here, not the extraction's. That is the one a
            // caller can act on: the extraction failing usually means the prose never carried
            // the fields, and the prose is what they would go and look at.
            .with_context(|| {
                format!("Failed to parse response from the original completion: {raw}")
            })?;
        self.signature.coerce(&mut value)?;
        self.signature.ensure(&value)?;
        Ok(Validated {
            usage: LmUsage::merge(asking, extracted.spend()),
            raw: extracted_text,
            value,
        })
    }

    /// One more ask on the same adapter carrying the rejected reply and its error; every
    /// failure past this point is final.
    async fn feedback_ask(
        &self,
        http: &reqwest::Client,
        lm: &dyn DynChatModel,
        inputs: &[Input<'_>],
        feedback: &Feedback,
        steering: &Steering,
    ) -> Result<(String, Value, Option<LmUsage>)> {
        let answered = self.ask(http, lm, inputs, Some(feedback), steering).await?;
        let answered_text = answered.response.first_text();
        let mut value = self.adapter.parse(&self.signature, &answered_text)?;
        self.signature.coerce(&mut value)?;
        self.signature.ensure(&value)?;
        let usage = answered.response.spend();
        Ok((answered_text, value, usage))
    }
}

impl Predict<Dynamic> {
    /// The validated reply as a caller-owned struct instead of loose JSON.
    pub async fn call_typed<T: DeserializeOwned>(&self, input: &str) -> Result<T> {
        typed(self.call(input).await?)
    }

    /// [`Self::call_typed`] through an explicit client and model.
    pub async fn call_typed_with<T: DeserializeOwned>(
        &self,
        http: &reqwest::Client,
        lm: &dyn DynChatModel,
        input: &str,
    ) -> Result<T> {
        typed(self.call_with(http, lm, input).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::JsonAdapter;
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
        async fn forward(
            &self,
            _http: &reqwest::Client,
            _request: &api::LmRequest,
        ) -> Result<api::LmResponse> {
            Ok(api::LmResponse::completions(self.0.iter().map(|reply| reply.to_string())))
        }
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
            .with_config(LmConfig {
                completions: Some(3),
                ..LmConfig::default()
            });

        let candidates = predict
            .forward_completions(input! { request: "pick a colour" })
            .await
            .expect("candidates");

        assert_eq!(candidates.len(), 3, "every completion is read");
        let colors: Vec<_> = candidates.iter().filter_map(|c| c.get("color").cloned()).collect();
        assert_eq!(colors, [json!("red"), json!("blue"), json!("green")]);
    }

    /// `Refine` writes advice onto a predictor between attempts, and the model has to actually
    /// see it — as one more input field, which is what upstream appends.
    #[tokio::test]
    async fn a_hint_reaches_the_prompt_as_one_more_input_field() {
        let lm = Scripted::new(&[MARKER_REPLY, MARKER_REPLY]);

        let mut plain = Predict::from_signature(signature());
        plain
            .call_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("valid reply");

        for predictor in plain.named_predictors() {
            *predictor.hint = Some("name a warm colour".to_owned());
        }
        plain
            .call_with(&reqwest::Client::new(), &lm, "draft it")
            .await
            .expect("valid reply");

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
        let value = Predict::from_signature(signature())
            .call_with(&reqwest::Client::new(), &lm, "draft it")
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
            .call_with(&reqwest::Client::new(), &lm, "draft it")
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
        assert!(
            predict
                .call_with(&reqwest::Client::new(), &lm, "draft it")
                .await
                .is_err()
        );
        assert_eq!(
            lm.calls().len(),
            1,
            "no second ask when the fallback is off"
        );
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
            Predict::from_signature(signature())
                .call_with(&reqwest::Client::new(), &lm, "draft it")
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
        let wrong: Result<Wrong> = Predict::from_signature(signature())
            .call_typed_with(&reqwest::Client::new(), &lm, "draft it")
            .await;
        assert!(wrong.is_err());
    }

    #[tokio::test]
    async fn a_typed_task_renders_every_input_and_returns_the_outputs_struct() {
        let lm = Scripted::new(&[MARKER_REPLY]);
        let outputs = Predict::task::<RoomTask>()
            .call_inputs_with(&reqwest::Client::new(), &lm, &room_inputs())
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
            .call_inputs_with(&reqwest::Client::new(), &lm, &room_inputs())
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
            .call_inputs_with(&reqwest::Client::new(), &lm, &inputs)
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
                config: LmConfig::default(),
                hint: None,
                signature: typed_signature(),
                adapter: Box::new(JsonAdapter::default()),
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
        let (http, lm) = self.asking()?;
        let pairs: Vec<Input<'_>> =
            inputs.fields().map(|(name, value)| Input::new(name, value.clone())).collect();
        let validated = self.call_with_inputs(&http, lm.as_ref(), &pairs, steering).await?;
        Ok(Prediction::new(prediction_example(&validated.value), validated.raw)
            .with_usage(validated.usage))
    }
}

impl<S: Send + Sync> Module for Predict<S> {
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        Box::pin(async move { self.forward_with_steering(inputs, &Steering::default()).await })
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
    use crate::{call, predict};

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

        let declared = predict!("request -> color, why");
        let derived = predict!(RoomTask);

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
    use crate::{input, predict};

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

        let asking = predict!("question -> answer");
        assert!(asking.lm().is_none(), "it defers until given one");
        let default = asking
            .forward(input! { question: "q" })
            .await
            .expect("asks");
        assert_eq!(default.get("answer").unwrap(), "from the default");

        let mine = predict!("question -> answer").with_lm(Arc::new(DummyLM::new([
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

        let defaults = predict!("question -> answer").with_lm(lm.clone());
        defaults
            .forward(input! { question: "q" })
            .await
            .expect("asks");

        let varied = predict!("question -> answer")
            .with_lm(lm.clone())
            .with_config(LmConfig {
                temperature: Some(1.0),
                max_tokens: Some(64),
                ..LmConfig::default()
            });
        varied
            .forward(input! { question: "q" })
            .await
            .expect("asks");

        let asked = lm.asked();
        assert_eq!(asked[0].config, LmConfig::default());
        assert_eq!(asked[1].config.temperature, Some(1.0));
        assert_eq!(asked[1].config.max_tokens, Some(64));
    }

    /// LmConfig travels with the module the same way the model override does.
    #[test]
    fn the_sampling_survives_being_given_a_task() {
        let carried =
            Predict::from_signature("q -> a".parse().expect("parses")).with_config(LmConfig {
                temperature: Some(0.5),
                ..LmConfig::default()
            });
        assert_eq!(carried.into_task::<()>().config().temperature, Some(0.5));
    }
}
