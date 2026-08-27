//! How a configured [`LM`] reaches a provider.
//!
//! The [`ChatModel`] implementation, the streaming factory beside it, and the `match` that turns a
//! model reference into the wire that serves it. Separated from what an `LM` *is* because nothing
//! outside reaches any of it except through the trait: a caller configures a model in one place and
//! calls it in another, and the two have different reasons to change.
//!
//! The dispatch itself is inherent rather than a design choice. dspy does the same thing in
//! `infer_provider`, and litellm does it inside its own dispatch — *something* has to map a model
//! string to a wire format. The trait is what makes the built-ins interchangeable with a caller's
//! own provider.

use std::future::Future;

use anyhow::Result;
use futures_util::Stream;

use super::{
    Capabilities, ChatModel, LM, OpenAiWire, Provider, anthropic, api, cache, global, ollama,
    openai, retry, usage,
};

impl ChatModel for LM {
    /// What this model states about itself in a saved program. dspy's `LM.dump_state`, which
    /// overrides `BaseLM`'s to add the three finetuning keys and to rename a reasoning model's
    /// token cap back. See [`saved::dump`](crate::lm::saved::dump).
    fn dump_state(&self) -> Option<serde_json::Map<String, serde_json::Value>> {
        Some(crate::lm::saved::dump(self))
    }

    /// The settings this model fills a request's gaps from — the `LmConfig` `with_defaults`
    /// applies, stated so a module can read it before the request is built.
    fn defaults(&self) -> api::LmConfig {
        self.config.clone()
    }

    async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
        self.answer(&self.with_defaults(request)).await
    }

    /// The provider's own stream, rather than the default's one-delta stand-in.
    ///
    /// On the configured client, as [`forward`](ChatModel::forward) is — dspy's do not name one
    /// either. The defaults are applied here too, so a streamed call asks exactly what the same
    /// call would ask unstreamed.
    fn forward_stream<'a>(
        &'a self,
        request: &'a api::LmRequest,
    ) -> impl futures_util::Stream<Item = Result<api::LmStreamEvent>> + Send + 'a {
        // Everything the provider stream needs is built here and owned by it: `RequestBuilder`
        // carries its own clone of the client, so nothing below borrows either of these.
        self.forward_stream_on(&global::client(), &self.with_defaults(request))
    }

    /// The watchers this model carries, told about its calls beside the process-wide ones — dspy's
    /// `LM(model, callbacks=[…])`. See [`LM::callbacks`].
    fn callbacks(&self) -> &[std::sync::Arc<dyn crate::Callback>] {
        &self.callbacks
    }

    /// What this model's provider will honour natively: what the caller stated, else the registry
    /// dspy consults, else — for an ollama model the registry does not list — the server itself.
    fn capabilities(&self) -> impl Future<Output = Capabilities> + Send {
        async move {
            if let Some(stated) = self.capabilities {
                return stated;
            }
            match self.model.provider {
                // The `/api/generate` wire cannot carry a native tool call — litellm renders tools
                // into the prompt on this route rather than sending them — so this configured LM
                // has no native feature to offer, whatever the registry says about the model.
                Provider::Ollama => Capabilities::default(),
                // The chat route can. litellm keeps ollama's rows under an `ollama/` key and falls
                // through to the server for a model it does not list; so does this.
                Provider::OllamaChat => Capabilities::listed(&format!("ollama/{}", self.model.id))
                    .unwrap_or(
                        ollama::capabilities(
                            &global::client(),
                            &self.ollama_host,
                            self.ollama_api_key.as_deref(),
                            self.timeout,
                            &self.model.id,
                        )
                        .await,
                    ),
                _ => Capabilities::listed(&self.model.reference()).unwrap_or_default(),
            }
        }
    }

    /// dspy `Citations.adapt_to_native_lm_feature`: `lm.model.startswith("anthropic/")`. The prefix
    /// names the provider, and `ModelRef` has already parsed it into one.
    fn native_citations_usable(&self) -> bool {
        matches!(self.model.provider, Provider::Anthropic)
    }

    /// dspy `Reasoning.adapt_to_native_lm_feature`'s caveat: `"gpt-5" in lm.model and lm.model_type
    /// == "chat"`. The chat-completions route is [`OpenAiWire::Chat`] for a compatible endpoint and
    /// the only route OpenRouter speaks; the Responses API, and every non-OpenAI provider, is
    /// unaffected.
    fn native_reasoning_usable(&self) -> bool {
        let on_chat_completions = match self.model.provider {
            Provider::OpenAiCompatible => matches!(self.openai.wire, OpenAiWire::Chat),
            Provider::OpenRouter => true,
            _ => false,
        };
        !(on_chat_completions && self.model.id.contains("gpt-5"))
    }
}

impl LM {
    /// The typed streaming boundary — dspy's stream of `LMStreamEvent`s.
    ///
    /// An OpenAI-shaped service streams real Server-Sent Events; a provider that does not stream
    /// answers once, and its reply is handed back as the events it would have arrived as, so a
    /// caller consuming a stream need not know which kind it asked. Streaming bypasses the
    /// response cache, as upstream's does — a stream is not a value to store and replay.
    ///
    /// The boxed stream is the same factory the non-streaming dispatch is: the arms return
    /// different stream types, and a `dyn Stream` is what makes them one return type.
    pub fn forward_stream_on(
        &self,
        http: &reqwest::Client,
        request: &api::LmRequest,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<api::LmStreamEvent>> + Send + 'static>> {
        match self.model.provider {
            // `Endpoint::stream` already boxes — it picks the chat or Responses wire, whose stream
            // types differ — so these arms hand its stream straight back rather than box it again.
            Provider::OpenAiCompatible => {
                openai::Endpoint::configured(&self.model.id, &self.openai, self.timeout)
                    .stream(http, request)
            }
            Provider::OpenRouter => openai::Endpoint::openrouter(
                &self.model.id,
                self.openrouter_api_key.as_deref(),
                self.timeout,
            )
            .stream(http, request),
            Provider::Anthropic => Box::pin(anthropic::stream(
                http,
                &self.model.id,
                self.anthropic_api_key.as_deref(),
                self.timeout,
                request,
            )),
            Provider::Ollama => Box::pin(ollama::generate_stream(
                http,
                &self.model.id,
                &self.ollama_host,
                self.ollama_api_key.as_deref(),
                self.timeout,
                request,
            )),
            Provider::OllamaChat => Box::pin(ollama::chat_stream(
                http,
                &self.model.id,
                &self.ollama_host,
                self.ollama_api_key.as_deref(),
                self.timeout,
                request,
            )),
        }
    }

    /// The reply, from the cache or from the provider — everything [`ChatModel::forward`] watches.
    async fn answer(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
        if !self.cache {
            let answered = self.ask_provider_retrying(request).await?;
            usage::record(&self.model.id, answered.spend());
            return Ok(answered);
        }
        let key = request.cache_key(&self.model.id);
        if let Some(replayed) = cache::shared().replay(&key) {
            return Ok(replayed);
        }
        let answered = self.ask_provider_retrying(request).await?;
        usage::record(&self.model.id, answered.spend());
        cache::shared().keep(key, answered.clone());
        Ok(answered)
    }

    /// The call, asked again while it fails the way dspy retries — see [`retry`].
    ///
    /// Inside the cache and not around it, which is where upstream puts it: `request_cache` wraps
    /// the function that carries `num_retries`, so a replayed answer is never retried and an answer
    /// a retry finally won is kept.
    async fn ask_provider_retrying(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
        retry::asking(self.retry, || self.ask_provider(request)).await
    }

    /// The call itself, on whichever wire format this model's provider speaks.
    async fn ask_provider(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
        // Every arm resolves the model reference and this LM's credentials into a provider — each
        // its own [`ChatModel`] — then makes the one uniform call. The match is the factory that
        // maps a model string to its provider, which is inherent: dspy does the same in
        // `infer_provider`, and litellm does it inside its own dispatch. The trait is what makes
        // the four interchangeable, and a caller's own provider indistinguishable from these.
        match self.model.provider {
            Provider::Anthropic => {
                anthropic::Anthropic {
                    model: &self.model.id,
                    api_key: self.anthropic_api_key.as_deref(),
                    timeout: self.timeout,
                }
                .forward(request)
                .await
            }
            Provider::OpenRouter => {
                openai::Endpoint::openrouter(
                    &self.model.id,
                    self.openrouter_api_key.as_deref(),
                    self.timeout,
                )
                .forward(request)
                .await
            }
            Provider::OpenAiCompatible => {
                openai::Endpoint::configured(&self.model.id, &self.openai, self.timeout)
                    .forward(request)
                    .await
            }
            Provider::Ollama => {
                ollama::Generate {
                    api_key: self.ollama_api_key.as_deref(),
                    model: &self.model.id,
                    host: &self.ollama_host,
                    timeout: self.timeout,
                }
                .forward(request)
                .await
            }
            Provider::OllamaChat => {
                ollama::Chat {
                    api_key: self.ollama_api_key.as_deref(),
                    model: &self.model.id,
                    host: &self.ollama_host,
                    timeout: self.timeout,
                }
                .forward(request)
                .await
            }
        }
    }
}
