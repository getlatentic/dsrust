//! dspy `clients/embedding.py::Embedder`: text to vectors, from a hosted model or a function.
//!
//! Upstream wraps litellm's `embedding` call or any callable, batches the inputs, and caches each
//! batch's answer. The same shape here: a model named `provider/id` reaches the provider's
//! OpenAI-compatible `/embeddings` endpoint, a function is called with each batch, and a batch
//! already answered is not asked again while `caching` is on.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Result, bail};
use serde_json::{Map, Value};

use super::openai::{OpenAiConfig, embeddings};
use super::routing::{ModelRef, Provider};

/// A function standing in for a model: the batch of texts and the call's keyword arguments, as
/// upstream calls `model(batch_inputs, **kwargs)`.
pub type EmbeddingFn =
    dyn Fn(&[String], &Map<String, Value>) -> Result<Vec<Vec<f32>>> + Send + Sync;

/// What embeds: a hosted model by name, or a function.
#[derive(Clone)]
pub enum EmbedderModel {
    /// `provider/model-id`, or a bare OpenAI model id as litellm takes one.
    Named(String),
    Callable(Arc<EmbeddingFn>),
}

impl fmt::Debug for EmbedderModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedderModel::Named(model) => write!(f, "Named({model:?})"),
            EmbedderModel::Callable(_) => write!(f, "Callable"),
        }
    }
}

/// The options one call may override: upstream's `batch_size=`, `caching=` and `**kwargs`.
#[derive(Debug, Clone, Default)]
pub struct EmbedCall {
    pub batch_size: Option<usize>,
    pub caching: Option<bool>,
    pub kwargs: Map<String, Value>,
}

pub struct Embedder {
    model: EmbedderModel,
    /// dspy's default of 200 inputs per request.
    batch_size: usize,
    caching: bool,
    default_kwargs: Map<String, Value>,
    http: reqwest::Client,
    timeout: Duration,
    /// Where an OpenAI-compatible model is reached; unset, the environment's endpoint and key.
    openai: Option<OpenAiConfig>,
    cache: Mutex<HashMap<String, Vec<Vec<f32>>>>,
}

impl fmt::Debug for Embedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Embedder")
            .field("model", &self.model)
            .field("batch_size", &self.batch_size)
            .field("caching", &self.caching)
            .finish_non_exhaustive()
    }
}

impl Embedder {
    /// `dspy.Embedder("openai/text-embedding-3-small")`.
    pub fn new(model: impl Into<String>) -> Self {
        Self::from_model(EmbedderModel::Named(model.into()))
    }

    /// `dspy.Embedder(my_function)`.
    pub fn callable(
        function: impl Fn(&[String], &Map<String, Value>) -> Result<Vec<Vec<f32>>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self::from_model(EmbedderModel::Callable(Arc::new(function)))
    }

    fn from_model(model: EmbedderModel) -> Self {
        Self {
            model,
            batch_size: 200,
            caching: true,
            default_kwargs: Map::new(),
            http: reqwest::Client::new(),
            timeout: Duration::from_secs(60),
            openai: None,
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn caching(mut self, caching: bool) -> Self {
        self.caching = caching;
        self
    }

    /// Upstream's `**kwargs`: sent with every request unless a call overrides a key.
    pub fn kwargs(mut self, kwargs: Map<String, Value>) -> Self {
        self.default_kwargs = kwargs;
        self
    }

    pub fn http(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The endpoint and key an OpenAI-compatible model is reached at, in place of the
    /// environment's — upstream's `api_base=` and `api_key=` keyword arguments.
    pub fn openai_config(mut self, config: OpenAiConfig) -> Self {
        self.openai = Some(config);
        self
    }

    pub fn model(&self) -> &EmbedderModel {
        &self.model
    }

    /// `embedder(inputs)`: one vector per input.
    pub async fn call(&self, inputs: &[impl AsRef<str>]) -> Result<Vec<Vec<f32>>> {
        self.call_with(inputs, EmbedCall::default()).await
    }

    /// `embedder("one input")`: the one vector.
    pub async fn call_one(&self, input: &str) -> Result<Vec<f32>> {
        let mut vectors = self.call(&[input]).await?;
        Ok(vectors.pop().unwrap_or_default())
    }

    /// `embedder(inputs, batch_size=..., caching=..., **kwargs)`.
    pub async fn call_with(
        &self,
        inputs: &[impl AsRef<str>],
        call: EmbedCall,
    ) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<String> = inputs.iter().map(|text| text.as_ref().to_owned()).collect();
        let batch_size = call.batch_size.unwrap_or(self.batch_size).max(1);
        let caching = call.caching.unwrap_or(self.caching);
        let mut kwargs = self.default_kwargs.clone();
        for (key, value) in call.kwargs {
            kwargs.insert(key, value);
        }
        let mut vectors = Vec::with_capacity(inputs.len());
        for batch in inputs.chunks(batch_size) {
            vectors.extend(self.batch(batch, caching, &kwargs).await?);
        }
        Ok(vectors)
    }

    async fn batch(
        &self,
        batch: &[String],
        caching: bool,
        kwargs: &Map<String, Value>,
    ) -> Result<Vec<Vec<f32>>> {
        let key = self.cache_key(batch, kwargs);
        let cached = caching
            .then(|| self.cache.lock().expect("not poisoned").get(&key).cloned())
            .flatten();
        if let Some(answered) = cached {
            return Ok(answered);
        }
        let answered = match &self.model {
            EmbedderModel::Callable(function) => function(batch, kwargs)?,
            EmbedderModel::Named(model) => self.hosted(model, batch, kwargs).await?,
        };
        if caching {
            self.cache
                .lock()
                .expect("not poisoned")
                .insert(key, answered.clone());
        }
        Ok(answered)
    }

    /// The model, the batch and the keyword arguments — less the reach arguments upstream's cache
    /// ignores: `api_key`, `api_base`, `base_url`.
    fn cache_key(&self, batch: &[String], kwargs: &Map<String, Value>) -> String {
        let model = match &self.model {
            EmbedderModel::Named(model) => Value::String(model.clone()),
            EmbedderModel::Callable(function) => {
                Value::String(format!("callable@{:p}", Arc::as_ptr(function)))
            }
        };
        let kept: Map<String, Value> = kwargs
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "api_key" | "api_base" | "base_url"))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        serde_json::json!({ "model": model, "input": batch, "kwargs": kept }).to_string()
    }

    async fn hosted(
        &self,
        model: &str,
        batch: &[String],
        kwargs: &Map<String, Value>,
    ) -> Result<Vec<Vec<f32>>> {
        // litellm takes a bare model id as OpenAI's, which is how dspy's own tests name one.
        let reference = match model.contains('/') {
            true => ModelRef::parse(model)?,
            false => ModelRef::parse(&format!("openai/{model}"))?,
        };
        match reference.provider {
            Provider::OpenAiCompatible => {
                let config = match &self.openai {
                    Some(config) => config.clone(),
                    None => OpenAiConfig::from_env(),
                };
                embeddings::embed(
                    &self.http,
                    &config,
                    &reference.id,
                    batch,
                    kwargs,
                    self.timeout,
                )
                .await
            }
            other => bail!(
                "`{model}` names a {other:?} model, and this crate reaches embeddings through the \
                 OpenAI-compatible endpoint only"
            ),
        }
    }
}
