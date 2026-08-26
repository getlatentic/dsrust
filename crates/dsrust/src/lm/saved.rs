//! The `lm` block of a saved program: what a model states about itself, and how one is rebuilt.
//!
//! dspy's `BaseLM.dump_state` and `LM.load_state`. A compiled program records the model each
//! predictor was pinned to, so loading it gets back the model the person who compiled it meant —
//! not whichever one happens to be configured in the process that opens the file. That is the
//! whole point of the block, and this crate used to write `null` there.
//!
//! **Key order is upstream's and is load-bearing.** `dspy.load` reads a dict and does not care,
//! but the claim on the README is that a file written here is the file dspy would have written,
//! and a diff is how anyone checks that. The order is nobody's rule — it falls out of the order
//! `LM.__init__` fills `self.kwargs`, then `LM.dump_state` appending three finetuning keys, then a
//! reasoning model's `max_tokens` being popped and re-set, which moves it to the end. Reproduced
//! here in that order and pinned byte for byte against a generated corpus.
//!
//! **No credential is ever written.** Upstream filters `api_key` out of the block; a saved program
//! is something people pass around, and this crate holds four provider keys where dspy holds one.
//! None of the four is dumped, and `tests/saved_lm.rs` asserts it of every case.

use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value, json};

use super::LM;
use super::openai::OpenAiWire;
use super::token_limit::is_openai_reasoning_model;

/// dspy's `LM_CLASS_STATE_KEY`: which `BaseLM` subclass wrote the block.
pub const CLASS_KEY: &str = "_dspy_lm_class";

/// dspy's `_BUILTIN_LM_CLASS_PATH`. A block naming anything else was written by a subclass, and
/// upstream refuses to import one without an explicit opt-in.
pub const BUILTIN_CLASS: &str = "dspy.clients.lm.LM";

/// dspy's `UNSAFE_LM_STATE_KEYS`. These three decide *where* a call goes, so a compiled program
/// obtained from anywhere could point a reader's calls — and their credential — at someone else's
/// endpoint. Dropped on load unless the caller says the file is trusted.
pub const UNSAFE_KEYS: [&str; 3] = ["api_base", "base_url", "model_list"];

/// What this model states about itself, in the block dspy's loader reads.
///
/// `finetuning_model`, `launch_kwargs` and `train_kwargs` are in every block upstream writes and
/// are written here at their defaults: this crate has no finetuning surface, so there is nothing
/// else they could say, and omitting them would make the file one dspy did not write.
pub fn dump(lm: &LM) -> Map<String, Value> {
    let model = lm.model.reference();
    let reasoning = is_openai_reasoning_model(&model);
    let mut block = Map::new();

    block.insert(CLASS_KEY.to_owned(), json!(BUILTIN_CLASS));
    block.insert("model".to_owned(), json!(model));
    block.insert("model_type".to_owned(), json!(model_type(lm)));
    block.insert("cache".to_owned(), json!(lm.cache));
    block.insert("num_retries".to_owned(), json!(lm.retry.attempts));
    block.insert("temperature".to_owned(), json!(lm.config.temperature));
    if !reasoning {
        block.insert("max_tokens".to_owned(), json!(lm.config.max_tokens));
    }
    extras(lm, &mut block);
    block.insert("finetuning_model".to_owned(), Value::Null);
    block.insert("launch_kwargs".to_owned(), json!({}));
    block.insert("train_kwargs".to_owned(), json!({}));
    if lm.use_developer_role {
        block.insert("use_developer_role".to_owned(), json!(true));
    }
    if reasoning {
        // Upstream sets `max_completion_tokens` for this family and `dump_state` renames it back,
        // by popping and re-inserting — which puts it after the three keys above.
        block.insert("max_tokens".to_owned(), json!(lm.config.max_tokens));
    }
    block
}

/// dspy's `model_type`, which selects the wire rather than describing the model.
fn model_type(lm: &LM) -> &'static str {
    match lm.openai.wire {
        OpenAiWire::Responses => "responses",
        OpenAiWire::Chat => "chat",
        OpenAiWire::Text => "text",
    }
}

/// The settings beyond the two dspy names in its constructor.
///
/// Upstream carries these because they were `**kwargs`, so their order in the block is whatever
/// order the caller wrote them in. There is no Rust equivalent of that — they are typed fields —
/// so they go in declaration order. A block for a model that set only `temperature` and
/// `max_tokens`, which is what the corpus covers and what almost every caller writes, is
/// unaffected; one that set `top_p` differs from dspy's in key order alone.
///
/// `api_base` is here rather than beside the wire settings because that is where dspy puts it: it
/// was a keyword, so it sits among the keywords. Written only when the endpoint is not the default
/// one, which is as close as this gets to upstream's "the caller passed it" — a caller who passed
/// OpenAI's own URL explicitly gets a block dspy would have named it in.
fn extras(lm: &LM, block: &mut Map<String, Value>) {
    let config = &lm.config;
    if lm.openai.base_url != super::openai::DEFAULT_OPENAI_BASE_URL {
        block.insert("api_base".to_owned(), json!(lm.openai.base_url));
    }
    if let Some(top_p) = config.top_p {
        block.insert("top_p".to_owned(), json!(top_p));
    }
    if let Some(stop) = &config.stop {
        block.insert("stop".to_owned(), json!(stop));
    }
    if let Some(n) = config.n {
        block.insert("n".to_owned(), json!(n));
    }
}

/// dspy's `_sanitize_lm_state`: drop what decides where a call goes, unless the file is trusted.
///
/// Returns the block unchanged when it carries none of them, as upstream does — so a block that
/// was never redirected is not rebuilt, and cannot pick up a key ordering change on the way
/// through.
pub fn sanitize(block: &Map<String, Value>, trusted: bool) -> Map<String, Value> {
    if trusted || !UNSAFE_KEYS.iter().any(|key| block.contains_key(*key)) {
        return block.clone();
    }
    block
        .iter()
        .filter(|(key, _)| !UNSAFE_KEYS.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Rebuild the model a block names, so a loaded program asks what it was compiled against.
///
/// dspy's `LM.load_state`. The credential is not in the block and comes from the environment on
/// both sides. A block naming a `BaseLM` subclass is refused with upstream's own message: dspy
/// imports the class when the caller says the file is trusted, which is not something a Rust
/// binary can do at all, so `trusted` widens what is *kept* here without widening what is loaded.
pub fn rebuild(block: &Map<String, Value>, trusted: bool) -> Result<LM> {
    let class = block.get(CLASS_KEY).and_then(Value::as_str);
    if let Some(class) = class
        && class != BUILTIN_CLASS
    {
        if !trusted {
            bail!(
                "Refusing to import custom serialized LM class `{class}`. Pass \
                 allow_unsafe_lm_state=True when loading trusted files to enable custom LM classes."
            );
        }
        bail!(
            "the saved program names the LM class `{class}`, which dspy would import. A Rust \
             binary has no such loader, so this model cannot be rebuilt — construct it and set it \
             with `set_lm` instead."
        );
    }

    let model = block
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("the saved lm block names no model"))?;
    let mut lm = LM::new(model)?;

    if block.get("model_type").and_then(Value::as_str) == Some("responses") {
        lm = lm.openai_responses_api();
    }
    if let Some(cache) = block.get("cache").and_then(Value::as_bool) {
        lm = lm.cache(cache);
    }
    if let Some(retries) = block.get("num_retries").and_then(Value::as_u64) {
        lm = lm.retry(super::Retry::attempts(retries as usize));
    }
    if block.get("use_developer_role").and_then(Value::as_bool) == Some(true) {
        lm = lm.use_developer_role(true);
    }
    // Present only on a trusted load — `sanitize` has already dropped it otherwise. `base_url` and
    // `model_list` survive that same load and are *not* applied: the first is litellm's alias for
    // a field this crate has one of, and the second is its router's, which has no counterpart here
    // at all. A trusted round-trip therefore keeps the endpoint and loses those two, which is
    // narrower than dspy and is the direction to be narrow in.
    if let Some(base_url) = block.get("api_base").and_then(Value::as_str) {
        lm = lm.openai_base_url(base_url);
    }

    lm.config.temperature = block.get("temperature").and_then(Value::as_f64);
    lm.config.max_tokens = block
        .get("max_tokens")
        .or_else(|| block.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .map(|tokens| tokens as u32);
    lm.config.top_p = block.get("top_p").and_then(Value::as_f64);
    lm.config.n = block.get("n").and_then(Value::as_u64).map(|n| n as u32);
    lm.config.stop = block.get("stop").and_then(|stop| {
        Some(
            stop.as_array()?
                .iter()
                .filter_map(|one| Some(one.as_str()?.to_owned()))
                .collect(),
        )
    });
    Ok(lm)
}
