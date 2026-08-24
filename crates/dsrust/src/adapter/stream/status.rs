//! Naming each stage a program passes through — dspy's `StatusMessageProvider`.
//!
//! A streamed run yields the watched field's text, which says nothing while the program is doing
//! something else: calling a tool, waiting on a model. These are the sentences that fill that
//! silence, and a caller overrides the ones it wants.
//!
//! **Four of the six say nothing by default, and that is upstream's shape rather than an
//! omission.** Only the two tool messages have wording; module and LM stages are there for a
//! caller to fill in and are silent until it does. A port that invented sentences for the other
//! four would put text on a stream dspy leaves empty.

use std::sync::Arc;

use anyhow::Error;
use serde_json::Value;

use crate::callback::{CallId, Callback};
use crate::example::{Example, Prediction};
use crate::lm::api;

/// dspy's `finish` tool, whose start is not announced — it ends the loop rather than doing work.
const FINISH: &str = "finish";

/// dspy's `Completed.`, the finish tool's own output, whose end is not announced either.
const COMPLETED: &str = "Completed.";

/// What to say as a program passes through each stage. Every method is defaulted, so a caller
/// overrides only the stages it cares about — upstream's base class, whose subclass does the same.
///
/// `None` means say nothing, which is what four of these do until overridden.
pub trait StatusMessages: Send + Sync {
    /// dspy `tool_start_status_message`. Its wording, exactly.
    fn tool_start(&self, tool: &str, args: &Value) -> Option<String> {
        let _ = args;
        Some(format!("Calling tool {tool}..."))
    }

    /// dspy `tool_end_status_message`. Its wording, exactly.
    fn tool_end(&self, outputs: &Value) -> Option<String> {
        let _ = outputs;
        Some("Tool calling finished! Querying the LLM with tool calling results...".to_owned())
    }

    /// dspy `module_start_status_message`, which says nothing until a caller gives it words.
    fn module_start(&self, module: &str, inputs: &Example) -> Option<String> {
        let _ = (module, inputs);
        None
    }

    /// dspy `module_end_status_message`.
    fn module_end(&self, answered: &Prediction) -> Option<String> {
        let _ = answered;
        None
    }

    /// dspy `lm_start_status_message`.
    fn lm_start(&self, request: &api::LmRequest) -> Option<String> {
        let _ = request;
        None
    }

    /// dspy `lm_end_status_message`.
    fn lm_end(&self, answered: &api::LmResponse) -> Option<String> {
        let _ = answered;
        None
    }
}

/// The stock wording, for a caller who wants the tool messages and nothing else.
pub struct DefaultStatus;

impl StatusMessages for DefaultStatus {}

/// Publishes a provider's sentences onto a streamed run — dspy's `StatusStreamingCallback`.
pub(super) struct Announcing<S> {
    pub(super) messages: Arc<dyn StatusMessages>,
    pub(super) sink: S,
}

impl<S: Fn(String) + Send + Sync> Callback for Announcing<S> {
    fn on_tool_start(&self, _call: &CallId, tool: &str, args: &Value) {
        // Upstream skips `finish` by name: it closes the loop rather than doing work, so
        // announcing it would report an action the program did not take.
        if tool == FINISH {
            return;
        }
        self.say(self.messages.tool_start(tool, args));
    }

    fn on_tool_end(&self, _call: &CallId, answered: Result<&Value, &Error>) {
        let Ok(outputs) = answered else { return };
        // And skips the finish tool's own output for the same reason, which upstream matches by
        // value rather than by name because `on_tool_end` is not told which tool it was.
        if outputs.as_str() == Some(COMPLETED) {
            return;
        }
        self.say(self.messages.tool_end(outputs));
    }

    fn on_module_start(&self, _call: &CallId, module: &str, inputs: &Example) {
        self.say(self.messages.module_start(module, inputs));
    }

    fn on_module_end(&self, _call: &CallId, answered: Result<&Prediction, &Error>) {
        if let Ok(answered) = answered {
            self.say(self.messages.module_end(answered));
        }
    }

    fn on_lm_start(&self, _call: &CallId, request: &api::LmRequest) {
        self.say(self.messages.lm_start(request));
    }

    fn on_lm_end(&self, _call: &CallId, answered: Result<&api::LmResponse, &Error>) {
        if let Ok(answered) = answered {
            self.say(self.messages.lm_end(answered));
        }
    }
}

impl<S: Fn(String) + Send + Sync> Announcing<S> {
    /// Upstream sends only a non-empty message, so a provider returning nothing is silence rather
    /// than a blank line on the stream.
    fn say(&self, message: Option<String>) {
        if let Some(message) = message.filter(|message| !message.is_empty()) {
            (self.sink)(message);
        }
    }
}
