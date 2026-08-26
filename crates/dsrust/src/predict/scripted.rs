//! A provider that answers from a script, with the hand-built signature and the derived task
//! the modules in this tree script it against.

use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::{Result, anyhow};

use crate::lm::api;
use crate::lm::{ChatModel, LmUsage};
use crate::signature::{OutField, Signature};

pub(super) fn signature() -> Signature {
    Signature::single_input(
        "Pick a color.",
        vec![
            OutField {
                name: "color".into(),
                desc: "the chosen color".into(),
                values: Some(vec!["red".into(), "blue".into()]),
                ..Default::default()
            },
            OutField {
                name: "why".into(),
                desc: "one short sentence".into(),
                ..Default::default()
            },
        ],
    )
}

/// Scripted stand-in for a provider: pops one canned reply per call and records what
/// each call asked, so tests can assert on the retry conversation.
pub(super) struct Scripted {
    replies: Mutex<VecDeque<&'static str>>,
    calls: Mutex<Vec<Call>>,
    /// Reported on every reply, so a test forcing several calls can assert on their sum.
    usage: Option<LmUsage>,
}

/// One call as the model received it. The messages, not a system prompt beside a set of turns:
/// the split that stood here re-derived a pair the adapter had stopped producing, and mapped every
/// role that was not `assistant` to `user`, so a tool result recorded as a user turn.
#[derive(Clone)]
pub(super) struct Call {
    pub(super) messages: Vec<api::LmMessage>,
    pub(super) json_mode: bool,
}

impl Call {
    /// The system prompt, or `""` when there is none.
    pub(super) fn system(&self) -> &str {
        api::system_of(&self.messages)
    }

    /// The conversation without the system prompt, so a test counting turns counts from the first
    /// thing anyone said.
    pub(super) fn turns(&self) -> &[api::LmMessage] {
        api::after_system(&self.messages)
    }
}

impl Scripted {
    pub(super) fn new(replies: &[&'static str]) -> Self {
        Self {
            replies: Mutex::new(replies.iter().copied().collect()),
            calls: Mutex::new(Vec::new()),
            usage: None,
        }
    }

    /// Report this cost on every reply. A provider charges each call, so a module that asks
    /// twice should answer with twice this.
    pub(super) fn costing(mut self, input_tokens: u32, output_tokens: u32) -> Self {
        self.usage = Some(LmUsage::counted(input_tokens, output_tokens));
        self
    }

    pub(super) fn calls(&self) -> Vec<Call> {
        self.calls.lock().expect("not poisoned").clone()
    }
}

impl ChatModel for Scripted {
    async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
        self.calls.lock().expect("not poisoned").push(Call {
            messages: request.messages.clone(),
            json_mode: request.output_schema().is_some(),
        });
        self.replies
            .lock()
            .expect("not poisoned")
            .pop_front()
            .map(|reply| api::LmResponse::text(reply).usage(self.usage.clone()))
            .ok_or_else(|| anyhow!("script exhausted"))
    }
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct Pick {
    pub(super) color: String,
    pub(super) why: String,
}

/// The derive is declaration data; the struct itself is never built.
#[allow(dead_code)]
#[derive(crate::signature::Signature)]
#[signature(instructions = "Pick a color for the room.")]
pub(super) struct RoomTask {
    #[input(desc = "the room being painted")]
    room: String,
    #[input(desc = "the mood to set")]
    mood: String,
    #[output(desc = "the chosen color", values("red", "blue"))]
    color: String,
    #[output(desc = "one short sentence")]
    why: String,
}

pub(super) fn room_inputs() -> RoomTaskInputs {
    RoomTaskInputs {
        room: "the study".into(),
        mood: "calm focus".into(),
    }
}
