//! A provider that answers from a script, with the hand-built signature and the derived task
//! the modules in this tree script it against.

use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::{Result, anyhow};

use crate::lm::api::{self, content_of};
use crate::lm::{ChatModel, ChatTurn, Content, LmUsage, Role};
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

#[derive(Clone)]
pub(super) struct Call {
    pub(super) system: String,
    pub(super) turns: Vec<ChatTurn>,
    pub(super) json_mode: bool,
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
            system: request.system().to_owned(),
            turns: recorded_turns(request),
            json_mode: request.output_schema().is_some(),
        });
        self.replies
            .lock()
            .expect("not poisoned")
            .pop_front()
            .map(|reply| api::LmResponse::text(reply).with_usage(self.usage.clone()))
            .ok_or_else(|| anyhow!("script exhausted"))
    }
}

/// The non-system messages as the turns a test asserts on, each part collapsed to its prose.
fn recorded_turns(request: &api::LmRequest) -> Vec<ChatTurn> {
    request
        .messages
        .iter()
        .filter(|message| message.role != "system")
        .map(|message| ChatTurn {
            role: match message.role.as_str() {
                "assistant" => Role::Assistant,
                _ => Role::User,
            },
            content: content_of(&message.parts).unwrap_or_else(|_| Content::Text(String::new())),
        })
        .collect()
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
