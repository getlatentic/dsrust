//! A coding agent behind dsrust's `ChatModel`.

use std::future::Future;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow};
use dsrust::lm::ChatModel;
use dsrust::lm::api::{LmRequest, LmResponse};
use harness::{Harness, RunHandle, RunMode, RunRequest, RunTuning, ToolAccess};

use crate::{collect, prompt};

/// Standing instructions every request carries, unless a caller replaces them.
///
/// dsrust's adapters read a reply by its field markers, and a chat-tuned coding
/// agent reflexively wraps its answer in Markdown. The JSON inside a field is
/// forgiven by dsrust's repair; the envelope is not — `**[[ ## answer ## ]]**`
/// or a fenced reply is a parse failure, which a `ReActV2` loop records as
/// `termination_reason: "parse_error"` with no answer at all. So the one thing
/// worth telling the agent every time is how to spell the markers.
pub const MARKER_DISCIPLINE: &str = "Reply in plain text only: no Markdown fences, headings, bold or \
    bullet formatting anywhere in the reply. When the request shows field markers of the form \
    `[[ ## name ## ]]`, reproduce each marker exactly — same brackets, same spaces — on its own \
    line, followed by that field's value and nothing else.";

/// Any [`Harness`] — Claude Code, Codex, an ACP agent, the built-in
/// OpenAI-compatible runtime — as the model a dsrust program calls.
///
/// Every request runs with [`ToolAccess::None`]. The agent's own tools would
/// otherwise run behind dsrust's back: for an ordinary `Predict` that is wasted
/// turns and a filesystem the prompt never mentioned; for a `ReActV2` loop it is
/// fatal, because the loop reads tool calls out of the reply and a reply that
/// already acted on them carries none. `ToolAccess::None` is a guarantee on the
/// adapters that advertise it and a refusal on the rest — see
/// `Features::withheld_tools` — so this never silently degrades.
///
/// Capabilities are dsrust's defaults, all off: tools are rendered into the
/// prompt and read back from text, which is the path an agent CLI can serve.
/// A schema still travels — `JsonAdapter`'s `response_format` becomes the run's
/// `output_schema` — and an adapter that advertises `structured_output` answers
/// with data, which this model hands back as the reply's text in place of the
/// agent's narration.
pub struct HarnessModel<H> {
    harness: H,
    tools: ToolAccess,
    model: Option<String>,
    cwd: Option<PathBuf>,
    max_turns: Option<u32>,
    max_thinking_tokens: Option<u32>,
    instructions: Option<String>,
    calls: AtomicU64,
}

impl<H: Harness> HarnessModel<H> {
    /// `harness` as a model, with [`MARKER_DISCIPLINE`] as its standing instructions.
    pub fn new(harness: H) -> Self {
        Self {
            harness,
            tools: ToolAccess::None,
            model: None,
            cwd: None,
            max_turns: None,
            max_thinking_tokens: None,
            instructions: Some(MARKER_DISCIPLINE.to_owned()),
            calls: AtomicU64::new(0),
        }
    }

    /// Let the agent use its tools — its own, and any [`ToolServer`](harness::ToolServer)
    /// attached to the harness — and hand back the answer it reaches with them.
    ///
    /// This is the agent as a *module*: a `Predict` over a model built this way
    /// gets one reply per call, arrived at by whatever tool loop the agent ran,
    /// and stays an ordinary predictor an optimizer can rewrite. It is the wrong
    /// choice under a `ReActV2`, whose own loop needs the tool calls back as
    /// calls; see [`HarnessModel`] for why the default withholds them.
    pub fn with_agent_tools(mut self) -> Self {
        self.tools = ToolAccess::Default;
        self
    }

    /// The model the agent should use, as its CLI names it (`sonnet`, `o3`). Overrides
    /// whatever the request names; `None` leaves the CLI's own default.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The working directory each run gets. Named rather than inherited: an
    /// agent's reach is its cwd, and the host process's is rarely what a prompt
    /// meant.
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Cap on the agent's turns per call, where the adapter honours one.
    pub fn with_max_turns(mut self, turns: u32) -> Self {
        self.max_turns = Some(turns);
        self
    }

    /// Cap the agent's extended thinking, in tokens; `0` turns it off. Measured
    /// on Claude Code: thinking off took a judge-shaped call from 10.0s to 7.9s
    /// and its output from 240 tokens to 72. A judge or a classifier rarely
    /// needs it; a reflection model usually does — leave it unset there.
    pub fn with_max_thinking_tokens(mut self, tokens: u32) -> Self {
        self.max_thinking_tokens = Some(tokens);
        self
    }

    /// Replace the standing instructions. `None` sends only what the request's
    /// own system messages say.
    pub fn with_instructions(mut self, instructions: Option<String>) -> Self {
        self.instructions = instructions;
        self
    }

    fn run_request(&self, request: &LmRequest, rendered: prompt::Rendered) -> RunRequest {
        let n = self.calls.fetch_add(1, Ordering::Relaxed);
        // The instructions — the standing ones and the request's own system
        // messages — are the whole system prompt when the agent is being a model
        // (no tools), and an addition to the agent's own when it is being an
        // agent. Measured on Claude Code: replacing it takes a call from ~7,000
        // prompt tokens to a few hundred, which at optimizer scale is the bill.
        let instructions = join(
            self.instructions.as_deref(),
            rendered.instructions.as_deref(),
        );
        let (system_prompt, extra_instructions) = match self.tools {
            ToolAccess::None => (instructions, None),
            _ => (None, instructions),
        };
        RunRequest {
            run_id: format!("dsrust-{}-{n}", std::process::id()),
            prompt: rendered.prompt,
            attachments: rendered.attachments,
            cwd: self.cwd.clone(),
            mode: RunMode::Ask,
            tools: self.tools,
            tuning: RunTuning {
                model: self.model.clone().or_else(|| nonblank(&request.model)),
                max_turns: self.max_turns,
                max_thinking_tokens: self.max_thinking_tokens,
                output_schema: request.output_schema().cloned(),
                system_prompt,
                extra_instructions,
                ..RunTuning::default()
            },
            resume: None,
        }
    }
}

impl<H: Harness + 'static> ChatModel for HarnessModel<H> {
    fn forward<'a>(
        &'a self,
        request: &'a LmRequest,
    ) -> impl Future<Output = Result<LmResponse>> + Send + 'a {
        let rendered = prompt::render(request);
        let run = self.run_request(request, rendered);
        let model = run.tuning.model.clone();
        // The run starts here, not on first poll, so the guard is built here too:
        // a future dropped before it is ever polled has still started an agent,
        // and that agent must stop.
        let started = self
            .harness
            .run(run)
            .map(|(handle, events)| (CancelOnDrop(handle), events))
            .context("the agent could not be started");
        async move {
            let (_cancel, events) = started?;
            // The run is on threads of its own and answers over a channel. Draining
            // it on one more thread, and awaiting a one-shot, keeps this future
            // free of any particular runtime.
            let (done, answered) = futures_channel::oneshot::channel();
            std::thread::spawn(move || {
                let _ = done.send(collect::drain(events, model));
            });
            answered
                .await
                .map_err(|_| anyhow!("the run's collector thread ended without answering"))?
        }
    }
}

/// Stops the agent when the call that started it is dropped mid-flight — the
/// caller went away, so the tokens it was spending should stop too. Best-effort
/// and idempotent, so a run that already finished costs nothing to "cancel".
struct CancelOnDrop(RunHandle);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let _ = self.0.cancel();
    }
}

fn nonblank(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn join(a: Option<&str>, b: Option<&str>) -> Option<String> {
    match (
        a.filter(|s| !s.trim().is_empty()),
        b.filter(|s| !s.trim().is_empty()),
    ) {
        (Some(a), Some(b)) => Some(format!("{a}\n\n{b}")),
        (Some(one), None) | (None, Some(one)) => Some(one.to_owned()),
        (None, None) => None,
    }
}
