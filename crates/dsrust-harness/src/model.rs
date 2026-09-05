//! A coding agent behind dsrust's `ChatModel`.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow};
use dsrust::lm::ChatModel;
use dsrust::lm::api::{LmRequest, LmResponse};
use harness::{Harness, RunHandle, RunMode, RunRequest, RunTuning, ToolAccess};

use crate::builder::{HarnessModelBuilder, Settings};
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
/// Built with [`HarnessModel::new`] for the defaults or [`HarnessModel::builder`]
/// to name them, the way `LM::new` and `LM::builder` do.
///
/// By default every request runs with [`ToolAccess::None`]. The agent's own tools
/// would otherwise run behind dsrust's back: for an ordinary `Predict` that is
/// wasted turns and a filesystem the prompt never mentioned; for a `ReActV2` loop
/// it is fatal, because the loop reads tool calls out of the reply and a reply
/// that already acted on them carries none. `ToolAccess::None` is a guarantee on
/// the adapters that advertise it and a refusal on the rest — see
/// `Features::withheld_tools` — so `build` turns such a harness away rather than
/// letting it degrade silently.
///
/// Capabilities are dsrust's defaults, all off: tools are rendered into the
/// prompt and read back from text, which is the path an agent CLI can serve.
/// A schema still travels — `JsonAdapter`'s `response_format` becomes the run's
/// `output_schema` — and an adapter that advertises `structured_output` answers
/// with data, which this model hands back as the reply's text in place of the
/// agent's narration.
pub struct HarnessModel<H> {
    harness: H,
    settings: Settings,
    calls: AtomicU64,
}

impl<H: Harness> HarnessModel<H> {
    /// `harness` as a model with every setting at its default, or the reason it cannot
    /// be one — see [`HarnessModelBuilder::build`].
    pub fn new(harness: H) -> Result<Self> {
        Self::builder(harness).build()
    }

    /// `harness` as a model, its settings named one by one.
    pub fn builder(harness: H) -> HarnessModelBuilder<H> {
        HarnessModelBuilder::new(harness)
    }

    pub(crate) fn from_parts(harness: H, settings: Settings) -> Self {
        Self {
            harness,
            settings,
            calls: AtomicU64::new(0),
        }
    }

    fn run_request(&self, request: &LmRequest, rendered: prompt::Rendered) -> RunRequest {
        let n = self.calls.fetch_add(1, Ordering::Relaxed);
        // The instructions — the standing ones and the request's own system
        // messages — are the whole system prompt when the agent is being a model
        // (no tools), and an addition to the agent's own when it is being an
        // agent. Measured on Claude Code: replacing it takes a call from ~7,000
        // prompt tokens to a few hundred, which at optimizer scale is the bill.
        let settings = &self.settings;
        let instructions = join(
            settings.instructions.as_deref(),
            rendered.instructions.as_deref(),
        );
        let (system_prompt, extra_instructions) = match settings.tools {
            ToolAccess::None => (instructions, None),
            _ => (None, instructions),
        };
        RunRequest {
            run_id: format!("dsrust-{}-{n}", std::process::id()),
            prompt: rendered.prompt,
            attachments: rendered.attachments,
            cwd: settings.cwd.clone(),
            mode: RunMode::Ask,
            tools: settings.tools,
            tuning: RunTuning {
                model: settings.model.clone().or_else(|| nonblank(&request.model)),
                max_turns: settings.max_turns,
                max_thinking_tokens: settings.max_thinking_tokens,
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
