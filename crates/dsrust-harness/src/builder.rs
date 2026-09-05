//! Building a [`HarnessModel`], with the one thing it cannot do without named first.
//!
//! The same shape as `LM::builder(model)`: the harness is positional, the settings are a
//! chain named as dsrust names its own (`.model(..)`, `.max_turns(..)`, `.cache(..)`-style
//! plain verbs), and `build` is where a request the harness cannot honour is refused — once,
//! here, rather than at the first call.

use std::path::PathBuf;

use anyhow::{Result, bail};
use harness::{Harness, ToolAccess};

use crate::model::{HarnessModel, MARKER_DISCIPLINE};

/// What a [`HarnessModel`] carries into every run.
pub(crate) struct Settings {
    pub(crate) tools: ToolAccess,
    pub(crate) model: Option<String>,
    pub(crate) cwd: Option<PathBuf>,
    pub(crate) max_turns: Option<u32>,
    pub(crate) max_thinking_tokens: Option<u32>,
    pub(crate) instructions: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            tools: ToolAccess::None,
            model: None,
            cwd: None,
            max_turns: None,
            max_thinking_tokens: None,
            instructions: Some(MARKER_DISCIPLINE.to_owned()),
        }
    }
}

/// A [`HarnessModel`] under construction. Reached through [`HarnessModel::builder`].
///
/// ```no_run
/// use dsrust_harness::HarnessModel;
/// use dsrust_harness::harness::Claude;
///
/// let model = HarnessModel::builder(Claude::new())
///     .cwd(std::env::temp_dir())
///     .max_turns(2)
///     .max_thinking_tokens(0)
///     .build()?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct HarnessModelBuilder<H> {
    harness: H,
    settings: Settings,
}

impl<H: Harness> HarnessModelBuilder<H> {
    pub(crate) fn new(harness: H) -> Self {
        Self {
            harness,
            settings: Settings::default(),
        }
    }

    /// What the agent may reach. [`ToolAccess::None`], the default, is the agent as a
    /// *model*: nothing but the prompt, which is what a `Predict` or a `ReActV2` loop
    /// needs — see [`HarnessModel`]. [`ToolAccess::Default`] is the agent as a *module*:
    /// its own tools, and any [`ToolServer`](harness::ToolServer) attached to the harness,
    /// run behind one reply.
    pub fn tools(mut self, tools: ToolAccess) -> Self {
        self.settings.tools = tools;
        self
    }

    /// The model the agent should use, as its CLI names it (`sonnet`, `o3`). Overrides
    /// whatever the request names; unset leaves the CLI's own default.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.settings.model = Some(model.into());
        self
    }

    /// The working directory each run gets. Named rather than inherited: an agent's
    /// reach is its cwd, and the host process's is rarely what a prompt meant.
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.settings.cwd = Some(cwd.into());
        self
    }

    /// Cap on the agent's turns per call, where the adapter honours one.
    pub fn max_turns(mut self, turns: u32) -> Self {
        self.settings.max_turns = Some(turns);
        self
    }

    /// Cap the agent's extended thinking, in tokens; `0` turns it off. Measured on
    /// Claude Code: thinking off took a judge-shaped call from 10.0s to 7.9s and its
    /// output from 240 tokens to 72. A judge or a classifier rarely needs it; a
    /// reflection model usually does — leave it unset there.
    pub fn max_thinking_tokens(mut self, tokens: u32) -> Self {
        self.settings.max_thinking_tokens = Some(tokens);
        self
    }

    /// Replace the standing [`MARKER_DISCIPLINE`]. Blank sends only what the request's
    /// own system messages say.
    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        let text = text.into();
        self.settings.instructions = (!text.trim().is_empty()).then_some(text);
        self
    }

    /// The model, or the reason this harness cannot be one.
    ///
    /// An adapter that cannot withhold its agent's tools — Codex, ACP — says so through
    /// `Features::withheld_tools`, and a model asking for [`ToolAccess::None`] on it would
    /// be refused at every call. Refused here instead, where the caller can still choose
    /// `.tools(ToolAccess::Default)` and run the agent as a module.
    pub fn build(self) -> Result<HarnessModel<H>> {
        if self.settings.tools == ToolAccess::None && !self.harness.features().withheld_tools {
            bail!(
                "{} cannot withhold its tools, so it cannot run as a model under ToolAccess::None; \
                 build it with `.tools(ToolAccess::Default)` and it runs as an agent instead",
                self.harness.info().display_name
            );
        }
        Ok(HarnessModel::from_parts(self.harness, self.settings))
    }
}
