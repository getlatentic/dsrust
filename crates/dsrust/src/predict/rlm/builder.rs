//! `RLM`'s builder: everything a caller sets before the loop runs.
//!
//! Its own file for the reason the loop is its own: the settings are a list that grows with the
//! module's surface, and the loop is one algorithm.
use std::sync::Arc;

use super::Rlm;
use super::signatures::signatures;
use crate::predict::Predict;

impl Rlm {
    pub fn max_iters(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }

    /// The budget the model is told about, which is stated in the action instructions — so
    /// changing it rebuilds them.
    pub fn max_llm_calls(mut self, max_llm_calls: usize) -> Self {
        self.max_llm_calls = max_llm_calls;
        let (action, _) = signatures(
            &self.signature,
            &self.tools,
            max_llm_calls,
            self.interpreter_factory.execution_instructions(),
        );
        self.generate_action = Predict::from_signature(action);
        self
    }

    /// Ask both steps of this model.
    pub fn set_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.generate_action = self.generate_action.set_lm(lm.clone());
        self.extract = self.extract.set_lm(lm);
        self
    }

    /// Ask the REPL turns of this model, leaving the extract step on whatever it had.
    ///
    /// The two steps are separable because upstream's are: `rlm.generate_action` and `rlm.extract`
    /// are attributes its own tests replace one at a time, and a caller wanting a cheaper model to
    /// read back a finished session wants the same seam.
    pub fn action_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.generate_action = self.generate_action.set_lm(lm);
        self
    }

    /// Ask the extract step of this model. See [`Self::action_lm`].
    pub fn extract_lm(mut self, lm: Arc<dyn crate::lm::DynChatModel>) -> Self {
        self.extract = self.extract.set_lm(lm);
        self
    }
}
