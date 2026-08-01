//! dspy `GroundedProposer`: the loop that proposes every predictor's instruction candidates.
//!
//! One level above [`super::proposer`], which owns the signature a single proposal is asked
//! through. This owns how many are asked for, in what order, and what grounding each one carries —
//! the tip drawn per proposal, the dataset summary drawn once, and the rollout id that makes two
//! proposals from the same prompt differ.

use std::sync::Arc;

use anyhow::Result;

use super::super::rng::Rng;
use super::proposer::GenerateModuleInstruction;
use super::signatures::InstructionInputs;
use crate::lm::DynChatModel;
use crate::signature::Signature;

/// dspy GroundedProposer's tips, in declaration order — the order `random.choice` indexes into. The
/// empty `none` tip is a real member: choosing it turns the tip field off for that candidate.
pub(super) const TIPS: [&str; 6] = [
    "",
    "Don't be afraid to be creative when creating the new instruction!",
    "Keep the instruction clear and concise.",
    "Make sure your instruction is very informative and descriptive.",
    "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "Include a persona that is relevant to the task in the instruction (ie. \"You are a ...\")",
];

/// The grounded proposer, in its zero-shot form: propose `num_candidates` instructions per predictor,
/// each optionally program-aware and carrying a randomly chosen tip.
pub(super) struct GroundedProposer {
    /// dspy's `data_summary`, produced once in `GroundedProposer.__init__` before any candidate is
    /// proposed. `None` is both "not asked for" and "the summarising failed", which is upstream's
    /// own fallback: it catches the exception and runs without the summary rather than losing the
    /// compile.
    pub(super) dataset_summary: Option<String>,
    pub(super) program_code: Option<String>,
    pub(super) tip_aware: bool,
    pub(super) prompt_model: Arc<dyn DynChatModel>,
    /// dspy `init_temperature`, carried from the optimizer to the one call that proposes.
    pub(super) init_temperature: f64,
}

impl GroundedProposer {
    /// dspy `propose_instructions_for_program`: for each predictor, `num_candidates` proposals, with
    /// candidate zero forced back to the predictor's current instruction — the baseline the search
    /// starts from. The RNG is CPython's, drawing a tip then a rollout id per proposal, in dspy's order.
    pub(super) async fn propose(
        &self,
        predictors: &[Signature],
        num_candidates: usize,
        rng: &mut Rng,
    ) -> Result<Vec<Vec<String>>> {
        let mut proposed = Vec::with_capacity(predictors.len());
        for signature in predictors {
            let mut candidates = Vec::with_capacity(num_candidates);
            for _ in 0..num_candidates {
                let tip = self.select_tip(rng);
                // dspy asks for each proposal through `prompt_model.copy(rollout_id=…,
                // temperature=init_temperature)`. The draw also advances the shared generator, which
                // is why it happens whether or not the id is used.
                let rollout = rng.randint(0, 1_000_000_000);
                let sampling = crate::lm::Sampling {
                    temperature: Some(self.init_temperature),
                    ..crate::lm::Sampling::rollout(rollout)
                };
                let inputs = InstructionInputs {
                    dataset_summary: self.dataset_summary.is_some(),
                    program_aware: self.program_code.is_some(),
                    instruct_history: false,
                    tip: tip.is_some(),
                };
                let generator = GenerateModuleInstruction::new(
                    self.program_code.clone(),
                    inputs,
                    self.prompt_model.clone(),
                    sampling,
                );
                let instruction = generator
                    .forward(
                        signature,
                        "No task demos provided.",
                        self.dataset_summary.as_deref().unwrap_or_default(),
                        "",
                        tip,
                    )
                    .await?;
                candidates.push(instruction);
            }
            if !candidates.is_empty() {
                candidates[0] = signature.instructions.clone();
            }
            proposed.push(candidates);
        }
        Ok(proposed)
    }

    /// dspy's `random.choice(list(TIPS.keys()))` when tips are on: a draw is made regardless of which
    /// tip lands, and the empty `none` tip reads as no tip. Off, no draw is made and there is no tip.
    fn select_tip(&self, rng: &mut Rng) -> Option<&'static str> {
        if !self.tip_aware {
            return None;
        }
        let tip = TIPS[rng.choice_index(TIPS.len())];
        (!tip.is_empty()).then_some(tip)
    }
}
