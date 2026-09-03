//! COPRO's two meta-signatures: the prompts it sends to a model to propose better instructions.
//!
//! Their bytes reach the model, so they are byte-verified against dspy in
//! `tests/conformance/basic_generate_instruction.json` and
//! `tests/conformance/generate_instruction_given_attempts.json`. The strings below are the source
//! those fixtures were generated from — the same class docstrings and field descriptions dspy
//! declares in `teleprompt/copro_optimizer.py`.

use crate::signature::{InField, OutField, Signature};

/// The two outputs both meta-signatures ask for, declared identically on each: the improved
/// instruction, and the prefix that would sit at the end of the prompt.
fn proposed_fields() -> Vec<OutField> {
    vec![
        OutField {
            name: "proposed_instruction".into(),
            desc: "The improved instructions for the language model".into(),
            ..Default::default()
        },
        OutField {
            name: "proposed_prefix_for_output_field".into(),
            desc: "The string at the end of the prompt, which will help the model start solving the task".into(),
            ..Default::default()
        },
    ]
}

/// dspy `BasicGenerateInstruction`: propose an instruction from the current one alone. COPRO's
/// zero-shot seed step asks this `breadth - 1` times per predictor.
pub(super) fn basic_generate_instruction() -> Signature {
    Signature {
        instructions: "You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative.".into(),
        inputs: vec![InField {
            name: "basic_instruction".into(),
            desc: "The initial instructions before optimization".into(),
            ..Default::default()
        }],
        outputs: proposed_fields(),
    }
}

/// dspy `GenerateInstructionGivenAttempts`: propose a better instruction given past attempts and
/// their scores, laid out worst-first. COPRO's depth step asks this `breadth` times per predictor.
///
/// `attempted_instructions` is declared with no description: dspy carries the `${name}` default
/// there and renders it blank, which this crate spells as an empty desc for the same blank line.
pub(super) fn generate_instruction_given_attempts() -> Signature {
    Signature {
        instructions: "You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.\n\nYour task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative.".into(),
        inputs: vec![InField {
            name: "attempted_instructions".into(),
            ..Default::default()
        }],
        outputs: proposed_fields(),
    }
}
