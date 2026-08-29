//! The three signatures `ProgramOfThought` asks with, built from the caller's own.
//!
//! Split from the module because they answer a different question: the module decides *when* to
//! generate, re-generate or answer, and this decides *what each of those asks look like*. dspy
//! keeps both in one file, which is a fact about Python's file sizes rather than about the code.

use crate::signature::{FieldKind, InField, OutField, Signature};

use super::Mode;

pub(super) fn mode_signature(signature: &Signature, mode: Mode) -> Signature {
    let mut inputs = signature.inputs.clone();
    let outputs = match mode {
        Mode::Generate => vec![generated_code()],
        Mode::Regenerate => {
            inputs.push(input_field(
                "previous_code",
                "previously-generated python code that errored",
            ));
            inputs.push(input_field(
                "error",
                "error message from previously-generated python code",
            ));
            vec![generated_code()]
        }
        Mode::Answer => {
            inputs.push(input_field(
                "final_generated_code",
                "python code that answers the question",
            ));
            inputs.push(input_field(
                "code_output",
                "output of previously-generated python code",
            ));
            signature.outputs.clone()
        }
    };
    Signature {
        instructions: instructions(signature, mode, &inputs, &outputs),
        inputs,
        outputs,
    }
}

fn input_field(name: &str, desc: &str) -> InField {
    InField {
        name: name.to_owned(),
        desc: desc.to_owned(),
        ..Default::default()
    }
}

fn generated_code() -> OutField {
    OutField {
        name: "generated_code".to_owned(),
        desc: "python code that answers the question".to_owned(),
        kind: FieldKind::Str,
        ..Default::default()
    }
}

/// dspy `_generate_instruction`: what each of the three asks is told to do.
fn instructions(
    signature: &Signature,
    mode: Mode,
    inputs: &[InField],
    outputs: &[OutField],
) -> String {
    let mode_inputs = backticked(inputs.iter().map(|field| field.name.as_str()));
    let mode_outputs = backticked(outputs.iter().map(|field| field.name.as_str()));
    let lines = match mode {
        Mode::Generate => {
            let final_outputs =
                backticked(signature.outputs.iter().map(|field| field.name.as_str()));
            vec![
                format!(
                    "You will be given {mode_inputs} and you will respond with {mode_outputs}."
                ),
                format!(
                    "Generating executable Python code that programmatically computes the correct \
                     {mode_outputs}."
                ),
                "After you're done with the computation and think you have the final output, make \
                 sure to submit your output by calling the preloaded function `SUBMIT()`."
                    .to_owned(),
                format!(
                    "You must structure your output in a dict, like {{\"field_a\": value_a, ...}}, \
                     with the correct value mapping for the field(s): {final_outputs}."
                ),
            ]
        }
        Mode::Regenerate => vec![
            format!("You are given {mode_inputs} due to an error in previous code."),
            "Your task is to correct the error and provide the new `generated_code`.".to_owned(),
        ],
        Mode::Answer => {
            vec![format!(
                "Given the final code {mode_inputs}, provide the final {mode_outputs}."
            )]
        }
    };
    lines.join("\n")
}

fn backticked<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `ProgramOfThought!("question -> answer")` — the model writes code, is shown what it produced, and states the answer.
///
/// Takes a string signature or a task declared with `#[derive(Signature)]`, as every other module
/// macro does; the declared form carries its doc comment as the signature's instructions.
/// `max_iters = N` caps the loop.
#[macro_export]
macro_rules! ProgramOfThought {
    ($signature:literal $(,)?) => {
        $crate::ProgramOfThought::new($crate::make_signature!($signature))
    };
    ($signature:literal, max_iters = $max:expr $(,)?) => {
        $crate::ProgramOfThought::new($crate::make_signature!($signature)).max_iters($max)
    };
    ($task:ty $(,)?) => {
        $crate::ProgramOfThought::new(<$task as $crate::signature::SignatureSpec>::signature())
    };
    ($task:ty, max_iters = $max:expr $(,)?) => {
        $crate::ProgramOfThought::new(<$task as $crate::signature::SignatureSpec>::signature())
            .max_iters($max)
    };
}
