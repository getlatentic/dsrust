//! GroundedProposer's instruction-proposal signatures — the prompts it sends to describe a program
//! and its module, then propose an instruction for it.
//!
//! Their bytes reach the model, so each is byte-verified as an ordinary ChatAdapter fixture (see
//! `tests/conformance/{describe_program,describe_module,generate_module_instruction}.json`). The
//! strings here are the source those fixtures were generated from — dspy's
//! `propose/grounded_proposer.py`. The dataset-summary signatures live with that pipeline.

use crate::signature::{InField, OutField, Signature};

fn input(name: &str, desc: &str) -> InField {
    InField {
        name: name.into(),
        desc: desc.into(),
        ..Default::default()
    }
}

fn output(name: &str, desc: &str) -> OutField {
    OutField {
        name: name.into(),
        desc: desc.into(),
        ..Default::default()
    }
}

/// dspy `DescribeProgram`: describe what task the whole program solves and how.
pub(crate) fn describe_program() -> Signature {
    Signature {
        instructions: "Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe what type of task this program appears to be designed to solve, and how it appears to work.".into(),
        inputs: vec![
            input("program_code", "Pseudocode for a language model program designed to solve a particular task."),
            input("program_example", "An example of the program in use."),
        ],
        outputs: vec![output(
            "program_description",
            "Describe what task the program is designed to solve, and how it goes about solving this task.",
        )],
    }
}

/// dspy `DescribeModule`: describe one module's role within the program.
pub(crate) fn describe_module() -> Signature {
    Signature {
        instructions: "Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe the purpose of one of the specified module in this pipeline.".into(),
        inputs: vec![
            input("program_code", "Pseudocode for a language model program designed to solve a particular task."),
            input("program_example", "An example of the program in use."),
            input("program_description", "Summary of the task the program is designed to solve, and how it goes about solving it."),
            input("module", "The module in the program that we want to describe."),
        ],
        outputs: vec![output(
            "module_description",
            "Description of the module's role in the broader program.",
        )],
    }
}

/// Which of `GenerateSingleModuleInstruction`'s optional inputs are present. dspy's
/// `generate_instruction_class` builds the signature field by field behind these flags; the order
/// below is upstream's exact order, which the ChatAdapter renders positionally. `task_demos` and
/// `basic_instruction` are unconditional there, so they carry no flag.
#[derive(Clone, Copy)]
pub(crate) struct InstructionInputs {
    pub dataset_summary: bool,
    pub program_aware: bool,
    pub instruct_history: bool,
    pub tip: bool,
}

/// dspy `generate_instruction_class(...).signature`: the instruction proposer, assembled from the
/// inputs the flags turn on.
pub(crate) fn generate_module_instruction(inputs: InstructionInputs) -> Signature {
    let mut fields = Vec::new();
    if inputs.dataset_summary {
        fields.push(input(
            "dataset_description",
            "A description of the dataset that we are using.",
        ));
    }
    if inputs.program_aware {
        fields.push(input(
            "program_code",
            "Language model program designed to solve a particular task.",
        ));
        fields.push(input("program_description", "Summary of the task the program is designed to solve, and how it goes about solving it."));
        fields.push(input("module", "The module to create an instruction for."));
        fields.push(input(
            "module_description",
            "Description of the module to create an instruction for.",
        ));
    }
    fields.push(input("task_demos", "Example inputs/outputs of our module."));
    if inputs.instruct_history {
        fields.push(input(
            "previous_instructions",
            "Previous instructions we've attempted, along with their associated scores.",
        ));
    }
    fields.push(input("basic_instruction", "Basic instruction."));
    if inputs.tip {
        fields.push(input(
            "tip",
            "A suggestion for how to go about generating the new instruction.",
        ));
    }
    Signature {
        instructions: "Use the information below to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task.".into(),
        inputs: fields,
        outputs: vec![output(
            "proposed_instruction",
            "Propose an instruction that will be used to prompt a Language Model to perform this task.",
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// The dspy-generated fixture a signature is verified against, loaded for a direct field
    /// comparison — the construction here must equal the strings the ChatAdapter conformance renders.
    fn fixture(name: &str) -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/conformance/{name}.json"));
        let text = std::fs::read_to_string(&path).expect("the fixture is committed");
        serde_json::from_str(&text).expect("the fixture parses")
    }

    /// Every field of a constructed signature against its fixture: instructions, and each input and
    /// output name paired with its description. A mis-transcribed byte here fails before it could
    /// ever reach a prompt.
    fn assert_matches(name: &str, signature: &Signature) {
        let fixture = fixture(name);
        assert_eq!(
            signature.instructions,
            fixture["instructions"].as_str().expect("instructions"),
            "{name} instructions"
        );
        let inputs: Vec<(&str, &str)> = signature
            .inputs
            .iter()
            .map(|f| (f.name.as_str(), f.desc.as_str()))
            .collect();
        let expected_inputs: Vec<(&str, &str)> = fixture["inputs"]
            .as_array()
            .expect("inputs")
            .iter()
            .map(|f| {
                (
                    f["name"].as_str().unwrap(),
                    f["desc"].as_str().unwrap_or_default(),
                )
            })
            .collect();
        assert_eq!(inputs, expected_inputs, "{name} inputs");
        let outputs: Vec<(&str, &str)> = signature
            .outputs
            .iter()
            .map(|f| (f.name.as_str(), f.desc.as_str()))
            .collect();
        let expected_outputs: Vec<(&str, &str)> = fixture["outputs"]
            .as_array()
            .expect("outputs")
            .iter()
            .map(|f| {
                (
                    f["name"].as_str().unwrap(),
                    f["desc"].as_str().unwrap_or_default(),
                )
            })
            .collect();
        assert_eq!(outputs, expected_outputs, "{name} outputs");
    }

    #[test]
    fn every_signature_matches_its_dspy_fixture() {
        assert_matches("describe_program", &describe_program());
        assert_matches("describe_module", &describe_module());
        // The fixture is the every-flag-on variant, MIPROv2's default.
        assert_matches(
            "generate_module_instruction",
            &generate_module_instruction(InstructionInputs {
                dataset_summary: true,
                program_aware: true,
                instruct_history: true,
                tip: true,
            }),
        );
    }
}
