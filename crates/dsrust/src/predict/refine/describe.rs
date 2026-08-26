//! Describing a program to the model that will advise it.
//!
//! `Refine` asks `OfferFeedback` what each predictor should do differently, and that model needs
//! to know what the predictors *are*. dspy's `inspect_modules` renders them: a separator, then
//! each module's input fields, output fields and instructions, indented with tabs.
//!
//! Every byte here is upstream's, including two that reading its code would not suggest. The
//! separator is eighty dashes. And a field line keeps its trailing space after `(str):` except on
//! the last of a group, because upstream strips each group as a whole rather than line by line.
//! Regenerate the reference rather than reasoning about it:
//!
//! ```text
//! .dspy-venv/bin/python -c "import dspy; from dspy.predict.refine import inspect_modules; \
//!   print(repr(inspect_modules(dspy.ChainOfThought('question -> answer'))))"
//! ```

use crate::adapter::prompt::{numbered_input_lines, numbered_output_lines};
use crate::module::NamedPredictor;

/// dspy's `"-" * 80`.
const SEPARATOR_WIDTH: usize = 80;

/// Every predictor in a program, as the advising model reads them.
pub(super) fn modules(predictors: &[NamedPredictor<'_>]) -> String {
    let separator = "-".repeat(SEPARATOR_WIDTH);
    let mut out = vec![separator.clone()];
    for predictor in predictors {
        out.push(format!("Module {}", predictor.name));
        out.push("\tInput Fields:".to_owned());
        out.push(indented(&numbered_input_lines(predictor.signature)));
        out.push("\tOutput Fields:".to_owned());
        out.push(indented(&numbered_output_lines(predictor.signature)));
        out.push(format!(
            "\tOriginal Instructions: {}",
            indented(&predictor.signature.instructions)
        ));
        out.push(separator.clone());
    }
    // Upstream's closing `[o.strip("\n") for o in output]`. Each element is built with the
    // leading newline its join produces, and then loses it — except inside the instructions line,
    // where the newline sits after text and so survives.
    out.iter()
        .map(|part| part.trim_matches('\n'))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A block moved in by two tabs, each line on its own.
///
/// Upstream joins onto an empty first element, so the block starts on the line *after* whatever
/// introduced it — which is why the instructions line ends in a space with nothing after it.
fn indented(block: &str) -> String {
    block.lines().map(|line| format!("\n\t\t{line}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Module;
    use crate::signature::Signature;
    use crate::{ChainOfThought, Predict};

    /// Two predictors, so the rule *between* blocks and the per-predictor names both show — the
    /// field names are the ones the golden's Python program uses, since `named_predictors` reads
    /// them off the struct.
    #[derive(crate::Module)]
    struct Pair {
        draft: Predict,
        settle: Predict,
    }

    impl crate::module::Forward for Pair {
        async fn forward(&self, inputs: crate::Example) -> anyhow::Result<crate::Prediction> {
            self.settle.forward(inputs).await
        }
    }

    /// Byte for byte against dspy's own `inspect_modules` for the same programs.
    ///
    /// This is prompt content, not a debug view: `Refine`'s feedback ask carries it as
    /// `modules_defn`, so a tab that became spaces or a dropped trailing space changes what the
    /// feedback model reads. It was a hand-written expected string until
    /// `scripts/generate_inspect_modules_fixture.py` replaced it — right, as it happened, but a
    /// hand-written expectation agrees with the code it tests by construction and would go on
    /// agreeing after a pin bump moved the format.
    ///
    /// Writing the original found two divergences in `ChainOfThought` itself, both since fixed: it
    /// named its predictor `self` where dspy names it `predict`, and gave `reasoning` a prose
    /// description where dspy gives a sentinel that renders as nothing.
    #[test]
    fn every_program_shape_reads_the_way_upstream_renders_it() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/conformance/predict/inspect_modules.json"
        ))
        .expect("the golden parses");

        for case in fixture["cases"].as_array().expect("cases") {
            let name = case["name"].as_str().expect("a name");
            let theirs = case["rendered"].as_str().expect("dspy's rendering");
            let ours = match name {
                "a_bare_predict" => modules(&Predict!("question -> answer").named_predictors()),
                "a_chain_of_thought" => modules(
                    &ChainOfThought::from_signature(
                        "question -> answer".parse().expect("a signature"),
                    )
                    .named_predictors(),
                ),
                "a_described_field" => {
                    let mut signature: Signature =
                        "question -> answer".parse().expect("a signature");
                    signature.outputs[0].desc = "One word only.".to_owned();
                    modules(&Predict::from_signature(signature).named_predictors())
                }
                "multiline_instructions" => {
                    let signature: Signature = "question -> answer".parse().expect("a signature");
                    let signature = signature
                        .with_instructions("Answer the question.\nBe brief.\n\nNever guess.");
                    modules(&Predict::from_signature(signature).named_predictors())
                }
                "two_predictors" => {
                    let draft: Signature = "question -> draft".parse().expect("a signature");
                    let mut program = Pair {
                        draft: Predict::from_signature(draft.with_instructions("Draft an answer.")),
                        settle: Predict::from_signature(
                            "draft -> answer".parse().expect("a signature"),
                        ),
                    };
                    modules(&program.named_predictors())
                }
                other => panic!("the golden names a case this test does not build: {other}"),
            };
            assert_eq!(ours, theirs, "case {name}");
        }
    }

    /// The separator repeats per module, so two predictors are two blocks rather than one long
    /// one — and the name on each is the one `named_predictors` gave it.
    #[test]
    fn each_predictor_gets_its_own_block() {
        let mut program = Predict!("question -> answer");
        let rendered = modules(&program.named_predictors());
        assert_eq!(rendered.matches(&"-".repeat(SEPARATOR_WIDTH)).count(), 2);
        assert!(rendered.contains("Module self"));
    }
}
