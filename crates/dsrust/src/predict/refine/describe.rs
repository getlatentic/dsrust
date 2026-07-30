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
    use crate::{ChainOfThought, Predict};

    /// Byte for byte against dspy 3.2.1's `inspect_modules` for the same program. The command in
    /// this module's header regenerates it.
    ///
    /// Writing this found two divergences in `ChainOfThought` itself, both since fixed: it named
    /// its predictor `self` where dspy names it `predict`, and gave `reasoning` a prose
    /// description where dspy gives a sentinel that renders as nothing.
    #[test]
    fn a_chain_of_thought_reads_the_way_upstream_renders_it() {
        let mut program =
            ChainOfThought::from_signature("question -> answer".parse().expect("a signature"));
        let rendered = modules(&program.named_predictors());

        assert_eq!(
            rendered,
            "--------------------------------------------------------------------------------\n\
             Module predict\n\
             \tInput Fields:\n\
             \t\t1. `question` (str):\n\
             \tOutput Fields:\n\
             \t\t1. `reasoning` (str): \n\
             \t\t2. `answer` (str):\n\
             \tOriginal Instructions: \n\
             \t\tGiven the fields `question`, produce the fields `answer`.\n\
             --------------------------------------------------------------------------------"
        );
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
