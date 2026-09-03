//! dspy `MultiModalInstructionProposer`: GEPA's proposer for components whose examples carry media.
//!
//! The default reflection prompt stringifies every value into one block of text, which turns an
//! image into its serialized marker and asks a model to improve an instruction from a wall of
//! base64. This proposer replaces each custom type with a numbered placeholder and sends the
//! objects *alongside* the text, so the reflection model sees the picture it is being asked about.
//!
//! Everything here is prompt bytes: the signature's own instructions, the markdown
//! ([`rendering`]), and the keyword summary prepended to it ([`patterns`]). All three are held to
//! `optimize/multimodal_proposal.json`.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use gepa::Candidate;
use serde_json::Value;

use super::proposer::{InstructionProposer, ReflectiveDataset};
use crate::example::Example;
use crate::lm::DynChatModel;
use crate::module::Module;
use crate::predict::Predict;
use crate::signature::{InField, OutField, Signature};

mod patterns;
mod rendering;

/// dspy `MultiModalInstructionProposer`.
///
/// ```
/// use std::sync::Arc;
/// use dsrust::optimize::{Feedback, MetricContext, MultiModalInstructionProposer};
/// use dsrust::{Example, GEPA, Prediction};
///
/// let metric = |_: &Example, _: &Prediction, _: &MetricContext<'_>| Feedback::score_only(1.0);
/// let gepa = GEPA::new(&metric, Arc::new(dsrust::DummyLM::new([])))
///     .instruction_proposer(Arc::new(MultiModalInstructionProposer::new()));
/// # let _ = gepa;
/// ```
pub struct MultiModalInstructionProposer {
    signature: Signature,
}

impl Default for MultiModalInstructionProposer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiModalInstructionProposer {
    pub fn new() -> Self {
        Self {
            signature: signature(),
        }
    }

    /// One component: the markdown, the media, and what the reflection model answered.
    ///
    /// The value handed to `examples_with_feedback` is the text alone when nothing multimodal was
    /// found, and otherwise a list whose head is the text and whose tail is every custom type in
    /// the order the examples met them — which is how the adapter splits them into content blocks.
    async fn propose_one(
        &self,
        reflection: &Arc<dyn DynChatModel>,
        current: &str,
        dataset: &ReflectiveDataset,
    ) -> Option<String> {
        let (markdown, media) = rendering::formatted(dataset);
        let examples = match patterns::summary(dataset) {
            Some(summary) => format!("{summary}\n\n{markdown}"),
            None => markdown,
        };
        let value = match media.is_empty() {
            true => Value::String(examples),
            false => Value::Array(
                std::iter::once(Value::String(examples))
                    .chain(media.into_iter().map(Value::String))
                    .collect(),
            ),
        };
        let prediction = Predict::from_signature(self.signature.clone())
            .set_lm(reflection.clone())
            .forward(Example::new([
                ("current_instruction", Value::String(current.to_owned())),
                ("examples_with_feedback", value),
            ]))
            .await
            .ok()?;
        prediction
            .example
            .get("improved_instruction")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }
}

impl InstructionProposer for MultiModalInstructionProposer {
    /// One proposal per component that is named in *both* the candidate and the datasets.
    ///
    /// Upstream's `if component_name in candidate and component_name in reflective_dataset` — a
    /// component missing from either is left with the text it had, rather than proposed for blind.
    fn propose<'a>(
        &'a self,
        reflection: &'a Arc<dyn DynChatModel>,
        candidate: &'a Candidate,
        components: &'a [String],
        datasets: &'a BTreeMap<String, ReflectiveDataset>,
    ) -> Pin<Box<dyn Future<Output = Candidate> + Send + 'a>> {
        Box::pin(async move {
            let mut proposed = Candidate::new();
            for name in components {
                let (Some(current), Some(dataset)) = (candidate.get(name), datasets.get(name))
                else {
                    continue;
                };
                if let Some(improved) = self.propose_one(reflection, current, dataset).await {
                    proposed.insert(name.clone(), improved);
                }
            }
            proposed
        })
    }
}

/// dspy `GenerateEnhancedMultimodalInstructionFromFeedback`, whose docstring is the prompt.
fn signature() -> Signature {
    Signature {
        instructions: INSTRUCTIONS.to_owned(),
        inputs: vec![
            InField {
                name: "current_instruction".to_owned(),
                desc: CURRENT_INSTRUCTION_DESC.to_owned(),
                ..Default::default()
            },
            InField {
                name: "examples_with_feedback".to_owned(),
                desc: EXAMPLES_DESC.to_owned(),
                ..Default::default()
            },
        ],
        outputs: vec![OutField {
            name: "improved_instruction".to_owned(),
            desc: IMPROVED_DESC.to_owned(),
            ..Default::default()
        }],
    }
}

/// Held to the golden rather than retyped: the class docstring, through `inspect.cleandoc`.
const INSTRUCTIONS: &str = include_str!("multimodal/instructions.txt");

const CURRENT_INSTRUCTION_DESC: &str =
    "The current instruction that was provided to the assistant to perform the multimodal task";

const EXAMPLES_DESC: &str = "Task examples with visual content showing inputs, assistant outputs, \
    and feedback. Pay special attention to feedback about visual analysis accuracy, \
    visual-textual integration, and any domain-specific visual knowledge that the assistant \
    missed.";

const IMPROVED_DESC: &str = "A better instruction for the assistant that addresses visual analysis \
    issues, provides clear guidance on how to process and integrate visual and textual \
    information, includes necessary visual domain knowledge, and prevents the visual analysis \
    mistakes shown in the examples.";

#[cfg(test)]
mod tests {
    use super::*;
    use gepa::Reflective;

    fn golden() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/conformance/optimize/multimodal_proposal.json"
        ))
        .expect("the multimodal golden is valid JSON")
    }

    /// A serialized `Image`, which is what a custom type looks like once it is in a field.
    fn image() -> Reflective {
        Reflective::Text(crate::adapter::types::base::serialized(
            &crate::Image::new(PIXEL).expect("a data url"),
        ))
    }

    const PIXEL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    fn text(value: &str) -> Reflective {
        Reflective::Text(value.to_owned())
    }

    fn map(entries: Vec<(&str, Reflective)>) -> Reflective {
        Reflective::Map(
            entries
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value))
                .collect(),
        )
    }

    fn sample(entries: Vec<(&str, Reflective)>) -> Vec<(String, Reflective)> {
        entries
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .collect()
    }

    /// The datasets the golden recorded, rebuilt here — the Python side holds objects and this side
    /// holds their serialized form, which is the same text either way.
    fn dataset(name: &str) -> ReflectiveDataset {
        match name {
            "text_only" => vec![sample(vec![
                ("Inputs", map(vec![("question", text("what?"))])),
                ("Generated Outputs", map(vec![("answer", text("this"))])),
                ("Feedback", text("fine")),
            ])],
            "one_image_in_the_inputs" => vec![sample(vec![
                (
                    "Inputs",
                    map(vec![
                        ("question", text("what is this?")),
                        ("photo", image()),
                    ]),
                ),
                ("Generated Outputs", map(vec![("answer", text("a pixel"))])),
                ("Feedback", text("incorrect, the colour is wrong")),
            ])],
            "images_across_two_examples_share_one_total" => vec![
                sample(vec![
                    ("Inputs", map(vec![("a", image()), ("b", image())])),
                    ("Generated Outputs", map(vec![("answer", text("two"))])),
                    ("Feedback", text("good")),
                ]),
                sample(vec![
                    ("Inputs", map(vec![("c", image())])),
                    ("Generated Outputs", map(vec![("answer", text("one"))])),
                    ("Feedback", text("wrong")),
                ]),
            ],
            "nesting_is_capped_at_heading_six" => vec![sample(vec![
                (
                    "Inputs",
                    map(vec![(
                        "a",
                        map(vec![(
                            "b",
                            map(vec![(
                                "c",
                                map(vec![(
                                    "d",
                                    map(vec![(
                                        "e",
                                        map(vec![("f", map(vec![("g", text("deep"))]))]),
                                    )]),
                                )]),
                            )]),
                        )]),
                    )]),
                ),
                ("Generated Outputs", map(vec![("answer", text("x"))])),
                ("Feedback", text("fine")),
            ])],
            "lists_and_tuples_are_numbered_items" => vec![sample(vec![
                (
                    "Inputs",
                    map(vec![
                        ("passages", Reflective::List(vec![text("one"), text("two")])),
                        ("pair", Reflective::List(vec![text("left"), text("right")])),
                    ]),
                ),
                ("Generated Outputs", map(vec![("answer", text("x"))])),
                ("Feedback", text("fine")),
            ])],
            "empty_containers_still_emit_a_line" => vec![sample(vec![
                (
                    "Inputs",
                    map(vec![
                        ("nothing", Reflective::Map(Vec::new())),
                        ("none_either", Reflective::List(Vec::new())),
                    ]),
                ),
                ("Generated Outputs", Reflective::Map(Vec::new())),
                ("Feedback", text("fine")),
            ])],
            "a_leaf_is_stripped" => vec![sample(vec![
                ("Inputs", map(vec![("padded", text("   spaced out   \n"))])),
                (
                    "Generated Outputs",
                    map(vec![("answer", text("\n\ntrailing\n\n"))]),
                ),
                ("Feedback", text("fine")),
            ])],
            "outputs_can_be_a_bare_string" => vec![sample(vec![
                ("Inputs", map(vec![("question", text("what?"))])),
                ("Generated Outputs", text("Couldn't parse the output.\n")),
                ("Feedback", text("failed to parse")),
            ])],
            feedback => vec![sample(vec![
                ("Inputs", map(vec![("q", text("x"))])),
                ("Generated Outputs", map(vec![("answer", text("y"))])),
                ("Feedback", text(feedback_for(feedback))),
            ])],
        }
    }

    fn feedback_for(case: &str) -> &'static str {
        match case {
            "one_feedback_counts_in_every_bucket" => {
                "incorrect, though it read well given the context"
            }
            "no_keyword_means_no_summary" => "hm",
            "keywords_are_matched_case_insensitively" => "WRONG",
            "a_substring_match_counts" => "wellington",
            other => panic!("no dataset for {other}"),
        }
    }

    #[test]
    fn every_recorded_example_renders_the_same_markdown() {
        let golden = golden();
        let cases = golden["cases"].as_array().expect("cases");
        assert!(cases.len() >= 12, "the golden lost cases: {}", cases.len());
        for case in cases {
            let name = case["name"].as_str().expect("a name");
            let (rendered, media) = rendering::formatted(&dataset(name));
            let rendered = match patterns::summary(&dataset(name)) {
                Some(summary) => format!("{summary}\n\n{rendered}"),
                None => rendered,
            };
            assert_eq!(
                rendered,
                case["formatted"].as_str().expect("formatted"),
                "the markdown for {name}"
            );
            let expected: usize = case["images_per_example"]
                .as_object()
                .expect("images_per_example")
                .values()
                .map(|n| n.as_u64().expect("a count") as usize)
                .sum();
            assert_eq!(media.len(), expected, "the media found in {name}");
        }
    }

    /// The signature is prompt too, and every word of it was recorded rather than retyped.
    #[test]
    fn the_signature_is_the_one_dspy_declares() {
        let golden = golden();
        let signature = super::signature();
        assert_eq!(
            signature.instructions,
            golden["signature"]["instructions"]
                .as_str()
                .expect("instructions"),
            "the class docstring"
        );
        let recorded = |side: &str| -> Vec<(String, String)> {
            golden["signature"][side]
                .as_array()
                .expect("a side")
                .iter()
                .map(|field| {
                    (
                        field["name"].as_str().expect("a name").to_owned(),
                        field["desc"].as_str().unwrap_or_default().to_owned(),
                    )
                })
                .collect()
        };
        assert_eq!(
            signature
                .inputs
                .iter()
                .map(|field| (field.name.clone(), field.desc.clone()))
                .collect::<Vec<_>>(),
            recorded("inputs")
        );
        assert_eq!(
            signature
                .outputs
                .iter()
                .map(|field| (field.name.clone(), field.desc.clone()))
                .collect::<Vec<_>>(),
            recorded("outputs")
        );
    }
}
