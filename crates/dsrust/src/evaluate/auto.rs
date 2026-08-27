//! dspy's built-in LM judges: score a free-text answer without a string match.
//!
//! `evaluate/auto_evaluation.py`. Four signatures a [`ChainOfThought`] asks, one arithmetic
//! function, and two metrics that combine the numbers — [`SemanticF1`], which most dspy programs
//! reach for, and [`CompleteAndGrounded`], which scores an answer against retrieved context.
//!
//! The instructions are lifted from a class docstring, so their line breaks are
//! `inspect.cleandoc`'s rather than anybody's typing, and every field's description is prompt text
//! a model reads. All four are held to `evaluate/auto_evaluation.json`, recorded by rendering the
//! real signatures through the real adapter.
//!
//! These are the reason [`Metric`] is a trait: they call a model to score, so a
//! metric bound that could not await would have had them as values no caller could pass to
//! [`Evaluate`](super::Evaluate) or to any optimizer.
//!
//! [`Metric`]: super::Metric

use std::pin::Pin;

use anyhow::Result;

use super::Metric;
use crate::Module;
use crate::example::{Example, Prediction};
use crate::predict::ChainOfThought;
use crate::signature::{FieldKind, InField, OutField, Signature};

/// dspy's default `threshold=0.66`: the score an optimizer's trace pass must reach to accept.
pub const DEFAULT_THRESHOLD: f64 = 0.66;

fn asked(name: &str) -> InField {
    InField {
        name: name.to_owned(),
        ..Default::default()
    }
}

fn answered(name: &str, desc: &str, kind: FieldKind) -> OutField {
    OutField {
        name: name.to_owned(),
        desc: desc.to_owned(),
        kind,
        ..Default::default()
    }
}

const RECALL: &str = "fraction (out of 1.0) of ground truth covered by the system response";
const PRECISION: &str = "fraction (out of 1.0) of system response covered by the ground truth";
const GROUND_TRUTH_IDEAS: &str = "enumeration of key ideas in the ground truth";
const RESPONSE_IDEAS: &str = "enumeration of key ideas in the system response";
const OVERLAP: &str = "discussion of the overlap between ground truth and system response";

/// dspy `SemanticRecallPrecision`: recall and precision in one ask.
pub fn semantic_recall_precision() -> Signature {
    Signature {
        instructions: "Compare a system's response to the ground truth to compute its recall and \
                       precision.\nIf asked to reason, enumerate key ideas in each response, and \
                       whether they are present in the other response."
            .into(),
        inputs: vec![
            asked("question"),
            asked("ground_truth"),
            asked("system_response"),
        ],
        outputs: vec![
            answered("recall", RECALL, FieldKind::Float),
            answered("precision", PRECISION, FieldKind::Float),
        ],
    }
}

/// dspy `DecompositionalSemanticRecallPrecision`: the same numbers, after the model has been made
/// to enumerate and compare the key ideas first.
pub fn decompositional_semantic_recall_precision() -> Signature {
    Signature {
        instructions: "Compare a system's response to the ground truth to compute recall and \
                       precision of key ideas.\nYou will first enumerate key ideas in each \
                       response, discuss their overlap, and then report recall and precision."
            .into(),
        inputs: vec![
            asked("question"),
            asked("ground_truth"),
            asked("system_response"),
        ],
        outputs: vec![
            answered("ground_truth_key_ideas", GROUND_TRUTH_IDEAS, FieldKind::Str),
            answered("system_response_key_ideas", RESPONSE_IDEAS, FieldKind::Str),
            answered("discussion", OVERLAP, FieldKind::Str),
            answered("recall", RECALL, FieldKind::Float),
            answered("precision", PRECISION, FieldKind::Float),
        ],
    }
}

/// dspy `AnswerCompleteness`: how much of the ground truth the answer covers.
pub fn answer_completeness() -> Signature {
    Signature {
        instructions: "Estimate the completeness of a system's responses, against the ground \
                       truth.\nYou will first enumerate key ideas in each response, discuss their \
                       overlap, and then report completeness."
            .into(),
        inputs: vec![
            asked("question"),
            asked("ground_truth"),
            asked("system_response"),
        ],
        outputs: vec![
            answered("ground_truth_key_ideas", GROUND_TRUTH_IDEAS, FieldKind::Str),
            answered("system_response_key_ideas", RESPONSE_IDEAS, FieldKind::Str),
            answered("discussion", OVERLAP, FieldKind::Str),
            answered("completeness", RECALL, FieldKind::Float),
        ],
    }
}

/// dspy `AnswerGroundedness`: how much of the answer the retrieved context supports.
pub fn answer_groundedness() -> Signature {
    Signature {
        instructions: "Estimate the groundedness of a system's responses, against real retrieved \
                       documents written by people.\nYou will first enumerate whatever \
                       non-trivial or check-worthy claims are made in the system response, and \
                       then\ndiscuss the extent to which some or all of them can be deduced from \
                       the retrieved context and basic commonsense."
            .into(),
        inputs: vec![
            asked("question"),
            asked("retrieved_context"),
            asked("system_response"),
        ],
        outputs: vec![
            answered(
                "system_response_claims",
                "enumeration of non-trivial or check-worthy claims in the system response",
                FieldKind::Str,
            ),
            answered(
                "discussion",
                "discussion of how supported the claims are by the retrieved context",
                FieldKind::Str,
            ),
            answered(
                "groundedness",
                "fraction (out of 1.0) of system response supported by the retrieved context",
                FieldKind::Float,
            ),
        ],
    }
}

/// dspy `f1_score`: the harmonic mean, with **both arguments clamped to `[0, 1]` first**.
///
/// The clamp is the half a reading skips: a model answering `precision=1.4` scores as 1.0 rather
/// than pushing the mean above one, and a negative answer becomes 0.0 rather than a negative score.
/// Two zeros are 0.0 by an explicit guard, not by dividing.
pub fn f1_score(precision: f64, recall: f64) -> f64 {
    let precision = precision.clamp(0.0, 1.0);
    let recall = recall.clamp(0.0, 1.0);
    match precision + recall == 0.0 {
        true => 0.0,
        false => 2.0 * (precision * recall) / (precision + recall),
    }
}

/// The fields a judge reads off an example or a prediction — dspy's `example.question`,
/// `example.response`, `pred.response` and `pred.context`, read by name. A field the caller's
/// dataset does not carry reads as empty, as `getattr` on a `dspy.Example` without it would raise
/// and upstream's judges simply assume the three names.
fn field(fields: &Example, name: &str) -> String {
    fields
        .get(name)
        .map(|value| match value {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default()
}

/// dspy `SemanticF1`: the F1 of an LM-judged recall and precision.
///
/// `decompositional` picks the signature that makes the model enumerate the key ideas before
/// scoring, which is upstream's second mode and a different prompt rather than a different metric.
pub struct SemanticF1 {
    judge: ChainOfThought,
    /// dspy's `threshold`. Read only by [`accepts`](Self::accepts), which is what upstream's
    /// `trace is not None` arm answers with.
    pub threshold: f64,
}

impl SemanticF1 {
    pub fn new() -> Self {
        Self {
            judge: ChainOfThought::from_signature(semantic_recall_precision()),
            threshold: DEFAULT_THRESHOLD,
        }
    }

    /// dspy's `decompositional=True`.
    pub fn decompositional() -> Self {
        Self {
            judge: ChainOfThought::from_signature(decompositional_semantic_recall_precision()),
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// The score, or an error the caller can see — where [`Metric::score`] must answer with a
    /// number and reports a failed judgement as `0.0`, which is what a metric raising inside
    /// dspy's `Evaluate` scores as under its own `failure_score`.
    pub async fn judge(&self, example: &Example, prediction: &Prediction) -> Result<f64> {
        let asked = Example::new([
            ("question", field(example, "question").into()),
            ("ground_truth", field(example, "response").into()),
            (
                "system_response",
                field(&prediction.example, "response").into(),
            ),
        ]);
        let scored = self.judge.forward(asked).await?;
        Ok(f1_score(
            number(&scored, "precision"),
            number(&scored, "recall"),
        ))
    }

    /// dspy's `trace is not None` arm: during an optimizer's bootstrapping a metric answers a
    /// *bool*, and this is that question.
    pub fn accepts(&self, score: f64) -> bool {
        score >= self.threshold
    }
}

impl Default for SemanticF1 {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for SemanticF1 {
    fn score<'a>(
        &'a self,
        example: &'a Example,
        prediction: &'a Prediction,
    ) -> Pin<Box<dyn std::future::Future<Output = f64> + Send + 'a>> {
        Box::pin(async move { self.judge(example, prediction).await.unwrap_or(0.0) })
    }
}

/// dspy `CompleteAndGrounded`: completeness against the ground truth and groundedness against the
/// retrieved context, combined by the same harmonic mean.
///
/// Upstream calls `f1_score(groundedness, completeness)` in that order. The mean is symmetric so
/// the number is the same either way, and the order is kept because it is the one upstream wrote.
pub struct CompleteAndGrounded {
    completeness: ChainOfThought,
    groundedness: ChainOfThought,
    pub threshold: f64,
}

impl CompleteAndGrounded {
    pub fn new() -> Self {
        Self {
            completeness: ChainOfThought::from_signature(answer_completeness()),
            groundedness: ChainOfThought::from_signature(answer_groundedness()),
            threshold: DEFAULT_THRESHOLD,
        }
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    pub async fn judge(&self, example: &Example, prediction: &Prediction) -> Result<f64> {
        let complete = self
            .completeness
            .forward(Example::new([
                ("question", field(example, "question").into()),
                ("ground_truth", field(example, "response").into()),
                (
                    "system_response",
                    field(&prediction.example, "response").into(),
                ),
            ]))
            .await?;
        let grounded = self
            .groundedness
            .forward(Example::new([
                ("question", field(example, "question").into()),
                (
                    "retrieved_context",
                    field(&prediction.example, "context").into(),
                ),
                (
                    "system_response",
                    field(&prediction.example, "response").into(),
                ),
            ]))
            .await?;
        Ok(f1_score(
            number(&grounded, "groundedness"),
            number(&complete, "completeness"),
        ))
    }

    pub fn accepts(&self, score: f64) -> bool {
        score >= self.threshold
    }
}

impl Default for CompleteAndGrounded {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for CompleteAndGrounded {
    fn score<'a>(
        &'a self,
        example: &'a Example,
        prediction: &'a Prediction,
    ) -> Pin<Box<dyn std::future::Future<Output = f64> + Send + 'a>> {
        Box::pin(async move { self.judge(example, prediction).await.unwrap_or(0.0) })
    }
}

/// A judged field as the number it is meant to be. A model that answered with prose scores zero
/// for that half rather than failing the row, which is what a `float` field coerced by upstream's
/// parser does with an unparseable answer once the adapter has already accepted it.
fn number(prediction: &Prediction, name: &str) -> f64 {
    prediction
        .get(name)
        .and_then(|value| match value {
            serde_json::Value::Number(number) => number.as_f64(),
            serde_json::Value::String(text) => text.trim().parse().ok(),
            _ => None,
        })
        .unwrap_or(0.0)
}
