//! GEPA's feedback metric: unlike a plain metric it returns text alongside the score, and GEPA
//! reflects on that text to rewrite an instruction.

/// dspy's `ScoreWithFeedback` — what a GEPA metric returns for one example. The score drives
/// acceptance and the Pareto front (summed over a minibatch, averaged over the valset); the feedback
/// is the text GEPA reflects on. `None` feedback becomes `This trajectory got a score of {score}.`,
/// dspy's fallback in `feedback_fn_creator`.
pub struct Feedback {
    pub score: f64,
    pub feedback: Option<String>,
}

impl Feedback {
    /// A score with explicit feedback text for GEPA to reflect on.
    pub fn new(score: f64, feedback: impl Into<String>) -> Self {
        Self { score, feedback: Some(feedback.into()) }
    }

    /// A bare score; GEPA fills the default feedback (dspy's metric returning a plain float).
    pub fn score_only(score: f64) -> Self {
        Self { score, feedback: None }
    }

    /// The feedback text GEPA reflects on, defaulting to dspy's score sentence.
    pub(super) fn text(&self) -> String {
        self.feedback.clone().unwrap_or_else(|| format!("This trajectory got a score of {}.", python_float(self.score)))
    }
}

/// Format a score the way Python's `str(float)` does inside dspy's fallback feedback: an integral
/// value keeps a trailing `.0` (`str(1.0) == "1.0"`), where Rust's `Display` would drop it. Rust's
/// shortest-round-trip `Display` matches Python's `repr` for the non-integral values GEPA scores use.
fn python_float(score: f64) -> String {
    if score.is_finite() && score == score.trunc() {
        format!("{score:.1}")
    } else {
        format!("{score}")
    }
}
