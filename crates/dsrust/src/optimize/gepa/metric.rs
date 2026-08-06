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
        Self {
            score,
            feedback: Some(feedback.into()),
        }
    }

    /// A bare score; GEPA fills the default feedback (dspy's metric returning a plain float).
    pub fn score_only(score: f64) -> Self {
        Self {
            score,
            feedback: None,
        }
    }

    /// The feedback text GEPA reflects on, defaulting to dspy's score sentence.
    pub(super) fn text(&self) -> String {
        self.feedback.clone().unwrap_or_else(|| {
            format!(
                "This trajectory got a score of {}.",
                python_float(self.score)
            )
        })
    }
}

/// Format a score the way Python's `str(float)` does inside dspy's fallback feedback: an integral
/// value keeps a trailing `.0` (`str(1.0) == "1.0"`), where Rust's `Display` would drop it. Rust's
/// shortest-round-trip `Display` matches Python's `repr` for the non-integral values GEPA scores use.
fn python_float(score: f64) -> String {
    if score.is_nan() {
        // Rust's Display says `NaN`; Python's str says `nan`, and this string reaches a prompt.
        return "nan".to_owned();
    }
    if score.is_finite() && score == score.trunc() {
        format!("{score:.1}")
    } else {
        format!("{score}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python's `str(float)`, which is what the fallback feedback sentence carries: integral
    /// values keep `.0`, non-integral print shortest-round-trip, and the non-finite spellings are
    /// Python's — `inf` matches Rust's Display by luck, `nan` does not and was `NaN` until this.
    #[test]
    fn scores_print_the_way_python_str_prints_them() {
        assert_eq!(python_float(1.0), "1.0");
        assert_eq!(python_float(0.0), "0.0");
        assert_eq!(python_float(-3.0), "-3.0");
        assert_eq!(python_float(0.5), "0.5");
        assert_eq!(python_float(f64::INFINITY), "inf");
        assert_eq!(python_float(f64::NEG_INFINITY), "-inf");
        assert_eq!(python_float(f64::NAN), "nan");
    }

    /// The fallback sentence itself, byte for byte, and the caller's own feedback verbatim when
    /// there is one. Both replacement mutants of `text` survived because nothing read it.
    #[test]
    fn feedback_text_is_the_callers_or_dspys_score_sentence() {
        let scored = Feedback {
            score: 1.0,
            feedback: None,
        };
        assert_eq!(scored.text(), "This trajectory got a score of 1.0.");
        let spoken = Feedback {
            score: 0.25,
            feedback: Some("wrong city".to_owned()),
        };
        assert_eq!(spoken.text(), "wrong city");
    }
}
