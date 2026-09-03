//! The arm `bootstrap_trace_data` takes when a forward ends in a parse failure.
//!
//! Its own file because it is the one place a run that produced no answer still produces a
//! trajectory — and because it has two arms that behave differently enough to be read together.

use super::reflecting::Captured;
use crate::example::{Example, Prediction};
use crate::module::TraceStep;
use crate::optimize::gepa::metric::Feedback;

/// What the caller wants kept from this run, which is the only state the arm reads.
#[derive(Clone, Copy)]
pub(super) struct Capture {
    pub(super) traces: bool,
    pub(super) track_best_outputs: bool,
}

/// A run that ended in a parse failure, as `bootstrap_trace_data` treats one.
///
/// dspy does not lose the example. It keeps the steps recorded before the failure, appends a
/// `FailedPrediction` carrying the text nobody could read, and scores it with the format
/// reward — so a program that answered unparseably still has something to reflect on.
///
/// Except on one arm, which is why the two are told apart here. Upstream grades the reward by
/// how much of the answer did parse:
///
/// ```text
/// format_failure_score + (failure_score - format_failure_score) * (present / expected)
/// ```
///
/// where `present` and `expected` are both `list(...)`, so the expression raises `TypeError`
/// whenever a declared field *was* parsed. `Evaluate` swallows it, `bootstrap_trace_data`
/// cannot unpack the result, and with GEPA's `raise_on_error=False` the example is dropped
/// from the batch entirely. That is reproduced as the behaviour it is — the example is
/// dropped — rather than as the crash that causes it; `optimize/failed_parse.json` records a
/// batch of three coming back empty. If upstream repairs the arithmetic the golden goes stale
/// and says so.
///
/// Reaching that arm needs the adapter fallback: `ChatAdapter` catches its own parse error and
/// retries through `JSONAdapter`, so what escapes is the fallback's, and `parsed_result` is
/// non-empty only when the retry found some of the declared fields.
pub(super) fn did_not_parse(
    example: &Example,
    inputs: Example,
    mut trace: Vec<TraceStep>,
    error: &anyhow::Error,
    predictors: &[(String, crate::signature::Signature)],
    failure_score: f64,
    capture: Capture,
) -> (Option<f64>, Option<Captured>, Option<Prediction>) {
    let mismatch = error.downcast_ref::<crate::adapter::parse::FieldMismatch>();
    let some_field_parsed = mismatch.is_some_and(|mismatch| {
        mismatch
            .parsed
            .as_object()
            .is_some_and(|parsed| !parsed.is_empty())
    });
    let failed = Prediction::new(Example::default(), String::new());
    if some_field_parsed {
        // Dropped: no trajectory, no score, no output. The batch comes back shorter.
        return (None, None, None);
    }
    let unparsed = crate::FailedPrediction {
        completion_text: mismatch.map_or_else(|| error.to_string(), |m| m.lm_response.clone()),
        // GEPA passes `format_failure_score=failure_score`, so the two are one number
        // here. `FailedPrediction::score` still reads it through Python's `or`, because that
        // is where a zero would be discarded.
        format_reward: Some(failure_score),
    };
    let score = unparsed.score(failure_score);
    // The step upstream appends: the predictor whose signature failed, the inputs it was
    // given, and the completion in place of an answer. Upstream finds that predictor by
    // comparing signatures and raises when none matches; `predictors` is the same walk, taken
    // once by the caller because it needs `&mut` and this does not.
    if let Some(mismatch) = mismatch {
        let named = predictors
            .iter()
            .find(|(_, signature)| signature.equals(&mismatch.signature));
        if let Some((name, signature)) = named {
            trace.push(TraceStep {
                predictor: name.clone(),
                inputs,
                outputs: crate::StepOutputs::Unparsed(unparsed.clone()),
                signature: signature.clone(),
            });
        }
    }
    let captured = capture.traces.then(|| Captured {
        example: example.clone(),
        prediction: failed.clone(),
        trace,
        scored: Feedback::score_only(score),
        unparsed: Some(unparsed),
    });
    let answered = capture.track_best_outputs.then_some(failed);
    (Some(score), captured, answered)
}
