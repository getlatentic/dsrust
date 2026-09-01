//! The engine's boundary to the LLM work (`core/adapter.py`): a candidate is a map from component
//! name to its text, and the adapter evaluates candidates and proposes new component texts. The
//! reflective flow's `make_reflective_dataset` is folded into `propose_new_texts` — the engine only
//! ever calls the two back-to-back, passing the captured evaluation between them.

use std::future::Future;

pub use crate::candidate::Candidate;

/// What the engine reads back from an evaluation: the per-example scores (their sum drives the
/// minibatch accept test; their mean over the valset drives selection and the best program), and
/// whether traces were captured — a `capture_traces=true` evaluation with no traces skips the
/// iteration (`reflective_mutation.py`: "No trajectories captured. Skipping.").
pub struct EvalBatch<O> {
    pub scores: Vec<f64>,
    pub captured_traces: bool,
    /// gepa's `outputs_by_val_id`: what each example's run produced, in the order the scores are
    /// in. `None` unless the caller asked for `track_best_outputs` — an adapter pays to keep these
    /// and nothing reads them otherwise.
    pub outputs: Option<Vec<O>>,
}

impl<O> EvalBatch<O> {
    /// An evaluation carrying scores and (for a `capture_traces=true` call) captured traces.
    pub fn traced(scores: Vec<f64>) -> Self {
        Self {
            scores,
            captured_traces: true,
            outputs: None,
        }
    }

    /// A plain scoring evaluation (`capture_traces=false`), as the valset and new-candidate evals do.
    pub fn scored(scores: Vec<f64>) -> Self {
        Self {
            scores,
            captured_traces: false,
            outputs: None,
        }
    }
}

/// GEPA's `GEPAAdapter`: the system-specific work the engine drives. `evaluate_minibatch` scores a
/// candidate on a trainset subsample (with traces, for reflection), `evaluate_valset` scores it on
/// the whole validation set (dspy's `FullEvaluationPolicy`), and `propose_new_texts` reflects on a
/// captured evaluation to rewrite the given components.
///
/// The methods are async with `Send` futures: a real adapter runs an LLM program and a reflection LM,
/// which in dsrs is async and multi-threaded. The engine awaits each call before the next, so a
/// method may borrow `&mut self` for the duration of its future.
pub trait GepaAdapter {
    /// gepa's `RolloutOutput`: what one example's run produced. Only ever kept when the caller
    /// tracks best outputs, so an adapter with nothing worth reporting can make this `()`.
    type Output: Clone + Send;

    fn evaluate_minibatch(
        &mut self,
        ids: &[usize],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> impl Future<Output = EvalBatch<Self::Output>> + Send;

    fn evaluate_valset(
        &mut self,
        candidate: &Candidate,
    ) -> impl Future<Output = EvalBatch<Self::Output>> + Send;

    /// Score a candidate on the given validation ids only — dspy's `cached_evaluate_full` over a
    /// merge subsample. The returned scores are in the order the ids were given, and the eval is
    /// counted as exactly that many metric calls, not a whole valset. Merge is the only caller.
    fn evaluate_valset_ids(
        &mut self,
        ids: &[usize],
        candidate: &Candidate,
    ) -> impl Future<Output = EvalBatch<Self::Output>> + Send;

    /// Replacement text for the components named, or `None` when there was nothing to reflect on
    /// at all.
    ///
    /// `None` is dspy raising `"No valid predictions found for any module."` from
    /// `make_reflective_dataset`, which gepa catches and turns into a skipped iteration. It is not
    /// the same as answering with an empty map: an empty map is a reflection that ran and proposed
    /// nothing for the components it was asked about, and upstream scores the unchanged candidate
    /// for it. Answering `None` where the map is merely empty spends an extra minibatch evaluation
    /// on a candidate identical to its parent, and `max_metric_calls` is what pays for it.
    fn propose_new_texts(
        &mut self,
        candidate: &Candidate,
        components: &[String],
        captured: &EvalBatch<Self::Output>,
    ) -> impl Future<Output = Option<Candidate>> + Send;
}
