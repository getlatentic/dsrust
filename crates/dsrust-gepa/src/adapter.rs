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
    /// gepa 0.1.4's `num_metric_calls`: how many metric calls the evaluation actually made, where
    /// an adapter counts them — a cached example costs none. `None` charges one per example, as
    /// the engine always did.
    pub num_metric_calls: Option<usize>,
}

impl<O> EvalBatch<O> {
    /// An evaluation carrying scores and (for a `capture_traces=true` call) captured traces.
    pub fn traced(scores: Vec<f64>) -> Self {
        Self {
            scores,
            captured_traces: true,
            outputs: None,
            num_metric_calls: None,
        }
    }

    /// A plain scoring evaluation (`capture_traces=false`), as the valset and new-candidate evals do.
    pub fn scored(scores: Vec<f64>) -> Self {
        Self {
            scores,
            captured_traces: false,
            outputs: None,
            num_metric_calls: None,
        }
    }
}

/// Why a reflective proposal produced nothing — the two stages gepa 0.1.4 tells apart, since it
/// answers them differently: a reflective dataset that could not be built skips the task, while a
/// reflection that failed is tried once more, task by task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposalFailure {
    /// dspy's `make_reflective_dataset` raised — *"No valid predictions found for any module."*
    /// among them. gepa: *"Iteration {i}: Exception building reflective dataset: {e}"*.
    ReflectiveDataset(String),
    /// The reflection model failed partway through.
    Reflection(String),
}

impl ProposalFailure {
    pub fn message(&self) -> &str {
        match self {
            Self::ReflectiveDataset(message) | Self::Reflection(message) => message,
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

    /// Replacement text for the components named, or which stage failed and why.
    ///
    /// `Err` is an exception out of dspy's proposal, which gepa catches — its message is the `{e}`
    /// upstream formats into its line. Two reach it: `make_reflective_dataset` raising `"No valid
    /// predictions found for any module."`, and the reflection model failing partway through.
    /// Upstream's `try` wraps each stage whole, so the *first* failure ends it — a second component
    /// is not attempted, and the components already proposed for are discarded with the rest.
    ///
    /// An empty map is not a failure: it is a reflection that ran and proposed nothing for the
    /// components it was asked about. gepa 0.1.4 skips the task on it rather than scoring a
    /// candidate identical to its parent.
    fn propose_new_texts(
        &mut self,
        candidate: &Candidate,
        components: &[String],
        captured: &EvalBatch<Self::Output>,
    ) -> impl Future<Output = Result<Candidate, ProposalFailure>> + Send;
}
