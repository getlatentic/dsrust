//! Watching one evaluation: the run, the rows inside it, and the score folded back on.
//!
//! Split from the other points because the shape is different. A module call or a model call is one
//! event with a beginning and an end; an evaluation is a *container* — every row's call belongs
//! inside it, and getting that wrong reports five hundred separate runs instead of one.

use std::future::Future;

use tracing::{Instrument, field};

use crate::callback::{self, Under};
use crate::evaluate::Pass;
use crate::example::Example;

use super::{TARGET, Watch, opening};

/// dspy `on_evaluate_start`: one whole run over a devset, with every module call it made inside it.
///
/// Upstream decorates `Evaluate.__call__`, and this wraps [`Evaluate::run`](crate::Evaluate::run) —
/// the same method under a different name. The outermost point of an optimizer's search, so a reader
/// filtering to `evaluate` sees one line per scoring pass rather than one per row.
///
/// `pass` is dspy's `callback_metadata`, and it is what separates the passes from each other: a
/// search alternates whole-valset scoring with subsamples, and the two mean different things.
///
/// The devset is handed over whole rather than counted, because upstream's is: `with_callbacks`
/// gives a handler the `inputs` dict of `Evaluate.__call__`, `devset` among its keys. A count is
/// what the span records and what a handler can take for itself.
pub fn evaluating(devset: &[Example], threads: usize, pass: Option<Pass>) -> Watch {
    let watch = opening(tracing::info_span!(
        target: TARGET,
        "evaluate",
        rows = devset.len(),
        threads = threads,
        pass = pass.map(|pass| match pass {
            Pass::Full => "full",
            Pass::Minibatch => "minibatch",
        }),
        inputs = field::Empty,
        outputs = field::Empty,
        error = field::Empty,
    ));
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_evaluate_start(&watch.call, devset, threads, pass)
        });
    }
    watch
}

/// Run an evaluation's rows inside the open point, so every module call they make is a child of it.
///
/// Both halves, as [`watching`](super::watching) does: instrumented on the future rather than entered across an
/// await, which would attribute whatever the runtime polled next to this evaluation, and run under
/// the call id, which is what makes each row's `on_module_start` name this evaluation as its parent.
/// Wrapping only the span left the callbacks reporting every row as an outermost call.
///
/// ```no_run
/// # use dsrust::observe::{evaluated_within, evaluating};
/// # async fn wrapper(devset: Vec<dsrust::Example>) {
/// let watching = evaluating(&devset, 1, None);
/// // Every module call the rows make is a child of this point, so a handler can tell "one
/// // evaluation of five hundred rows" from "five hundred separate runs".
/// let scored = evaluated_within(&watching, async { 42 }).await;
/// assert_eq!(scored, 42);
/// # }
/// ```
pub async fn evaluated_within<T>(watch: &Watch, rows: impl Future<Output = T>) -> T {
    Under::new(watch.call, rows)
        .instrument(watch.span.clone())
        .await
}

/// dspy `on_evaluate_end`: what an evaluation found, or why it gave up.
///
/// Its own function rather than [`watching`](super::watching) because the rows stream *inside* the
/// point rather than resolving as one future the wrapper can await.
///
/// The error arm is upstream's too, and is not a row's failure: a failing row scores
/// `failure_score` and the run carries on. It is the whole run giving up once `max_errors` rows
/// have failed, which upstream raises from `Evaluate.__call__` — so its decorator fires this
/// handler with `outputs=None` and the exception set.
pub fn scored(watch: &Watch, evaluated: Result<&crate::evaluate::Evaluation, &anyhow::Error>) {
    watch.finished(evaluated, describe);
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_evaluate_end(&watch.call, evaluated)
        });
    }
}

/// What the span records for a run that finished.
fn describe(evaluation: &crate::evaluate::Evaluation) -> String {
    format!(
        "{{\"score\":{},\"rows\":{},\"failed\":{}}}",
        evaluation.score,
        evaluation.results.len(),
        evaluation.failure_count(),
    )
}
