//! A GEPA run, reported through the observability this crate already has.
//!
//! `dsrust_gepa` takes no logging dependency, so its [`Progress`] default says nothing. This is the
//! subscriber that puts a run on the same `tracing` spine as every model call and module — one
//! event per decision, carrying the values rather than a formatted line, and upstream's own
//! sentence on `message` for a subscriber that wants to print it.

use gepa::progress::{Event, Progress};

/// Reports each decision as a `tracing` event under the `dsrust` target.
///
/// dspy reaches Python's `logging` here. `tracing` is what this crate reaches everywhere else, so
/// a caller already collecting spans gets the optimization run in the same stream rather than in a
/// second one they have to merge.
pub struct Reported;

impl Progress for Reported {
    fn report(&self, event: Event<'_>) {
        // The fields are what a subscriber acts on; `message` is upstream's line, for one that
        // only prints. Recording both is what lets a UI and a log share one event.
        tracing::info!(
            target: "dsrust",
            iteration = event.iteration(),
            decision = decision(&event),
            message = %event.message(),
            "gepa"
        );
    }
}

/// The decision, as one word a subscriber can match on without reading the sentence.
fn decision(event: &Event<'_>) -> &'static str {
    match event {
        Event::Proposed { .. } => "proposed",
        Event::ProposedNothing { .. } => "proposed_nothing",
        Event::NothingToLearnFrom { .. } => "nothing_to_learn_from",
        Event::NoTrajectories { .. } => "no_trajectories",
        Event::ReflectionFailed { .. } => "reflection_failed",
        Event::Rejected { .. } => "rejected",
        Event::Accepted { is_best: true, .. } => "accepted_best",
        Event::Accepted { .. } => "accepted",
        Event::Merged { .. } => "merged",
        Event::NoMergeCandidates { .. } => "no_merge_candidates",
    }
}
