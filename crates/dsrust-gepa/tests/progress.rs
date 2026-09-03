//! A run reports what it decided, and reports it only when someone is listening.
//!
//! The engine's conformance tests hold the *search* to the real `gepa` package. Nothing held the
//! reporting, which is the half a caller streaming a run to a user interface actually consumes —
//! and a seam that compiles while emitting nothing is the failure mode a type cannot catch.

use std::sync::{Arc, Mutex};

use gepa::Candidate;
use gepa::progress::Rejection;
use gepa::progress::{Event, Progress, Silent};

#[derive(Default)]
struct Collected {
    seen: Mutex<Vec<String>>,
}

impl Progress for Collected {
    fn report(&self, event: Event<'_>) {
        self.seen.lock().expect("not poisoned").push(format!(
            "{}@{}",
            decision(&event),
            event.iteration()
        ));
    }
}

fn decision(event: &Event<'_>) -> &'static str {
    match event {
        Event::Selected { .. } => "selected",
        Event::Proposed { .. } => "proposed",
        Event::ProposedNothing { .. } => "proposed_nothing",
        Event::NoTextUpdates { .. } => "no_text_updates",
        Event::NothingToLearnFrom { .. } => "nothing_to_learn_from",
        Event::NoTrajectories { .. } => "no_trajectories",
        Event::ReflectiveDatasetFailed { .. } => "reflective_dataset_failed",
        Event::BatchedReflectionFailed { .. } => "batched_reflection_failed",
        Event::ReflectionFailed { .. } => "reflection_failed",
        Event::Rejected { .. } => "rejected",
        Event::AcceptedOnMinibatch { .. } => "accepted_on_minibatch",
        Event::Accepted { is_best: true, .. } => "accepted_best",
        Event::Accepted { .. } => "accepted",
        Event::Merged { .. } => "merged",
        Event::NoMergeCandidates { .. } => "no_merge_candidates",
    }
}

/// Every variant reaches a subscriber, carrying its iteration — which is what orders the stream a
/// caller renders.
#[test]
fn a_subscriber_receives_every_decision() {
    // The winning candidate the event borrows, declared so it outlives the event.
    let won: Candidate = [("step".to_owned(), "Answer it.".to_owned())]
        .into_iter()
        .collect();

    let collected = Arc::new(Collected::default());
    let progress: Arc<dyn Progress> = collected.clone();
    for event in [
        Event::Proposed {
            iteration: 1,
            component: "answer",
            text: "try harder",
        },
        Event::ProposedNothing { iteration: 2 },
        Event::NoTextUpdates { iteration: 2 },
        Event::NothingToLearnFrom {
            iteration: 3,
            parent: 0,
        },
        Event::NoTrajectories {
            iteration: 4,
            parent: 0,
        },
        Event::ReflectiveDatasetFailed {
            iteration: 4,
            error: "No valid predictions found for any module.",
        },
        Event::BatchedReflectionFailed {
            iteration: 4,
            error: "no answer",
        },
        Event::ReflectionFailed {
            iteration: 4,
            error: "no answer",
        },
        Event::Selected {
            iteration: 4,
            candidate: 0,
            score: 0.5,
        },
        Event::Rejected {
            iteration: 4,
            before: 2.0,
            after: 1.0,
            reason: Rejection::NotBetter,
        },
        Event::AcceptedOnMinibatch {
            iteration: 5,
            before: 1.0,
            after: 2.0,
        },
        Event::Accepted {
            iteration: 5,
            candidate: 1,
            score: 0.9,
            is_best: true,
            program: &won,
        },
        Event::Merged {
            iteration: 6,
            first: 1,
            second: 2,
            ancestor: 0,
        },
        Event::NoMergeCandidates { iteration: 7 },
    ] {
        progress.report(event);
    }
    assert_eq!(
        *collected.seen.lock().expect("not poisoned"),
        [
            "proposed@1",
            "proposed_nothing@2",
            "no_text_updates@2",
            "nothing_to_learn_from@3",
            "no_trajectories@4",
            "reflective_dataset_failed@4",
            "batched_reflection_failed@4",
            "reflection_failed@4",
            "selected@4",
            "rejected@4",
            "accepted_on_minibatch@5",
            "accepted_best@5",
            "merged@6",
            "no_merge_candidates@7",
        ]
    );
}

/// The default says nothing and costs nothing — gepa's `logger=None`.
#[test]
fn the_default_reports_nothing() {
    // The winning candidate the event borrows, declared so it outlives the event.
    let won: Candidate = [("step".to_owned(), "Answer it.".to_owned())]
        .into_iter()
        .collect();

    // Nothing to assert but that it accepts every variant without panicking: a `Silent` that
    // errored would take down a run nobody asked to watch.
    Silent.report(Event::ProposedNothing { iteration: 1 });
    Silent.report(Event::Accepted {
        iteration: 2,
        candidate: 0,
        score: 1.0,
        is_best: false,
        program: &won,
    });
}

/// The line is gepa's own, so a subscriber that only prints gets what upstream's log said.
#[test]
fn the_message_is_the_upstream_line() {
    assert_eq!(
        Event::Proposed {
            iteration: 9,
            component: "reasoning",
            text: "be brief",
        }
        .message(),
        "Iteration 9: Proposed new text for reasoning: be brief"
    );
}

/// Every event this crate defines is one the engine can actually report.
///
/// Four of nine were not. `NothingToLearnFrom`, `Proposed`, `Merged` and `NoMergeCandidates` were
/// each defined, rendered by `message()`, matched in this file and in dsrust's `reported.rs` — and
/// never passed to `report`. They could not occur, and a subscriber waiting on one waited forever.
/// Nothing caught it because every test constructed the events instead of running the engine, and
/// a match arm reads exactly the same whether or not the arm is reachable.
///
/// This reads the source rather than driving a run, deliberately: reaching all nine through the
/// engine needs a merge, a rejection and an acceptance in one run, and a test that elaborate would
/// be the thing that rots. What it costs is that a variant reported from dead code would still pass
/// — so it is a floor, not a proof.
#[test]
fn every_event_the_crate_defines_is_reported_somewhere() {
    let progress = include_str!("../src/progress.rs");
    let engine = include_str!("../src/engine.rs");

    // Both enums, and each stopped at its own closing brace: `Rejection` follows `Event` in the
    // same file, so a scan that only skips *to* `pub enum Event` reads its variants as Event's and
    // reports them missing. Reading both is better than stopping at one — a rejection reason
    // nothing constructs is as unreachable as an event nothing reports.
    let variants = |declaration: &str| -> Vec<String> {
        progress
            .lines()
            .skip_while(|line| !line.starts_with(declaration))
            .skip(1)
            .take_while(|line| !line.starts_with('}'))
            .filter_map(|line| {
                let name = line.strip_prefix("    ")?;
                let name = name.split([' ', '{', '(', ',']).next()?;
                name.starts_with(|c: char| c.is_ascii_uppercase())
                    .then(|| name.to_owned())
            })
            .collect()
    };
    let events = variants("pub enum Event");
    let reasons = variants("pub enum Rejection");
    assert!(
        events.len() >= 9,
        "only found {events:?} — the parse of the enum has drifted"
    );
    assert!(
        reasons.len() >= 3,
        "only found {reasons:?} — the parse of the enum has drifted"
    );
    let defined: Vec<String> = events
        .into_iter()
        .map(|name| format!("Event::{name}"))
        .chain(reasons.into_iter().map(|name| format!("Rejection::{name}")))
        .collect();

    for variant in defined {
        // `Event::Proposed` is a prefix of `Event::ProposedNothing`, so a substring test reports a
        // variant as reachable because a *different* one is — which is how the first version of
        // this test passed while `Proposed` was still unreachable.
        let reported = engine.match_indices(&variant).any(|(at, matched)| {
            engine[at + matched.len()..]
                .chars()
                .next()
                .is_none_or(|next| !next.is_alphanumeric() && next != '_')
        });
        assert!(
            reported,
            "`{variant}` is defined and the engine never constructs it, so it cannot occur"
        );
    }
}
