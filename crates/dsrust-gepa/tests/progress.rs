//! A run reports what it decided, and reports it only when someone is listening.
//!
//! The engine's conformance tests hold the *search* to the real `gepa` package. Nothing held the
//! reporting, which is the half a caller streaming a run to a user interface actually consumes —
//! and a seam that compiles while emitting nothing is the failure mode a type cannot catch.

use std::sync::{Arc, Mutex};

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

/// Every variant reaches a subscriber, carrying its iteration — which is what orders the stream a
/// caller renders.
#[test]
fn a_subscriber_receives_every_decision() {
    let collected = Arc::new(Collected::default());
    let progress: Arc<dyn Progress> = collected.clone();
    for event in [
        Event::Proposed {
            iteration: 1,
            component: "answer",
            text: "try harder",
        },
        Event::ProposedNothing { iteration: 2 },
        Event::NothingToLearnFrom { iteration: 3 },
        Event::NoTrajectories { iteration: 4 },
        Event::ReflectionFailed {
            iteration: 4,
            error: "No valid predictions found for any module.",
        },
        Event::Rejected {
            iteration: 4,
            before: 2.0,
            after: 1.0,
        },
        Event::Accepted {
            iteration: 5,
            candidate: 1,
            score: 0.9,
            is_best: true,
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
            "nothing_to_learn_from@3",
            "no_trajectories@4",
            "reflection_failed@4",
            "rejected@4",
            "accepted_best@5",
            "merged@6",
            "no_merge_candidates@7",
        ]
    );
}

/// The default says nothing and costs nothing — gepa's `logger=None`.
#[test]
fn the_default_reports_nothing() {
    // Nothing to assert but that it accepts every variant without panicking: a `Silent` that
    // errored would take down a run nobody asked to watch.
    Silent.report(Event::ProposedNothing { iteration: 1 });
    Silent.report(Event::Accepted {
        iteration: 2,
        candidate: 0,
        score: 1.0,
        is_best: false,
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
