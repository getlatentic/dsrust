//! What a run says while it happens — gepa's `LoggerProtocol`, as an event rather than a line.
//!
//! Upstream reports through `logger.log(f"Iteration {i}: …")`: a formatted string, and dspy passes
//! an adapter that hands it to Python's `logging`. That is enough for a terminal and not enough for
//! anything else — a caller streaming a run to a user interface has to parse the numbers back out
//! of prose that was never meant to be read by a program.
//!
//! So the seam is the same and the payload is not: one report per decision, carrying the values
//! upstream formats into its line. [`Event::message`] renders upstream's own text from them, so
//! nothing is lost to a caller that only wants the line.
//!
//! Nothing is reported unless a caller asks: the default is a no-op, and this crate takes no
//! logging dependency to provide it.

/// One decision, as it is made.
///
/// Every variant corresponds to a `logger.log` upstream reaches, and carries what that line
/// formats — so a subscriber can act on the numbers and still print the sentence.
#[derive(Debug, Clone, PartialEq)]
pub enum Event<'a> {
    /// The reflection proposed replacement text for one component. gepa: *"Iteration {i}: Proposed
    /// new text for {name}: {text}"*.
    Proposed {
        iteration: i64,
        component: &'a str,
        text: &'a str,
    },
    /// The reflection ran and produced nothing. gepa: *"Iteration {i}: Reflective mutation did not
    /// propose a new candidate"*.
    ProposedNothing { iteration: i64 },
    /// Every sampled score was already perfect, so there was nothing to reflect on. gepa:
    /// *"Iteration {i}: All subsample scores perfect. Skipping."*
    NothingToLearnFrom { iteration: i64 },
    /// The proposal scored no better than its parent on the minibatch and was dropped before it
    /// cost a validation pass. Upstream logs no line here; the decision is the one a caller
    /// watching a run most wants to see, and its absence upstream is why a run looks stalled.
    Rejected {
        iteration: i64,
        before: f64,
        after: f64,
    },
    /// The proposal was kept and scored on the validation set. gepa logs only the better-program
    /// case; `is_best` is what that line means, and the score is what it prints.
    Accepted {
        iteration: i64,
        candidate: usize,
        score: f64,
        is_best: bool,
    },
    /// Two candidates were merged through a common ancestor. gepa: *"Iteration {i}: Merged
    /// programs {id1} and {id2} via ancestor {ancestor}"*.
    Merged {
        iteration: i64,
        first: usize,
        second: usize,
        ancestor: usize,
    },
    /// A merge was due and no mergeable pair existed. gepa: *"Iteration {i}: No merge candidates
    /// found"*.
    NoMergeCandidates { iteration: i64 },
}

impl Event<'_> {
    /// gepa's own line for this decision, for a caller that wants the sentence rather than the
    /// numbers. `Rejected` has no upstream line, so it reads in the same voice as the others.
    pub fn message(&self) -> String {
        match self {
            Event::Proposed {
                iteration,
                component,
                text,
            } => format!("Iteration {iteration}: Proposed new text for {component}: {text}"),
            Event::ProposedNothing { iteration } => {
                format!(
                    "Iteration {iteration}: Reflective mutation did not propose a new candidate"
                )
            }
            Event::NothingToLearnFrom { iteration } => {
                format!("Iteration {iteration}: All subsample scores perfect. Skipping.")
            }
            Event::Rejected {
                iteration,
                before,
                after,
            } => format!(
                "Iteration {iteration}: Proposal scored {after} against {before} on the minibatch \
                 and was dropped."
            ),
            Event::Accepted {
                iteration,
                candidate,
                score,
                is_best,
            } => match is_best {
                true => format!(
                    "Iteration {iteration}: Found a better program on the valset with score {score}."
                ),
                false => format!(
                    "Iteration {iteration}: Kept program {candidate} with valset score {score}."
                ),
            },
            Event::Merged {
                iteration,
                first,
                second,
                ancestor,
            } => format!(
                "Iteration {iteration}: Merged programs {first} and {second} via ancestor {ancestor}"
            ),
            Event::NoMergeCandidates { iteration } => {
                format!("Iteration {iteration}: No merge candidates found")
            }
        }
    }

    pub fn iteration(&self) -> i64 {
        match self {
            Event::Proposed { iteration, .. }
            | Event::ProposedNothing { iteration }
            | Event::NothingToLearnFrom { iteration }
            | Event::Rejected { iteration, .. }
            | Event::Accepted { iteration, .. }
            | Event::Merged { iteration, .. }
            | Event::NoMergeCandidates { iteration } => *iteration,
        }
    }
}

/// Where a run's events go. gepa's `LoggerProtocol`, and the default is upstream's `None`: a run
/// with no subscriber reports nothing and pays nothing.
pub trait Progress: Send + Sync {
    fn report(&self, event: Event<'_>);
}

/// The default: a run nobody is watching.
pub struct Silent;

impl Progress for Silent {
    fn report(&self, _event: Event<'_>) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lines are gepa's, so a caller printing them gets what upstream's log would have said.
    #[test]
    fn the_message_is_upstreams_line() {
        assert_eq!(
            Event::Accepted {
                iteration: 3,
                candidate: 7,
                score: 0.8,
                is_best: true,
            }
            .message(),
            "Iteration 3: Found a better program on the valset with score 0.8."
        );
        assert_eq!(
            Event::Merged {
                iteration: 5,
                first: 1,
                second: 2,
                ancestor: 0,
            }
            .message(),
            "Iteration 5: Merged programs 1 and 2 via ancestor 0"
        );
    }

    /// Every event names its iteration, which is what orders a stream a UI is rendering.
    #[test]
    fn every_event_carries_its_iteration() {
        for event in [
            Event::ProposedNothing { iteration: 1 },
            Event::NothingToLearnFrom { iteration: 2 },
            Event::Rejected {
                iteration: 3,
                before: 1.0,
                after: 0.5,
            },
            Event::NoMergeCandidates { iteration: 4 },
        ] {
            assert_eq!(
                event.iteration(),
                event.message()["Iteration ".len()..][..1]
                    .parse::<i64>()
                    .expect("the line names it too")
            );
        }
    }
}
