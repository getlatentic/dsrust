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

use crate::Candidate;

/// One decision, as it is made.
///
/// Every variant corresponds to something upstream *does*, and `message` renders the sentence
/// upstream would have logged — so a subscriber can act on the numbers and still print the line.
///
/// Not every variant is a transcription, and it was never only that. `Rejected` has no line
/// upstream at all: the decision is the one a caller watching a run most wants to see, and its
/// absence is why a run looks stalled. `is_best` is what upstream's better-program line *means*
/// rather than what it prints. And `program` on `Accepted` is what upstream's `new_program_idx`
/// refers to — an index into a structure no subscriber can see is a reference with no referent.
/// A field that completes the report belongs here; one that invents a decision does not.
#[derive(Debug, Clone, PartialEq)]
pub enum Event<'a> {
    /// The parent an iteration reflects on, once its minibatch has been scored. gepa 0.1.4:
    /// *"Iteration {i}: Selected program {idx} score: {score}"* — logged after the parent
    /// evaluations, where 0.1.1 logged it before.
    Selected {
        iteration: i64,
        candidate: usize,
        score: f64,
    },
    /// The reflection proposed replacement text for one component. gepa: *"Iteration {i}: Proposed
    /// new text for {name}: {text}"*.
    Proposed {
        iteration: i64,
        component: &'a str,
        text: &'a str,
    },
    /// No task produced a proposal this iteration. gepa: *"Iteration {i}: Reflective mutation did
    /// not propose a new candidate"*.
    ProposedNothing { iteration: i64 },
    /// The reflection ran and rewrote nothing, so the child would equal its parent and is not
    /// scored. gepa 0.1.4: *"Iteration {i}: Reflection returned no text updates; skipping proposal
    /// for this task."*
    NoTextUpdates { iteration: i64 },
    /// Every sampled score was already perfect, so there was nothing to reflect on. gepa 0.1.4:
    /// *"Iteration {i}: All subsample scores perfect for parent {idx}. Skipping."*
    NothingToLearnFrom { iteration: i64, parent: usize },
    /// The parent was run and recorded no trajectory, so there was nothing to reflect *on* —
    /// distinct from reflecting and finding nothing to say. gepa 0.1.4: *"Iteration {i}: No
    /// trajectories for parent {idx}. Skipping."*
    ///
    /// A program that records no trace produces this on every iteration, and a caller seeing only
    /// [`Event::ProposedNothing`] cannot tell that from a reflection that ran and declined.
    NoTrajectories { iteration: i64, parent: usize },
    /// The reflective dataset could not be built — dspy raises when no module has a valid
    /// prediction to learn from, and gepa catches it. gepa 0.1.4: *"Iteration {i}: Exception
    /// building reflective dataset: {e}"*.
    ReflectiveDatasetFailed { iteration: i64, error: &'a str },
    /// The reflection model failed over the iteration's tasks, and each is about to be tried on
    /// its own. gepa 0.1.4: *"Batched reflection failed ({e}); retrying per task."*
    BatchedReflectionFailed { iteration: i64, error: &'a str },
    /// The reflection model failed for one task on its own, which ends that task. gepa 0.1.4:
    /// *"Per-task reflection failed: {e}"*.
    ReflectionFailed { iteration: i64, error: &'a str },
    /// A proposal was not kept, and why. gepa 0.1.4 logs each kind in its own words.
    Rejected {
        iteration: i64,
        before: f64,
        after: f64,
        reason: Rejection,
    },
    /// A proposal passed the minibatch test and is about to be scored on the validation set. gepa
    /// 0.1.4: *"Iteration {i}: Accepted candidate (subsample score {before} -> {after}); running
    /// full eval."*
    AcceptedOnMinibatch {
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
        /// What the winning candidate *says*, not just where it sits.
        ///
        /// gepa's line prints `new_program_idx`, an index into `state.program_candidates` — a
        /// structure a subscriber cannot see, so the index alone is a reference with no referent.
        /// A caller checkpointing the best program so far has the text only on [`Self::Proposed`],
        /// and pairing the two by iteration is inference rather than a record.
        ///
        /// Borrowed, like `error` and `text` on the other variants: a subscriber that ignores it
        /// pays nothing, and one that persists clones at the moment it decides to.
        program: &'a Candidate,
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

/// Why gepa 0.1.4 did not keep a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection {
    /// The acceptance criterion refused it: *"New subsample score {after} is not better than old
    /// score {before}, skipping"*.
    NotBetter,
    /// It passed the criterion and the selection strategy left it out — `BestImprovement` or
    /// `TopKImprovements`: *"Passed acceptance (score {before} -> {after}) but was not selected by
    /// the selection strategy ({strategy}), skipping"*.
    NotSelected(&'static str),
    /// An identical candidate was selected earlier in the same iteration.
    Duplicate,
}

impl Event<'_> {
    /// gepa's own line for this decision, for a caller that wants the sentence rather than the
    /// numbers.
    pub fn message(&self) -> String {
        match self {
            Event::Selected {
                iteration,
                candidate,
                score,
            } => format!(
                "Iteration {iteration}: Selected program {candidate} score: {}",
                py_float(*score)
            ),
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
            Event::NoTextUpdates { iteration } => format!(
                "Iteration {iteration}: Reflection returned no text updates; skipping proposal for \
                 this task."
            ),
            Event::NoTrajectories { iteration, parent } => {
                format!("Iteration {iteration}: No trajectories for parent {parent}. Skipping.")
            }
            Event::ReflectiveDatasetFailed { iteration, error } => {
                format!("Iteration {iteration}: Exception building reflective dataset: {error}")
            }
            Event::BatchedReflectionFailed { error, .. } => {
                format!("Batched reflection failed ({error}); retrying per task.")
            }
            Event::ReflectionFailed { error, .. } => {
                format!("Per-task reflection failed: {error}")
            }
            Event::NothingToLearnFrom { iteration, parent } => format!(
                "Iteration {iteration}: All subsample scores perfect for parent {parent}. Skipping."
            ),
            Event::Rejected {
                iteration,
                before,
                after,
                reason,
            } => {
                let (before, after) = (py_float(*before), py_float(*after));
                match reason {
                    Rejection::NotBetter => format!(
                        "Iteration {iteration}: New subsample score {after} is not better than old \
                         score {before}, skipping"
                    ),
                    Rejection::NotSelected(strategy) => format!(
                        "Iteration {iteration}: Passed acceptance (score {before} -> {after}) but \
                         was not selected by the selection strategy ({strategy}), skipping"
                    ),
                    Rejection::Duplicate => format!(
                        "Iteration {iteration}: Duplicate of another candidate selected this \
                         iteration, skipping"
                    ),
                }
            }
            Event::AcceptedOnMinibatch {
                iteration,
                before,
                after,
            } => format!(
                "Iteration {iteration}: Accepted candidate (subsample score {} -> {}); running full \
                 eval.",
                py_float(*before),
                py_float(*after)
            ),
            Event::Accepted {
                program: _,
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
            Event::Selected { iteration, .. }
            | Event::Proposed { iteration, .. }
            | Event::ProposedNothing { iteration }
            | Event::NoTextUpdates { iteration }
            | Event::NothingToLearnFrom { iteration, .. }
            | Event::NoTrajectories { iteration, .. }
            | Event::ReflectiveDatasetFailed { iteration, .. }
            | Event::BatchedReflectionFailed { iteration, .. }
            | Event::ReflectionFailed { iteration, .. }
            | Event::Rejected { iteration, .. }
            | Event::AcceptedOnMinibatch { iteration, .. }
            | Event::Accepted { iteration, .. }
            | Event::Merged { iteration, .. }
            | Event::NoMergeCandidates { iteration } => *iteration,
        }
    }
}

/// A float as Python's f-string prints one — `2.0` for a whole number, where Rust prints `2`.
fn py_float(value: f64) -> String {
    let spelled = format!("{value}");
    match value.is_finite() && !spelled.contains(['.', 'e']) {
        true => format!("{spelled}.0"),
        false => spelled,
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
        // The winning candidate the event borrows; the sentence upstream logs names the index, not
        // the text, so it does not appear below.
        let won: Candidate = [("step".to_owned(), "Answer it.".to_owned())]
            .into_iter()
            .collect();
        assert_eq!(
            Event::Accepted {
                iteration: 3,
                candidate: 7,
                score: 0.8,
                is_best: true,
                program: &won,
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
            Event::NothingToLearnFrom {
                iteration: 2,
                parent: 0,
            },
            Event::Rejected {
                iteration: 3,
                before: 1.0,
                after: 0.5,
                reason: Rejection::NotBetter,
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
