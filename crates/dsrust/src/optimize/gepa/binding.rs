//! Writing a candidate back onto the program it came from.
//!
//! One function, and it is the seam where GEPA's two kinds of component meet the same program: a
//! predictor takes an instruction, a `Flex` takes a whole module's source. Split out because that
//! is a different question from *scoring* a candidate or *proposing* one, and because getting it
//! half-right is invisible — a search that proposes and accepts but never binds reports a better
//! program and hands back the one it started with.

use gepa::Candidate;

use crate::module::Module;

/// Apply a candidate to the student: an instruction to each predictor, a source to each `Flex`.
///
/// Both kinds, because a candidate holds both. This wrote instructions only until 2026-08-25, so a
/// code component could be proposed, scored and accepted and the winning source never reached the
/// program — the search ran, reported a better candidate, and left the module it started with.
///
/// A source that does not parse is dropped rather than raised: upstream's `rebind_flex_code` raises
/// and `DspyAdapter.evaluate` catches it to score the batch as a failure, which is the same outcome
/// by a shorter route — a candidate whose code will not bind cannot score.
pub(super) fn set_instructions<S: Module + ?Sized>(student: &mut S, candidate: &Candidate) {
    for predictor in student.named_predictors() {
        if let Some(instruction) = candidate.get(&predictor.name) {
            predictor.signature.instructions = instruction.clone();
        }
    }
    for named in student.named_flexes() {
        if let Some(source) = candidate.get(&named.name) {
            let _ = named.flex.bind(source.clone());
        }
    }
}
