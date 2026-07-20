//! dspy `format_demos`: which few-shot examples reach the prompt, and how each one reads.
//!
//! An optimizer bootstraps demos from real runs, so one can arrive missing a field its run never
//! produced. Rendering them all alike would teach the model that a blank answer is acceptable,
//! so dspy sorts them three ways: whole demos stand as solved turns, partial ones announce
//! themselves first, and a demo with no input or no output is dropped rather than shown.

use crate::example::Example;
use crate::lm::ChatTurn;
use crate::signature::Signature;

use super::exchange::{Answer, ask};

/// dspy `incomplete_demo_prefix`, opening the user turn of a demo that is missing something.
const PARTIAL_PREFIX: &str =
    "This is an example of the task, though some input or output fields are not supplied.";

/// dspy's `missing_field_message`, standing in for an output the demo's run never recorded.
/// The trailing space is upstream's and is load-bearing: the field block is stripped as a whole,
/// so the space survives on every line but the last.
const NOT_SUPPLIED: &str = "Not supplied for this particular example. ";

/// Where a demo lands in dspy's three-way split.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Every signature field supplied, so the demo reads as an ordinary solved turn.
    Complete,
    /// Missing a field, but with an input to ask and an output to show.
    Partial,
    /// Nothing a solved turn could demonstrate.
    Dropped,
}

/// The demo turns preceding the real request, partial demos first — dspy renders the two groups
/// in that order regardless of the order they were handed in.
pub(super) fn demo_turns(
    signature: &Signature,
    demos: &[Example],
    answer: Answer,
) -> Vec<ChatTurn> {
    let matching = |kind: Kind| {
        demos
            .iter()
            .filter(move |demo| classify(signature, demo) == kind)
    };
    let partial = matching(Kind::Partial).flat_map(|demo| {
        [
            ask(signature, demo, Some(PARTIAL_PREFIX)),
            answer(signature, demo, Some(NOT_SUPPLIED)),
        ]
    });
    let complete = matching(Kind::Complete)
        .flat_map(|demo| [ask(signature, demo, None), answer(signature, demo, None)]);
    partial.chain(complete).collect()
}

/// dspy counts a field as supplied only when it is present and not `None`, but counts it as
/// *there* on presence alone — so a demo carrying an explicit null is partial rather than
/// dropped, and renders that null for the model to read.
fn classify(signature: &Signature, demo: &Example) -> Kind {
    let supplied = |name| demo.get(name).is_some_and(|value| !value.is_null());
    let complete = signature.inputs.iter().all(|field| supplied(&field.name))
        && signature.outputs.iter().all(|field| supplied(&field.name));
    if complete {
        return Kind::Complete;
    }
    let present = |name| demo.get(name).is_some();
    let shows_both = signature.inputs.iter().any(|field| present(&field.name))
        && signature.outputs.iter().any(|field| present(&field.name));
    match shows_both {
        true => Kind::Partial,
        false => Kind::Dropped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::{FieldKind, InField, OutField};
    use serde_json::json;

    /// Two fields on each side, so a demo can miss one and still have something to show, and so
    /// a missing output can land somewhere other than the last line.
    fn paint() -> Signature {
        let input = |name: &str, desc: &str| InField {
            name: name.into(),
            desc: desc.into(),
            kind: FieldKind::Str,
            values: None,
        };
        let output = |name: &str, desc: &str| OutField {
            name: name.into(),
            desc: desc.into(),
            kind: FieldKind::Str,
            values: None,
            schema: None,
        };
        Signature {
            instructions: "Pick a colour.".into(),
            inputs: vec![
                input("room", "the room being painted"),
                input("mood", "the mood to set"),
            ],
            outputs: vec![
                output("colour", "the chosen colour"),
                output("why", "one short sentence"),
            ],
        }
    }

    fn rendered(demos: &[Example]) -> Vec<String> {
        demo_turns(&paint(), demos, crate::adapter::exchange::answer)
            .into_iter()
            .map(|turn| turn.content.text().unwrap().to_owned())
            .collect()
    }

    fn whole() -> Example {
        Example::new([
            ("room", json!("the study")),
            ("mood", json!("calm focus")),
            ("colour", json!("blue")),
            ("why", json!("It reads calm.")),
        ])
    }

    #[test]
    fn a_whole_demo_reads_as_an_ordinary_solved_turn() {
        assert_eq!(
            rendered(&[whole()]),
            [
                "[[ ## room ## ]]\nthe study\n\n[[ ## mood ## ]]\ncalm focus",
                "[[ ## colour ## ]]\nblue\n\n[[ ## why ## ]]\nIt reads calm.\n\n\
                 [[ ## completed ## ]]\n",
            ]
        );
    }

    /// A demo missing its last output still writes that output's marker, so the model reads the
    /// full set of sections it is being asked for rather than inferring one is optional.
    #[test]
    fn a_partial_demo_announces_itself_and_marks_what_it_never_recorded() {
        let demo = Example::new([
            ("room", json!("the study")),
            ("mood", json!("calm focus")),
            ("colour", json!("blue")),
        ]);
        assert_eq!(
            rendered(&[demo]),
            [
                "This is an example of the task, though some input or output fields are not \
                 supplied.\n\n[[ ## room ## ]]\nthe study\n\n[[ ## mood ## ]]\ncalm focus",
                "[[ ## colour ## ]]\nblue\n\n[[ ## why ## ]]\nNot supplied for this particular \
                 example.\n\n[[ ## completed ## ]]\n",
            ]
        );
    }

    /// dspy strips the output block once, as a whole, so the space its missing-field message
    /// ends in survives everywhere except the last line.
    #[test]
    fn the_missing_field_note_keeps_its_trailing_space_off_the_last_line() {
        assert!(
            NOT_SUPPLIED.ends_with(' '),
            "the space this test pins is upstream's, and is invisible in the expectation below"
        );
        let demo = Example::new([
            ("room", json!("the study")),
            ("mood", json!("calm focus")),
            ("why", json!("It reads calm.")),
        ]);
        assert_eq!(
            rendered(&[demo])[1],
            format!(
                "[[ ## colour ## ]]\n{NOT_SUPPLIED}\n\n[[ ## why ## ]]\nIt reads calm.\n\n\
                 [[ ## completed ## ]]\n"
            )
        );
    }

    /// The two sides are not symmetric: a missing output is marked, a missing input is simply
    /// absent, because the prefix has already told the model the demo is partial.
    #[test]
    fn a_demo_missing_an_input_shows_only_the_inputs_it_has() {
        let demo = Example::new([
            ("room", json!("the study")),
            ("colour", json!("blue")),
            ("why", json!("It reads calm.")),
        ]);
        assert_eq!(
            rendered(&[demo])[0],
            "This is an example of the task, though some input or output fields are not \
             supplied.\n\n[[ ## room ## ]]\nthe study"
        );
    }

    /// An explicit null is present but not supplied: enough to keep the demo, not enough to call
    /// it whole. The model reads Python's spelling of it, which is what upstream prints.
    #[test]
    fn a_null_field_makes_a_demo_partial_and_prints_as_python_none() {
        let demo = Example::new([
            ("room", json!("the study")),
            ("mood", json!(null)),
            ("colour", json!("blue")),
            ("why", json!("It reads calm.")),
        ]);
        assert_eq!(
            rendered(&[demo])[0],
            "This is an example of the task, though some input or output fields are not \
             supplied.\n\n[[ ## room ## ]]\nthe study\n\n[[ ## mood ## ]]\nNone"
        );
    }

    /// Half an example teaches the wrong lesson, so dspy shows neither half.
    #[test]
    fn a_demo_with_only_one_side_is_dropped_rather_than_half_shown() {
        let inputs_only = Example::new([("room", json!("the study")), ("mood", json!("calm"))]);
        let outputs_only = Example::new([("colour", json!("blue")), ("why", json!("Calm."))]);
        assert!(rendered(&[inputs_only, outputs_only]).is_empty());
    }

    #[test]
    fn partial_demos_are_rendered_before_whole_ones() {
        let partial = Example::new([
            ("room", json!("the nursery")),
            ("mood", json!("gentle")),
            ("colour", json!("green")),
        ]);
        let turns = rendered(&[whole(), partial]);
        assert!(turns[0].starts_with(PARTIAL_PREFIX), "got: {}", turns[0]);
        assert!(turns[1].contains("[[ ## colour ## ]]\ngreen"));
        assert_eq!(
            turns[2],
            "[[ ## room ## ]]\nthe study\n\n[[ ## mood ## ]]\ncalm focus"
        );
    }
}
