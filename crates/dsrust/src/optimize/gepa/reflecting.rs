//! Turning a run into the shapes reflection reads.
//!
//! Split from the adapter because the two change for different reasons: `adapter.rs` decides *when*
//! a dataset is built and *who* is asked to propose from it, and this decides what one *looks like*.
//!
//! Two shapes, not one. The reflective form is ordered pairs, because a rendered prompt cares about
//! order; the code proposer's is a JSON object, because upstream `repr`s a Python dict into its
//! prompt. They are the same records under two renderers.
//!
//! `code_reflective_records` is a free function taking the captured runs, which is what upstream's
//! is — `code_reflective_records(eval_batch)`, not a method reaching into the adapter. Its sibling
//! `make_reflective_dataset` *is* a method on `DspyAdapter` and stays one. The shapes follow
//! upstream's rather than whichever was convenient: the first draft kept both on the adapter, which
//! made the free one need private fields it had no business reading.

use gepa::{Reflective, ReflectiveSample};

use crate::example::{Example, Prediction};
use crate::module::TraceStep;

use super::metric::Feedback;

/// A reflective sample as the code proposer's `format_failures` reads it.
///
/// The reflective shape is ordered pairs because a rendered prompt cares about order; the code
/// proposer's is a JSON object because upstream `repr`s a Python dict. The two are the same records
/// seen by two renderers, and this is the one conversion between them.
pub(super) fn reflective_records(
    samples: &[ReflectiveSample],
) -> Vec<serde_json::Map<String, serde_json::Value>> {
    fn value_of(reflective: &Reflective) -> serde_json::Value {
        match reflective {
            Reflective::Text(text) => serde_json::Value::String(text.clone()),
            Reflective::Map(entries) => serde_json::Value::Object(
                entries
                    .iter()
                    .map(|(name, inner)| (name.clone(), value_of(inner)))
                    .collect(),
            ),
            Reflective::List(items) => {
                serde_json::Value::Array(items.iter().map(value_of).collect())
            }
        }
    }
    samples
        .iter()
        .map(|sample| {
            sample
                .iter()
                .map(|(name, value)| (name.clone(), value_of(value)))
                .collect()
        })
        .collect()
}

/// A predictor's *inputs* as a GEPA reflective map.
///
/// dspy lifts a `History` input out of the field map into a fenced `Context` block and drops the
/// original key, so the conversation reaches the reflection model as one numbered listing rather
/// than as a JSON blob in a field. Upstream decides by `isinstance(input_val, History)`; this
/// decides by the signature's annotation, which is the same question asked of the declaration
/// rather than of the value — and the only one available once a value is an untyped `Value`.
///
/// Each message is `str(message)` on a Python dict: single-quoted, `None`/`True` spellings, which
/// is what [`crate::python::repr`] already renders for the prompts that show Python source.
pub(super) fn rendered_inputs(
    inputs: &Example,
    signature: &crate::signature::Signature,
) -> Reflective {
    let history = crate::adapter::history::field_name(signature);
    let mut entries: Vec<(String, Reflective)> = Vec::new();
    if let Some(value) = history.and_then(|name| inputs.get(name)) {
        entries.push(("Context".to_owned(), Reflective::Text(context_block(value))));
    }
    for (name, rendered) in inputs.rendered() {
        if Some(name.as_str()) == history {
            continue;
        }
        entries.push((name, Reflective::Text(rendered)));
    }
    Reflective::Map(entries)
}

/// dspy's `Context` value: a ```` ```json ```` fence around one `  {i}: {message}` line per
/// exchange. The fence says json and the contents are Python dict reprs; that is upstream's, not a
/// transcription slip.
pub(super) fn context_block(history: &serde_json::Value) -> String {
    let mut block = String::from("```json\n");
    let messages = history
        .get("messages")
        .and_then(serde_json::Value::as_array);
    for (index, message) in messages.into_iter().flatten().enumerate() {
        block.push_str(&format!("  {index}: {}\n", crate::python::repr(message)));
    }
    block.push_str("```");
    block
}

/// An example's fields as a GEPA reflective map: field name → its rendered value, in declaration
/// order (dspy's `{k: str(v) for k, v in inputs.items()}`).
pub(super) fn rendered_map(example: &Example) -> Reflective {
    Reflective::Map(
        example
            .rendered()
            .into_iter()
            .map(|(name, value)| (name, Reflective::Text(value)))
            .collect(),
    )
}

/// One example's captured run, kept from a `capture_traces=true` evaluation so [`Adapter::propose_new_texts`]
/// can build the reflective dataset — dspy's `eval_batch.trajectories`.
///
/// The example and its prediction travel with the trace because the feedback text is *not* computed
/// here: dspy calls the metric again at reflection time, once per record, with the predictor it
/// drew. So what scoring keeps is the run, not a sentence about it.
pub(super) struct Captured {
    pub(super) example: Example,
    pub(super) prediction: Prediction,
    pub(super) trace: Vec<TraceStep>,
    /// What scoring said about this run — kept, unlike the per-predictor path, because a *code*
    /// component reflects on the score it already got rather than asking the metric again.
    ///
    /// The two paths genuinely differ. Reflecting on a predictor draws one step and calls the metric
    /// a second time with that step, because the question is "what should this predictor have done".
    /// Reflecting on source asks "what should this program have done", which scoring already
    /// answered.
    pub(super) scored: Feedback,
}

/// dspy `code_reflective_records`: the reflective dataset for a *code* component.
///
/// Whole-program inputs and outputs per example, where the per-predictor path takes one drawn
/// step's. That difference is the point: a predictor's component is its instruction and reflects
/// on what that predictor did, while a `Flex`'s component is the source of the whole module and
/// reflects on what the program did.
///
/// The feedback is the one scoring produced, falling back to upstream's sentence when the metric
/// said nothing — a score with no words is still a signal, and saying so is better than sending
/// the model an empty field.
pub(super) fn code_reflective_records(captured: &[Captured]) -> Vec<ReflectiveSample> {
    captured
        .iter()
        .map(|captured| {
            // `Feedback::text` is upstream's `feedback or "This trajectory got a score of …"`,
            // already spelled once for the per-predictor path.
            let feedback = captured.scored.text();
            vec![
                ("Inputs".to_owned(), rendered_map(&captured.example)),
                (
                    "Generated Outputs".to_owned(),
                    rendered_map(&captured.prediction.example),
                ),
                ("Feedback".to_owned(), Reflective::Text(feedback)),
            ]
        })
        .collect()
}
