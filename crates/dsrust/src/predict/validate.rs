//! Turning one reply into an accepted value: parse it, recover from what dspy recovers from, and
//! coerce it against the signature.
//!
//! Lifted out of `predict.rs` when that file crossed 400 lines. It is one job — everything between
//! the model answering and a `Prediction` existing — and the pieces it reaches for (the JSON
//! fallback, the feedback ask, the native channels) each already live beside it.

use anyhow::Result;
use serde_json::{Map, Value};

use super::{Feedback, Input, Predict, Steering};
use crate::adapter::parse::FieldMismatch;
use crate::example::Example;
use crate::lm::{DynChatModel, LmUsage};

/// One accepted reply: the value that passed coercion and validation, the raw text it was
/// parsed from, and the adapter that produced it — enough for a typed caller to push a
/// deeper failure back through the same feedback path.
pub(super) struct Validated {
    pub(super) raw: String,
    pub(super) value: Value,
    /// The candidates beside the first, when `n > 1` asked for more than one — dspy's
    /// `Prediction.from_completions` holds them and reads its own fields off the first.
    ///
    /// Best-effort, as `forward_completions` is: a candidate the model malformed is dropped rather
    /// than failing the call, since the answer is the first one and the rest are alternatives.
    pub(super) siblings: Vec<Example>,
    /// What every call behind this one cost together — a fallback and a feedback retry each add
    /// their own, so the accepted reply carries the whole exchange rather than its last ask.
    pub(super) usage: Option<LmUsage>,
}

impl<S> Predict<S> {
    pub(super) async fn call_with_inputs(
        &self,
        lm: &dyn DynChatModel,
        inputs: &[Input<'_>],
        steering: &Steering,
    ) -> Result<Validated> {
        // dspy's ChatAdapter catches a parse failure and re-asks the whole exchange through
        // the JSON adapter; `use_json_adapter_fallback` turns that off. The adapter states the
        // policy, this module carries it out, because only the module can call the model.
        let answered = self.ask(lm, inputs, None, steering).await?;
        // A provider that returned no completions at all did not return an empty one. Upstream's
        // `Adapter.__call__` loops over the outputs, so an empty list never reaches its parser and
        // `Predict` hands back a prediction with no fields; parsing `""` instead would report a
        // malformed reply for a call that produced none. `forward_completions` already answers with
        // an empty list here, and the two paths cannot disagree about the same response.
        if answered.response.outputs.is_empty() {
            return Ok(Validated {
                raw: String::new(),
                value: Value::Object(Map::new()),
                siblings: Vec::new(),
                usage: answered.response.spend(),
            });
        }
        // dspy `_call_postprocess`: a reply whose fields left the render for a native feature is
        // filled from those channels and skips the whole text path — parse, coercion, feedback —
        // exactly as upstream returns the native reply without re-validating it.
        if let Some(value) = self.native_value(&answered)? {
            return Ok(Validated {
                raw: answered.response.first_text(),
                value,
                // A native reply's fields come off the provider's own channels, which carry one
                // answer however many completions were asked for.
                siblings: Vec::new(),
                usage: answered.response.spend(),
            });
        }
        let usage = answered.response.spend();
        let raw = answered.response.first_text();
        // An adapter that answers in prose has a second model read the fields out of it. The
        // adapter says what to ask and who to ask; only this module can do the asking.
        if let Some(extraction) = self.adapter.extraction(&self.signature) {
            return self.extract(extraction, raw, usage).await;
        }
        // Every candidate the model returned, the way `forward_completions` reads them: each
        // parsed on its own, and one it malformed dropped rather than failing the call. dspy's
        // `Adapter.__call__` returns one parsed dict per output and `from_completions` keeps them
        // all, reading the prediction's own fields off the first.
        let siblings = self.candidates(&answered.response);
        let parsed = crate::observe::parsing(self.adapter.as_ref(), &raw, || {
            self.adapter.parse(&self.signature, &raw)
        });
        let (raw, mut value, usage) = match parsed {
            Ok(value) => (raw, value, usage),
            // A reply that spoke the format but left a field out. Upstream raises
            // `AdapterParseError` here and `ChatAdapter.__call__` re-asks through `JSONAdapter`,
            // which is the arm below — so this one only runs where a caller asked for the
            // feedback ask instead, and it carries the partial forward for `ensure` to name.
            Err(error) if self.feedback_retry && error.is::<FieldMismatch>() => {
                let partial = error
                    .downcast::<FieldMismatch>()
                    .map(|mismatch| mismatch.parsed)
                    .unwrap_or(Value::Null);
                (raw, partial, usage)
            }
            Err(error) => match self.adapter.json_fallback() {
                None => return Err(error),
                Some(fallback) => {
                    tracing::warn!(%error, "reply did not parse; re-asking through the fallback");
                    let answered = self
                        .ask_through(fallback.as_ref(), lm, inputs, None, steering)
                        .await?;
                    let answered_text = answered.response.first_text();
                    let value = crate::observe::parsing(fallback.as_ref(), &answered_text, || {
                        fallback.parse(&self.signature, &answered_text)
                    })?;
                    let merged = LmUsage::merge(usage, answered.response.spend());
                    (answered_text, value, merged)
                }
            },
        };
        // A value that will not coerce is upstream's `AdapterParseError` too — `parse_value`
        // raises inside `parse`, so the JSON fallback is what answers it there. Without the
        // feedback ask there is nothing left to try, and the error is the caller's.
        match self
            .signature
            .coerce(&mut value)
            .and_then(|()| self.signature.ensure(&value))
        {
            Ok(()) => Ok(Validated {
                raw,
                value,
                siblings,
                usage,
            }),
            Err(error) if !self.feedback_retry => Err(error),
            Err(error) => {
                tracing::warn!(%error, "retrying with feedback");
                let feedback = Feedback {
                    previous: raw,
                    error: error.to_string(),
                };
                let (raw, value, retried) =
                    self.feedback_ask(lm, inputs, &feedback, steering).await?;
                Ok(Validated {
                    raw,
                    value,
                    // A feedback ask re-asks for one answer, so the candidates the first reply
                    // carried no longer describe it.
                    siblings: Vec::new(),
                    usage: LmUsage::merge(usage, retried),
                })
            }
        }
    }
}
