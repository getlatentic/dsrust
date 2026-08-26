//! `LMOutputBuilder` — streamed events assembled back into an [`LmResponse`].

use std::collections::HashMap;

use anyhow::{Result, bail};
use serde_json::Value;

use super::delta::LmDelta;
use super::event::LmStreamEvent;
use super::part::{LmPart, Metadata};
use super::response::{LmOutput, LmResponse};
use crate::lm::LmUsage;

/// Where a tool call's arguments accumulate while they are still arriving in fragments.
const ARGS_BUFFER: &str = "args_buffer";

/// dspy's `LMOutputBuilder`: the accumulator that folds a stream's deltas into a finished response.
///
/// Deltas carry their own indices rather than arriving in order, so the builder holds them by index
/// and [`to_response`](Self::to_response) refuses to finish while the outputs are not contiguous
/// from zero — a gap means a stream is still in flight, not that an output is empty.
///
/// ```
/// use dsrust::lm::api::{LmDelta, LmOutputBuilder, LmStreamEvent};
///
/// let mut folding = LmOutputBuilder::new();
/// folding
///     .apply(LmStreamEvent::Delta {
///         output_index: 1,
///         part_index: 0,
///         delta: LmDelta::TextDelta { text: "second".to_owned() },
///     })
///     .expect("a delta applies");
/// // Output 0 never arrived, so finishing here would invent an empty one in its place.
/// assert!(folding.to_response(None, None).is_err());
/// ```
#[derive(Debug, Default)]
pub struct LmOutputBuilder {
    model: Option<String>,
    /// A hole is a part index no delta has reached yet, which `to_response` refuses to finish on.
    parts: HashMap<usize, Vec<Option<LmPart>>>,
    finish_reasons: HashMap<usize, Option<String>>,
    truncated: HashMap<usize, bool>,
}

impl LmOutputBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// `Some` once the stream has ended and the reply is complete.
    pub fn apply(&mut self, event: LmStreamEvent) -> Result<Option<LmResponse>> {
        match event {
            LmStreamEvent::Start { model } => {
                self.model = model;
                Ok(None)
            }
            LmStreamEvent::Delta {
                output_index,
                part_index,
                delta,
            } => {
                self.apply_delta(output_index, part_index, delta)?;
                Ok(None)
            }
            LmStreamEvent::OutputEnd {
                output_index,
                finish_reason,
                truncated,
            } => {
                self.finish_reasons.insert(output_index, finish_reason);
                self.truncated.insert(output_index, truncated);
                Ok(None)
            }
            LmStreamEvent::End {
                usage,
                cost,
                response,
            } => match response {
                Some(response) => Ok(Some(*response)),
                None => Ok(Some(self.to_response(usage, cost)?)),
            },
            LmStreamEvent::Error { error } => bail!(error),
        }
    }

    pub fn to_response(&self, usage: Option<LmUsage>, cost: Option<f64>) -> Result<LmResponse> {
        let mut indices: Vec<usize> = self
            .parts
            .keys()
            .chain(self.finish_reasons.keys())
            .chain(self.truncated.keys())
            .copied()
            .collect();
        indices.sort_unstable();
        indices.dedup();
        let last = indices.last().copied().unwrap_or(0);

        let missing: Vec<usize> = (0..=last)
            .filter(|index| !indices.is_empty() && !indices.contains(index))
            .collect();
        if !missing.is_empty() {
            bail!("stream output indices must be contiguous from 0; missing indices: {missing:?}");
        }

        let outputs = (0..=last)
            .map(|index| self.output(index))
            .collect::<Result<Vec<_>>>()?;
        Ok(LmResponse {
            model: self.model.clone(),
            outputs,
            usage,
            cost,
            ..LmResponse::default()
        })
    }

    fn output(&self, index: usize) -> Result<LmOutput> {
        let buffer = self.parts.get(&index).cloned().unwrap_or_default();
        let missing: Vec<usize> = buffer
            .iter()
            .enumerate()
            .filter_map(|(at, part)| part.is_none().then_some(at))
            .collect();
        if !missing.is_empty() {
            bail!(
                "stream part indices for output {index} must be contiguous; missing indices: {missing:?}"
            );
        }
        Ok(LmOutput {
            parts: buffer.into_iter().flatten().map(finalized).collect(),
            finish_reason: self.finish_reasons.get(&index).cloned().flatten(),
            truncated: self.truncated.get(&index).copied().unwrap_or(false),
            ..LmOutput::default()
        })
    }

    fn apply_delta(&mut self, output: usize, at: usize, delta: LmDelta) -> Result<()> {
        let parts = self.parts.entry(output).or_default();
        if parts.len() <= at {
            parts.resize(at + 1, None);
        }
        parts[at] = Some(merged(parts[at].take(), delta)?);
        Ok(())
    }
}

/// One increment folded onto whatever that slot already held.
fn merged(current: Option<LmPart>, delta: LmDelta) -> Result<LmPart> {
    match delta {
        LmDelta::TextDelta { text } => match current {
            None => Ok(LmPart::text(text)),
            Some(LmPart::Text { text: held, .. }) => Ok(LmPart::text(held + &text)),
            Some(_) => bail!("cannot apply a text delta to a non-text stream part"),
        },
        LmDelta::ThinkingDelta { text } => {
            let held = match current {
                None => String::new(),
                Some(LmPart::Thinking { text: held, .. }) => held,
                Some(_) => bail!("cannot apply a thinking delta to a non-thinking stream part"),
            };
            Ok(LmPart::Thinking {
                text: held + &text,
                redacted: false,
                metadata: Metadata::new(),
            })
        }
        LmDelta::ToolCallDelta {
            id,
            name,
            args_delta,
        } => tool_call(current, id, name, args_delta),
        LmDelta::CitationDelta { citation } => replaced(current, citation, |part| {
            matches!(part, LmPart::Citation { .. })
        }),
        LmDelta::ImageDelta { image } => {
            replaced(current, image, |part| matches!(part, LmPart::Image { .. }))
        }
        LmDelta::AudioDelta { audio } => {
            replaced(current, audio, |part| matches!(part, LmPart::Audio { .. }))
        }
    }
}

fn replaced(
    current: Option<LmPart>,
    incoming: LmPart,
    is_same_kind: fn(&LmPart) -> bool,
) -> Result<LmPart> {
    match current {
        Some(held) if !is_same_kind(&held) => {
            bail!("cannot apply a delta to a different stream part type")
        }
        _ => Ok(incoming),
    }
}

fn tool_call(
    current: Option<LmPart>,
    id: Option<String>,
    name: Option<String>,
    args_delta: Option<String>,
) -> Result<LmPart> {
    let (held_id, held_name, mut buffer) = match current {
        None => (None, String::new(), String::new()),
        Some(LmPart::ToolCall {
            id,
            name,
            provider_data,
            ..
        }) => {
            let buffer = provider_data
                .get(ARGS_BUFFER)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            (id, name, buffer)
        }
        Some(_) => bail!("cannot apply a tool-call delta to a non-tool-call stream part"),
    };
    buffer.push_str(args_delta.as_deref().unwrap_or_default());

    let mut provider_data = Metadata::new();
    provider_data.insert(ARGS_BUFFER.to_owned(), Value::String(buffer.clone()));
    Ok(LmPart::ToolCall {
        id: id.or(held_id),
        name: name.unwrap_or(held_name),
        // Half-arrived arguments are not an error mid-stream, only at the end.
        args: object_or_empty(&buffer),
        provider_data,
        metadata: Metadata::new(),
    })
}

fn finalized(part: LmPart) -> LmPart {
    let LmPart::ToolCall {
        id,
        name,
        provider_data,
        metadata,
        ..
    } = part
    else {
        return part;
    };
    let args = provider_data
        .get(ARGS_BUFFER)
        .and_then(Value::as_str)
        .map(object_or_empty)
        .unwrap_or_default();
    LmPart::ToolCall {
        id,
        name,
        args,
        provider_data,
        metadata,
    }
}

fn object_or_empty(raw: &str) -> Metadata {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(object)) => object,
        _ => Metadata::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn built(events: Vec<LmStreamEvent>) -> Result<LmResponse> {
        let mut builder = LmOutputBuilder::new();
        let mut last = None;
        for event in events {
            last = builder.apply(event)?.or(last);
        }
        last.ok_or_else(|| anyhow::anyhow!("the stream never ended"))
    }

    #[test]
    fn text_fragments_accumulate_into_one_part() {
        let response = built(vec![
            LmStreamEvent::Start {
                model: Some("openai/gpt-4o".to_owned()),
            },
            LmStreamEvent::delta(0, LmDelta::text("Par")),
            LmStreamEvent::delta(0, LmDelta::text("is")),
            LmStreamEvent::end(),
        ])
        .expect("builds");

        assert_eq!(response.model.as_deref(), Some("openai/gpt-4o"));
        assert_eq!(response.first_text(), "Paris");
    }

    /// Arguments arrive split across frames, so they are only valid JSON once the last one lands.
    #[test]
    fn tool_call_arguments_are_reassembled_across_frames() {
        let response = built(vec![
            LmStreamEvent::delta(
                0,
                LmDelta::ToolCallDelta {
                    id: Some("call_1".to_owned()),
                    name: Some("search".to_owned()),
                    args_delta: Some("{\"q\": \"Pa".to_owned()),
                },
            ),
            LmStreamEvent::delta(
                0,
                LmDelta::ToolCallDelta {
                    id: None,
                    name: None,
                    args_delta: Some("ris\"}".to_owned()),
                },
            ),
            LmStreamEvent::end(),
        ])
        .expect("builds");

        let LmPart::ToolCall { id, name, args, .. } = &response.outputs[0].parts[0] else {
            panic!("expected a tool call")
        };
        assert_eq!(
            id.as_deref(),
            Some("call_1"),
            "carried from the first frame"
        );
        assert_eq!(name, "search");
        assert_eq!(args["q"], json!("Paris"));
    }

    #[test]
    fn a_delta_of_the_wrong_kind_is_refused_rather_than_silently_replacing() {
        let mut builder = LmOutputBuilder::new();
        builder
            .apply(LmStreamEvent::delta(0, LmDelta::text("hi")))
            .expect("first applies");
        let clash = builder.apply(LmStreamEvent::delta(0, LmDelta::thinking("hmm")));
        assert!(clash.is_err(), "got {clash:?}");
    }

    #[test]
    fn thinking_fragments_accumulate_separately_from_text() {
        let response = built(vec![
            LmStreamEvent::delta(0, LmDelta::thinking("we")),
            LmStreamEvent::delta(0, LmDelta::thinking("ll")),
            LmStreamEvent::delta(1, LmDelta::text("Paris")),
            LmStreamEvent::end(),
        ])
        .expect("builds");

        let parts = &response.outputs[0].parts;
        assert!(matches!(&parts[0], LmPart::Thinking { text, .. } if text == "well"));
        assert_eq!(response.first_text(), "Paris", "thinking is not prose");
    }

    /// A gap means a frame was lost, and finishing on it would hand back a reply with a hole in
    /// the middle rather than an error.
    #[test]
    fn a_gap_in_the_part_indices_refuses_to_finish() {
        let built = built(vec![
            LmStreamEvent::delta(1, LmDelta::text("second")),
            LmStreamEvent::end(),
        ]);
        assert!(built.is_err(), "got {built:?}");
    }

    #[test]
    fn a_gap_in_the_output_indices_refuses_to_finish() {
        let built = built(vec![
            LmStreamEvent::Delta {
                output_index: 1,
                part_index: 0,
                delta: LmDelta::text("second candidate"),
            },
            LmStreamEvent::end(),
        ]);
        assert!(built.is_err(), "got {built:?}");
    }

    #[test]
    fn the_end_event_carries_why_the_reply_stopped_and_what_it_cost() {
        let response = built(vec![
            LmStreamEvent::delta(0, LmDelta::text("as far as")),
            LmStreamEvent::OutputEnd {
                output_index: 0,
                finish_reason: Some("length".to_owned()),
                truncated: true,
            },
            LmStreamEvent::End {
                usage: Some(LmUsage::counted(10, 4)),
                cost: Some(0.002),
                response: None,
            },
        ])
        .expect("builds");

        assert_eq!(response.outputs[0].finish_reason.as_deref(), Some("length"));
        assert!(response.outputs[0].truncated);
        assert_eq!(response.usage.and_then(|usage| usage.total()), Some(14));
        assert_eq!(response.cost, Some(0.002));
    }

    /// A provider that sends the whole reply on the end event wins over anything assembled.
    #[test]
    fn a_response_on_the_end_event_is_used_as_it_stands() {
        let response = built(vec![
            LmStreamEvent::delta(0, LmDelta::text("assembled")),
            LmStreamEvent::End {
                usage: None,
                cost: None,
                response: Some(Box::new(LmResponse::text("authoritative"))),
            },
        ])
        .expect("builds");
        assert_eq!(response.first_text(), "authoritative");
    }

    #[test]
    fn an_error_event_ends_the_stream_as_an_error() {
        let built = built(vec![LmStreamEvent::Error {
            error: "upstream refused".to_owned(),
        }]);
        assert!(
            built.expect_err("errors").to_string().contains("refused"),
            "the provider's own message survives"
        );
    }
}
