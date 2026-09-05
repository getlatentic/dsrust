//! Fold a run's event stream into the one reply a model call returns.

use std::sync::mpsc::Receiver;

use anyhow::{Result, anyhow};
use dsrust::lm::LmUsage;
use dsrust::lm::api::{LmOutput, LmPart, LmResponse};
use harness::RunEvent;

/// Read events until the run exits, then answer as a model would: the text as
/// one output, reasoning as a thinking part beside it, the CLI's token counts
/// as usage.
///
/// A run that exits non-zero, or reports an error, without having said
/// anything is a failed call. One that said something and then hit an error is
/// answered with what it said and the error as the finish reason: the caller
/// decides whether a partial answer is worth having, the way it does for a
/// provider that cut a stream short.
pub(crate) fn drain(events: Receiver<RunEvent>, model: Option<String>) -> Result<LmResponse> {
    let mut text = String::new();
    let mut structured: Option<serde_json::Value> = None;
    let mut thinking = String::new();
    let mut usage = None;
    let mut cost = None;
    let mut session = None;
    let mut error = None;
    let mut exit_code = None;
    for event in events {
        match event {
            RunEvent::Text { delta, .. } => text.push_str(&delta),
            RunEvent::Thinking { delta, .. } => thinking.push_str(&delta),
            RunEvent::Usage {
                input_tokens,
                output_tokens,
                total_tokens,
                cache_read_tokens,
                cache_write_tokens,
                cost_usd,
                ..
            } => {
                usage = Some(LmUsage {
                    input_tokens: narrow(input_tokens),
                    output_tokens: narrow(output_tokens),
                    total_tokens: narrow(total_tokens),
                    prompt_tokens: narrow(input_tokens),
                    completion_tokens: narrow(output_tokens),
                    cache_read_tokens: narrow(cache_read_tokens),
                    cache_write_tokens: narrow(cache_write_tokens),
                    ..LmUsage::default()
                });
                cost = cost_usd;
            }
            RunEvent::StructuredOutput { value, .. } => structured = Some(value),
            RunEvent::Session { session_id, .. } => session = session_id,
            RunEvent::Error { message, .. } => error = Some(message),
            RunEvent::Exited {
                exit_code: code, ..
            } => exit_code = code,
            _ => {}
        }
    }

    let failed = error.is_some() || exit_code.is_some_and(|code| code != 0);
    if failed && text.trim().is_empty() {
        return Err(anyhow!(
            "the agent answered nothing: {}",
            error.unwrap_or_else(|| format!("exit code {exit_code:?}"))
        ));
    }
    let mut parts = Vec::new();
    if !thinking.is_empty() {
        parts.push(LmPart::thinking(thinking, false));
    }
    // An agent asked for a shape may still narrate before it fills it. The
    // shape is the answer; the narration is for a reader of the transcript, not
    // for the parser.
    parts.push(LmPart::text(match structured {
        Some(value) => value.to_string(),
        None => text,
    }));
    Ok(LmResponse {
        model,
        outputs: vec![LmOutput {
            parts,
            finish_reason: Some(if failed {
                "error".to_owned()
            } else {
                "stop".to_owned()
            }),
            ..LmOutput::default()
        }],
        usage,
        cost,
        response_id: session,
        ..LmResponse::default()
    })
}

/// Token counts arrive as `u64`; dsrust keeps `u32`. A count past four billion is
/// not a real one, so it is dropped rather than wrapped into a small lie.
fn narrow(count: Option<u64>) -> Option<u32> {
    count.and_then(|n| u32::try_from(n).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn run(events: Vec<RunEvent>) -> Result<LmResponse> {
        let (tx, rx) = mpsc::channel();
        for event in events {
            tx.send(event).unwrap();
        }
        drop(tx);
        drain(rx, Some("m".into()))
    }
    fn text(delta: &str) -> RunEvent {
        RunEvent::Text {
            run_id: "r".into(),
            delta: delta.into(),
        }
    }
    fn exited(code: i32) -> RunEvent {
        RunEvent::Exited {
            run_id: "r".into(),
            exit_code: Some(code),
            cancelled: false,
        }
    }

    #[test]
    fn deltas_are_one_text_output_and_reasoning_rides_beside_it() {
        let reply = run(vec![
            RunEvent::Thinking {
                run_id: "r".into(),
                delta: "hm".into(),
            },
            text("hel"),
            text("lo"),
            exited(0),
        ])
        .unwrap();
        let parts = &reply.outputs[0].parts;
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1].as_text(), Some("hello"));
        assert_eq!(reply.outputs[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(reply.model.as_deref(), Some("m"));
    }

    #[test]
    fn a_structured_answer_replaces_the_prose_as_the_text_a_parser_reads() {
        let reply = run(vec![
            text("Let me put that in the shape you asked for."),
            RunEvent::StructuredOutput {
                run_id: "r".into(),
                value: serde_json::json!({ "code": "q" }),
            },
            exited(0),
        ])
        .unwrap();
        assert_eq!(reply.outputs[0].parts[0].as_text(), Some(r#"{"code":"q"}"#));
    }

    #[test]
    fn the_agents_session_id_is_the_response_id() {
        // A host that wants to resume the conversation needs the id the CLI
        // gave the session, and this is the only field it can travel in.
        let reply = run(vec![
            RunEvent::Session {
                run_id: "r".into(),
                session_id: Some("ses-9".into()),
                model: None,
            },
            text("x"),
            exited(0),
        ])
        .unwrap();
        assert_eq!(reply.response_id.as_deref(), Some("ses-9"));
    }

    #[test]
    fn usage_and_cost_are_carried_under_both_spellings() {
        let reply = run(vec![
            text("x"),
            RunEvent::Usage {
                run_id: "r".into(),
                input_tokens: Some(10),
                output_tokens: Some(3),
                total_tokens: Some(13),
                cache_read_tokens: Some(4),
                cache_write_tokens: Some(2),
                cost_usd: Some(0.25),
            },
            exited(0),
        ])
        .unwrap();
        let usage = reply.usage.unwrap();
        assert_eq!(
            (usage.input_tokens, usage.prompt_tokens),
            (Some(10), Some(10))
        );
        assert_eq!(
            (usage.output_tokens, usage.completion_tokens),
            (Some(3), Some(3))
        );
        assert_eq!(usage.total_tokens, Some(13));
        assert_eq!(
            (usage.cache_read_tokens, usage.cache_write_tokens),
            (Some(4), Some(2))
        );
        assert_eq!(reply.cost, Some(0.25));
    }

    #[test]
    fn a_run_that_said_nothing_and_failed_is_an_error_naming_why() {
        let err = run(vec![
            RunEvent::Error {
                run_id: "r".into(),
                message: "not signed in".into(),
            },
            exited(1),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("not signed in"), "{err}");
        // A bare non-zero exit with no message still names the code.
        let err = run(vec![exited(2)]).unwrap_err();
        assert!(err.to_string().contains("2"), "{err}");
    }

    #[test]
    fn a_partial_answer_is_kept_and_the_failure_is_the_finish_reason() {
        let reply = run(vec![
            text("half"),
            RunEvent::Error {
                run_id: "r".into(),
                message: "cut".into(),
            },
            exited(1),
        ])
        .unwrap();
        assert_eq!(reply.outputs[0].parts[0].as_text(), Some("half"));
        assert_eq!(reply.outputs[0].finish_reason.as_deref(), Some("error"));
    }
}
