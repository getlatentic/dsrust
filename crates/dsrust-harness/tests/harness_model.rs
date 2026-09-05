//! `HarnessModel` against a scripted harness: what it asks the agent for, and
//! what it hands back — no CLI, no network.

use std::sync::{Arc, Mutex};

use dsrust::lm::ChatModel;
use dsrust::lm::api::{LmMessage, LmRequest};
use dsrust_harness::{HarnessModel, MARKER_DISCIPLINE};
use harness::{
    CredentialSpec, Error, Harness, Info, Readiness, RunCallback, RunControl, RunEvent, RunHandle,
    RunRequest, ToolAccess,
};
use serde_json::json;

/// Records the request it was started with and replays scripted events.
struct Scripted {
    events: Vec<RunEvent>,
    seen: Arc<Mutex<Vec<RunRequest>>>,
    cancelled: Arc<Mutex<bool>>,
}

struct Control(Arc<Mutex<bool>>);
impl RunControl for Control {
    fn cancel(&self) -> Result<(), Error> {
        *self.0.lock().unwrap() = true;
        Ok(())
    }
    fn was_cancelled(&self) -> bool {
        *self.0.lock().unwrap()
    }
}

impl Harness for Scripted {
    fn info(&self) -> Info {
        Info {
            id: "scripted".into(),
            display_name: "Scripted".into(),
            description: String::new(),
            install_hint: None,
        }
    }
    fn readiness(&self) -> Readiness {
        unreachable!("not exercised")
    }
    fn start(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, Error> {
        self.seen.lock().unwrap().push(request);
        for event in &self.events {
            on_event(event.clone());
        }
        Ok(Box::new(Control(Arc::clone(&self.cancelled))))
    }
    fn credential(&self) -> CredentialSpec {
        unreachable!("not exercised")
    }
}

fn answering(text: &str) -> (Scripted, Arc<Mutex<Vec<RunRequest>>>) {
    let seen = Arc::default();
    let harness = Scripted {
        events: vec![
            RunEvent::Text {
                run_id: "r".into(),
                delta: text.into(),
            },
            RunEvent::Exited {
                run_id: "r".into(),
                exit_code: Some(0),
                cancelled: false,
            },
        ],
        seen: Arc::clone(&seen),
        cancelled: Arc::default(),
    };
    (harness, seen)
}

fn ask(messages: Vec<LmMessage>) -> LmRequest {
    LmRequest::from_messages("sonnet", messages)
}

#[tokio::test]
async fn every_call_withholds_the_agents_tools_and_carries_the_marker_discipline() {
    let (harness, seen) = answering("[[ ## answer ## ]]\n4");
    let model = HarnessModel::new(harness);
    let reply = model
        .forward(&ask(vec![
            LmMessage::system(["be brief"]),
            LmMessage::user(["2+2?"]),
        ]))
        .await
        .unwrap();

    assert_eq!(
        reply.outputs[0].parts[0].as_text(),
        Some("[[ ## answer ## ]]\n4")
    );
    let seen = seen.lock().unwrap();
    let run = &seen[0];
    assert_eq!(
        run.tools,
        ToolAccess::None,
        "the agent's own tools never run behind dsrust's back"
    );
    assert_eq!(run.prompt, "2+2?");
    let instructions = run.tuning.extra_instructions.as_deref().unwrap();
    assert!(
        instructions.starts_with(MARKER_DISCIPLINE),
        "standing instructions first: {instructions}"
    );
    assert!(
        instructions.ends_with("be brief"),
        "then the request's system message: {instructions}"
    );
    assert_eq!(
        run.tuning.model.as_deref(),
        Some("sonnet"),
        "the request's model reaches the CLI"
    );
}

#[tokio::test]
async fn a_configured_model_cwd_and_turn_cap_override_the_request_and_a_schema_passes_through() {
    let (harness, seen) = answering("{}");
    let model = HarnessModel::new(harness)
        .with_model("opus")
        .with_cwd("/tmp/work")
        .with_max_turns(3);
    let mut request = ask(vec![LmMessage::user(["q"])]);
    request.config.response_format = Some(json!({ "type": "object" }));
    model.forward(&request).await.unwrap();

    let run = &seen.lock().unwrap()[0];
    assert_eq!(run.tuning.model.as_deref(), Some("opus"));
    assert_eq!(
        run.cwd.as_deref().map(|p| p.to_str().unwrap()),
        Some("/tmp/work")
    );
    assert_eq!(run.tuning.max_turns, Some(3));
    assert_eq!(run.tuning.output_schema, Some(json!({ "type": "object" })));
}

#[tokio::test]
async fn instructions_can_be_replaced_or_removed() {
    let (harness, seen) = answering("x");
    let model = HarnessModel::new(harness).with_instructions(None);
    model
        .forward(&ask(vec![LmMessage::user(["q"])]))
        .await
        .unwrap();
    assert_eq!(
        seen.lock().unwrap()[0].tuning.extra_instructions,
        None,
        "nothing standing, nothing in the request"
    );
}

#[tokio::test]
async fn a_failed_run_is_an_error_the_caller_can_read() {
    let harness = Scripted {
        events: vec![
            RunEvent::Error {
                run_id: "r".into(),
                message: "Claude Code is not signed in".into(),
            },
            RunEvent::Exited {
                run_id: "r".into(),
                exit_code: Some(1),
                cancelled: false,
            },
        ],
        seen: Arc::default(),
        cancelled: Arc::default(),
    };
    let err = HarnessModel::new(harness)
        .forward(&ask(vec![LmMessage::user(["q"])]))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not signed in"), "{err}");
}

/// A run that never ends on its own: it holds the event channel open until it is
/// cancelled, then exits. The shape of an agent still working when the caller
/// gives up — and the only shape under which "dropping the call cancels it" can
/// be observed rather than raced.
struct Hanging {
    cancelled: Arc<Mutex<bool>>,
}

impl Harness for Hanging {
    fn info(&self) -> Info {
        Info {
            id: "hanging".into(),
            display_name: "Hanging".into(),
            description: String::new(),
            install_hint: None,
        }
    }
    fn readiness(&self) -> Readiness {
        unreachable!("not exercised")
    }
    fn start(&self, _request: RunRequest, on_event: RunCallback) -> Result<RunHandle, Error> {
        let cancelled = Arc::clone(&self.cancelled);
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !*cancelled.lock().unwrap() && std::time::Instant::now() < deadline {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            on_event(RunEvent::Exited {
                run_id: "r".into(),
                exit_code: None,
                cancelled: true,
            });
        });
        Ok(Box::new(Control(Arc::clone(&self.cancelled))))
    }
    fn credential(&self) -> CredentialSpec {
        unreachable!("not exercised")
    }
}

#[tokio::test]
async fn dropping_the_call_cancels_the_agent_whether_or_not_it_was_polled() {
    // A caller that gives up on the answer must not leave an agent spending
    // tokens. The run starts when `forward` is called, so the guard has to be
    // in place before the first poll — a future dropped un-polled is the case
    // that once slipped past it.
    let cancelled: Arc<Mutex<bool>> = Arc::default();
    let model = HarnessModel::new(Hanging {
        cancelled: Arc::clone(&cancelled),
    });
    let request = ask(vec![LmMessage::user(["q"])]);

    let never_polled = model.forward(&request);
    drop(never_polled);
    assert!(
        *cancelled.lock().unwrap(),
        "dropped before its first poll, the run was still cancelled"
    );

    *cancelled.lock().unwrap() = false;
    let polled_once = tokio::time::timeout(
        std::time::Duration::from_millis(20),
        model.forward(&request),
    )
    .await;
    assert!(
        polled_once.is_err(),
        "a run that has not ended cannot have answered"
    );
    assert!(
        *cancelled.lock().unwrap(),
        "dropped after polling, cancelled too"
    );
}

#[tokio::test]
async fn each_call_gets_its_own_run_id() {
    let (harness, seen) = answering("x");
    let model = HarnessModel::new(harness);
    model
        .forward(&ask(vec![LmMessage::user(["a"])]))
        .await
        .unwrap();
    model
        .forward(&ask(vec![LmMessage::user(["b"])]))
        .await
        .unwrap();
    let seen = seen.lock().unwrap();
    assert_ne!(
        seen[0].run_id, seen[1].run_id,
        "a host correlating events by run id must not see two runs as one"
    );
}
