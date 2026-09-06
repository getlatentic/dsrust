//! `HarnessModel` against a scripted harness: what it asks the agent for, and
//! what it hands back — no CLI, no network.

use std::sync::{Arc, Mutex};

use dsrust::lm::ChatModel;
use dsrust::lm::api::{LmMessage, LmReasoningConfig, LmRequest};
use dsrust_harness::{HarnessModel, MARKER_DISCIPLINE, Temperature};
use harness::{
    CredentialSpec, Error, Features, Harness, Info, Readiness, ReasoningEffort, RunCallback,
    RunControl, RunEvent, RunHandle, RunRequest, ToolAccess,
};
use serde_json::json;

/// Records the request it was started with and replays scripted events.
struct Scripted {
    events: Vec<RunEvent>,
    seen: Arc<Mutex<Vec<RunRequest>>>,
    cancelled: Arc<Mutex<bool>>,
    features: Features,
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
    fn features(&self) -> Features {
        self.features.clone()
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
        features: Features {
            withheld_tools: true,
            ..Features::default()
        },
    };
    (harness, seen)
}

fn ask(messages: Vec<LmMessage>) -> LmRequest {
    LmRequest::from_messages("sonnet", messages)
}

#[tokio::test]
async fn every_call_withholds_the_agents_tools_and_carries_the_marker_discipline() {
    let (harness, seen) = answering("[[ ## answer ## ]]\n4");
    let model = HarnessModel::new(harness).unwrap();
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
    // As a model (no tools) the instructions ARE the system prompt: the agent's
    // own ~7,000-token envelope is not sent at all.
    assert_eq!(
        run.tuning.extra_instructions, None,
        "nothing appended to an agent prompt that is not sent"
    );
    let instructions = run.tuning.system_prompt.as_deref().unwrap();
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
    let model = HarnessModel::builder(harness)
        .model("opus")
        .cwd("/tmp/work")
        .max_turns(3)
        .build()
        .unwrap();
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
    let model = HarnessModel::builder(harness)
        .instructions("")
        .build()
        .unwrap();
    model
        .forward(&ask(vec![LmMessage::user(["q"])]))
        .await
        .unwrap();
    let run = &seen.lock().unwrap()[0];
    assert_eq!(
        (
            run.tuning.system_prompt.as_deref(),
            run.tuning.extra_instructions.as_deref()
        ),
        (None, None),
        "nothing standing, nothing in the request"
    );
}

#[tokio::test]
async fn as_an_agent_the_instructions_are_added_beside_the_agents_own_prompt() {
    // With tools, the agent needs its own prompt to drive them; ours rides as
    // an addition, and a thinking cap travels as given. The adapter has to
    // advertise that it takes them, or `build` turns it away — an agent that
    // discards the addition is one whose replies do not parse.
    let (harness, seen) = advertising(
        Features {
            withheld_tools: true,
            custom_instructions: true,
            ..Features::default()
        },
        "x",
    );
    let model = HarnessModel::builder(harness)
        .tools(ToolAccess::Default)
        .max_thinking_tokens(0)
        .build()
        .unwrap();
    model
        .forward(&ask(vec![
            LmMessage::system(["be brief"]),
            LmMessage::user(["q"]),
        ]))
        .await
        .unwrap();
    let run = &seen.lock().unwrap()[0];
    assert_eq!(run.tuning.system_prompt, None, "the agent keeps its prompt");
    assert!(
        run.tuning
            .extra_instructions
            .as_deref()
            .is_some_and(|i| i.ends_with("be brief")),
        "{:?}",
        run.tuning.extra_instructions
    );
    assert_eq!(run.tuning.max_thinking_tokens, Some(0));
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
        features: Features {
            withheld_tools: true,
            ..Features::default()
        },
    };
    let err = HarnessModel::new(harness)
        .unwrap()
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
    fn features(&self) -> Features {
        Features {
            withheld_tools: true,
            ..Features::default()
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
    })
    .unwrap();
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
    let model = HarnessModel::new(harness).unwrap();
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

#[test]
fn a_harness_that_cannot_withhold_its_tools_is_refused_at_build_not_at_the_first_call() {
    // Codex and ACP cannot un-offer their tools; `Features::withheld_tools` says so.
    // A model over one of them would fail every `forward`, so `build` is where it
    // is turned away — and where `.tools(ToolAccess::Default)` is still a choice.
    // Codex-shaped: it takes instructions even though it cannot withhold tools.
    let cannot = || Scripted {
        events: Vec::new(),
        seen: Arc::default(),
        cancelled: Arc::default(),
        features: Features {
            withheld_tools: false,
            custom_instructions: true,
            ..Features::default()
        },
    };
    let Err(refused) = HarnessModel::new(cannot()) else {
        panic!("a harness that cannot withhold its tools was accepted as a model");
    };
    let refused = refused.to_string();
    assert!(refused.contains("Scripted"), "names the harness: {refused}");
    assert!(
        refused.contains("ToolAccess::None"),
        "and what was asked: {refused}"
    );
    assert!(
        refused.contains("ToolAccess::Default"),
        "and the way out: {refused}"
    );
    assert!(
        HarnessModel::builder(cannot())
            .tools(ToolAccess::Default)
            .build()
            .is_ok(),
        "as an agent it is fine"
    );
}

/// A harness advertising exactly these features, answering with `text`.
fn advertising(features: Features, text: &str) -> (Scripted, Arc<Mutex<Vec<RunRequest>>>) {
    let (mut harness, seen) = answering(text);
    harness.features = features;
    (harness, seen)
}

#[test]
fn an_adapter_that_ignores_instructions_cannot_run_as_an_agent() {
    // ACP honours neither `system_prompt` nor `extra_instructions`. Under
    // `ToolAccess::Default` the marker discipline travels as the latter, so a run
    // there is one whose reply dsrust cannot parse — and the parse failure names a
    // field, never the setting that went missing. Refused where the cause is legible.
    let acp = || {
        advertising(
            Features {
                withheld_tools: false,
                custom_instructions: false,
                ..Features::default()
            },
            "",
        )
        .0
    };
    let Err(refused) = HarnessModel::builder(acp())
        .tools(ToolAccess::Default)
        .build()
    else {
        panic!("an adapter that drops the marker discipline was accepted as an agent");
    };
    let refused = refused.to_string();
    assert!(refused.contains("Scripted"), "names the harness: {refused}");
    assert!(
        refused.contains("marker discipline"),
        "and what would be lost: {refused}"
    );
}

#[tokio::test]
async fn a_sampling_knob_is_refused_by_name_rather_than_answered_as_though_it_applied() {
    // The failure this prevents is not an error but a number: a judge pinned to 0.0
    // that quietly samples scores one program differently on each run.
    let (harness, seen) = answering("[[ ## answer ## ]]\n4");
    let model = HarnessModel::new(harness).unwrap();
    let mut request = ask(vec![LmMessage::user(["2+2?"])]);
    request.config.temperature = Some(0.0);

    let refused = model.forward(&request).await.unwrap_err().to_string();
    assert!(
        refused.contains("temperature"),
        "names the knob it cannot honour: {refused}"
    );
    assert!(
        seen.lock().unwrap().is_empty(),
        "and refuses before an agent is ever started"
    );
}

#[tokio::test]
async fn a_reasoning_effort_travels_where_the_adapter_advertises_one() {
    // The mirror of the refusals: the honour clause pinned, so a later tightening
    // cannot quietly swallow a knob that does have a destination.
    let (harness, seen) = advertising(
        Features {
            withheld_tools: true,
            effort: true,
            ..Features::default()
        },
        "[[ ## answer ## ]]\n4",
    );
    let model = HarnessModel::new(harness).unwrap();
    let mut request = ask(vec![LmMessage::user(["2+2?"])]);
    request.config.reasoning = Some(LmReasoningConfig {
        effort: Some("medium".into()),
        ..LmReasoningConfig::default()
    });

    model.forward(&request).await.unwrap();
    assert_eq!(
        seen.lock().unwrap()[0].tuning.effort,
        Some(ReasoningEffort::Medium)
    );
}

#[tokio::test]
async fn an_effort_the_harness_cannot_spell_is_refused_rather_than_rounded() {
    // dsrust's effort is free text and `RunTuning::effort` is four variants, so the
    // tempting bug is to round. Rounding answers a question nobody asked, and the
    // caller never learns their setting was approximated.
    let (harness, _) = advertising(
        Features {
            withheld_tools: true,
            effort: true,
            ..Features::default()
        },
        "",
    );
    let model = HarnessModel::new(harness).unwrap();
    let mut request = ask(vec![LmMessage::user(["2+2?"])]);
    request.config.reasoning = Some(LmReasoningConfig {
        effort: Some("xhigh".into()),
        ..LmReasoningConfig::default()
    });

    let refused = model.forward(&request).await.unwrap_err().to_string();
    assert!(refused.contains("xhigh"), "names the value: {refused}");
    assert!(refused.contains("high"), "and what it does take: {refused}");
}

#[tokio::test]
async fn an_effort_is_refused_where_the_adapter_has_nowhere_to_put_it() {
    // Claude Code has no effort flag, so `Features::effort` is false there. Sending
    // it anyway is the silent drop; the caller hears about it instead.
    let (harness, _) = answering("");
    let model = HarnessModel::new(harness).unwrap();
    let mut request = ask(vec![LmMessage::user(["2+2?"])]);
    request.config.reasoning = Some(LmReasoningConfig {
        effort: Some("high".into()),
        ..LmReasoningConfig::default()
    });

    let refused = model.forward(&request).await.unwrap_err().to_string();
    assert!(
        refused.contains("reasoning effort"),
        "names the knob: {refused}"
    );
}

#[tokio::test]
async fn a_knob_that_is_not_a_sampling_one_is_refused_on_the_same_terms() {
    // `extensions` is a map rather than an option, so it needs its own reading of
    // "the caller asked" — an empty one is nobody asking.
    let (harness, _) = answering("");
    let model = HarnessModel::new(harness).unwrap();
    let mut request = ask(vec![LmMessage::user(["q"])]);
    request
        .config
        .extensions
        .insert("service_tier".into(), json!("flex"));

    let refused = model.forward(&request).await.unwrap_err().to_string();
    assert!(refused.contains("extensions"), "names the knob: {refused}");
}

#[tokio::test]
async fn a_rollout_runs_when_the_caller_takes_the_agents_own_variation() {
    // `Sampling::rollout` is temperature 1.0 and a fresh id — how every retry-shaped
    // module makes attempt two differ. Refused by default; opted into by name, the
    // agent's own variation stands in and `BestOfN` and `Refine` can run at all.
    let (harness, seen) = answering("[[ ## answer ## ]]\n4");
    let model = HarnessModel::builder(harness)
        .temperature(Temperature::FromTheAgent)
        .build()
        .unwrap();
    let mut request = ask(vec![LmMessage::user(["2+2?"])]);
    request.config.temperature = Some(1.0);

    model.forward(&request).await.unwrap();
    assert_eq!(
        seen.lock().unwrap().len(),
        1,
        "the run started rather than being turned away"
    );
}

#[tokio::test]
async fn the_opt_in_does_not_reach_a_knob_the_agent_cannot_stand_in_for() {
    // COPRO asks for several completions from one call. An agent answers once, so
    // tolerating that would hand it one proposal where it asked for ten — which no
    // amount of run-to-run variation substitutes for.
    let (harness, seen) = answering("");
    let model = HarnessModel::builder(harness)
        .temperature(Temperature::FromTheAgent)
        .build()
        .unwrap();
    let mut request = ask(vec![LmMessage::user(["q"])]);
    request.config.temperature = Some(0.7);
    request.config.n = Some(10);

    let refused = model.forward(&request).await.unwrap_err().to_string();
    assert!(
        refused.contains("n"),
        "still names the completions: {refused}"
    );
    assert!(
        !refused.contains("temperature"),
        "but not the knob that was opted into: {refused}"
    );
    assert!(seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn replaced_instructions_are_the_ones_that_travel() {
    // The other half of `instructions`: the existing test pins that blank removes the
    // standing text and never that non-blank text arrives. A mutant that inverted the
    // blank check dropped every real instruction and kept the empty one, and nothing
    // noticed, because `None` and a blank both read as "nothing" downstream.
    let (harness, seen) = answering("x");
    let model = HarnessModel::builder(harness)
        .instructions("be terse")
        .build()
        .unwrap();
    model
        .forward(&ask(vec![LmMessage::user(["q"])]))
        .await
        .unwrap();
    let run = &seen.lock().unwrap()[0];
    let prompt = run
        .tuning
        .system_prompt
        .as_deref()
        .expect("the replacement travels");
    assert!(prompt.contains("be terse"), "{prompt:?}");
    assert!(
        !prompt.contains(MARKER_DISCIPLINE),
        "replaced, not appended to: {prompt:?}"
    );
}

#[tokio::test]
async fn a_thinking_cap_reaches_the_run_and_the_request_beats_the_builder() {
    // Every earlier assertion on the cap used zero, so a body of `Some(0)` survived
    // mutation. Non-zero in both directions: the builder's value travels when the
    // request names none, and a request's `reasoning.max_tokens` wins over it —
    // dspy's order, per-call kwargs over `lm.kwargs`.
    let (harness, seen) = answering("x");
    let model = HarnessModel::builder(harness)
        .max_thinking_tokens(7)
        .build()
        .unwrap();

    model
        .forward(&ask(vec![LmMessage::user(["q"])]))
        .await
        .unwrap();
    let mut request = ask(vec![LmMessage::user(["q"])]);
    request.config.reasoning = Some(LmReasoningConfig {
        max_tokens: Some(3),
        ..LmReasoningConfig::default()
    });
    model.forward(&request).await.unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(
        seen[0].tuning.max_thinking_tokens,
        Some(7),
        "the builder's cap, when the request names none"
    );
    assert_eq!(
        seen[1].tuning.max_thinking_tokens,
        Some(3),
        "the request's cap beats the builder's"
    );
}
