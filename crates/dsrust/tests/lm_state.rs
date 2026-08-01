//! What a saved LM block does *not* survive, and why nothing gates it here.
//!
//! dspy 3.3 sanitises the block on load — `_sanitize_lm_state` drops `api_base`, `base_url` and
//! `model_list` unless `allow_unsafe_lm_state=True` — because those three decide *where* a call
//! goes, and a compiled program is something people pass around. The rest of the block names a
//! model or a sampling setting and is kept, so a dspy round-trip preserves the model pin.
//!
//! This crate drops the **whole** block instead: `load_state` restores a signature and its demos,
//! and `dump_state` rebuilds from the predictor, which has nowhere to hold a saved LM dict. So
//! there is nothing for a trust flag to gate — a redirect cannot be acted on, and cannot be
//! laundered onward either. What is lost is the pin dspy would have kept.
//!
//! Both halves are measured here rather than argued, because the first reading of this went the
//! other way: a `ProgramState` deserialised and reserialised *does* carry the block, and that was
//! briefly mistaken for a program round-trip preserving it. It is a different path.

use dsrust::{Module, predict::Predict};
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/state/lm_state.json");
    let text = std::fs::read_to_string(&path).expect("the lm-state golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

/// A saved program carrying the golden's `lm` block, as one written elsewhere would.
///
/// The name *and* the process id are both in the path. The id keeps two copies of this binary
/// apart; the name keeps the tests in this one apart, since they run in parallel and one removing
/// its file while the other reads it is the same empty-file race one process wide. Reproduced here
/// before it was fixed here.
fn saved_with_lm(name: &str, lm: &Value) -> std::path::PathBuf {
    let mut program = Predict::parse("question -> answer").expect("parses");
    let mut state = serde_json::to_value(program.dump_state()).expect("serialises");
    state["self"]["lm"] = lm.clone();

    let path = std::env::temp_dir().join(format!(
        "dsrust-lm-state-{name}-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, state.to_string()).expect("writes");
    path
}

/// A loaded program carries no LM state at all, so re-saving it cannot pass on an endpoint
/// redirect that a dspy load would have dropped.
///
/// This is what makes `allow_unsafe_lm_state` have nothing to gate here. It is a stronger position
/// than upstream's rather than a weaker one — dspy keeps the safe keys and drops three; this keeps
/// none — and the cost is on the other side, in `the_model_pin_does_not_survive_a_round_trip`.
#[test]
fn a_loaded_program_carries_no_saved_lm_state() {
    let fixture = fixture();
    let path = saved_with_lm("carries-none", &fixture["saved_lm"]);

    let mut program = Predict::parse("question -> answer").expect("parses");
    program.load(&path).expect("loads");

    let carried = serde_json::to_value(program.dump_state()).expect("serialises");
    assert_eq!(
        carried["self"]["lm"],
        Value::Null,
        "a re-saved program should carry no LM block at all"
    );
    let _ = std::fs::remove_file(&path);
}

/// The divergence that costs something: dspy's own load-then-save keeps `model`, `temperature` and
/// the rest, so a program that pinned a model still names it. Here the pin is gone.
///
/// Asserted against what dspy actually kept, so it says exactly what is lost rather than "we drop
/// it". Filed as `saved-lm-round-trip`; when that lands this test says so.
#[test]
fn the_model_pin_does_not_survive_a_round_trip() {
    let fixture = fixture();
    let kept = fixture["sanitized"].as_object().expect("what dspy kept");
    assert!(
        kept.contains_key("model") && kept.contains_key("temperature"),
        "dspy should keep the model pin through its own round-trip"
    );

    let path = saved_with_lm("pin-lost", &fixture["saved_lm"]);
    let mut program = Predict::parse("question -> answer").expect("parses");
    program.load(&path).expect("loads");
    let carried = serde_json::to_value(program.dump_state()).expect("serialises");
    assert_eq!(
        carried["self"]["lm"],
        Value::Null,
        "the pin now survives — resolve `saved-lm-round-trip` and assert what is kept instead"
    );
    let _ = std::fs::remove_file(&path);
}
