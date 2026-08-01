//! What a saved LM block survives, and what `allow_unsafe_lm_state` gates.
//!
//! dspy 3.3 sanitises the block on load — `_sanitize_lm_state` drops `api_base`, `base_url` and
//! `model_list` unless the caller vouches for the file — because those three decide *where* a call
//! goes, and a compiled program is something people pass around. The rest names a model or a
//! sampling setting and is kept, then rebuilt into a live LM the predictor actually asks.
//!
//! This crate does the same, through [`dsrust::module::Trust`]. It did not until
//! `saved-lm-round-trip`: `load_state` restored a signature and its demos and dropped the block
//! whole, so a dspy-compiled program lost the model its author pinned it to. Both halves are
//! measured here — what a default load drops, and what it keeps — because a port that dropped
//! everything would pass a test that only checked the redirect was gone.
//!
//! The block's own bytes are `tests/saved_lm.rs`; this is the program-level path around them.

use dsrust::module::Trust;
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

/// The block a program carries after loading the golden and saving itself again.
fn round_tripped(name: &str, lm: &Value, load: fn(&mut Predict, &std::path::Path)) -> Value {
    let path = saved_with_lm(name, lm);
    let mut program = Predict::parse("question -> answer").expect("parses");
    load(&mut program, &path);
    let carried = serde_json::to_value(program.dump_state()).expect("serialises");
    let _ = std::fs::remove_file(&path);
    carried["self"]["lm"].clone()
}

/// An ordinary load keeps the model pin, which is the whole reason dspy writes the block.
///
/// Compared against what dspy's own load-then-save kept, key for key — not against a list written
/// here, which would agree with whatever this crate happened to do.
#[test]
fn an_ordinary_load_keeps_what_dspy_keeps() {
    let fixture = fixture();
    let kept = fixture["sanitized"].as_object().expect("what dspy kept");
    let ours = round_tripped("keeps", &fixture["saved_lm"], |program, path| {
        program.load(path).expect("loads");
    });
    let ours = ours.as_object().expect("a block survives an ordinary load");

    for (key, value) in kept {
        assert_eq!(
            ours.get(key),
            Some(value),
            "dspy kept {key} through its own round-trip and this did not"
        );
    }
    assert!(
        kept.contains_key("model") && kept.contains_key("temperature"),
        "the golden must carry a pin for this to be testing anything"
    );
}

/// An ordinary load drops the three keys that decide where a call goes, so re-saving the program
/// cannot pass a redirect on to whoever opens it next.
#[test]
fn an_ordinary_load_drops_the_redirect() {
    let fixture = fixture();
    let unsafe_keys: Vec<&str> = fixture["unsafe_keys"]
        .as_array()
        .expect("the golden names them")
        .iter()
        .map(|key| key.as_str().expect("a string"))
        .collect();
    assert_eq!(unsafe_keys.len(), 3, "dspy names three");

    let saved = fixture["saved_lm"].as_object().expect("the block");
    for key in &unsafe_keys {
        assert!(saved.contains_key(*key), "the golden must carry {key}");
    }

    let ours = round_tripped("drops", &fixture["saved_lm"], |program, path| {
        program.load(path).expect("loads");
    });
    let ours = ours.as_object().expect("a block");
    for key in &unsafe_keys {
        assert!(!ours.contains_key(*key), "{key} was laundered onward");
    }
}

/// Vouching for the file keeps the redirect, which is what dspy's flag is for.
///
/// The flag is only observable as the difference between the two loads, so both are asserted
/// against the golden's own two arms rather than against each other.
#[test]
fn a_trusted_load_keeps_the_redirect() {
    let fixture = fixture();
    let trusted = fixture["trusted"]
        .as_object()
        .expect("what dspy kept when trusted");
    let ours = round_tripped("trusted", &fixture["saved_lm"], |program, path| {
        program.load_trusted(path).expect("loads");
    });
    let ours = ours.as_object().expect("a block");

    assert!(
        trusted.contains_key("api_base"),
        "the golden's trusted arm must differ from its default one"
    );
    assert_eq!(
        ours.get("api_base"),
        trusted.get("api_base"),
        "a trusted load should reach the endpoint it was compiled against"
    );

    // The other two are kept by the sanitiser and then not honoured, so they do not come back out.
    // `base_url` is litellm's alias for the field `api_base` already set — restoring both would
    // mean deciding which wins, which is a question only litellm's resolution order answers.
    // `model_list` configures its router, and this crate has no router to configure. Asserted
    // rather than left unsaid: a trusted round-trip here is narrower than dspy's, and the place to
    // find that out is a test rather than a diff of two saved files.
    for key in ["base_url", "model_list"] {
        assert!(
            trusted.contains_key(key),
            "the golden's trusted arm should carry {key}"
        );
        assert!(
            !ours.contains_key(key),
            "{key} came back, so this comment is now wrong"
        );
    }
}

/// `Trust` says what it gates rather than leaving a `true` at the call site to explain itself.
#[test]
fn trust_names_what_it_widens() {
    assert!(!Trust::Default.allows_redirect());
    assert!(Trust::File.allows_redirect());
}
