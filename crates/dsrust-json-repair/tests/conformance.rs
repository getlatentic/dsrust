//! Every case in `tests/conformance/json_repair.json`, against the library that produced it.
//!
//! The comparison is on the bytes `json.dumps` writes, not on a structural copy: that is what
//! keeps `7` apart from `7.0`, keeps a key at the position it was first assigned, and keeps an
//! integer wider than a machine word exact. A structural comparison would agree in all three
//! places while the values differed.

use json_repair::{Repair, Value};
use serde_json::Value as Json;

fn fixture() -> Json {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/json_repair.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error} — run scripts/generate_json_repair_fixture.py",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("the fixture is JSON")
}

fn options(case: &Json) -> Repair {
    let flag = |name: &str| case["options"][name].as_bool().unwrap_or(false);
    Repair::new()
        .strict(flag("strict"))
        .stream_stable(flag("stream_stable"))
        .skip_json_loads(flag("skip_json_loads"))
}

/// What one case answered, as the fixture spells it: the dumped value, or the refusal.
fn answer(case: &Json) -> Result<String, String> {
    let input = case["input"].as_str().expect("an input");
    options(case)
        .loads(input)
        .map(|value| value.to_string())
        .map_err(|error| error.message().to_owned())
}

#[test]
fn every_recorded_input_parses_to_the_bytes_json_repair_produced() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    let mut checked = 0;

    for case in cases {
        let name = case["name"].as_str().expect("a name");
        if case.get("diverges").is_some() {
            continue;
        }
        let why = case["why"].as_str().unwrap_or("");
        match (case["ok"].as_bool().expect("ok"), answer(case)) {
            (true, Ok(ours)) => {
                let expected = case["dumps"].as_str().expect("a dumps");
                assert_eq!(
                    ours, expected,
                    "{name}: {why}\n  input: {:?}",
                    case["input"]
                );
            }
            (false, Err(_)) => {}
            (true, Err(error)) => panic!(
                "{name}: {why}\n  json_repair returned {}\n  we refused: {error}",
                case["dumps"]
            ),
            (false, Ok(ours)) => panic!(
                "{name}: {why}\n  json_repair raised {}\n  we returned: {ours}",
                case["message"]
            ),
        }
        checked += 1;
    }

    // A fixture that lost its cases would otherwise pass in silence.
    assert!(
        checked > 100,
        "only {checked} cases — the fixture is not the one that was generated"
    );
    eprintln!("  {checked} cases against {}", fixture["source"]);
}

#[test]
fn the_declared_divergences_still_diverge() {
    // Asserted the *other* way round, so closing one of these turns this test red and says which.
    // A gap recorded as an equality would be a gap nobody is told about when it closes.
    let fixture = fixture();
    let mut declared = 0;
    for case in fixture["cases"].as_array().expect("cases") {
        let Some(reason) = case.get("diverges").and_then(Json::as_str) else {
            continue;
        };
        let name = case["name"].as_str().expect("a name");
        assert_eq!(
            reason, "lone-surrogate",
            "{name}: an undeclared kind of divergence"
        );
        let ours = answer(case).expect("the parse itself still succeeds");
        assert_ne!(
            ours,
            case["dumps"].as_str().expect("a dumps"),
            "{name}: this now agrees with json_repair — drop the `diverges` marker in \
             scripts/json_repair_corpus.py"
        );
        assert!(
            ours.contains("\\ufffd"),
            "{name}: the substitute for a surrogate a Rust char cannot hold is U+FFFD, got {ours}"
        );
        declared += 1;
    }
    assert_eq!(
        declared, 1,
        "the fixture declares a different number of divergences than this test"
    );
}

#[test]
fn the_generator_measured_what_the_corpus_reaches() {
    // The fixture carries the line counts its own generator traced through `json_repair`. They are
    // not an assertion about this crate; they are what stops the corpus being quietly narrowed to
    // whatever still passes.
    let fixture = fixture();
    let coverage = fixture["coverage"].as_object().expect("a coverage block");
    let string_lines = coverage["parse_string.py"].as_u64().expect("a count");
    assert!(
        string_lines > 350,
        "the corpus reaches {string_lines} lines of parse_string.py, which is where every known \
         disagreement came from"
    );
}

#[test]
fn a_value_reads_back_the_way_a_caller_would_use_it() {
    let fields =
        json_repair::loads(r#"{answer: "Paris", "why": 'the capital',}"#).expect("repaired");
    assert_eq!(fields.get("answer"), Some(&Value::Str("Paris".into())));
    assert_eq!(fields.get("why"), Some(&Value::Str("the capital".into())));
    assert_eq!(fields.get("missing"), None);
}
