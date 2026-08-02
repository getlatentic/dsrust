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
    read(case)
        .map(|value| value.to_string())
        .map_err(|error| error.message().to_owned())
}

/// The value, through whichever entry point the case was recorded from. `load(fd)` is a different
/// parse from `loads(text)`, not an alias for it.
fn read(case: &Json) -> Result<json_repair::Value, json_repair::Error> {
    let input = case["input"].as_str().expect("an input");
    match case["from_file"].as_bool().unwrap_or(false) {
        true => options(case).from_reader(input.as_bytes()),
        false => options(case).loads(input),
    }
}

/// The repair log, as the fixture spells it: one `{"text", "context"}` object per entry.
fn repair_log(case: &Json) -> Vec<Json> {
    let input = case["input"].as_str().expect("an input");
    let (_, log) = options(case).loads_logged(input).expect("this case parses");
    log.into_iter()
        .map(|entry| serde_json::json!({ "text": entry.text, "context": entry.context }))
        .collect()
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
            // The message too, not merely that both refused: two different refusals are two
            // different behaviours, and comparing only the fact of one let a mutation that swapped
            // them survive in the schema suite.
            (false, Err(error)) => assert_eq!(
                error,
                case["message"].as_str().expect("a message"),
                "{name}: {why}\n  refused for a different reason"
            ),
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
fn reading_a_file_is_not_the_same_call_as_reading_a_string() {
    // Upstream ties the suffix fast path to where the input came from, so the same bytes read two
    // ways can answer two things. Asserted as a *difference* on the cases where one was measured,
    // because an equality here would pass just as well if `from_file` were an alias for `loads`.
    let fixture = fixture();
    let mut differing = 0;
    for case in fixture["cases"].as_array().expect("cases") {
        if !case["from_file"].as_bool().unwrap_or(false) {
            continue;
        }
        let input = case["input"].as_str().expect("an input");
        let as_file = read(case).expect("the file path parses").to_string();
        assert_eq!(
            as_file,
            case["dumps"].as_str().expect("a dumps"),
            "{}",
            case["name"]
        );
        if json_repair::loads(input)
            .expect("the string path parses")
            .to_string()
            != as_file
        {
            differing += 1;
        }
    }
    assert!(
        differing >= 3,
        "only {differing} file cases read differently from the string path — the corpus stopped \
         covering the distinction, or the two paths have become the same"
    );
}

#[test]
fn repair_json_returns_the_text_json_dumps_writes() {
    // `repair_json` is `loads` written back out, which is the same bytes the fixture already
    // records — except for the empty string, which upstream returns as itself rather than as a
    // pair of quotes, and which nothing else in the suite reaches.
    let fixture = fixture();
    let mut empty_strings = 0;
    for case in fixture["cases"].as_array().expect("cases") {
        if case.get("diverges").is_some()
            || !case["ok"].as_bool().expect("ok")
            || case["from_file"].as_bool().unwrap_or(false)
        {
            continue;
        }
        let name = case["name"].as_str().expect("a name");
        let input = case["input"].as_str().expect("an input");
        let ours = options(case).repair_json(input).expect("this case parses");
        let dumped = case["dumps"].as_str().expect("a dumps");
        if dumped == "\"\"" {
            assert_eq!(ours, "", "{name}: the empty string comes back as itself");
            empty_strings += 1;
            continue;
        }
        assert_eq!(ours, dumped, "{name}");
    }
    assert!(
        empty_strings > 0,
        "no case reaches the empty-string branch, so it is untested"
    );
}

#[test]
fn every_repair_is_logged_the_way_json_repair_logs_it() {
    // The stronger of the two oracles. A value says where the parse arrived; the log says which
    // branch it took to get there, so a port that reaches the right answer by the wrong route is
    // caught here and nowhere else. The context window — ten code points either side of the
    // cursor — pins *where* it was when it decided.
    let fixture = fixture();
    let mut entries = 0;
    for case in fixture["cases"].as_array().expect("cases") {
        if case.get("diverges").is_some()
            || !case["ok"].as_bool().expect("ok")
            || case["from_file"].as_bool().unwrap_or(false)
        {
            continue;
        }
        let name = case["name"].as_str().expect("a name");
        let expected = case["log"].as_array().expect("a log");
        let ours = repair_log(case);
        assert_eq!(
            ours.len(),
            expected.len(),
            "{name}: {} log entries against json_repair's {}\n  ours: {ours:#?}\n  theirs: {expected:#?}",
            ours.len(),
            expected.len(),
        );
        assert_eq!(&ours, expected, "{name}: the repairs differ");
        entries += ours.len();
    }
    // Cases that repair *nothing* log nothing, so a fixture of only-valid inputs would pass this
    // while pinning no branch at all. A floor at the measured count rather than a round number:
    // adding cases raises it, and losing coverage fails here instead of quietly passing.
    assert!(
        entries >= 191,
        "only {entries} logged repairs across the corpus"
    );
    eprintln!("  {entries} logged repairs match");
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

#[test]
fn the_two_free_functions_are_the_default_arguments_and_nothing_else() {
    // `loads` and `repair_json` are the whole API most callers touch, and each is the builder with
    // no options set. The rest of this file goes through `Repair`, so without this the two
    // shorthands are reachable and unexercised.
    assert_eq!(
        json_repair::repair_json("{a: 1,}").expect("repaired"),
        r#"{"a": 1}"#
    );
    assert_eq!(
        json_repair::repair_json("prose").expect("repaired"),
        "",
        "the empty string is itself"
    );
    assert_eq!(
        json_repair::loads("{a: 1,}").expect("repaired"),
        Repair::new().loads("{a: 1,}").expect("repaired"),
    );
}

#[test]
fn a_refusal_prints_the_message_upstream_raises_with() {
    // Upstream's is a `ValueError`, so its text *is* what a caller sees. Reading it off `Display`
    // rather than off `message()` is what pins the impl.
    let error = Repair::new()
        .strict(true)
        .loads("{\"a\" 1}")
        .expect_err("strict mode refuses a missing separator");
    assert_eq!(
        error.to_string(),
        "Missing ':' after key in strict mode while parsing object."
    );
}
