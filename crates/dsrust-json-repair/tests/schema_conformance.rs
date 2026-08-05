//! Schema-guided repair, against `json_repair` — with `jsonschema`'s answers replayed.
//!
//! The repairer asks a validator questions and decides what to do with the answers. Reproducing
//! the validator would be porting a fourth library; reproducing the *decisions* is this crate's
//! job, and this separates the two. The fixture records every question upstream asked and what it
//! was told, and the validator below replays that table.
//!
//! **A question that was never asked is a failure**, which is what makes this an oracle rather
//! than a stub: a port that validates a different value, validates one time fewer, or skips
//! validation altogether cannot get past it.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use json_repair::{Repair, SchemaRepairMode, SchemaValidator, ValidationError, Value};
use serde_json::Value as Json;

fn fixture() -> Json {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/json_repair_schema.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{}: {error} — run scripts/generate_json_repair_schema_fixture.py",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("the fixture is JSON")
}

/// Answers only the questions upstream was recorded asking, and remembers which it was asked.
struct Replay {
    answers: HashMap<(String, String, String), Json>,
    asked: RefCell<BTreeSet<(String, String, String)>>,
}

impl Replay {
    fn new(calls: &[Json]) -> Self {
        let answers = calls
            .iter()
            .map(|call| {
                let key = (
                    call["method"].as_str().expect("a method").to_owned(),
                    call["value"].as_str().expect("a value").to_owned(),
                    call["schema"].as_str().expect("a schema").to_owned(),
                );
                (key, call.clone())
            })
            .collect();
        Self {
            answers,
            asked: RefCell::new(BTreeSet::new()),
        }
    }

    fn answer(&self, method: &str, value: &Value, schema: &Value) -> Json {
        let key = (method.to_owned(), value.to_string(), schema.to_string());
        self.asked.borrow_mut().insert(key.clone());
        self.answers.get(&key).cloned().unwrap_or_else(|| {
            panic!(
                "the repairer asked `{method}` a question json_repair never asked:\n  value:  {}\n  schema: {}\n  recorded: {:#?}",
                key.1,
                key.2,
                self.answers.keys().collect::<Vec<_>>()
            )
        })
    }
}

impl SchemaValidator for Replay {
    fn is_valid(&self, value: &Value, schema: &Value) -> Result<bool, ValidationError> {
        let answer = self.answer("is_valid", value, schema);
        match answer["raised"].as_str() {
            Some(message) => Err(raised(&answer, message)),
            None => Ok(answer["ok"].as_bool().expect("a recorded answer")),
        }
    }

    fn validate(&self, value: &Value, schema: &Value) -> Result<(), ValidationError> {
        let answer = self.answer("validate", value, schema);
        if let Some(message) = answer["raised"].as_str() {
            return Err(raised(&answer, message));
        }
        match answer["ok"].as_bool().expect("a recorded answer") {
            true => Ok(()),
            false => Err(ValidationError::Invalid(
                answer["message"].as_str().expect("a message").to_owned(),
            )),
        }
    }
}

/// The exception the validator raised, as its class decides. `json_repair` catches `TypeError` in
/// one place and `jsonschema`'s own errors nowhere, so replaying them as one kind would hide which.
fn raised(answer: &Json, message: &str) -> ValidationError {
    match answer["raised_type"].as_str() {
        Some("TypeError") => ValidationError::Type(message.to_owned()),
        _ => ValidationError::Unreadable(message.to_owned()),
    }
}

/// The fixture's schema, in the crate's own `Value` — the same text on both sides.
fn schema_of(case: &Json) -> Value {
    from_json(&case["schema"])
}

fn from_json(node: &Json) -> Value {
    match node {
        Json::Null => Value::Null,
        Json::Bool(flag) => Value::Bool(*flag),
        Json::Number(number) => match number.as_i64() {
            Some(whole) if !node.to_string().contains(['.', 'e', 'E']) => Value::Int(whole),
            _ => Value::Float(number.as_f64().expect("a float")),
        },
        Json::String(text) => Value::Str(text.clone()),
        Json::Array(items) => Value::Array(items.iter().map(from_json).collect()),
        Json::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(key, value)| (key.clone(), from_json(value)))
                .collect(),
        ),
    }
}

#[test]
fn every_schema_guided_repair_decides_what_json_repair_decided() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    let (mut asked, mut logged) = (0, 0);

    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let calls = case["calls"].as_array().expect("the recorded calls");
        let replay = Rc::new(Replay::new(calls));
        let mode = match case["mode"].as_str() {
            Some("salvage") => SchemaRepairMode::Salvage,
            _ => SchemaRepairMode::Standard,
        };
        let repair = Repair::new()
            .schema(schema_of(case))
            .schema_repair_mode(mode)
            .validator(replay.clone());

        match (
            case["ok"].as_bool().expect("ok"),
            repair.loads(case["input"].as_str().expect("input")),
        ) {
            (true, Ok(ours)) => assert_eq!(
                ours.to_string(),
                case["dumps"].as_str().expect("a dumps"),
                "{name}: {}",
                case["why"]
            ),
            // The *message* too. Comparing only that both refused let a `TypeError` from iterating
            // a number pass as "No schema matched the value" — two different refusals, and the
            // mutation that swapped them survived because nothing looked.
            (false, Err(error)) => assert_eq!(
                error.message(),
                case["message"].as_str().expect("a message"),
                "{name}: refused for a different reason"
            ),
            (true, Err(error)) => {
                panic!(
                    "{name}: json_repair returned {} and we refused: {error}",
                    case["dumps"]
                )
            }
            (false, Ok(ours)) => {
                panic!(
                    "{name}: json_repair raised {} and we returned {ours}",
                    case["message"]
                )
            }
        }

        // The questions matter as much as the answers: asking fewer means a check was skipped.
        let ours = replay.asked.borrow().len();
        let theirs = replay.answers.len();
        assert_eq!(
            ours, theirs,
            "{name}: asked {ours} distinct questions, json_repair asked {theirs}"
        );
        asked += ours;
        logged += assert_the_log_matches(case, &repair, name);
    }

    assert!(
        asked > 40,
        "only {asked} validator calls across the corpus — the seam is barely used"
    );
    assert!(
        logged > 30,
        "only {logged} logged schema repairs, so most of the narration is unpinned"
    );
    eprintln!(
        "  {} cases, {asked} replayed validator calls, {logged} logged repairs, from {}",
        cases.len(),
        fixture["source"]
    );
}

/// The repair log for one case, which says *which* rule produced the value rather than only that
/// something did. Answers how many entries it checked.
fn assert_the_log_matches(case: &Json, repair: &Repair, name: &str) -> usize {
    let Some(expected) = case["log"].as_array() else {
        return 0;
    };
    let (_, ours) = repair
        .loads_logged(case["input"].as_str().expect("input"))
        .expect("this case parses");
    let ours: Vec<Json> = ours
        .into_iter()
        .map(|entry| serde_json::json!({ "text": entry.text, "context": entry.context }))
        .collect();
    assert_eq!(&ours, expected, "{name}: the repairs differ");
    ours.len()
}

#[test]
fn without_a_validator_the_crate_answers_as_python_without_jsonschema() {
    // Upstream raises `ValueError("jsonschema is required when using schema-aware repair.")` when
    // the package is absent, and the message reaches the caller through the same path: the outer
    // `except` in `repair_json` swallows it on the fast path, so the parser runs and the *final*
    // `validate` is the one that raises.
    let schema = Value::Object(
        [("type".to_owned(), Value::Str("object".to_owned()))]
            .into_iter()
            .collect(),
    );
    let error = Repair::new()
        .schema(schema)
        .loads("{a: 1}")
        .expect_err("no validator is plugged in");
    assert_eq!(
        error.message(),
        "jsonschema is required when using schema-aware repair."
    );
}
