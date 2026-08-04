//! Differential fuzzing against `json_repair` itself: random malformed JSON, both sides.
//!
//! The committed fixture holds cases chosen by reading the library's branches, which is exactly
//! the method that leaves the branches nobody thought of. This closes that: both sides are pure
//! functions from a string to a value, and the library is on hand as the reference.
//!
//! **Runs only when a campaign has been run.** `scripts/fuzz_json_repair.py` writes
//! `target/json_repair_fuzz.json`, a campaign artifact rather than a golden — twenty thousand
//! random strings are evidence, not documentation. Run the script, run this, and promote whatever
//! disagrees into `scripts/json_repair_corpus.py` as a named case with its reason.
//!
//! `target/` is gitignored, and cargo-mutants copies only the source tree, so under mutation this
//! corpus is always absent and every one of those cases scores nothing. `sweep.rs` is what covers
//! the regime there: five hundred generated cases committed under `tests/conformance/`. The absent
//! path below asserts that file is present rather than returning quietly, so this can never be the
//! last thing standing between a change and the differential oracle.

use json_repair::Repair;
use serde_json::Value as Json;

fn corpus() -> Option<Json> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/json_repair_fuzz.json");
    let text = std::fs::read_to_string(&path).ok()?;
    // A corpus that is present and unreadable must not skip as quietly as one that is absent.
    Some(serde_json::from_str(&text).unwrap_or_else(|error| {
        panic!(
            "{}: {error} — the corpus is there and does not parse",
            path.display()
        )
    }))
}

/// How a case disagreed.
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Shape {
    /// Both refused, for different reasons.
    DifferentRefusal,
    /// json_repair parsed it and we refused.
    WeRefused,
    /// We parsed it and json_repair refused.
    WeAccepted,
    /// Both parsed it, differently.
    DifferentValue,
}

/// The committed differential corpus, which is what covers this regime when no campaign has run.
const SWEEP: &str = "tests/conformance/json_repair_sweep.json";

/// One case, through the door it was recorded from and under the options it was recorded with.
///
/// `load(fd)` is not an alias for `loads(text)`: upstream turns the valid-JSON suffix fast path off
/// for file input, so the two disagree on roughly one input in four hundred. A campaign that sent
/// every case through `loads` could not see that, and it is a divergence this repo already
/// documents. The answer is a string either way so the comparison stays one line — a value dumped,
/// or the text `repair_json` writes.
fn answer(case: &Json, input: &str) -> Result<String, json_repair::Error> {
    let flag = |name: &str| case["options"][name].as_bool().unwrap_or(false);
    let repair = Repair::new()
        .strict(flag("strict"))
        .skip_json_loads(flag("skip_json_loads"))
        .stream_stable(flag("stream_stable"))
        .ensure_ascii(case["options"]["ensure_ascii"].as_bool().unwrap_or(true));
    match case["entry"].as_str().unwrap_or("loads") {
        "load" => repair.from_reader(input.as_bytes()).map(|v| v.to_string()),
        "repair_json" => repair.repair_json(input),
        _ => repair.loads(input).map(|v| v.to_string()),
    }
}

/// A campaign small enough to be a leftover rather than a run.
const A_CAMPAIGN: usize = 100;

#[test]
fn a_fuzz_campaign_agrees_with_json_repair_when_one_has_been_run() {
    let Some(corpus) = corpus() else {
        // `eprintln!` is captured by the harness, so the old name plus a swallowed line read as
        // "no random input repairs differently" having compared nothing at all. The name says what
        // is conditional now, and the absent path still asserts the one thing that must hold: that
        // something covers this regime. `sweep.rs` reads the committed corpus below and panics
        // without it, and cargo-mutants copies only the source tree — so under mutation the
        // campaign artifact in `target/` is *always* absent and the sweep is the whole oracle.
        let sweep = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(SWEEP);
        assert!(
            sweep.exists(),
            "no campaign corpus and no {SWEEP} either — nothing in this crate compares random \
             input against json_repair. Run `.venv/bin/python scripts/fuzz_json_repair.py`, or \
             regenerate the sweep."
        );
        return;
    };
    let cases = corpus["cases"].as_array().expect("cases");
    assert!(
        cases.len() >= A_CAMPAIGN,
        "{} cases is not a campaign — a stale or truncated corpus passing here says as little as \
         an absent one",
        cases.len()
    );

    let mut disagreements: Vec<(Shape, String)> = Vec::new();
    for case in cases {
        let input = case["input"].as_str().expect("an input");
        let ours = answer(case, input);
        // What the case is compared against depends on the door it went through: `repair_json`
        // hands back the *text* json.dumps would write, the other two hand back a value.
        let expected = match case["entry"].as_str().unwrap_or("loads") {
            "repair_json" => &case["repaired"],
            _ => &case["dumps"],
        };
        let shape = match (case["ok"].as_bool().expect("ok"), &ours) {
            (true, Ok(ours)) if ours == expected.as_str().expect("an answer") => {
                continue;
            }
            // Including *why*: matching a refusal against a refusal without comparing the
            // message is how a wrong error passes for a right one, which the schema suite proved.
            (false, Err(ours))
                if ours.message() == case["message"].as_str().unwrap_or_default() =>
            {
                continue;
            }
            (false, Err(_)) => Shape::DifferentRefusal,
            (true, Err(_)) => Shape::WeRefused,
            (false, Ok(_)) => Shape::WeAccepted,
            (true, Ok(_)) => Shape::DifferentValue,
        };
        let ours = match &ours {
            Ok(value) => value.to_string(),
            Err(error) => format!("Err({error})"),
        };
        disagreements.push((
            shape,
            format!(
                "{input:?}\n      ours: {ours}\n      json_repair: {}",
                case["dumps"]
            ),
        ));
    }

    if disagreements.is_empty() {
        eprintln!(
            "  {} random inputs, seed {}: no disagreement",
            cases.len(),
            corpus["seed"]
        );
        return;
    }

    disagreements.sort_by(|left, right| left.0.cmp(&right.0));
    let mut tally: std::collections::BTreeMap<&str, usize> = Default::default();
    for (shape, _) in &disagreements {
        *tally.entry(label(shape)).or_default() += 1;
    }
    eprintln!();
    for (label, count) in &tally {
        eprintln!("  {count:>5}  {label}");
    }
    eprintln!();
    for (shape, detail) in disagreements.iter().take(30) {
        eprintln!("  {}\n      {detail}\n", label(shape));
    }
    panic!(
        "{} of {} random inputs disagree with json_repair (seed {})",
        disagreements.len(),
        cases.len(),
        corpus["seed"]
    );
}

fn label(shape: &Shape) -> &'static str {
    match shape {
        Shape::WeRefused => "json_repair parsed, we refused",
        Shape::WeAccepted => "we parsed, json_repair refused",
        Shape::DifferentValue => "both parsed, differently",
        Shape::DifferentRefusal => "both refused, differently",
    }
}
