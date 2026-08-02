//! Differential fuzzing against `json_repair` itself: random malformed JSON, both sides.
//!
//! The committed fixture holds cases chosen by reading the library's branches, which is exactly
//! the method that leaves the branches nobody thought of. This closes that: both sides are pure
//! functions from a string to a value, and the library is on hand as the reference.
//!
//! **Skips unless the corpus is there.** `scripts/fuzz_json_repair.py` writes
//! `target/json_repair_fuzz.json`, a campaign artifact rather than a golden — twenty thousand
//! random strings are evidence, not documentation. Run the script, run this, and promote whatever
//! disagrees into `scripts/json_repair_corpus.py` as a named case with its reason.

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

#[test]
fn no_random_input_repairs_differently_from_json_repairs() {
    let Some(corpus) = corpus() else {
        eprintln!(
            "no target/json_repair_fuzz.json — run `.venv/bin/python scripts/fuzz_json_repair.py`"
        );
        return;
    };
    let cases = corpus["cases"].as_array().expect("cases");

    let mut disagreements: Vec<(Shape, String)> = Vec::new();
    for case in cases {
        let input = case["input"].as_str().expect("an input");
        let ours = json_repair::loads(input);
        let shape = match (case["ok"].as_bool().expect("ok"), &ours) {
            (true, Ok(ours)) if ours.to_string() == case["dumps"].as_str().expect("dumps") => {
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
