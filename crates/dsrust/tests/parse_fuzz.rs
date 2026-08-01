//! Differential fuzzing against dspy's parsers: random replies, both sides, every disagreement.
//!
//! The committed goldens hold 51 cases, each chosen by reading a branch of `parse` — which is
//! exactly the method that leaves the branches nobody thought of. This closes that: both sides are
//! pure functions from a string to a value-or-error and dspy is on hand as the reference.
//!
//! **Skips unless the corpus is there.** `scripts/fuzz_parse.py` writes `target/parse_fuzz.json`,
//! which is a campaign artifact rather than a golden — ten thousand random strings are evidence, not
//! documentation, and they do not belong in git. Run the script, run this, promote whatever
//! disagrees into `generate_parse_fixture.py` as a named case with its reason.
//!
//! Reports **every** disagreement rather than failing on the first, because the useful output is the
//! shape of the disagreements, not the earliest one.

use dsrust::signature::Signature;
use dsrust::{Adapter, ChatAdapter, JsonAdapter, XmlAdapter};
use serde_json::Value;

fn corpus() -> Option<Value> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/parse_fuzz.json");
    let text = std::fs::read_to_string(&path).ok()?;
    // A corpus that is *present* and unreadable used to skip as quietly as one that was absent,
    // which would hide the corpus dspy writes for a value JSON has no literal for.
    Some(serde_json::from_str(&text).unwrap_or_else(|error| {
        panic!(
            "{}: {error} — the corpus is there and does not parse",
            path.display()
        )
    }))
}

fn adapter_for(name: &str) -> Box<dyn Adapter> {
    match name {
        "xml" => Box::new(XmlAdapter::default()),
        "json" => Box::new(JsonAdapter::default()),
        _ => Box::new(ChatAdapter::default()),
    }
}

/// How a case disagreed, so the summary can group rather than list.
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Shape {
    /// dspy parsed it and we refused.
    WeRefused,
    /// We parsed it and dspy refused — the worse direction: a wrong value reaches the caller.
    WeAccepted,
    /// Both parsed it, differently.
    DifferentValue,
}

#[test]
fn no_random_reply_parses_differently_from_dspys() {
    let Some(corpus) = corpus() else {
        eprintln!(
            "no target/parse_fuzz.json — run `.venv/bin/python scripts/fuzz_parse.py` to generate one"
        );
        return;
    };
    let signature: Signature = corpus["signature"]
        .as_str()
        .expect("a signature")
        .parse()
        .expect("the signature parses");
    let cases = corpus["cases"].as_array().expect("cases");

    let mut disagreements: Vec<(Shape, &str, String)> = Vec::new();
    for case in cases {
        let which = case["adapter"].as_str().unwrap_or("chat");
        let completion = case["completion"].as_str().expect("a completion");
        let expected = &case["expected"];
        let ours = adapter_for(which).parse(&signature, completion);

        let shape = match (expected["ok"].as_bool().expect("ok"), &ours) {
            (true, Ok(ours)) if *ours == expected["fields"] => continue,
            (false, Err(_)) => continue,
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
            which,
            format!("{completion:?}\n      ours: {ours}\n      dspy: {expected}"),
        ));
    }

    if disagreements.is_empty() {
        eprintln!(
            "  {} random replies, seed {}: no disagreement",
            cases.len(),
            corpus["seed"]
        );
        return;
    }

    disagreements.sort_by(|left, right| (&left.0, left.1).cmp(&(&right.0, right.1)));
    // The counts first: which adapter, and which direction. That is the finding; the examples
    // below are only how to reproduce it.
    let mut tally: std::collections::BTreeMap<(&str, &str), usize> = Default::default();
    for (shape, which, _) in &disagreements {
        let label = match shape {
            Shape::WeRefused => "dspy parsed, we refused",
            Shape::WeAccepted => "we parsed, dspy refused",
            Shape::DifferentValue => "both parsed, differently",
        };
        *tally.entry((which, label)).or_default() += 1;
    }
    eprintln!();
    for ((which, label), count) in &tally {
        eprintln!("  {count:>5}  [{which}] {label}");
    }
    eprintln!(
        "\n{} of {} random replies disagree (seed {}):\n",
        disagreements.len(),
        cases.len(),
        corpus["seed"]
    );
    for (shape, which, detail) in disagreements.iter().take(40) {
        let label = match shape {
            Shape::WeRefused => "dspy parsed, we refused",
            Shape::WeAccepted => "we parsed, dspy refused",
            Shape::DifferentValue => "both parsed, differently",
        };
        eprintln!("  [{which}] {label}\n      {detail}\n");
    }
    if disagreements.len() > 40 {
        eprintln!("  … and {} more", disagreements.len() - 40);
    }

    // The rule is now every disagreement, with no category excused. It used to allow exactly one —
    // `[json] dspy parsed, we refused`, where `json_repair` recovered a reply this crate's own
    // repair could not — and that allowance is what `dsrust-json-repair` was written to delete.
    panic!(
        "{} of {} random replies disagree with dspy (seed {})",
        disagreements.len(),
        cases.len(),
        corpus["seed"]
    );
}
