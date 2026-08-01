//! Differential fuzzing against dspy's parsers: random replies, both sides, every disagreement.
//!
//! The committed goldens hold a case per branch of `parse`, each chosen by reading it — which is
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

/// The committed slice: a fixed seed of the same grammar, compared against dspy's own answers.
///
/// It exists because the campaign corpus lives in `target/`, and a tree without one — a fresh
/// clone, a copied source tree, every single cargo-mutants run — silently skipped this whole
/// comparison. The parser's strongest oracle contributed nothing to any survivor count, which is
/// most of why `parse.rs` had so many. Regenerate with
/// `.venv/bin/python scripts/fuzz_parse.py 1500 0 --sweep`.
fn sweep() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/parse/fuzz_sweep.json");
    let text = std::fs::read_to_string(&path).expect("the fuzz sweep is committed");
    serde_json::from_str(&text).expect("the sweep parses")
}

/// The scratch corpus a campaign leaves behind, when there is one. Deliberately not committed: ten
/// thousand random strings are evidence, not documentation.
fn campaign() -> Option<Value> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/parse_fuzz.json");
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
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
    // The committed sweep always, the campaign corpus as well when one is lying around. Never a
    // silent skip: a comparison that returns early when its input is missing reports the same
    // green as one that ran, and this one was doing that in every copied tree.
    for corpus in [Some(sweep()), campaign()].into_iter().flatten() {
        check(&corpus);
    }
}

fn check(corpus: &Value) {
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

    // One category is known and named: `json_repair` recovers JSON this crate's own repair does
    // not, which `json-repair-port` tracks. Everything else must be zero, and the rules are
    // deliberately not "no disagreements at all" — a blanket assertion against a known gap either
    // blocks every run or gets switched off, and neither says anything.
    //
    //   - **`we parsed, dspy refused` is zero for every adapter, always.** That is the direction
    //     that hands a caller a wrong value instead of an error, and no repair gap excuses it.
    //   - **the marker and tag parsers agree completely.** They did on the first run of this and
    //     there is no reason for that to stop.
    let mut unexpected: Vec<&(Shape, &str, String)> = disagreements
        .iter()
        .filter(|(shape, which, _)| {
            *shape == Shape::WeAccepted || (*which != "json" || *shape != Shape::WeRefused)
        })
        .collect();
    unexpected.dedup_by(|left, right| std::ptr::eq(*left, *right));
    assert!(
        unexpected.is_empty(),
        "{} disagreements outside the known `json-repair-port` gap — the first is [{}] {}",
        unexpected.len(),
        unexpected[0].1,
        unexpected[0].2
    );
    eprintln!(
        "  all {} disagreements are the known json-repair gap; the other two parsers agree",
        disagreements.len()
    );
}
