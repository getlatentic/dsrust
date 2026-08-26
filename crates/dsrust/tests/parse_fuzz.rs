//! Differential fuzzing against dspy's parsers: random replies, both sides, every disagreement.
//!
//! The committed goldens hold a case per branch of `parse`, each chosen by reading it — which is
//! exactly the method that leaves the branches nobody thought of. This closes that: both sides are
//! pure functions from a string to a value-or-error and dspy is on hand as the reference.
//!
//! **Never skips.** A committed 1500-case sweep at a fixed seed always runs, and the campaign
//! corpus at `target/parse_fuzz.json` runs as well when a campaign has left one — that one is
//! evidence rather than documentation and stays out of git. This test *did* skip, for as long as
//! the sweep did not exist: cargo-mutants copies the source tree, a copied tree has no `target/`,
//! and so did a fresh clone. The comparison was absent from every mutation run of the parser and
//! nothing said so, because an early return reports the same green as a run.
//!
//! Run `scripts/fuzz_parse.py` for a campaign, run this, and promote whatever disagrees into
//! `generate_parse_fixture.py` as a named case with its reason. Regenerate the sweep with
//! `--sweep` when the grammar widens.
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
    /// Both refused, for different reasons.
    ///
    /// dspy raises `AdapterParseError` for every refusal these generators produce, so matching the
    /// class alone matched everything — across the 88% of each campaign dspy rejects. The message
    /// names the missing field and the adapter that was reading, which is the part a caller acts on.
    DifferentRefusal,
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
            (false, Err(ours))
                if ours.to_string() == expected["message"].as_str().unwrap_or_default() =>
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
            Shape::DifferentRefusal => "both refused, differently",
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
            Shape::DifferentRefusal => "both refused, differently",
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
