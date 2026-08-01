//! What `ChatAdapter` reads *out* of a reply, against dspy's own parser.
//!
//! Nineteen fixtures in this repo pin the prompt the crate **sends**. Until this one, none pinned
//! what it **reads** — and mutation testing put a number on that: 35 survivors in `adapter/parse.rs`,
//! twenty in `next_tag` alone, which could be made to return `Some(("xyzzy", "xyzzy", "xyzzy"))`
//! with the whole suite green. The byte claim runs both ways and only one way had an oracle.
//!
//! The corpus is dspy's branches rather than a well-behaved model's output. For the marker parser: a
//! marker with content on the same line, a repeated field, an undeclared field, a missing one, prose
//! before the first marker, an indented marker, a marker inside a value. For the XML one, whose scan
//! is `<(?P<name>\w+)>(.*?)</\1>` under DOTALL: a non-greedy body, a same-name nest, a hyphenated
//! name, an attribute, an unclosed tag, a mismatched close, and tags buried in prose. None of these
//! are shapes the end-to-end tests produce, which is exactly why the parsers went unchecked.
//!
//! **Refusals are compared too.** A reply the crate accepts where dspy raises reaches the caller as
//! a wrong value instead of an error, which is the worse direction to diverge in.

use dsrust::signature::Signature;
use dsrust::{Adapter, ChatAdapter, XmlAdapter};
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/parse/chat_parse.json");
    let text = std::fs::read_to_string(&path).expect("the parse golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

#[test]
fn the_parser_reads_what_dspys_reads_and_refuses_what_dspy_refuses() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the golden records no cases");

    let (mut accepted, mut refused, mut diverging) = (0, 0, 0);
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let signature: Signature = case["signature"]
            .as_str()
            .expect("a signature")
            .parse()
            .unwrap_or_else(|error| panic!("case {name}: signature does not parse: {error}"));
        let completion = case["completion"].as_str().expect("a completion");
        let expected = &case["chat"];

        let adapter: Box<dyn Adapter> = match case["adapter"].as_str().unwrap_or("chat") {
            "xml" => Box::new(XmlAdapter::default()),
            _ => Box::new(ChatAdapter::default()),
        };
        let ours = adapter.parse(&signature, completion);
        match expected["ok"].as_bool().expect("ok") {
            true if case["diverges"].as_bool().unwrap_or(false) => {
                // Accepted by both, but the crate hands back the text a scalar was written as
                // where dspy hands back the cast value. Same cause as the refusals below.
                diverging += 1;
                let ours = ours.expect("the crate parses this");
                assert_ne!(
                    ours, expected["fields"],
                    "case {name}: this now matches dspy — resolve `parse-time-casting` and drop \
                     the `diverges` flag"
                );
            }
            true => {
                accepted += 1;
                let ours = ours.unwrap_or_else(|error| {
                    panic!("case {name}: dspy parsed this, we refused it: {error}")
                });
                assert_eq!(ours, expected["fields"], "case {name}: parsed fields");
            }
            false if case["diverges"].as_bool().unwrap_or(false) => {
                // A recorded divergence, not an exemption. dspy casts a scalar while parsing and
                // raises when it will not fit; this crate casts during validation, which is what
                // `Predict::feedback_retry` is built on. Asserted the *other* way round so it goes
                // red the day `parse-time-casting` lands, rather than being skipped and forgotten.
                diverging += 1;
                assert!(
                    ours.is_ok(),
                    "case {name}: this now refuses at parse as dspy does — resolve \
                     `parse-time-casting`, drop the `diverges` flag from the generator, and assert \
                     the refusal"
                );
            }
            false => {
                refused += 1;
                assert!(
                    ours.is_err(),
                    "case {name}: dspy refused this and we accepted it as {ours:?} — a wrong value \
                     reaching the caller rather than an error"
                );
            }
        }
    }

    // A corpus of only-valid replies would pin nothing about the refusals, which is half of what
    // `parse` decides. The generator refuses to write one; this keeps a hand-edited golden honest.
    assert!(
        accepted > 0 && refused > 0,
        "the golden no longer exercises both arms: {accepted} accepted, {refused} refused"
    );
    assert_eq!(
        diverging, 3,
        "the recorded parse-time-casting divergences changed count; if one was fixed, say so"
    );
}
