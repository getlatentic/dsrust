//! What each adapter reads *out* of a reply, against dspy's own parser.
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
//! are shapes the end-to-end tests produce, which is exactly why the parsers went unchecked. And for
//! the JSON one, which is `json_repair.loads` followed by a *recursive* brace regex when that did not
//! yield an object: an object buried in prose or a fence, a nested object, an array holding one, a
//! trailing comma, single quotes, unquoted keys, a missing brace, and Python's literals.
//!
//! **Refusals are compared too.** A reply the crate accepts where dspy raises reaches the caller as
//! a wrong value instead of an error, which is the worse direction to diverge in.
//!
//! Both scans name their field with `\w+`, which is `str.isalnum()` plus `_` — neither ASCII nor
//! `char::is_alphanumeric`. The crate shipped one of each, in opposite directions, and 58 cases
//! said nothing about either: an all-ASCII corpus cannot see that predicate at all. The generator
//! refuses to write one now.

use dsrust::adapter::Input;
use dsrust::signature::Signature;
use dsrust::{Adapter, ChatAdapter, JsonAdapter, XmlAdapter};
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/conformance/parse/adapter_parse.json");
    let text = std::fs::read_to_string(&path).expect("the parse golden is committed");
    serde_json::from_str(&text).expect("the golden parses")
}

/// The two halves have to agree about what a field may be called.
///
/// dspy names a field with `\w+` on both sides, so a signature whose fields are not ASCII renders
/// markers it can also read back. The crate rendered them and then refused them — `split_header`
/// asked for `is_ascii_alphanumeric`, so every reply to such a signature came back as "reply has no
/// [[ ## field ## ]] sections". A render golden alone could not see it and a parse golden alone
/// would not have said the prompt was fine; the pair is the statement worth making.
#[test]
fn a_signature_whose_fields_are_not_ascii_renders_markers_it_can_read_back() {
    let signature: Signature = "question -> réponse: str, 答え: str"
        .parse()
        .expect("a signature may name its fields the way Python names an identifier");
    let (system, turns) = ChatAdapter::default()
        .format(
            &signature,
            &[],
            &[Input::record(
                "question",
                Value::from("Quelle est la capitale?"),
            )],
        )
        .expect("renders");

    // dspy==3.3.0b1, `ChatAdapter().format(U, [], {...})`, copied from its output.
    assert!(
        system.contains("[[ ## réponse ## ]]\n{réponse}\n\n[[ ## 答え ## ]]\n{答え}"),
        "the system prompt does not carry the markers dspy renders:\n{system}"
    );
    assert_eq!(
        turns[0].content,
        "[[ ## question ## ]]\nQuelle est la capitale?\n\nRespond with the corresponding output \
         fields, starting with the field `[[ ## réponse ## ]]`, then `[[ ## 答え ## ]]`, and then \
         ending with the marker for `[[ ## completed ## ]]`."
            .into()
    );

    let reply = "[[ ## réponse ## ]]\nParis\n\n[[ ## 答え ## ]]\nはい\n\n[[ ## completed ## ]]\n";
    let read = ChatAdapter::default()
        .parse(&signature, reply)
        .expect("and reads back what it asked for");
    assert_eq!(
        read,
        serde_json::json!({ "réponse": "Paris", "答え": "はい" })
    );
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
        let expected = &case["dspy"];

        let adapter: Box<dyn Adapter> = match case["adapter"].as_str().unwrap_or("chat") {
            "xml" => Box::new(XmlAdapter::default()),
            "json" => Box::new(JsonAdapter::default()),
            _ => Box::new(ChatAdapter::default()),
        };
        let ours = adapter.parse(&signature, completion);
        match expected["ok"].as_bool().expect("ok") {
            true if case["diverges"].as_bool().unwrap_or(false) => {
                // dspy accepts this and the crate does not agree — either it refuses, or it hands
                // back a different value. Asserted as *disagreement* rather than as a specific
                // wrong answer, so the case still says something while staying honest about which
                // way it differs.
                diverging += 1;
                let agrees = ours.as_ref().is_ok_and(|ours| *ours == expected["fields"]);
                assert!(
                    !agrees,
                    "case {name}: this now matches dspy — drop its `diverges` flag in the generator \
                     and let it be compared like the rest"
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
                // The message, not merely that both refused. Every refusal here is an
                // `AdapterParseError`, so comparing the fact of one compares nothing: it is the
                // text that names the adapter, the reply and the fields it expected.
                let refusal = ours
                    .as_ref()
                    .err()
                    .map(std::string::ToString::to_string)
                    .unwrap_or_else(|| panic!(
                        "case {name}: dspy refused this and we accepted it — a wrong value reaching \
                         the caller rather than an error"
                    ));
                // A cast failure carries pydantic's own rendering, down to a versioned docs URL.
                // Asserted as a *difference* so it goes red the day someone reproduces it, rather
                // than skipped and forgotten.
                match expected["message_diverges"].as_str() {
                    Some(reason) => {
                        assert_eq!(
                            reason, "pydantic-error-text",
                            "{name}: undeclared divergence"
                        );
                        assert_ne!(
                            refusal.as_str(),
                            expected["message"].as_str().expect("a message"),
                            "case {name}: this now matches dspy — drop `message_diverges`"
                        );
                        assert!(
                            !refusal.is_empty(),
                            "case {name}: refused with nothing to read"
                        );
                    }
                    None => assert_eq!(
                        refusal.as_str(),
                        expected["message"].as_str().expect("a message"),
                        "case {name}: refused for a different reason"
                    ),
                }
            }
        }
    }

    // A corpus of only-valid replies would pin nothing about the refusals, which is half of what
    // `parse` decides. The generator refuses to write one; this keeps a hand-edited golden honest.
    assert!(
        accepted > 0 && refused > 0,
        "the golden no longer exercises both arms: {accepted} accepted, {refused} refused"
    );
    // Three, all of them `parse-time-casting`. The fourth was
    // `json_unescaped_quote_inside_a_string`, which closed when `dsrust-json-repair` landed —
    // and this count, plus the assertion above it, is what said so rather than letting a golden
    // quietly go on recording a gap that no longer exists.
    assert_eq!(
        diverging, 3,
        "the recorded divergences changed count; if one was fixed, say so"
    );
}
