//! A signature declared from inside a `macro_rules!` reads its types.
//!
//! A macro fragment does not arrive as the type it captured: `$ty:ty` reaches the derive wrapped
//! in an invisible delimiter, so every match on a path type fell through and the field took the
//! fallback — its annotation became the token stream and its kind became the opaque `Json`.
//!
//! The result was a prompt with Rust syntax in it (`1. \`out\` (Vec < String >):`) and a
//! JSON-schema note on a plain `String`, from a program that compiled without a warning.
//!
//! Generating signatures from a macro is ordinary: a table of type shapes, or a set of tasks
//! differing in one field. Found writing exactly such a table.

use dsrust::Signature;
use dsrust::adapter::{Adapter, ChatAdapter, Input};
use dsrust::signature::SignatureSpec;

macro_rules! task {
    ($task:ident, $ty:ty) => {
        #[derive(Signature)]
        #[doc = "T."]
        #[allow(dead_code)]
        struct $task {
            #[input]
            q: String,
            #[output]
            out: $ty,
        }
    };
}

task!(Plain, String);
task!(Listed, Vec<String>);
task!(Counted, i64);

fn field_line(signature: &dsrust::signature::Signature) -> String {
    let inputs = [Input::new("q", "x".into())];
    ChatAdapter::default()
        .format(signature, &[], &inputs)
        .expect("renders")[0]
        .text()
        .expect("a system prompt")
        .lines()
        .find(|line| line.starts_with("1. `out`"))
        .unwrap_or_default()
        .to_owned()
}

/// The annotation is Python's, not the tokens the macro captured.
#[test]
fn a_macro_declared_field_reads_its_type() {
    assert_eq!(field_line(&Plain::signature()), "1. `out` (str):");
    assert_eq!(field_line(&Listed::signature()), "1. `out` (list[str]):");
    assert_eq!(field_line(&Counted::signature()), "1. `out` (int):");
}

/// A scalar declared through a macro is still a scalar, so it grows no schema note.
#[test]
fn a_macro_declared_scalar_is_not_json() {
    let inputs = [Input::new("q", "x".into())];
    let prompt = ChatAdapter::default()
        .format(&Plain::signature(), &[], &inputs)
        .expect("renders")[0]
        .text()
        .expect("a system prompt");
    assert!(
        !prompt.contains("JSON schema"),
        "a `String` output grew a schema note: {prompt}"
    );
}
