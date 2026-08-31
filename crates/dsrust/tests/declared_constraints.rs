//! Bounds declared on a field, rendered as dspy renders them.
//!
//! dspy passes a field's pydantic constraints through `InputField`/`OutputField` and prints them
//! as prose under the field's line. The crate could already render that string — it crossed the
//! bridge from Python as data — but nothing in Rust could declare one, so a signature written here
//! wrote a prompt missing a line upstream writes.
//!
//! Found by porting `gepa_trusted_monitor`, whose only output field is `ge=0, le=9`.

use dsrust::Signature;
use dsrust::adapter::{Adapter, ChatAdapter, Input};
use dsrust::signature::SignatureSpec;

#[derive(Signature)]
/// Score it.
struct Bounded {
    #[input]
    code: String,
    #[output(desc = "how suspicious", ge = 0, le = 9)]
    score: i64,
    /// Written `max_length` first on purpose.
    #[output(max_length = 200, min_length = 10)]
    rationale: String,
    #[output]
    plain: String,
}

fn rendered() -> String {
    let inputs = [Input::new("code", "x".into())];
    ChatAdapter::default()
        .format(&Bounded::signature(), &[], &inputs)
        .expect("renders")[0]
        .text()
        .expect("a system prompt")
}

/// The clause, the prose, and the placement — under the field, not beside it.
#[test]
fn a_declared_bound_reads_as_dspys_constraints_line() {
    let prompt = rendered();
    assert!(
        prompt.contains(
            "1. `score` (int): how suspicious\n\
             Constraints: greater than or equal to: 0, less than or equal to: 9"
        ),
        "got: {prompt}"
    );
}

/// **Declaration order is the rendered order.** Upstream walks its keyword arguments as written
/// rather than a fixed sequence, so a fixed order here would be a different prompt for the same
/// program — measured: `dspy.OutputField(le=9, ge=0)` renders `less than or equal to: 9, greater
/// than or equal to: 0`.
#[test]
fn the_order_written_is_the_order_rendered() {
    assert!(
        rendered().contains("Constraints: maximum length: 200, minimum length: 10"),
        "got: {}",
        rendered()
    );
}

/// A field with no bounds gains no line, which is what keeps every other prompt unchanged.
#[test]
fn an_unbounded_field_says_nothing() {
    let prompt = rendered();
    let plain = prompt
        .lines()
        .skip_while(|line| !line.contains("`plain`"))
        .nth(1)
        .unwrap_or_default();
    assert!(
        !plain.starts_with("Constraints:"),
        "an unbounded field grew a constraints line: {prompt}"
    );
}
