//! dspy's callback points, as `tracing` spans: watching a run without changing it.
//!
//! Upstream's `BaseCallback` fires at six places, each with a start and an end —
//! module, lm, adapter format, adapter parse, tool, evaluate — and `with_callbacks` gives each
//! call a uuid and links it to its parent through a context variable. That is a span tree with
//! extra steps, so the Rust shape is `tracing`: a span *is* the identifier, the parent linkage,
//! the start and the end, and a subscriber is what a Rust caller already knows how to write.
//!
//! A span cannot mutate what it sees, which answers upstream's own two worries: dspy's
//! documentation warns readers not to mutate what a callback is handed, and it wraps every handler
//! in `try/except` so a broken one cannot break the run.
//!
//! Nothing is serialized unless something is listening. `tracing`'s macros check subscriber interest
//! before evaluating a field, and [`shown`] and [`finished`] return immediately on a disabled span.
//!
//! **Two of the six points exist: module and lm.** Adapter format, adapter parse, tool and evaluate
//! have no span, and no unused constructor here either — a function nothing calls is what let the
//! ledger claim these existed while the tree had none. `tests/observe.rs` decides which exist, and
//! it can only see spans a run produced.

use std::fmt::Write as _;
use std::future::Future;

use anyhow::Result;
use tracing::{Instrument, Span, field};

use crate::example::{Example, Prediction};

/// The target every span here carries, so `RUST_LOG=dsrust::observe=info` is the whole of what a
/// caller needs to watch a run — and so a subscriber can select these without matching on names.
pub const TARGET: &str = "dsrust::observe";

/// dspy `on_module_start`/`on_module_end`: one module's run, with everything it did inside it.
///
/// `kind` is the module's type — `Predict`, `ReAct` — which is what upstream's `instance` is read
/// for. A composed program nests, so the span tree is the program's shape.
pub fn module(kind: &'static str) -> Span {
    tracing::info_span!(
        target: TARGET,
        "module",
        module = kind,
        inputs = field::Empty,
        outputs = field::Empty,
        error = field::Empty,
    )
}

/// A module's span with its inputs already on it — dspy's `on_module_start`, as one call.
///
/// Taken by reference and rendered here rather than at the span's creation, so a `forward` records
/// its inputs and then moves them on: `module_shown(kind, &inputs)` followed by a body that consumes
/// `inputs` is two statements and borrows nothing across them.
pub fn module_shown(kind: &'static str, inputs: &Example) -> Span {
    let span = module(kind);
    shown_example(&span, inputs);
    span
}

/// dspy `on_lm_start`/`on_lm_end`: one call to a model, inside whichever module made it.
pub fn lm(model: &str) -> Span {
    tracing::info_span!(
        target: TARGET,
        "lm",
        model = model,
        inputs = field::Empty,
        outputs = field::Empty,
        error = field::Empty,
    )
}

/// What dspy's `on_*_start` was shown, recorded on the span.
///
/// Separate from creating the span because a value worth rendering is a value worth not rendering
/// when nothing is listening, and `Span::record` evaluates its argument either way.
pub fn shown(span: &Span, inputs: &str) {
    if span.is_disabled() {
        return;
    }
    span.record("inputs", inputs);
}

/// As [`shown`], for an [`Example`] — a module's or a program's inputs, as dspy's `inputs` dict.
pub fn shown_example(span: &Span, inputs: &Example) {
    if span.is_disabled() {
        return;
    }
    span.record("inputs", as_json(inputs).as_str());
}

/// Run `work` inside `span`, recording what dspy's `on_*_end` receives: the outputs, or the failure.
///
/// One function rather than a start call and an end call, because an end that a `?` can skip is an
/// end that will be skipped. Every exit records something.
pub async fn watching<T, Work>(span: Span, describe: fn(&T) -> String, work: Work) -> Result<T>
where
    Work: Future<Output = Result<T>>,
{
    let answered = work.instrument(span.clone()).await;
    finished(&span, describe, &answered);
    answered
}

/// Record the outcome on a span: dspy's `on_*_end(outputs=…, exception=…)`, where exactly one of
/// the two is present.
///
/// A failure renders `{:#}` and not `{}`, because a parse failure keeps its cause in the chain and
/// the chain is the half naming the field.
pub fn finished<T>(span: &Span, describe: fn(&T) -> String, answered: &Result<T>) {
    if span.is_disabled() {
        return;
    }
    match answered {
        Ok(outputs) => span.record("outputs", describe(outputs).as_str()),
        Err(error) => span.record("error", format!("{error:#}").as_str()),
    };
}

/// An [`Example`]'s fields as a JSON object, which is the shape dspy's `inputs` dict has.
///
/// A field that will not render is dropped rather than raised: this is only watching the call.
pub fn as_json(example: &Example) -> String {
    let mut rendered = String::from("{");
    for (index, (name, value)) in example.fields().enumerate() {
        if index > 0 {
            rendered.push(',');
        }
        let _ = write!(rendered, "{}:{value}", serde_json::json!(name));
    }
    rendered.push('}');
    rendered
}

/// A [`Prediction`]'s parsed fields, for [`watching`]'s `describe`.
pub fn prediction(answered: &Prediction) -> String {
    as_json(&answered.example)
}

/// What a model answered, for [`watching`]'s `describe`: the text, whether it was replayed, and
/// what it cost.
///
/// Not the whole response. dspy's `on_lm_end` is handed the outputs, and a span field is a line in a
/// log — a reply's every part, its provider envelope and its logprobs would bury the four values a
/// reader is actually looking for. The response itself is the caller's, unchanged.
pub fn spent(answered: &crate::lm::api::LmResponse) -> String {
    let usage = answered
        .usage
        .as_ref()
        .and_then(|usage| usage.total_tokens)
        .map_or_else(|| "null".to_owned(), |tokens| tokens.to_string());
    format!(
        "{{\"text\":{},\"cache_hit\":{},\"total_tokens\":{usage}}}",
        serde_json::json!(answered.first_text()),
        answered.cache_hit,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_examples_fields_render_as_the_dict_dspy_passes() {
        let example = Example::new([
            ("question", serde_json::json!("capital of France?")),
            ("hops", serde_json::json!(2)),
        ]);
        assert_eq!(
            as_json(&example),
            r#"{"question":"capital of France?","hops":2}"#
        );
        assert_eq!(as_json(&Example::default()), "{}");
    }

    /// A field name with a quote in it still renders as JSON rather than breaking the object.
    #[test]
    fn a_field_name_is_escaped() {
        let example = Example::new([(r#"od"d"#, serde_json::json!(1))]);
        assert_eq!(as_json(&example), r#"{"od\"d":1}"#);
    }

    /// Nothing is recorded on a disabled span, which is every span in a program with no subscriber.
    #[test]
    fn a_disabled_span_records_nothing() {
        let span = Span::none();
        assert!(span.is_disabled());
        shown(&span, "ignored");
        finished::<()>(
            &span,
            |_| unreachable!("a disabled span asks for no description"),
            &Ok(()),
        );
    }
}
