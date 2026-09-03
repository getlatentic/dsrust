//! Adding context to an error without hiding what went wrong.
//!
//! `anyhow`'s `.context("…")` puts the new sentence in front and the cause behind: `{e}` prints the
//! context alone, and the cause is reachable only through `{e:#}` or `source()`. That is the wrong
//! way round for the sentence a caller sees first. Reported from an outside project and it cost
//! them a debugging cycle — `{e}` said "validated reply did not fit the requested type" and `{e:#}`
//! said ``missing field `name` ``, and only one of those tells you what to change.
//!
//! [`Explained::explain`] keeps both: the cause is folded into the message, and the original error
//! stays in the chain so `downcast_ref` still finds it.
//!
//! A context line that *categorises* is worth having. One that *replaces* the cause is not.

use std::fmt::Display;

use anyhow::{Error, Result};

pub(crate) trait Explained<T> {
    /// `.context`, with the cause kept in the message rather than only in the chain.
    fn explain(self, context: impl Display) -> Result<T>;

    /// The same, where building the sentence costs something worth skipping on the happy path.
    fn explain_with<C: Display>(self, context: impl FnOnce() -> C) -> Result<T>;
}

impl<T, E: Into<Error>> Explained<T> for std::result::Result<T, E> {
    fn explain(self, context: impl Display) -> Result<T> {
        self.explain_with(|| context)
    }

    fn explain_with<C: Display>(self, context: impl FnOnce() -> C) -> Result<T> {
        self.map_err(|source| {
            let source: Error = source.into();
            // `{:#}` is anyhow's own walk of the chain, joined by `: ` — the same fold
            // `LmFailure::from_transport` does by hand for a `reqwest::Error`, whose Display names
            // the url and stops.
            let message = format!("{}: {source:#}", context());
            source.context(message)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    fn parse_failure() -> std::result::Result<(), serde_json::Error> {
        serde_json::from_str::<serde_json::Value>("{").map(|_| ())
    }

    /// The sentence a caller sees first carries both halves. `.context` gives it only the first.
    #[test]
    fn the_cause_is_in_the_message_not_only_the_chain() {
        let plain = parse_failure()
            .context("validated reply did not fit")
            .unwrap_err();
        assert_eq!(format!("{plain}"), "validated reply did not fit");

        let explained = parse_failure()
            .explain("validated reply did not fit")
            .unwrap_err();
        let shown = format!("{explained}");
        assert!(
            shown.starts_with("validated reply did not fit: "),
            "got: {shown}"
        );
        assert!(shown.contains("EOF while parsing"), "got: {shown}");
    }

    /// The original error stays in the chain, so a caller can still branch on its type rather than
    /// on prose — which is the thing folding it into a string would otherwise cost.
    #[test]
    fn the_original_error_is_still_downcastable() {
        let explained = parse_failure()
            .explain("validated reply did not fit")
            .unwrap_err();
        assert!(explained.downcast_ref::<serde_json::Error>().is_some());
    }

    /// An `anyhow::Error` that already carries a chain folds all of it, not just the outermost.
    #[test]
    fn an_existing_chain_folds_whole() {
        let layered = parse_failure()
            .explain("reading the reply")
            .explain("answering the question")
            .unwrap_err();
        let shown = format!("{layered}");
        assert!(
            shown.starts_with("answering the question: reading the reply: "),
            "got: {shown}"
        );
        assert!(shown.contains("EOF while parsing"), "got: {shown}");
    }
}
