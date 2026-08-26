//! `make_signature!("subject -> haiku")`: the string spelling, checked while the crate compiles.
//!
//! dspy raises on a malformed signature when the program runs. A macro can do better, because
//! the string is a literal the compiler already holds: the two ways a spelling can be refused
//! are decided here, and a program that spells one wrong does not build.
//!
//! Only the refusals live here. What a good spelling *becomes* is `dsrust::signature::parse`, and
//! the expansion calls it, so the field lists have one implementation rather than two. The
//! refusals are checked against the same generated golden that parser is, which is what stops
//! the halves drifting apart.

use proc_macro::TokenStream;
use quote::quote;

/// Why a signature string cannot be parsed, in upstream's own words.
pub(crate) fn refusal(spelling: &str) -> Option<String> {
    if spelling.matches("->").count() != 1 {
        return Some(format!(
            "Invalid signature format: '{spelling}', must contain exactly one '->'."
        ));
    }
    let (before, after) = spelling.split_once("->").expect("one arrow, just counted");
    let mut shared: Vec<&str> = names(before)
        .into_iter()
        .filter(|name| names(after).contains(name))
        .collect();
    shared.sort_unstable();
    shared.dedup();
    if shared.is_empty() {
        return None;
    }
    Some(format!(
        "Input and output fields must have distinct names, but found duplicates: '{}'.",
        shared.join(", ")
    ))
}

/// The field names on one side, which is everything before a `:` in each top-level comma part.
fn names(side: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (at, character) in side.char_indices() {
        match character {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&side[start..at]);
                start = at + character.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&side[start..]);
    parts
        .into_iter()
        .map(|part| part.split_once(':').map_or(part, |(name, _)| name).trim())
        .filter(|name| !name.is_empty())
        .collect()
}

/// The signature expression a checked literal becomes.
fn checked(literal: &syn::LitStr) -> Result<proc_macro2::TokenStream, syn::Error> {
    if let Some(refusal) = refusal(&literal.value()) {
        return Err(syn::Error::new(literal.span(), refusal));
    }
    Ok(quote! {
        ::dsrust::signature::parse(#literal).expect("refused at compile time if it could fail")
    })
}

pub(crate) fn expand(input: TokenStream) -> TokenStream {
    let literal = syn::parse_macro_input!(input as syn::LitStr);
    match checked(&literal) {
        Ok(signature) => signature.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

/// `Predict!("subject -> haiku")` and its `ChainOfThought!` twin: the module a spelling
/// declares, built rather than described.
pub(crate) fn expand_module(literal: syn::LitStr, module: &str) -> TokenStream {
    let built = match checked(&literal) {
        Ok(signature) => signature,
        Err(error) => return error.into_compile_error().into(),
    };
    let module = syn::Ident::new(module, literal.span());
    quote! { ::dsrust::#module::from_signature(#built) }.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same golden the runtime parser answers to.
    ///
    /// Two implementations decide whether a spelling is legal — this one so a bad literal fails
    /// the build, and `dsrust::signature::parse` so a runtime string is answered. They agree here
    /// or they do not agree at all.
    #[test]
    fn refuses_exactly_what_dspy_refuses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../dsrust/tests/conformance/signature/signature.json");
        let text = std::fs::read_to_string(&path).expect("the signature golden is committed");
        let golden: serde_json::Value = serde_json::from_str(&text).expect("the golden parses");
        let cases = golden["parse"].as_array().expect("parse cases");
        assert!(!cases.is_empty(), "the golden records no signatures");

        for case in cases {
            let spelling = case["signature"].as_str().expect("a signature");
            assert_eq!(
                refusal(spelling).as_deref(),
                case["error"].as_str(),
                "verdict for {spelling:?}"
            );
        }
    }
}
