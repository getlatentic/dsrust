//! `#[derive(Signature)]`: a DSPy-style signature declared as one struct. Struct-level
//! instructions come from `#[signature(instructions = "...")]` or the doc comment; each
//! field is marked `#[input(...)]` or `#[output(...)]`. The derive expands to `<Name>Inputs`
//! and `<Name>Outputs` companion structs, a `SignatureSpec` impl for the host crate's typed
//! module entry points, and inherent `predict()` / `chain_of_thought()` constructors.
//!
//! The [`predict!`] and [`chain_of_thought!`] call macros are the matching call-site sugar:
//! one invocation names the task, fills its inputs, and evaluates to the module call's
//! future.

use proc_macro::TokenStream;

mod call;
mod emit;
mod parse;
mod signature_str;

/// `String`, `bool`, fixed-width integers, and floats travel as scalar wire fields; any
/// other field type — `Vec<String>`, your own structs, `Vec<Struct>` — travels as JSON.
/// The derive cannot check the trait bounds that requires; the generated code carries
/// them, so the compiler reports a missing impl at the derive site. Every field type needs
/// `Debug + Clone` (the companion structs derive both) plus `serde::Serialize`; a JSON
/// output additionally needs `serde::Deserialize` and `schemars::JsonSchema` (its schema
/// is embedded in the signature).
#[proc_macro_derive(Signature, attributes(signature, input, output))]
pub fn derive_signature(input: TokenStream) -> TokenStream {
    let item = syn::parse_macro_input!(input as syn::DeriveInput);
    match parse::model(&item) {
        Ok(model) => emit::expand(&model).into(),
        Err(error) => error.into_compile_error().into(),
    }
}

/// `predict!("subject -> haiku")` — the module a string signature declares, built. The spelling
/// is checked as this crate compiles, so there is no `?` to write and no runtime failure left.
///
/// `predict!(Task { field: value, ... })` — one `Predict` call on a derived task. Expands to
/// `Task::predict().call(&TaskInputs { field: (value).into(), ... })` and evaluates to that
/// call's future, so the caller writes `.await?`. Values coerce through `Into` toward each
/// field's declared type, and the inputs literal is exhaustive: a forgotten field is a
/// compile error.
#[proc_macro]
pub fn predict(input: TokenStream) -> TokenStream {
    if let Ok(spelling) = syn::parse::<syn::LitStr>(input.clone()) {
        return signature_str::expand_module(spelling, "Predict");
    }
    if let Ok(task) = syn::parse::<syn::Ident>(input.clone()) {
        return quote::quote! { ::dsrs::Predict::task::<#task>() }.into();
    }
    call::expand(input, call::Module::Predict)
}

/// `signature!("subject -> haiku")` — dspy's string spelling, refused while this crate compiles
/// rather than when the program runs. Evaluates to a `Signature`, so it drops straight into
/// `Predict::new` with no `?` to write and no failure left to handle.
#[proc_macro]
pub fn signature(input: TokenStream) -> TokenStream {
    signature_str::expand(input)
}

/// `chain_of_thought!(Task { field: value, ... })` — the [`predict!`] grammar driving the
/// task's `ChainOfThought` module instead.
#[proc_macro]
pub fn chain_of_thought(input: TokenStream) -> TokenStream {
    if let Ok(spelling) = syn::parse::<syn::LitStr>(input.clone()) {
        return signature_str::expand_module(spelling, "ChainOfThought");
    }
    if let Ok(task) = syn::parse::<syn::Ident>(input.clone()) {
        return quote::quote! { ::dsrs::ChainOfThought::task::<#task>() }.into();
    }
    call::expand(input, call::Module::ChainOfThought)
}
