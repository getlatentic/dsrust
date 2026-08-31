//! `#[derive(Signature)]`: a DSPy-style signature declared as one struct. Struct-level
//! instructions come from `#[signature(instructions = "...")]` or the doc comment; each
//! field is marked `#[input(...)]` or `#[output(...)]`. The derive expands to `<Name>Inputs`
//! and `<Name>Outputs` companion structs, a `SignatureSpec` impl for the host crate's typed
//! module entry points, and inherent `predict()` / `chain_of_thought()` constructors.
//!
//! The [`Predict!`] and [`ChainOfThought!`] call macros are the matching call-site sugar:
//! one invocation names the task, fills its inputs, and evaluates to the module call's
//! future.

use proc_macro::TokenStream;

mod annotate;
mod call;
mod constraints;
mod emit;
mod module;
mod parse;
mod signature_str;
mod tool;
mod ty;

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

/// `Predict!("subject -> haiku")` — the module a string signature declares, built. The spelling
/// is checked as this crate compiles, so there is no `?` to write and no runtime failure left.
///
/// `Predict!(Task { field: value, ... })` — one `Predict` call on a derived task. Expands to
/// `Task::predict().call(&TaskInputs { field: (value).into(), ... })` and evaluates to that
/// call's future, so the caller writes `.await?`. Values coerce through `Into` toward each
/// field's declared type, and the inputs literal is exhaustive: a forgotten field is a
/// compile error.
#[proc_macro]
// dspy names this Predict; a proc macro's name is a function name, so the lint fires.
#[allow(non_snake_case)]
pub fn Predict(input: TokenStream) -> TokenStream {
    if let Ok(spelling) = syn::parse::<syn::LitStr>(input.clone()) {
        return signature_str::expand_module(spelling, "Predict");
    }
    if let Ok(task) = syn::parse::<syn::Ident>(input.clone()) {
        return quote::quote! { ::dsrust::Predict::task::<#task>() }.into();
    }
    call::expand(input, call::Module::Predict)
}

/// `#[derive(Module)]` — a program of your own, given everything Python inherits: the walk an
/// optimizer works through, and being callable through `call!`. Write `dsrust::Forward` for how it
/// runs; every named field is treated as a step unless marked `#[not_a_step]`.
#[proc_macro_derive(Module, attributes(not_a_step, task))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let item = syn::parse_macro_input!(input as syn::DeriveInput);
    match module::expand(&item) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

/// `make_signature!("subject -> haiku")` — dspy's string spelling, refused while this crate compiles
/// rather than when the program runs. Evaluates to a `Signature`, so it drops straight into
/// `Predict::new` with no `?` to write and no failure left to handle.
#[proc_macro]
// dspy names this make_signature; a proc macro's name is a function name, so the lint fires.
#[allow(non_snake_case)]
pub fn make_signature(input: TokenStream) -> TokenStream {
    signature_str::expand(input)
}

/// `ChainOfThought!(Task { field: value, ... })` — the [`Predict!`] grammar driving the
/// task's `ChainOfThought` module instead.
#[proc_macro]
// dspy names this ChainOfThought; a proc macro's name is a function name, so the lint fires.
#[allow(non_snake_case)]
pub fn ChainOfThought(input: TokenStream) -> TokenStream {
    if let Ok(spelling) = syn::parse::<syn::LitStr>(input.clone()) {
        return signature_str::expand_module(spelling, "ChainOfThought");
    }
    if let Ok(task) = syn::parse::<syn::Ident>(input.clone()) {
        return quote::quote! { ::dsrust::ChainOfThought::task::<#task>() }.into();
    }
    call::expand(input, call::Module::ChainOfThought)
}

/// `tool!` — a tool written as a function, with its doc comment as the description.
///
/// A tool: its doc comment is the description the model reads, its typed parameters are the
/// argument schema. dspy reads both off a callable with `inspect`; a Rust doc comment is erased
/// before the program runs, so it is read here instead, while the code is still source.
///
/// ```ignore
/// #[tool]
/// /// Look one term up in the index.
/// ///
/// /// Give a single term, not a sentence.
/// fn search(term: String) -> anyhow::Result<String> {
///     Ok(index::lookup(&term))
/// }
///
/// let tools: Vec<Box<dyn Tool>> = vec![Box::new(Search)];
/// ```
///
/// The function stays callable as itself; the tool is a type of the same name in PascalCase, and
/// the name on the wire is still `search`. The description goes through the same `cleandoc` a
/// signature's instructions do, because it is prompt text either way. Arguments are deserialized
/// by name; one that is missing or the wrong type is **answered** with a refusal rather than
/// raised, since a loop can act on an answer.
///
/// **On a method it is a tool over the receiver.** Python captures a draft in a closure and hands
/// six closures to `dspy.ReAct`; a Rust `fn` captures nothing, so the state is `&self` instead. A
/// marked method keeps its own name and gains `<name>_tool()` beside it, holding an `Arc` of the
/// receiver. Putting the attribute on the impl block as well adds `tools()` — every marked method
/// in declaration order, so a roster is one call rather than one call per tool:
///
/// ```ignore
/// #[tool]
/// impl Composition {
///     pub fn new(unit: Unit) -> Self { /* … */ }   // unmarked: an ordinary method
///
///     /// Set the learner-facing section title. Call once, before writing blocks.
///     #[tool]
///     fn set_title(&self, title: String) -> anyhow::Result<String> {
///         self.draft.lock().expect("held").title = title.trim().to_owned();
///         Ok(format!("Title set to {:?}.", title.trim()))
///     }
/// }
///
/// let composition = Arc::new(Composition::new(unit));
/// let tools = composition.tools();
/// ```
///
/// Interior mutability stays the caller's, exactly as it is in Python: a tool takes `&self`
/// because the roster outlives any one call, so a draft it writes into is behind a `Mutex`.
#[proc_macro_attribute]
pub fn tool(attribute: TokenStream, input: TokenStream) -> TokenStream {
    match syn::parse::<syn::Item>(input).and_then(|item| tool::expand(attribute.into(), item)) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}
