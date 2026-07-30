//! The call-site macros behind `Predict!` / `ChainOfThought!`: one parsed grammar,
//! `Task { field: value, ... }`, expanded to a typed module call on the named task.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Expr, Ident, Token};

/// Which inherent constructor the expansion calls on the task.
pub enum Module {
    Predict,
    ChainOfThought,
}

/// A parsed call site: the task type and its `field: value` pairs in written order.
struct Call {
    task: Ident,
    fields: Punctuated<FieldValue, Token![,]>,
}

struct FieldValue {
    name: Ident,
    value: Expr,
}

impl Parse for Call {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let task = input.parse()?;
        let content;
        syn::braced!(content in input);
        Ok(Self {
            task,
            fields: content.parse_terminated(FieldValue::parse, Token![,])?,
        })
    }
}

impl Parse for FieldValue {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        Ok(Self {
            name,
            value: input.parse()?,
        })
    }
}

pub fn expand(input: proc_macro::TokenStream, module: Module) -> proc_macro::TokenStream {
    match syn::parse::<Call>(input) {
        Ok(call) => emit(&call, module).into(),
        Err(error) => grammar_error(&error).into_compile_error().into(),
    }
}

/// Every malformed call gets the one message that states the grammar, at the token where
/// syn stopped, instead of a syn internal like "expected curly braces".
fn grammar_error(error: &syn::Error) -> syn::Error {
    syn::Error::new(error.span(), "expected `Task { field: value, ... }`")
}

fn emit(call: &Call, module: Module) -> TokenStream {
    let task = &call.task;
    // format_ident! carries the task ident's span onto the synthesized companion name, so
    // a missing-field error points at the call site.
    let inputs = format_ident!("{}Inputs", task);
    let constructor = match module {
        Module::Predict => quote! { predict },
        Module::ChainOfThought => quote! { chain_of_thought },
    };
    let names = call.fields.iter().map(|f| &f.name);
    let values = call.fields.iter().map(|f| &f.value);
    // A plain exhaustive literal: a forgotten field stays a compile error. `(value).into()`
    // lets `&str` and `String` land in String fields and typed values — structs, Vecs —
    // land reflexively; the macro cannot know a field's type, so unsuffixed integer
    // literals fall back to i32 and need a suffix (61u32) when the field is unsigned, and
    // a non-empty vec! must already hold the field's element type (`Into` is not
    // element-wise: vec!["a"] cannot become Vec<String>).
    quote! {
        #task::#constructor().call_inputs(&#inputs {
            #( #names: (#values).into(), )*
        })
    }
}
