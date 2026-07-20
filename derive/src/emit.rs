use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{Field, Kind, Model};

/// Expansion: the two companion structs, the `SignatureSpec` impl, and the module
/// constructors. Generated paths name the library as `::dsrs`, which its
/// `extern crate self as dsrs` alias keeps valid inside the crate itself.
pub fn expand(model: &Model) -> TokenStream {
    let companions = companions(model);
    let spec = spec_impl(model);
    let constructors = constructor_impl(model);
    quote! {
        #companions
        #spec
        #constructors
    }
}

/// Inherent module constructors, so a call site reads `GiftTask::predict().call(&inputs)`.
/// dead_code is allowed because the derive cannot know which module the host uses.
fn constructor_impl(model: &Model) -> TokenStream {
    let name = &model.name;
    quote! {
        impl #name {
            #[allow(dead_code)]
            pub fn predict() -> ::dsrs::predict::TypedPredict<Self> {
                ::dsrs::predict::Predict::task::<Self>()
            }

            #[allow(dead_code)]
            pub fn chain_of_thought() -> ::dsrs::predict::TypedChainOfThought<Self> {
                ::dsrs::predict::ChainOfThought::task::<Self>()
            }
        }
    }
}

/// The companions carry the user's declared Rust types verbatim, so a `u32` input stays a
/// `u32` at the call site and a `f64` output deserializes as a number.
fn companions(model: &Model) -> TokenStream {
    let vis = &model.vis;
    let inputs_name = format_ident!("{}Inputs", model.name);
    let outputs_name = format_ident!("{}Outputs", model.name);
    let input_fields = model.inputs.iter().map(|f| companion_field(vis, f));
    let output_fields = model.outputs.iter().map(|f| companion_field(vis, f));
    quote! {
        #[derive(Debug, Clone, ::serde::Serialize)]
        #vis struct #inputs_name {
            #( #input_fields, )*
        }

        #[derive(Debug, Clone, ::serde::Serialize, ::serde::Deserialize)]
        #vis struct #outputs_name {
            #( #output_fields, )*
        }
    }
}

fn companion_field(vis: &syn::Visibility, field: &Field) -> TokenStream {
    let ident = &field.ident;
    let ty = &field.ty;
    quote! { #vis #ident: #ty }
}

fn spec_impl(model: &Model) -> TokenStream {
    let name = &model.name;
    let inputs_name = format_ident!("{}Inputs", model.name);
    let outputs_name = format_ident!("{}Outputs", model.name);
    let instructions = &model.instructions;
    let in_fields = model.inputs.iter().map(in_field);
    let out_fields = model.outputs.iter().map(out_field);
    let pair_names = model.inputs.iter().map(|f| f.ident.to_string());
    let pair_values = model.inputs.iter().map(pair_value);
    quote! {
        impl ::dsrs::signature::SignatureSpec for #name {
            type Inputs = #inputs_name;
            type Outputs = #outputs_name;

            fn signature() -> ::dsrs::signature::Signature {
                ::dsrs::signature::Signature {
                    instructions: #instructions.to_owned(),
                    inputs: ::std::vec![ #( #in_fields ),* ],
                    outputs: ::std::vec![ #( #out_fields ),* ],
                }
            }

            fn input_pairs(
                inputs: &Self::Inputs,
            ) -> ::std::vec::Vec<(&'static str, ::std::string::String)> {
                ::std::vec![ #( (#pair_names, #pair_values) ),* ]
            }
        }
    }
}

/// How one input renders for the adapters: `String` clones, scalar kinds go through
/// `to_string`, and a `Json` field serializes to JSON text. That serialization only fails
/// on a broken `Serialize` impl — programmer error, not model behavior — so the generated
/// code expects success and names the field.
fn pair_value(field: &Field) -> TokenStream {
    let ident = &field.ident;
    match field.kind {
        Kind::Str => quote! { ::std::clone::Clone::clone(&inputs.#ident) },
        Kind::Json => {
            let message = format!("input `{ident}` must serialize to JSON");
            quote! { ::serde_json::to_string(&inputs.#ident).expect(#message) }
        }
        _ => quote! { ::std::string::ToString::to_string(&inputs.#ident) },
    }
}

fn kind(field: &Field) -> TokenStream {
    let variant = match field.kind {
        Kind::Str => quote! { Str },
        Kind::Bool => quote! { Bool },
        Kind::Int => quote! { Int },
        Kind::Float => quote! { Float },
        Kind::Json => quote! { Json },
    };
    quote! { ::dsrs::signature::FieldKind::#variant }
}

fn in_field(field: &Field) -> TokenStream {
    let name = field.ident.to_string();
    let desc = &field.desc;
    let kind = kind(field);
    quote! {
        ::dsrs::signature::InField { name: #name, desc: #desc.to_owned(), kind: #kind }
    }
}

/// A `Json` output embeds the schema of its declared type, retrieved through the host
/// crate's schemars-backed helper; the `JsonSchema` bound that requires is checked right
/// here at the derive site.
fn out_field(field: &Field) -> TokenStream {
    let name = field.ident.to_string();
    let desc = &field.desc;
    let kind = kind(field);
    let values = match &field.values {
        Some(values) => quote! { ::std::option::Option::Some(::std::vec![ #( #values ),* ]) },
        None => quote! { ::std::option::Option::None },
    };
    let schema = match field.kind {
        Kind::Json => {
            let ty = &field.ty;
            quote! {
                ::std::option::Option::Some(
                    ::dsrs::signature::json_field_schema::<#ty>(),
                )
            }
        }
        _ => quote! { ::std::option::Option::None },
    };
    quote! {
        ::dsrs::signature::OutField {
            name: #name,
            desc: #desc.to_owned(),
            kind: #kind,
            values: #values,
            schema: #schema,
        }
    }
}
