//! What a tool's parameters become: the schema the model reads, and the bindings its body gets.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{Attribute, Error, Expr, FnArg, Pat, Result, Token, Type};

/// One named parameter: what the model sends, and what the body binds.
pub(crate) struct Argument {
    pub(crate) name: String,
    pub(crate) binding: syn::Ident,
    pub(crate) ty: Type,
    /// What stands in when the model omits this one, written `#[tool(default = ...)]`.
    ///
    /// Rust has no default arguments, so a tool ported from a Python one that has them cannot say
    /// so in its signature — and the difference is not cosmetic: dspy leaves an argument carrying
    /// a `default` out of the `required` list it sends the provider, so without this every ported
    /// argument becomes one the model is obliged to fill.
    pub(crate) default: Option<Expr>,
}

/// The named parameters of a tool, skipping a `&self` receiver.
///
/// Takes the inputs mutably because `#[tool(default = ...)]` has to come *off* them: the function
/// is re-emitted as it was written, and an attribute the compiler does not know would fail there.
pub(crate) fn take(inputs: &mut Punctuated<FnArg, Token![,]>) -> Result<Vec<Argument>> {
    inputs
        .iter_mut()
        .filter(|input| !matches!(input, FnArg::Receiver(_)))
        .map(|input| match input {
            FnArg::Receiver(_) => unreachable!("filtered above"),
            FnArg::Typed(typed) => {
                let default = take_default(&mut typed.attrs)?;
                match &*typed.pat {
                    Pat::Ident(ident) => Ok(Argument {
                        name: ident.ident.to_string(),
                        binding: format_ident!("{}", ident.ident),
                        ty: (*typed.ty).clone(),
                        default,
                    }),
                    other => Err(Error::new(
                        other.span(),
                        "a tool's parameters are named, because the model sends them by name",
                    )),
                }
            }
        })
        .collect()
}

/// Read `#[tool(default = ...)]` off one parameter and remove it.
fn take_default(attributes: &mut Vec<Attribute>) -> Result<Option<Expr>> {
    let mut default = None;
    for attribute in attributes.iter() {
        if !attribute.path().is_ident("tool") {
            continue;
        }
        attribute.parse_nested_meta(|meta| match meta.path.is_ident("default") {
            true => {
                default = Some(meta.value()?.parse::<Expr>()?);
                Ok(())
            }
            false => Err(meta.error("a tool parameter takes `default = ...` and nothing else")),
        })?;
    }
    attributes.retain(|attribute| !attribute.path().is_ident("tool"));
    Ok(default)
}

/// `Tool::args` as a value: each parameter name against the JSON Schema of its declared type.
///
/// A default is written onto the end of its parameter's schema, which is where pydantic puts it.
pub(crate) fn value(arguments: &[Argument]) -> TokenStream {
    let entries = arguments.iter().map(|argument| {
        let key = &argument.name;
        let ty = &argument.ty;
        let schema = match &argument.default {
            None => quote! { ::dsrust::signature::json_argument_schema::<#ty>() },
            Some(default) => quote! {{
                let mut __schema = ::dsrust::signature::json_argument_schema::<#ty>();
                if let ::std::option::Option::Some(__object) = __schema.as_object_mut() {
                    __object.insert(
                        "default".to_owned(),
                        ::dsrust::__macro_support::serde_json::json!(#default),
                    );
                }
                __schema
            }},
        };
        quote! { ((#key).to_owned(), #schema) }
    });
    quote! {
        ::dsrust::__macro_support::serde_json::Value::Object(
            ::dsrust::__macro_support::serde_json::Map::from_iter([#(#entries),*]),
        )
    }
}

/// The same map as `Tool::args`, behind a `static` so a tool that is a unit struct needs no field.
pub(crate) fn schema(arguments: &[Argument]) -> TokenStream {
    let built = value(arguments);
    quote! {
        fn args(&self) -> &::dsrust::__macro_support::serde_json::Value {
            static ARGS: ::std::sync::LazyLock<
                ::dsrust::__macro_support::serde_json::Value,
            > = ::std::sync::LazyLock::new(|| #built);
            &ARGS
        }
    }
}

/// One `let` per parameter, read out of what the model sent.
///
/// A wrong or missing argument is answered, not raised: the loop can read a refusal and try again,
/// where an error ends the turn. dspy's tools answer the same way.
pub(crate) fn bindings(arguments: &[Argument]) -> TokenStream {
    let each = arguments.iter().map(|argument| {
        let key = &argument.name;
        let binding = &argument.binding;
        let ty = &argument.ty;
        // A declared default stands in through the same JSON the schema states it as, so the value
        // the body binds and the value the model was promised cannot drift apart.
        let stated = match &argument.default {
            None => quote! { ::dsrust::__macro_support::serde_json::Value::Null },
            Some(default) => quote! {
                ::dsrust::__macro_support::serde_json::json!(#default)
            },
        };
        let absent = quote! {
            match ::dsrust::__macro_support::serde_json::from_value::<#ty>(#stated) {
                ::std::result::Result::Ok(__parsed) => __parsed,
                ::std::result::Result::Err(_) => {
                    return ::std::result::Result::Ok(::std::format!(
                        "Refused: this tool needs `{}`.", #key
                    ));
                }
            }
        };
        quote! {
            let #binding: #ty = match __args.get(#key) {
                ::std::option::Option::Some(__value) => {
                    match ::dsrust::__macro_support::serde_json::from_value::<#ty>(
                        __value.clone(),
                    ) {
                        ::std::result::Result::Ok(__parsed) => __parsed,
                        ::std::result::Result::Err(__error) => {
                            return ::std::result::Result::Ok(::std::format!(
                                "Refused: `{}` is not the type this tool takes ({}).",
                                #key, __error
                            ));
                        }
                    }
                }
                ::std::option::Option::None => #absent,
            };
        }
    });
    quote! { #(#each)* }
}

/// The names the bindings introduced, in the order the parameters were declared.
pub(crate) fn names(arguments: &[Argument]) -> Vec<&syn::Ident> {
    arguments.iter().map(|argument| &argument.binding).collect()
}
