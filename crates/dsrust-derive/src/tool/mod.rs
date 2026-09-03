//! `#[tool]` — a tool written as a function, with its doc comment as the description.
//!
//! dspy takes a plain callable and reads two things off it: `__doc__` becomes the description the
//! model is shown, and the type hints become the argument schema. Rust has both — a doc comment and
//! typed parameters — but a doc comment is erased long before the program runs, so there is no
//! `inspect.getdoc` to reach for. Reading it has to happen while the code is still source, which is
//! what an attribute macro is.
//!
//! The description is prompt text, so rustdoc's leading space comes off each line — " Call once"
//! with a leading space is a different string in front of the model. Nothing else does: upstream
//! sends `func.__doc__` unnormalised, and the `cleandoc` a *signature* runs its instructions
//! through would strip indentation the author of a tool wrote deliberately.

mod args;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{Error, FnArg, ImplItem, Item, ItemFn, ItemImpl, Result, Signature};

use crate::parse::tool_doc_text;

/// One attribute, two things it can sit on: the function that *is* a tool, and the impl block
/// whose marked methods are a roster of them.
pub fn expand(attribute: TokenStream, item: Item) -> Result<TokenStream> {
    let declared = declared_description(attribute)?;
    match item {
        Item::Fn(item) => function(item, declared),
        Item::Impl(item) => roster(item),
        other => Err(Error::new(
            other.span(),
            "a tool is a function, or an impl block whose `#[tool]` methods are a roster of them",
        )),
    }
}

/// A free function becomes a tool type of the same name in PascalCase, and stays callable itself.
/// A method takes `&self` instead, so it becomes an accessor beside itself: a free function has no
/// state to hold, and a method's is the receiver.
fn function(mut item: ItemFn, declared: Option<String>) -> Result<TokenStream> {
    if item
        .sig
        .inputs
        .iter()
        .any(|input| receiver_of(input).is_some())
    {
        let accessor = accessor(&item.attrs, &mut item.sig, &item.vis, declared)?;
        return Ok(quote! { #item #accessor });
    }
    let name = item.sig.ident.to_string();
    let description = description(&item.attrs, declared)?;
    let arguments = args::take(&mut item.sig.inputs)?;
    let called = &item.sig.ident;
    let bound = args::names(&arguments);
    let visibility = &item.vis;
    let declared = format_ident!("{}", pascal(&name));
    let documented = documented(&name);

    let implementation = implementation(
        &quote! { #declared },
        &name,
        &description,
        &arguments,
        &quote! { #called(#(#bound),*) },
        item.sig.asyncness.is_some(),
    );
    Ok(quote! {
        #item

        #[doc = #documented]
        #visibility struct #declared;

        #implementation
    })
}

/// An impl block becomes a roster: every `#[tool]` method is one tool over the shared receiver,
/// and the type gains a `tools` method answering with all of them.
fn roster(item: ItemImpl) -> Result<TokenStream> {
    let owner = owner(&item)?;
    let mut declared = Vec::new();
    let mut emitted = Vec::new();
    let mut stripped = item.clone();

    for member in &mut stripped.items {
        let ImplItem::Fn(method) = member else {
            continue;
        };
        let marked = method.attrs.iter().any(is_tool_attribute);
        let stated = method
            .attrs
            .iter()
            .filter(|attribute| is_tool_attribute(attribute))
            .try_fold(None, |found, attribute| match &attribute.meta {
                syn::Meta::Path(_) => Ok(found),
                _ => declared_description(attribute.parse_args::<TokenStream>()?),
            })?;
        method
            .attrs
            .retain(|attribute| !is_tool_attribute(attribute));
        if !marked {
            continue;
        }
        if !method
            .sig
            .inputs
            .iter()
            .any(|input| receiver_of(input).is_some())
        {
            return Err(Error::new(
                method.sig.span(),
                "a tool in a roster takes `&self`, which is what lets every tool in it write into \
                 one state",
            ));
        }
        emitted.push(accessor(
            &method.attrs,
            &mut method.sig,
            &method.vis,
            stated,
        )?);
        declared.push(accessor_name(&method.sig.ident));
    }

    if declared.is_empty() {
        return Err(Error::new(
            item.span(),
            "this impl block declares no tools: mark each one `#[tool]`, so a constructor and a \
             helper beside them stay ordinary methods",
        ));
    }
    Ok(quote! {
        #stripped

        impl #owner {
            #(#emitted)*

            /// The tools this type declares, in the order they are written.
            pub fn tools(
                self: &::std::sync::Arc<Self>,
            ) -> ::std::vec::Vec<::std::boxed::Box<dyn ::dsrust::Tool>> {
                ::std::vec![#(self.#declared()),*]
            }
        }
    })
}

/// A method's tool, as a method beside it. An attribute on a method may emit only more methods —
/// rustc refuses a `struct` or an `impl` there — so this closes over an `Arc` of the receiver
/// rather than declaring a type of its own, which is the sharing Python gets from a closure.
fn accessor(
    attributes: &[syn::Attribute],
    signature: &mut Signature,
    visibility: &syn::Visibility,
    declared: Option<String>,
) -> Result<TokenStream> {
    let name = signature.ident.to_string();
    let description = description(attributes, declared)?;
    let arguments = args::take(&mut signature.inputs)?;
    let schema = args::value(&arguments);
    let bindings = args::bindings(&arguments, &name, &quote! { &__declared });
    let bound = args::names(&arguments);
    let called = &signature.ident;
    let declared = accessor_name(called);
    let held = match signature.asyncness.is_some() {
        false => quote! {
            {
                let __declared = #schema;
                ::dsrust::FnTool::new(
                    #name,
                    #description,
                    __declared.clone(),
                    move |__args: &::dsrust::__macro_support::serde_json::Value| {
                        let __answered: ::dsrust::__macro_support::anyhow::Result<_> = (|| {
                            #bindings
                            ::std::result::Result::Ok(__held.#called(#(#bound),*)?)
                        })();
                        __answered
                    },
                )
            }
        },
        true => quote! {
            {
                let __declared = #schema;
                ::dsrust::AsyncFnTool::new(
                    #name,
                    #description,
                    __declared.clone(),
                    move |__args: ::dsrust::__macro_support::serde_json::Value| {
                        let __held = ::std::sync::Arc::clone(&__held);
                        let __declared = __declared.clone();
                        async move {
                            let __args = &__args;
                            let __answered: ::dsrust::__macro_support::anyhow::Result<_> = async {
                                #bindings
                                ::std::result::Result::Ok(__held.#called(#(#bound),*).await?)
                            }
                            .await;
                            __answered
                        }
                    },
                )
            }
        },
    };
    let documented = documented(&name);
    Ok(quote! {
        #[doc = #documented]
        // The `?` converts a tool that answers with some other error type; where it answers with
        // `anyhow` already there is nothing to convert, and clippy says so.
        #[allow(clippy::needless_question_mark)]
        #visibility fn #declared(
            self: &::std::sync::Arc<Self>,
        ) -> ::std::boxed::Box<dyn ::dsrust::Tool> {
            let __held = ::std::sync::Arc::clone(self);
            ::std::boxed::Box::new(#held)
        }
    })
}

/// `set_title` gives `set_title_tool`, because the method keeps its own name.
fn accessor_name(called: &syn::Ident) -> syn::Ident {
    format_ident!("{}_tool", called)
}

/// The `Tool` impl the free-function form declares a type for.
fn implementation(
    declared: &TokenStream,
    name: &str,
    description: &str,
    arguments: &[args::Argument],
    body: &TokenStream,
    awaited: bool,
) -> TokenStream {
    let schema = args::schema(arguments);
    let bindings = args::bindings(arguments, name, &quote! { ::dsrust::Tool::args(self) });
    let answering = match awaited {
        false => quote! {
            // The value is the answer — dspy's `Tool.__call__` returns whatever the tool produced,
            // and an agent observes it as such. `call` is its text form, for a caller that asked
            // for text.
            fn call_value(
                &self,
                __args: &::dsrust::__macro_support::serde_json::Value,
            ) -> ::dsrust::__macro_support::anyhow::Result<
                ::dsrust::__macro_support::serde_json::Value,
            > {
                let __answered: ::dsrust::__macro_support::anyhow::Result<_> = (|| {
                    #bindings
                    ::std::result::Result::Ok(#body?)
                })();
                __answered.and_then(|__value| {
                    ::dsrust::__macro_support::serde_json::to_value(__value)
                        .map_err(::std::convert::Into::into)
                })
            }

            fn call(
                &self,
                __args: &::dsrust::__macro_support::serde_json::Value,
            ) -> ::dsrust::__macro_support::anyhow::Result<::std::string::String> {
                ::dsrust::Tool::call_value(self, __args)
                    .map(::dsrust::__macro_support::observation_text)
            }
        },
        true => quote! {
            fn call(
                &self,
                _args: &::dsrust::__macro_support::serde_json::Value,
            ) -> ::dsrust::__macro_support::anyhow::Result<::std::string::String> {
                ::std::result::Result::Err(::dsrust::__macro_support::anyhow::anyhow!(
                    "`{}` is asynchronous, so it answers through `acall_value` — which is what \
                     every agent calls",
                    #name
                ))
            }

            fn acall_value<'__a>(
                &'__a self,
                __args: &'__a ::dsrust::__macro_support::serde_json::Value,
            ) -> ::std::pin::Pin<::std::boxed::Box<
                dyn ::std::future::Future<
                    Output = ::dsrust::__macro_support::anyhow::Result<
                        ::dsrust::__macro_support::serde_json::Value,
                    >,
                > + ::std::marker::Send + '__a,
            >> {
                ::std::boxed::Box::pin(async move {
                    let __answered: ::dsrust::__macro_support::anyhow::Result<_> = async {
                        #bindings
                        ::std::result::Result::Ok(#body.await?)
                    }
                    .await;
                    __answered.and_then(|__value| {
                        ::dsrust::__macro_support::serde_json::to_value(__value)
                            .map_err(::std::convert::Into::into)
                    })
                })
            }
        },
    };
    quote! {
        // The `?` converts a tool that answers with some other error type; where it answers with
        // `anyhow` already there is nothing to convert, and clippy says so.
        #[allow(clippy::needless_question_mark)]
        impl ::dsrust::Tool for #declared {
            fn name(&self) -> &str {
                #name
            }

            fn description(&self) -> &str {
                #description
            }

            #schema

            #answering
        }
    }
}

/// The rustdoc for a generated item, which is deliberately not the tool's own description.
///
/// A description is prompt text and may carry any shape at all, including the indented block that
/// rustdoc reads as Rust to compile and run. Repeating it here would make every such tool fail its
/// own doctests, so what is documented is where to find it: the description is `description()`,
/// and the function this was written on keeps whatever doc comment it had.
fn documented(name: &str) -> String {
    format!("The tool `{name}`. Its description is what `Tool::description` answers.")
}

/// What `#[tool(desc = "...")]` states, if anything.
///
/// dspy's `Tool` takes `desc` beside the callable and uses it in preference to `__doc__`, and the
/// need is sharper here: rustdoc reads an indented line in a doc comment as a code block and tries
/// to *run* it, so a description whose exact shape matters — one carrying an indented example, say
/// — cannot always be written as a doc comment at all.
fn declared_description(attribute: TokenStream) -> Result<Option<String>> {
    if attribute.is_empty() {
        return Ok(None);
    }
    let meta: syn::MetaNameValue = syn::parse2(attribute)?;
    if !meta.path.is_ident("desc") {
        return Err(Error::new(
            meta.path.span(),
            "`#[tool]` takes `desc = \"...\"`, the description the model is shown",
        ));
    }
    match meta.value {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(text),
            ..
        }) => Ok(Some(text.value())),
        other => Err(Error::new(other.span(), "`desc` is a string literal")),
    }
}

/// The description the model reads: what `desc` states, or the doc comment, or nothing — an
/// undocumented Python function makes a tool with no description, shown to the model by its name
/// and arguments alone, and so does an undocumented function here.
fn description(attributes: &[syn::Attribute], declared: Option<String>) -> Result<String> {
    match declared {
        Some(declared) => Ok(declared),
        None => Ok(tool_doc_text(attributes)),
    }
}

fn receiver_of(input: &FnArg) -> Option<proc_macro2::Span> {
    match input {
        FnArg::Receiver(receiver) => Some(receiver.span()),
        FnArg::Typed(_) => None,
    }
}

fn is_tool_attribute(attribute: &syn::Attribute) -> bool {
    attribute.path().is_ident("tool")
}

/// The type a roster hangs off. Named rather than taken whole, because each tool holds an
/// `Arc<Owner>` and a generic impl has no single type to put there.
fn owner(item: &ItemImpl) -> Result<syn::Ident> {
    if !item.generics.params.is_empty() {
        return Err(Error::new(
            item.generics.span(),
            "a roster is built for one type, because each tool holds an `Arc` of it",
        ));
    }
    let syn::Type::Path(path) = &*item.self_ty else {
        return Err(Error::new(
            item.self_ty.span(),
            "a roster hangs off a named type",
        ));
    };
    path.path
        .segments
        .last()
        .filter(|segment| segment.arguments.is_none())
        .map(|segment| segment.ident.clone())
        .ok_or_else(|| Error::new(item.self_ty.span(), "a roster hangs off a named type"))
}

/// `add_block` becomes `AddBlock`: the tool's wire name stays the function's, and the type it
/// declares reads as a Rust type.
fn pascal(name: &str) -> String {
    name.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut characters = part.chars();
            match characters.next() {
                Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
                None => String::new(),
            }
        })
        .collect()
}
