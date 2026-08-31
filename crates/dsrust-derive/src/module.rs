//! `#[derive(Module)]`: everything a program of your own owes the rest of the crate, except how
//! to run it.
//!
//! Python gets these by inheriting: the walk an optimizer works through, and being callable.
//! Rust has no inheritance, so a derive is where they come from — the same trade
//! `#[derive(Signature)]` already makes.
//!
//! `forward_traced` stays hand-written, because relabelling a trace needs the order the steps run
//! in and only `forward` knows that. A module without it still compiles; every step then receives
//! the same demo rather than the ones its own calls earned.
//!
//! A field is a step if it is a module, and there is no way to ask a type that at macro time, so
//! the rule is positional instead: every named field is walked unless it carries `#[not_a_step]`.
//! A struct whose fields are all modules therefore says nothing at all.

use quote::quote;
use syn::{Data, DeriveInput, Fields};

pub(crate) fn expand(item: &DeriveInput) -> Result<proc_macro2::TokenStream, syn::Error> {
    let name = &item.ident;
    let label = name.to_string();
    let steps = steps(item)?;
    let answering = match task(item)? {
        // The task arms of the built-in module macros wrap in `Typed`, which is the same thing:
        // the answer is read as the task's outputs rather than looked up in a `Prediction`.
        Some(task) => quote! {
            impl ::dsrust::Ask for #name {
                type Answer = <#task as ::dsrust::signature::SignatureSpec>::Outputs;

                fn ask<'a>(
                    &'a self,
                    inputs: ::dsrust::Example,
                ) -> ::std::pin::Pin<
                    ::std::boxed::Box<
                        dyn ::std::future::Future<Output = ::dsrust::__macro_support::anyhow::Result<Self::Answer>>
                            + Send
                            + 'a,
                    >,
                > {
                    ::std::boxed::Box::pin(async move {
                        ::dsrust::Module::forward(self, inputs)
                            .await?
                            .typed::<Self::Answer>()
                    })
                }
            }
        },
        None => quote! { ::dsrust::asks_with_a_prediction!(#name); },
    };

    // Each child's predictors are renamed after the field holding it, so a demo says which step
    // of the program earned it rather than which predictor inside that step.
    let walk = steps.iter().map(|field| {
        let label = field.to_string();
        quote! {
            for mut inner in ::dsrust::Module::named_predictors(&mut self.#field) {
                inner.name = #label.to_owned();
                found.push(inner);
            }
        }
    });

    Ok(quote! {
        impl ::dsrust::Module for #name {
            fn forward<'a>(
                &'a self,
                inputs: ::dsrust::Example,
            ) -> ::std::pin::Pin<
                ::std::boxed::Box<
                    dyn ::std::future::Future<Output = ::dsrust::__macro_support::anyhow::Result<::dsrust::Prediction>>
                        + Send
                        + 'a,
                >,
            > {
                ::std::boxed::Box::pin(async move {
                    // dspy's `on_module_start`/`on_module_end`, which upstream gets by decorating
                    // `Module.__call__`. Here the derive is that entry, so a module of the caller's
                    // own is watched without their having to ask.
                    let watch = ::dsrust::observe::module_shown(
                        #label,
                        &inputs,
                        ::dsrust::Module::callbacks(self),
                    );
                    ::dsrust::observe::watching(
                        watch,
                        ::dsrust::Forward::forward(self, inputs),
                    )
                    .await
                })
            }

            fn named_predictors(&mut self) -> ::std::vec::Vec<::dsrust::NamedPredictor<'_>> {
                let mut found = ::std::vec::Vec::new();
                #(#walk)*
                found
            }
        }

        #answering
    })
}

/// The task this module answers with, if it names one: `#[task(QA)]` beside the derive.
///
/// Without it `call!` answers with a `Prediction`, which is what a program whose outputs are not
/// one task's — a router, a pipeline ending in a different shape — has to answer with.
fn task(item: &DeriveInput) -> Result<Option<syn::Path>, syn::Error> {
    let Some(attribute) = item.attrs.iter().find(|a| a.path().is_ident("task")) else {
        return Ok(None);
    };
    attribute.parse_args::<syn::Path>().map(Some).map_err(|_| {
        syn::Error::new_spanned(
            attribute,
            "`#[task(..)]` names one signature type, as `#[task(QA)]`",
        )
    })
}

/// The named fields to walk, in declaration order.
fn steps(item: &DeriveInput) -> Result<Vec<syn::Ident>, syn::Error> {
    let Data::Struct(data) = &item.data else {
        return Err(syn::Error::new_spanned(
            item,
            "a module is a struct: its fields are the steps it runs",
        ));
    };
    let Fields::Named(named) = &data.fields else {
        return Err(syn::Error::new_spanned(
            &data.fields,
            "a module's steps are named fields, so an optimizer can name what it rewrote",
        ));
    };
    Ok(named
        .named
        .iter()
        .filter(|field| !field.attrs.iter().any(|a| a.path().is_ident("not_a_step")))
        .filter_map(|field| field.ident.clone())
        .collect())
}
