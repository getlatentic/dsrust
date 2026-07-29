//! `#[dsrust::main]`: an async `main` with the runtime already under it.
//!
//! Every entry point in this library is async, because a provider call is a network call. That
//! normally makes a runtime the caller's second dependency, and tokio's own `#[tokio::main]`
//! cannot be borrowed for it — the code that attribute writes names `::tokio` in *its* crate root,
//! so re-exporting the crate is not enough.
//!
//! Writing the runtime here instead names it through `::dsrust::tokio`, which the library owns. A
//! caller with `dsrust` as their only dependency gets an async `main` that runs.

use proc_macro2::TokenStream;
use quote::quote;

/// Turn `async fn main()` into a `fn main()` that drives it on a multi-threaded runtime.
pub fn expand(item: TokenStream) -> syn::Result<TokenStream> {
    let function: syn::ItemFn = syn::parse2(item)?;
    if function.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            function.sig.fn_token,
            "#[dsrust::main] takes an `async fn`; a plain `fn main` needs no runtime",
        ));
    }

    let attributes = &function.attrs;
    let visibility = &function.vis;
    let name = &function.sig.ident;
    let inputs = &function.sig.inputs;
    let output = &function.sig.output;
    let body = &function.block;

    Ok(quote! {
        #(#attributes)*
        #visibility fn #name(#inputs) #output {
            ::dsrust::tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("the async runtime starts")
                .block_on(async move #body)
        }
    })
}

#[cfg(test)]
mod tests {
    use quote::quote;

    /// The runtime is named through the library, never as `::tokio` — the caller does not have a
    /// crate by that name, which is the whole reason this exists rather than `#[tokio::main]`.
    #[test]
    fn the_runtime_is_reached_through_the_library() {
        let expanded = super::expand(quote! {
            async fn main() -> Result<(), Box<dyn std::error::Error>> { Ok(()) }
        })
        .expect("expands")
        .to_string();
        assert!(
            expanded.contains(":: dsrust :: tokio :: runtime"),
            "{expanded}"
        );
        assert!(
            !expanded.contains("async fn main"),
            "the async is gone: {expanded}"
        );
    }

    /// A `fn main` that is not async has nothing to drive, and saying so beats expanding to
    /// `block_on` over a value that is not a future.
    #[test]
    fn a_synchronous_main_is_refused_by_name() {
        let refused = super::expand(quote! { fn main() {} }).expect_err("refused");
        assert!(refused.to_string().contains("async fn"), "{refused}");
    }
}
