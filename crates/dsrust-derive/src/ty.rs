//! The declared type, seen through whatever wrapped it.
//!
//! A `macro_rules!` fragment does not arrive as the type it captured. `$ty:ty` reaches a derive
//! wrapped in an invisible delimiter, syn's grouped variant, so every match on a *path* type in
//! this crate fell through for a signature declared inside a macro and the field took the
//! fallback: its annotation became the token stream (`Vec < String >`, in a prompt) and its kind
//! became the opaque `Json`, so a `String` output grew a JSON-schema note dspy never writes.
//!
//! Generating signatures from a macro is an ordinary thing to want — a table of type shapes, a set
//! of tasks that differ only in one field — and it failed silently, which is the worst way.

/// The type inside any grouping a macro or a written pair of parentheses added.
pub fn peeled(ty: &syn::Type) -> &syn::Type {
    match ty {
        syn::Type::Group(group) => peeled(&group.elem),
        syn::Type::Paren(paren) => peeled(&paren.elem),
        other => other,
    }
}
