//! Items the derive macro expands into, referenced through `::dsrust::__macro_support::…`. Not a
//! public API — hidden from docs and exempt from semver.
//!
//! The crates below are re-exported because a macro expands into them. Naming `::serde` there
//! would resolve in the *caller's* crate root, so a caller who depends only on `dsrust` could not
//! use the derive at all — reported from a real project, and reproduced as a crate depending on
//! nothing else. `tool!` is held to the same rule: it names a `Value`, parses each argument out of
//! one, and gives a body with no stated return type an `anyhow::Result<String>`.

pub use anyhow;
pub use serde;
pub use serde_json;
pub use serde_json::json;

use core::marker::PhantomData;

use crate::adapter::Type;
use crate::signature::TypeDescription;

/// The custom-type prose a derived field states, found by autoref specialization.
///
/// The derive reads a field's Rust type but cannot see whether it is a [`Type`], so it asks
/// through this probe rather than branching on the type name. A type that implements `Type`
/// answers with its [`description`](Type::description); any other answers with nothing — so a
/// plain struct field carries the same empty `descriptions` it always has, and only a custom type
/// adds a line.
///
/// The two answers sit at different receiver levels so method resolution picks between them: the
/// [`Type`] one takes `self` by value and is tried first, the fallback takes `&self` and is
/// reached only by the autoref the first sidesteps. A bound on an *inherent* method would error
/// rather than fall through, so each answer is a trait.
pub struct TypeProbe<T>(pub PhantomData<T>);

/// The answer for a field whose type is a [`Type`]. By-value `self`, so it matches the probe
/// exactly and wins over the `&self` fallback whenever `T: Type`.
pub trait DescribeViaType {
    fn field_descriptions(self) -> Vec<TypeDescription>;
}

impl<T: Type> DescribeViaType for TypeProbe<T> {
    fn field_descriptions(self) -> Vec<TypeDescription> {
        <T as Type>::description().into_iter().collect()
    }
}

/// The answer for every other field type. `&self`, so it is reached only when the by-value
/// [`DescribeViaType`] does not apply — i.e. when the type is not a [`Type`].
pub trait DescribeFallback {
    fn field_descriptions(&self) -> Vec<TypeDescription> {
        Vec::new()
    }
}

impl<T> DescribeFallback for TypeProbe<T> {}

/// The schema a derived output field states, found the same way its prose is.
///
/// Three answers rather than two, because there are three cases and conflating any pair loses one:
///
/// 1. **One of dspy's own types.** Its schema is upstream's, recorded verbatim — pydantic renders
///    the model *and its class docstring*, which no Rust struct produces. `Code` also implements no
///    `JsonSchema` at all, which made it unusable as a derived output while `q -> out: dspy.Code`
///    worked: two spellings of one program, one of which did not compile.
/// 2. **A caller's own [`Type`].** It states prose *and* keeps the schema its Rust shape gives —
///    the seam carries a description, it does not silence the note. Collapsing this into (1) made
///    every caller's custom type lose its schema, which a test caught.
/// 3. **Any other field type**, schema'd from its shape as it always was.
///
/// Two receiver levels, not three. Autoref tries `Self` then `&Self` and stops, so a third would
/// never be reached — the first attempt put the fallback at `&&self` and every plain `JsonSchema`
/// field failed to resolve a method at all.
///
/// (2) and (3) therefore share `&self`, which is sound because they never both apply: a type that
/// is a `Type` *and* `JsonSchema` is answered by (1) at the by-value level before either is
/// consulted, and a type that is only one of them satisfies only one bound. A type that is neither
/// resolves no method, which is the error a caller should get.
pub trait SchemaViaType {
    fn field_schema(self) -> Option<serde_json::Value>;
}

impl<T: Type + schemars::JsonSchema> SchemaViaType for TypeProbe<T> {
    fn field_schema(self) -> Option<serde_json::Value> {
        <T as Type>::output_schema().or_else(|| Some(crate::signature::json_field_schema::<T>()))
    }
}

/// A [`Type`] with no `JsonSchema` — one of dspy's own, whose schema is recorded or absent.
pub trait SchemaViaTypeOnly {
    fn field_schema(&self) -> Option<serde_json::Value>;
}

impl<T: Type> SchemaViaTypeOnly for TypeProbe<T> {
    fn field_schema(&self) -> Option<serde_json::Value> {
        <T as Type>::output_schema()
    }
}

/// Every other field type: the schema its Rust shape gives, in pydantic's dialect.
pub trait SchemaFallback {
    fn field_schema(&self) -> Option<serde_json::Value>;
}

impl<T: schemars::JsonSchema> SchemaFallback for TypeProbe<T> {
    fn field_schema(&self) -> Option<serde_json::Value> {
        Some(crate::signature::json_field_schema::<T>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Code, Image};

    /// A custom type with a description answers with it; one without answers empty; a plain
    /// struct — not a `Type` at all — takes the fallback. The derive emits exactly this call,
    /// with both traits in scope.
    #[test]
    fn the_probe_finds_a_type_description_and_falls_back_otherwise() {
        use {DescribeFallback as _, DescribeViaType as _};

        let code = TypeProbe::<Code>(PhantomData).field_descriptions();
        assert_eq!(code.len(), 1);
        assert_eq!(code[0].name, "Code");
        assert!(code[0].replaces_schema);

        // `Image` is a `Type` but states no description.
        assert!(
            TypeProbe::<Image>(PhantomData)
                .field_descriptions()
                .is_empty()
        );

        // A plain struct is not a `Type`; the fallback answers.
        struct Plain;
        assert!(
            TypeProbe::<Plain>(PhantomData)
                .field_descriptions()
                .is_empty()
        );
    }
}
