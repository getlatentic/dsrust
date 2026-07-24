//! Items the derive macro expands into, referenced through `::dsrust::__macro_support::…`. Not a
//! public API — hidden from docs and exempt from semver.

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
        assert!(TypeProbe::<Image>(PhantomData).field_descriptions().is_empty());

        // A plain struct is not a `Type`; the fallback answers.
        struct Plain;
        assert!(TypeProbe::<Plain>(PhantomData).field_descriptions().is_empty());
    }
}
