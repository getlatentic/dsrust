//! The Python spelling of a Rust type, for the annotation a prompt carries.
//!
//! Every adapter prints a field's type on its own line — `1. \`ideas\` (list[Gift]):` — and dspy
//! prints Python's spelling there. The derive used to answer `json` for every non-scalar, which
//! named nothing a model could use.
//!
//! Each spelling below was read off dspy 3.2.1 rather than guessed, `Union[str, NoneType]`
//! included: it is what `typing.Optional[str]` prints as, where `Optional[str]` and `str | None`
//! are both what a reader would expect and neither is what upstream writes.

use quote::ToTokens;

/// How dspy would print this type.
pub fn python_spelling(ty: &syn::Type) -> String {
    let syn::Type::Path(path) = ty else {
        return ty.to_token_stream().to_string();
    };
    let Some(last) = path.path.segments.last() else {
        return ty.to_token_stream().to_string();
    };
    let name = last.ident.to_string();
    match arguments(last) {
        None => scalar(&name).map_or(name, str::to_owned),
        Some(args) => generic(&name, &args),
    }
}

/// The Python name for a Rust scalar, or nothing when the type is one of the caller's own —
/// which travels under the name it was declared with, as a pydantic model does.
fn scalar(name: &str) -> Option<&'static str> {
    match name {
        "String" | "str" => Some("str"),
        "bool" => Some("bool"),
        "f32" | "f64" => Some("float"),
        // What `serde_json::Value` holds is what pydantic spells `Any`.
        "Value" => Some("Any"),
        _ if INT_TYPES.contains(&name) => Some("int"),
        _ => None,
    }
}

const INT_TYPES: [&str; 10] = [
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "isize", "usize",
];

/// The type arguments of a path segment, if it has any.
fn arguments(segment: &syn::PathSegment) -> Option<Vec<&syn::Type>> {
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    Some(
        args.args
            .iter()
            .filter_map(|arg| match arg {
                syn::GenericArgument::Type(ty) => Some(ty),
                _ => None,
            })
            .collect(),
    )
}

/// A container, spelled the way Python spells it.
///
/// Anything unrecognised keeps its own name and drops its arguments: a name a reader can look up
/// beats a Rust spelling no Python program would print.
fn generic(name: &str, args: &[&syn::Type]) -> String {
    let spelled: Vec<String> = args.iter().map(|arg| python_spelling(arg)).collect();
    match (name, spelled.as_slice()) {
        ("Vec" | "VecDeque" | "HashSet" | "BTreeSet", [of]) => format!("list[{of}]"),
        ("HashMap" | "BTreeMap", [key, value]) => format!("dict[{key}, {value}]"),
        // dspy prints an optional as the union it is, naming the null arm `NoneType`.
        ("Option", [inner]) => format!("Union[{inner}, NoneType]"),
        ("Box" | "Arc" | "Rc", [inner]) => inner.clone(),
        _ => name.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn spelling(ty: syn::Type) -> String {
        python_spelling(&ty)
    }

    #[test]
    fn the_scalars_take_pythons_names() {
        assert_eq!(spelling(parse_quote!(String)), "str");
        assert_eq!(spelling(parse_quote!(i64)), "int");
        assert_eq!(spelling(parse_quote!(u8)), "int");
        assert_eq!(spelling(parse_quote!(f64)), "float");
        assert_eq!(spelling(parse_quote!(bool)), "bool");
    }

    /// One of the caller's own types travels under its own name, the way a pydantic model does.
    #[test]
    fn a_declared_type_keeps_its_name() {
        assert_eq!(spelling(parse_quote!(Gift)), "Gift");
        assert_eq!(spelling(parse_quote!(crate::gifts::Gift)), "Gift");
    }

    #[test]
    fn the_containers_are_spelled_the_way_dspy_prints_them() {
        assert_eq!(spelling(parse_quote!(Vec<Gift>)), "list[Gift]");
        assert_eq!(spelling(parse_quote!(Vec<String>)), "list[str]");
        assert_eq!(
            spelling(parse_quote!(HashMap<String, i32>)),
            "dict[str, int]"
        );
    }

    /// Read off dspy 3.2.1: `typing.Optional[str]` prints as `Union[str, NoneType]`, which is
    /// neither of the two spellings a reader would reach for.
    #[test]
    fn an_option_is_the_union_dspy_prints() {
        assert_eq!(
            spelling(parse_quote!(Option<String>)),
            "Union[str, NoneType]"
        );
        assert_eq!(
            spelling(parse_quote!(Option<Gift>)),
            "Union[Gift, NoneType]"
        );
    }

    #[test]
    fn nesting_is_spelled_all_the_way_down() {
        assert_eq!(
            spelling(parse_quote!(Vec<HashMap<String, Vec<Gift>>>)),
            "list[dict[str, list[Gift]]]"
        );
    }

    /// A pointer is not a type a model has to know about.
    #[test]
    fn a_smart_pointer_is_spelled_as_what_it_points_at() {
        assert_eq!(spelling(parse_quote!(Box<Gift>)), "Gift");
        assert_eq!(spelling(parse_quote!(Vec<Box<Gift>>)), "list[Gift]");
    }

    #[test]
    fn an_untyped_value_is_any() {
        assert_eq!(spelling(parse_quote!(serde_json::Value)), "Any");
    }
}
