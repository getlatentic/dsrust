use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::parse::{Field, Kind, Model};

/// Expansion: the two companion structs, the `SignatureSpec` impl, and the module
/// constructors. Generated paths name the library as `::dsrust`, which its
/// `extern crate self as dsrust` alias keeps valid inside the crate itself.
pub fn expand(model: &Model) -> TokenStream {
    let companions = companions(model);
    let spec = spec_impl(model);
    let constructors = constructor_impl(model);
    let declared = fields_are_read(model);
    quote! {
        #companions
        #spec
        #constructors
        #declared
    }
}

/// A declared signature is a *declaration*: a caller writes the struct to state the task and then
/// never constructs it — `Predict!(Task { … })` builds the generated `TaskInputs` instead. So every
/// field of every signature warns as dead code, in the caller's own crate, for writing the thing
/// the derive asked them to write. Reading each field once here is what makes the declaration count
/// as the use it is.
fn fields_are_read(model: &Model) -> TokenStream {
    let name = &model.name;
    let idents = model
        .inputs
        .iter()
        .chain(&model.outputs)
        .map(|field| &field.ident);
    quote! {
        const _: () = {
            #[allow(dead_code)]
            fn declared_fields_are_used(declared: &#name) {
                #( let _ = &declared.#idents; )*
            }
        };
    }
}

/// Inherent module constructors, so a call site reads `GiftTask::predict().call(&inputs)`.
/// dead_code is allowed because the derive cannot know which module the host uses.
fn constructor_impl(model: &Model) -> TokenStream {
    let name = &model.name;
    quote! {
        impl #name {
            #[allow(dead_code)]
            pub fn predict() -> ::dsrust::predict::TypedPredict<Self> {
                ::dsrust::predict::Predict::task::<Self>()
            }

            #[allow(dead_code)]
            pub fn chain_of_thought() -> ::dsrust::predict::TypedChainOfThought<Self> {
                ::dsrust::predict::ChainOfThought::task::<Self>()
            }
        }
    }
}

/// The companions carry the user's declared Rust types verbatim, so a `u32` input stays a
/// `u32` at the call site and a `f64` output deserializes as a number.
///
/// **`non_snake_case` is allowed on both.** A field's name is prompt text — `gepa_papillon`
/// declares `response_A` and `response_B`, and renaming them writes a different prompt — so a
/// caller has to be able to spell one. Without this the warning fires inside code the caller never
/// wrote and cannot annotate: an `#[allow]` on their own struct does not reach the companions.
fn companions(model: &Model) -> TokenStream {
    let vis = &model.vis;
    let inputs_name = format_ident!("{}Inputs", model.name);
    let outputs_name = format_ident!("{}Outputs", model.name);
    let input_fields = model.inputs.iter().map(|f| companion_field(vis, f));
    let output_fields = model.outputs.iter().map(|f| companion_field(vis, f));
    // Through the library's own re-export, not `::serde`: these are structs this macro writes, and
    // `::serde` would resolve in the caller's crate root — so a caller depending only on `dsrust`
    // could not use the derive. `serde(crate = …)` is what points the derived code at the same
    // re-export for its runtime.
    quote! {
        #[derive(Debug, Clone, ::dsrust::__macro_support::serde::Serialize)]
        #[serde(crate = "::dsrust::__macro_support::serde")]
        #[allow(non_snake_case)]
        #vis struct #inputs_name {
            #( #input_fields, )*
        }

        #[derive(
            Debug,
            Clone,
            ::dsrust::__macro_support::serde::Serialize,
            ::dsrust::__macro_support::serde::Deserialize,
        )]
        #[serde(crate = "::dsrust::__macro_support::serde")]
        #[allow(non_snake_case)]
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
    let instructions = instructions(model);
    let in_fields = model.inputs.iter().map(in_field);
    let out_fields = model.outputs.iter().map(out_field);
    let pair_inputs = model.inputs.iter().map(pair_input);
    quote! {
        impl ::dsrust::signature::SignatureSpec for #name {
            type Inputs = #inputs_name;
            type Outputs = #outputs_name;

            fn signature() -> ::dsrust::signature::Signature {
                ::dsrust::signature::Signature {
                    instructions: #instructions,
                    inputs: ::std::vec![ #( #in_fields ),* ],
                    outputs: ::std::vec![ #( #out_fields ),* ],
                }
            }

            fn input_pairs(
                inputs: &Self::Inputs,
            ) -> ::std::vec::Vec<::dsrust::adapter::Input<'static>> {
                ::std::vec![ #( #pair_inputs ),* ]
            }
        }
    }
}

/// How one input reaches the adapters. Every field crosses as a `Value`, which is what dspy
/// hands its adapters: rendering is the adapter's job, and a structured field that arrived
/// pre-rendered could not expand into the turns a `History` needs. Serialization only fails on
/// a broken `Serialize` impl — programmer error, not model behavior — so this expects success
/// and names the field.
fn pair_value(field: &Field) -> TokenStream {
    let ident = &field.ident;
    let message = format!("input `{ident}` must serialize to JSON");
    quote! { ::dsrust::__macro_support::serde_json::to_value(&inputs.#ident).expect(#message) }
}

/// One input as the adapters receive it, carrying whether it came from one of the caller's own
/// structs. That is what dspy reads off a value with `isinstance(value, BaseModel)`, and it is
/// gone by the time the value is JSON — so it is answered here, from the declared type.
fn pair_input(field: &Field) -> TokenStream {
    let name = field.ident.to_string();
    let value = pair_value(field);
    match crate::parse::is_record(&field.ty) {
        true => quote! { ::dsrust::adapter::Input::record(#name, #value) },
        false => quote! { ::dsrust::adapter::Input::new(#name, #value) },
    }
}

/// The prose a field's declared type states about itself, asked of the type through the host
/// crate's autoref probe: a custom [`Type`](dsrust::Type)'s `description()`, or nothing for a
/// plain structure that says nothing about itself. The derive cannot see whether a Rust type is a
/// custom type, so it asks rather than branching on the type's name.
fn type_descriptions(ty: &syn::Type) -> TokenStream {
    quote! {
        {
            use ::dsrust::__macro_support::{DescribeFallback as _, DescribeViaType as _};
            ::dsrust::__macro_support::TypeProbe::<#ty>(::core::marker::PhantomData)
                .field_descriptions()
        }
    }
}

/// The objective the signature carries, synthesised from the field names when nobody wrote one.
///
/// dspy's `_default_instructions` stands in for a missing docstring, and it is reached rather than
/// respelled here — the sentence is prompt text, and two copies of it drift.
fn instructions(model: &Model) -> TokenStream {
    if !model.instructions.is_empty() {
        let text = &model.instructions;
        return quote! { #text.to_owned() };
    }
    let inputs = model.inputs.iter().map(|field| field.ident.to_string());
    let outputs = model.outputs.iter().map(|field| field.ident.to_string());
    quote! {
        ::dsrust::signature::default_instructions(&[#(#inputs),*], &[#(#outputs),*])
    }
}

/// The host crate's `FieldKind` for this field. Every non-scalar becomes the opaque `Json`
/// kind: the derive reads the Rust type, which does not tell it the Python type dspy prints.
fn kind(field: &Field) -> TokenStream {
    let variant = match field.kind {
        Kind::Str => quote! { Str },
        Kind::Bool => quote! { Bool },
        Kind::Int => quote! { Int },
        Kind::Float => quote! { Float },
        Kind::Reasoning => quote! { Reasoning },
        Kind::Json => {
            let annotation = json_annotation(field);
            let descriptions = type_descriptions(&field.ty);
            return quote! {
                ::dsrust::signature::FieldKind::Json(
                    ::dsrust::signature::JsonType::plain(#annotation)
                        .descriptions(#descriptions),
                )
            };
        }
    };
    quote! { ::dsrust::signature::FieldKind::#variant }
}

/// An output field's kind, carrying the structure of its declared type as well as its name.
///
/// [`BamlAdapter`](dsrust::BamlAdapter) states a type instead of a schema of it, and without this
/// every Rust type reached it as the bare word `json`. The shape comes from `schemars`, whose
/// `JsonSchema` bound an output field already carries for its schema — so this asks nothing new
/// of a caller, and needs no annotation of the kind other ports require.
///
/// Only outputs. An input's structure is never stated: a request carries the value itself, and
/// requiring `JsonSchema` of every input type would be a bound no caller owes today.
fn out_kind(field: &Field) -> TokenStream {
    let Kind::Json = field.kind else {
        return kind(field);
    };
    let ty = &field.ty;
    let annotation = json_annotation(field);
    let descriptions = type_descriptions(ty);
    quote! {
        ::dsrust::signature::FieldKind::Json(
            ::dsrust::signature::JsonType::reflected(
                #annotation,
                ::dsrust::signature::json_field_reflection::<#ty>(),
            )
            .descriptions(#descriptions),
        )
    }
}

fn in_field(field: &Field) -> TokenStream {
    let name = field.ident.to_string();
    let desc = &field.desc;
    let kind = kind(field);
    // Not `enum_aware`, unlike an output: the note that names an enum's members is an output's,
    // and an input field already renders as its type's own name. Asking here would put a
    // `JsonSchema` bound on every input type, which the derive has never required — dspy states
    // an input's structure nowhere, because the request carries the value itself.
    let values = closed_set(field);
    let constraints = constraints(field);
    quote! {
        ::dsrust::signature::InField {
            name: #name.to_owned(),
            desc: #desc.to_owned(),
            kind: #kind,
            values: #values,
            constraints: #constraints,
            ..::std::default::Default::default()
        }
    }
}

/// The kind, decided at run time when the declared type turns out to be a string enumeration.
///
/// Outputs only, for the reason `in_field` gives.
///
/// The derive cannot tell an enum from a struct — both are a path — so it asks the type's schema
/// instead, which can. dspy renders the two differently, so a Rust enum reaching a prompt as a
/// JSON schema is a different prompt than dspy's for the same program.
fn enum_aware(field: &Field, otherwise: TokenStream) -> TokenStream {
    let Kind::Json = field.kind else {
        return otherwise;
    };
    let ty = &field.ty;
    let annotation = crate::annotate::python_spelling(ty);
    quote! {
        match ::dsrust::signature::declared_members::<#ty>() {
            // dspy prints the type's own name for an enum, not a schema of it.
            ::std::option::Option::Some(_) => {
                ::dsrust::signature::FieldKind::Enum(#annotation.to_owned())
            }
            ::std::option::Option::None => #otherwise,
        }
    }
}

/// The members an enumeration names, else whatever `values(...)` declared.
///
/// A `Literal` closed set and an enum are different renderings — upstream prints a `Literal`'s
/// members and asks for the spelling, and prints an enum's *name* and asks for one of its values —
/// so the two cannot be merged, only chosen between.
fn enum_members(field: &Field, otherwise: TokenStream) -> TokenStream {
    let Kind::Json = field.kind else {
        return otherwise;
    };
    let ty = &field.ty;
    quote! {
        match ::dsrust::signature::declared_members::<#ty>() {
            ::std::option::Option::Some(members) => ::std::option::Option::Some(members),
            ::std::option::Option::None => #otherwise,
        }
    }
}

/// The Python type a `Json` field prints, with a closed set folded into it.
///
/// A `values(...)` set on a list is `list[Literal['a', 'b']]` upstream, not a `list[str]` beside a
/// constraint: Python has no field-level closed set, so the members live in the annotation and in
/// the schema, and nowhere else.
fn json_annotation(field: &Field) -> String {
    let spelled = crate::annotate::python_spelling(&field.ty);
    match &field.values {
        Some(members) => {
            spelled.replace("[str]", &format!("[{}]", crate::annotate::literal(members)))
        }
        None => spelled,
    }
}

/// A declared `values(...)` set as the run-time `Vec<LiteralValue>` the field carries. The
/// members are string literals, which is the only closed set a typed Rust field can hold.
fn closed_set(field: &Field) -> TokenStream {
    match &field.values {
        Some(values) => quote! {
            ::std::option::Option::Some(::std::vec![
                #( ::dsrust::signature::LiteralValue::Str(#values.to_owned()) ),*
            ])
        },
        None => quote! { ::std::option::Option::None },
    }
}

/// A `Json` output embeds the schema of its declared type, retrieved through the host
/// crate's schemars-backed helper; the `JsonSchema` bound that requires is checked right
/// here at the derive site.
fn out_field(field: &Field) -> TokenStream {
    let name = field.ident.to_string();
    let desc = &field.desc;
    let kind = enum_aware(field, out_kind(field));
    // A constrained list states its set inside its schema, as `list[Literal[...]]` does upstream.
    // Carrying it as the field's own closed set too would render the note for a scalar `Literal`.
    let values = match constrained_list(field) {
        true => quote! { ::std::option::Option::None },
        false => enum_members(field, closed_set(field)),
    };
    let schema = out_schema(field);
    let constraints = constraints(field);
    quote! {
        ::dsrust::signature::OutField {
            name: #name.to_owned(),
            desc: #desc.to_owned(),
            kind: #kind,
            values: #values,
            schema: #schema,
            constraints: #constraints,
            ..::std::default::Default::default()
        }
    }
}

/// The bounds a field declared, as the prose the prompt carries. See [`crate::constraints`].
fn constraints(field: &Field) -> TokenStream {
    match &field.constraints {
        Some(prose) => quote! { ::std::option::Option::Some(#prose.to_owned()) },
        None => quote! { ::std::option::Option::None },
    }
}

/// Whether this field is a list of strings narrowed by `values(...)`.
fn constrained_list(field: &Field) -> bool {
    field.values.is_some() && crate::annotate::is_list_of_strings(&field.ty)
}

/// The schema a `Json` output prints in its note.
///
/// A constrained list is spelled out rather than read off the Rust type: schemars sees a
/// `Vec<String>` and cannot know the set, which lives on the attribute. dspy's `list[Literal[...]]`
/// puts the members in `items`, and the key order is `move_type_to_front`'s.
fn out_schema(field: &Field) -> TokenStream {
    let Kind::Json = field.kind else {
        return quote! { ::std::option::Option::None };
    };
    if let Some(members) = &field.values
        && constrained_list(field)
    {
        return quote! {
            ::std::option::Option::Some(::dsrust::__macro_support::serde_json::json!({
                "type": "array",
                "items": { "type": "string", "enum": [#(#members),*] },
            }))
        };
    }
    let ty = &field.ty;
    // Asked of the type rather than taken from its Rust shape. One of dspy's own types answers
    // with the schema upstream prints — recorded, because pydantic renders a model *and its class
    // docstring* — and everything else answers with `json_field_schema`. `Code` has no
    // `JsonSchema` at all, so this is also what lets it be a derived output: the bound sits on the
    // fallback, which is never instantiated for a type that answers the other way.
    quote! {
        {
            use ::dsrust::__macro_support::{
                SchemaFallback as _, SchemaViaType as _, SchemaViaTypeOnly as _,
            };
            ::dsrust::__macro_support::TypeProbe::<#ty>(::core::marker::PhantomData)
                .field_schema()
        }
    }
}

/// A caller's `Cargo.toml` is the derive's real interface: whatever crate the expansion names has
/// to be one they already depend on. Reported from a real project — a crate depending on `dsrust`
/// alone could not use the derive at all, because the companion structs derived `::serde::…`,
/// which resolves in *their* crate root.
#[cfg(test)]
mod tests {
    use crate::parse::model;
    use syn::parse_quote;

    /// The crates the expansion may name. `dsrust` is the library itself and `std`/`core` are the
    /// language; anything else is a dependency the derive would be silently demanding.
    const OWED: [&str; 3] = ["dsrust", "std", "core"];

    fn expanded(item: syn::DeriveInput) -> String {
        super::expand(&model(&item).expect("parses")).to_string()
    }

    /// Every leading `::name` in the expansion, which is the shape that resolves at a crate root.
    fn crates_named(expansion: &str) -> Vec<String> {
        let mut named = Vec::new();
        for at in expansion.match_indices(":: ").map(|(at, _)| at) {
            let before = expansion[..at].trim_end();
            // A leading `::` is one not preceded by an identifier — `::serde`, never `foo :: bar`.
            if before.ends_with(':') || before.chars().next_back().is_some_and(is_pathish) {
                continue;
            }
            let rest = expansion[at + 3..].trim_start();
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() && !named.contains(&name) {
                named.push(name);
            }
        }
        named
    }

    fn is_pathish(character: char) -> bool {
        character.is_alphanumeric() || character == '_' || character == '>'
    }

    #[test]
    fn the_expansion_names_no_crate_a_caller_does_not_already_have() {
        let expansion = expanded(parse_quote! {
            #[signature(instructions = "Do the task.")]
            struct Task {
                #[input]
                text: String,
                #[input]
                count: u32,
                #[output]
                answer: String,
                #[output]
                steps: Vec<String>,
            }
        });
        let named = crates_named(&expansion);
        assert!(
            named.contains(&"dsrust".to_owned()),
            "the library itself is named: {named:?}"
        );
        let owed: Vec<&String> = named
            .iter()
            .filter(|name| !OWED.contains(&name.as_str()))
            .collect();
        assert!(
            owed.is_empty(),
            "the expansion demands {owed:?} of the caller; go through ::dsrust::__macro_support"
        );
    }

    /// serde's derive needs telling where its runtime lives, or the code *it* generates names
    /// `::serde` itself and the leak reopens one level down.
    #[test]
    fn the_companions_point_serde_at_the_re_export() {
        let expansion = expanded(parse_quote! {
            #[signature(instructions = "Do the task.")]
            struct Task {
                #[input]
                text: String,
                #[output]
                answer: String,
            }
        });
        assert_eq!(
            expansion
                .matches("crate = \"::dsrust::__macro_support::serde\"")
                .count(),
            2,
            "both companion structs say where serde is: {expansion}"
        );
    }
}
