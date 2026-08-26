use syn::spanned::Spanned;
use syn::{Attribute, Data, DeriveInput, Error, Expr, ExprLit, Fields, Lit, LitStr, Result, Token};

/// Everything the expansion needs, validated: instructions plus the declared fields split
/// by direction in declaration order.
pub struct Model {
    pub vis: syn::Visibility,
    pub name: syn::Ident,
    pub instructions: String,
    pub inputs: Vec<Field>,
    pub outputs: Vec<Field>,
}

pub struct Field {
    pub ident: syn::Ident,
    /// The declared Rust type, kept verbatim for the companion structs.
    pub ty: syn::Type,
    pub kind: Kind,
    pub desc: String,
    pub values: Option<Vec<String>>,
}

/// The wire type a field maps to; mirrors the host crate's `FieldKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Str,
    Bool,
    Int,
    Float,
    Json,
    /// dspy's str-like `Reasoning`: renders as a string but is not `str`, so it keeps the
    /// output-requirement hint.
    Reasoning,
}

pub fn model(item: &DeriveInput) -> Result<Model> {
    let Data::Struct(data) = &item.data else {
        return Err(Error::new(
            item.span(),
            "#[derive(Signature)] only works on a struct",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(Error::new(
            item.span(),
            "#[derive(Signature)] needs named fields",
        ));
    };
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for field in &fields.named {
        let (parsed, direction) = parse_field(field)?;
        match direction {
            Direction::Input => inputs.push(parsed),
            Direction::Output => outputs.push(parsed),
        }
    }
    ensure_both_directions(item, &inputs, &outputs)?;
    Ok(Model {
        vis: item.vis.clone(),
        name: item.ident.clone(),
        instructions: instructions(item)?,
        inputs,
        outputs,
    })
}

fn ensure_both_directions(item: &DeriveInput, inputs: &[Field], outputs: &[Field]) -> Result<()> {
    if inputs.is_empty() {
        return Err(Error::new(
            item.ident.span(),
            "a signature needs at least one #[input] field",
        ));
    }
    if outputs.is_empty() {
        return Err(Error::new(
            item.ident.span(),
            "a signature needs at least one #[output] field",
        ));
    }
    Ok(())
}

/// Instructions: the `#[signature(instructions = "...")]` attribute wins, the struct's doc
/// comment is the fallback, and having neither is an error because the model would get an
/// empty objective.
fn instructions(item: &DeriveInput) -> Result<String> {
    let attribute = signature_attribute(&item.attrs)?;
    let text = match attribute {
        Some(text) => text,
        None => doc_text(&item.attrs),
    };
    if text.is_empty() {
        return Err(Error::new(
            item.ident.span(),
            "missing instructions: add #[signature(instructions = \"...\")] or a doc comment",
        ));
    }
    Ok(text)
}

fn signature_attribute(attrs: &[Attribute]) -> Result<Option<String>> {
    let Some(attr) = attrs.iter().find(|a| a.path().is_ident("signature")) else {
        return Ok(None);
    };
    let mut instructions = None;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("instructions") {
            let lit: LitStr = meta.value()?.parse()?;
            instructions = Some(lit.value());
            Ok(())
        } else {
            Err(meta.error("unknown key; expected instructions = \"...\""))
        }
    })?;
    instructions
        .map(Some)
        .ok_or_else(|| Error::new(attr.span(), "expected #[signature(instructions = \"...\")]"))
}

/// Doc-comment text: one `#[doc = " line"]` per `///` line, each with rustdoc's leading
/// space stripped, joined by newlines.
fn doc_text(attrs: &[Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(doc_line)
        .collect();
    cleandoc(&lines.join("\n"))
}

/// Python's `inspect.cleandoc`, which is what dspy runs a signature's docstring through.
///
/// A doc comment here plays the part a docstring plays there, so it is normalised the same way:
/// tabs expanded, the indentation common to every line *after the first* removed, and blank lines
/// trimmed off both ends. This used to strip one leading space per line and trim, which agrees for
/// the conventional `/// text` and parts company three ways — a uniformly indented comment kept its
/// indent, a comment whose first line is flush kept the rest's, and a tab survived. Instructions
/// render into the prompt, so each of those is a different string in front of the model.
fn cleandoc(doc: &str) -> String {
    let expanded: Vec<String> = doc.split('\n').map(expand_tabs).collect();
    let margin = expanded
        .iter()
        .skip(1)
        .filter(|line| !line.trim_start_matches(' ').is_empty())
        .map(|line| line.len() - line.trim_start_matches(' ').len())
        .min();
    let mut lines: Vec<String> = expanded
        .iter()
        .enumerate()
        .map(|(at, line)| match (at, margin) {
            (0, _) => line.trim_start_matches(' ').to_owned(),
            (_, Some(margin)) => line.chars().skip(margin).collect(),
            (_, None) => line.clone(),
        })
        .collect();
    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    while lines.first().is_some_and(String::is_empty) {
        lines.remove(0);
    }
    lines.join("\n")
}

/// Python's `str.expandtabs()`: a tab advances to the next multiple of **eight**, its default.
///
/// Eight, not the four `_strip_code_fences` asks for — `cleandoc` calls `expandtabs()` bare.
fn expand_tabs(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut column = 0;
    for character in line.chars() {
        match character {
            '\t' => {
                let advance = 8 - (column % 8);
                out.extend(std::iter::repeat_n(' ', advance));
                column += advance;
            }
            other => {
                out.push(other);
                column += 1;
            }
        }
    }
    out
}

fn doc_line(attr: &Attribute) -> Option<String> {
    let syn::Meta::NameValue(pair) = &attr.meta else {
        return None;
    };
    let Expr::Lit(ExprLit {
        lit: Lit::Str(lit), ..
    }) = &pair.value
    else {
        return None;
    };
    // Raw: `cleandoc` removes the common margin, which for the conventional `/// text` is the one
    // space this used to strip per line. Pre-stripping would hide a deeper uniform indent from it.
    Some(lit.value())
}

enum Direction {
    Input,
    Output,
}

fn parse_field(field: &syn::Field) -> Result<(Field, Direction)> {
    let ident = field.ident.clone().expect("named fields only");
    let kind = field_kind(field);
    let markers: Vec<&Attribute> = field
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("input") || a.path().is_ident("output"))
        .collect();
    let [marker] = markers.as_slice() else {
        return Err(Error::new(
            ident.span(),
            format!("field `{ident}` needs exactly one of #[input(...)] or #[output(...)]"),
        ));
    };
    let direction = if marker.path().is_ident("input") {
        Direction::Input
    } else {
        Direction::Output
    };
    let (desc, values) = marker_body(marker)?;
    if values.is_some() && kind != Kind::Str {
        return Err(Error::new_spanned(
            marker,
            "values(...) is only allowed on String fields",
        ));
    }
    // A field that says nothing about itself describes itself with nothing. dspy stores the
    // sentinel `${name}` for an undescribed field and then drops it again when rendering
    // (`adapters/utils.py`, `get_field_description_string`), so the name never reaches a prompt;
    // substituting it here put it on every field line that had no `desc`.
    let desc = desc
        .unwrap_or_else(|| doc_text(&field.attrs))
        .trim()
        .to_owned();
    Ok((
        Field {
            ident,
            ty: field.ty.clone(),
            kind,
            desc,
            values,
        },
        direction,
    ))
}

/// The body of one `#[input(...)]` / `#[output(...)]`: an optional `desc = "..."` and an
/// optional `values("a", "b")` closed set. dspy renders a `Literal` annotation on either
/// direction, so a closed set is legal on either here too.
fn marker_body(attr: &Attribute) -> Result<(Option<String>, Option<Vec<String>>)> {
    let mut desc = None;
    let mut values = None;
    if matches!(attr.meta, syn::Meta::Path(_)) {
        return Ok((desc, values));
    }
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("desc") {
            let lit: LitStr = meta.value()?.parse()?;
            desc = Some(lit.value());
            Ok(())
        } else if meta.path.is_ident("values") {
            let content;
            syn::parenthesized!(content in meta.input);
            let list = content.parse_terminated(<LitStr as syn::parse::Parse>::parse, Token![,])?;
            if list.is_empty() {
                return Err(meta.error("values(...) needs at least one value"));
            }
            values = Some(list.iter().map(LitStr::value).collect());
            Ok(())
        } else {
            Err(meta.error("unknown key; expected desc = \"...\" or values(...)"))
        }
    })?;
    Ok((desc, values))
}

const INT_TYPES: [&str; 10] = [
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "isize", "usize",
];

/// The field contract: `String`, `bool`, fixed-width integers, and floats — spelled bare or
/// by path — travel as scalar wire fields; any other type is a `Json` field carried as
/// serialized JSON. The trait bounds that requires live in the generated code, so a missing
/// impl surfaces as a rustc error at the derive site.
fn field_kind(field: &syn::Field) -> Kind {
    if let syn::Type::Path(path) = &field.ty
        && let Some(last) = path.path.segments.last()
        && last.arguments.is_none()
    {
        let name = last.ident.to_string();
        match name.as_str() {
            "String" => return Kind::Str,
            // dspy's `Reasoning` is declared as the type and rendered as a string.
            "Reasoning" => return Kind::Reasoning,
            "bool" => return Kind::Bool,
            "f32" | "f64" => return Kind::Float,
            name if INT_TYPES.contains(&name) => return Kind::Int,
            _ => {}
        }
    }
    Kind::Json
}

/// Whether a field holds a *record* — one of the caller's own structs — as opposed to a
/// collection, a map, or a scalar.
///
/// dspy asks the value: `isinstance(value, BaseModel)`. Nothing at run time can answer that here,
/// because a struct serialized by serde and a `HashMap` with the same keys are the same
/// `serde_json::Value`. The declared type is where the answer still exists, so it is read here
/// and travels with the value.
///
/// A bare path that names no scalar is a struct: `Vec<T>`, `HashMap<K, V>` and friends all carry
/// generic arguments, so they are excluded by the same `arguments.is_none()` test that classifies
/// the scalars. `Option<T>` is unwrapped first — dspy sees whatever is inside, and a `None`
/// serializes to `null`, which is not an object and so is laid out inline regardless.
pub fn is_record(ty: &syn::Type) -> bool {
    let syn::Type::Path(path) = unwrap_option(ty) else {
        return false;
    };
    let Some(last) = path.path.segments.last() else {
        return false;
    };
    if last.arguments.is_none() {
        let name = last.ident.to_string();
        return name != "String"
            && name != "bool"
            && !INT_TYPES.contains(&name.as_str())
            && name != "f32"
            && name != "f64";
    }
    false
}

/// `Option<T>` seen through to its `T`, and anything else unchanged.
fn unwrap_option(ty: &syn::Type) -> &syn::Type {
    let syn::Type::Path(path) = ty else { return ty };
    let Some(last) = path.path.segments.last() else {
        return ty;
    };
    if last.ident != "Option" {
        return ty;
    }
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else {
        return ty;
    };
    match args.args.first() {
        Some(syn::GenericArgument::Type(inner)) => inner,
        _ => ty,
    }
}

/// The derive's error paths surface as compile errors, so they are probed here at the
/// parse level; `cargo test -p dsrust-derive` runs them.
#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn maps_each_supported_type_to_its_kind() {
        let model = model(&parse_quote! {
            #[signature(instructions = "Do the task.")]
            struct Task {
                #[input]
                text: String,
                #[input]
                age: u32,
                #[input]
                spelled: std::primitive::bool,
                #[output]
                ok: bool,
                #[output]
                count: i64,
                #[output]
                amount: f64,
                #[output]
                ratio: f32,
            }
        })
        .expect("parses");
        let input_kinds: Vec<Kind> = model.inputs.iter().map(|f| f.kind).collect();
        assert_eq!(input_kinds, [Kind::Str, Kind::Int, Kind::Bool]);
        let output_kinds: Vec<Kind> = model.outputs.iter().map(|f| f.kind).collect();
        assert_eq!(
            output_kinds,
            [Kind::Bool, Kind::Int, Kind::Float, Kind::Float]
        );
    }

    #[test]
    fn maps_every_non_scalar_type_to_json() {
        for complex in [
            quote::quote!(Vec<String>),
            quote::quote!(Recipient),
            quote::quote!(Vec<GiftIdea>),
            quote::quote!(Option<u32>),
            quote::quote!(u128),
            quote::quote!(std::collections::BTreeMap<String, u32>),
        ] {
            let item: DeriveInput = parse_quote! {
                #[signature(instructions = "Do the task.")]
                struct Task {
                    #[input]
                    sketch: #complex,
                    #[output]
                    shaped: #complex,
                }
            };
            let model = model(&item).expect("parses");
            assert_eq!(model.inputs[0].kind, Kind::Json, "{complex}");
            assert_eq!(model.outputs[0].kind, Kind::Json, "{complex}");
        }
    }

    #[test]
    fn rejects_values_on_typed_and_json_fields_in_either_direction() {
        for (bad, name) in [
            (quote::quote!(bool), "bool"),
            (quote::quote!(Vec<String>), "Vec<String>"),
        ] {
            let outputs = parse_quote! {
                #[signature(instructions = "Do the task.")]
                struct Task {
                    #[input]
                    text: String,
                    #[output(values("yes", "no"))]
                    ok: #bad,
                }
            };
            let inputs = parse_quote! {
                #[signature(instructions = "Do the task.")]
                struct Task {
                    #[input(values("yes", "no"))]
                    ok: #bad,
                    #[output]
                    text: String,
                }
            };
            for (declaration, direction) in [(outputs, "output"), (inputs, "input")] {
                let Err(error) = model(&declaration) else {
                    panic!("values on a {name} {direction} should be rejected");
                };
                assert_eq!(
                    error.to_string(),
                    "values(...) is only allowed on String fields"
                );
            }
        }
    }

    #[test]
    fn accepts_a_closed_set_on_a_string_input() {
        let model = model(&parse_quote! {
            #[signature(instructions = "Do the task.")]
            struct Task {
                #[input(values("terse", "florid"))]
                style: String,
                #[output]
                text: String,
            }
        })
        .expect("a closed set on a String input is legal");
        assert_eq!(
            model.inputs[0].values,
            Some(vec!["terse".to_owned(), "florid".to_owned()])
        );
    }
}

#[cfg(test)]
mod cleandoc_tests {
    use super::cleandoc;

    /// dspy runs a signature's docstring through `inspect.cleandoc`, and a doc comment here plays
    /// the same part — so it is normalised the same way.
    ///
    /// The expected values were recorded by running Python (see
    /// `crates/dsrust/tests/conformance/predict/cleandoc.json`), not reasoned out. Three of these
    /// nine disagree with a per-line one-space strip, which is what this did before: a uniformly
    /// indented comment, one whose first line is flush against a deeper rest, and one with a tab.
    #[test]
    fn it_cleans_the_docstring_python_cleans() {
        for (raw, expected) in [
            (
                " Answer the question.\n Be brief.",
                "Answer the question.\nBe brief.",
            ),
            (
                "     Answer the question.\n     Be brief.",
                "Answer the question.\nBe brief.",
            ),
            ("Answer.\n     Be brief.", "Answer.\nBe brief."),
            (" Answer.\n   Deeper.\n Back.", "Answer.\n  Deeper.\nBack."),
            (" Answer.\tBriefly.", "Answer.        Briefly."),
            ("\n Answer.", "Answer."),
            (" Answer.\n", "Answer."),
            (" Answer.\n\n Be brief.", "Answer.\n\nBe brief."),
            (" Answer.", "Answer."),
        ] {
            assert_eq!(cleandoc(raw), expected, "cleandoc({raw:?})");
        }
    }
}
