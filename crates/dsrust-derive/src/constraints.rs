//! What a field's declared bounds read as in the prompt — dspy's `PYDANTIC_CONSTRAINT_MAP`.
//!
//! dspy passes a field's pydantic constraints through `InputField`/`OutputField` and renders them
//! as prose under the field's line:
//!
//! ```text
//! 1. `suspicion_score` (int): How likely the code is to be written by an attacker
//! Constraints: greater than or equal to: 0, less than or equal to: 9
//! ```
//!
//! The crate could already render that string — it crossed the bridge from Python as data. What it
//! could not do was *declare* one, and a signature that cannot state a bound writes a prompt
//! missing a line upstream writes. `gepa_trusted_monitor` is an official tutorial and this is its
//! only output field.

use syn::{Error, Expr, Lit, Result, UnOp};

/// The prose dspy gives each constraint, and the only keys it renders.
const PROSE: [(&str, &str); 8] = [
    ("gt", "greater than: "),
    ("ge", "greater than or equal to: "),
    ("lt", "less than: "),
    ("le", "less than or equal to: "),
    ("min_length", "minimum length: "),
    ("max_length", "maximum length: "),
    ("multiple_of", "a multiple of the given number: "),
    ("allow_inf_nan", "allow 'inf', '-inf', 'nan' values: "),
];

/// Whether this attribute key names a constraint.
pub fn is_constraint(key: &str) -> bool {
    PROSE.iter().any(|(name, _)| *name == key)
}

/// One constraint as its clause, or an error naming what the key can hold.
///
/// **Declaration order is the rendered order.** Upstream walks the keyword arguments as written
/// rather than a fixed sequence, so `le` before `ge` reads that way in the prompt — which makes
/// the attribute's order prompt text too.
pub fn clause(key: &str, value: &Expr) -> Result<String> {
    let prose = PROSE
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, prose)| *prose)
        .ok_or_else(|| Error::new_spanned(value, format!("{key} is not a constraint")))?;
    Ok(format!("{prose}{}", literal(value)?))
}

/// A constraint's value, spelled as Python spells it — which is the spelling that reaches a model.
fn literal(value: &Expr) -> Result<String> {
    if let Expr::Unary(unary) = value
        && matches!(unary.op, UnOp::Neg(_))
    {
        return Ok(format!("-{}", literal(&unary.expr)?));
    }
    let Expr::Lit(lit) = value else {
        return Err(Error::new_spanned(
            value,
            "a constraint takes a number or a bool literal",
        ));
    };
    match &lit.lit {
        Lit::Int(int) => Ok(int.base10_digits().to_owned()),
        Lit::Float(float) => Ok(float.base10_digits().to_owned()),
        // Python's `f"{True}"`, which is what upstream interpolates.
        Lit::Bool(boolean) => Ok(match boolean.value {
            true => "True".to_owned(),
            false => "False".to_owned(),
        }),
        _ => Err(Error::new_spanned(
            value,
            "a constraint takes a number or a bool literal",
        )),
    }
}

/// The clauses as one `Constraints:` line's worth of prose, or nothing when none were declared.
pub fn joined(clauses: &[String]) -> Option<String> {
    match clauses.is_empty() {
        true => None,
        false => Some(clauses.join(", ")),
    }
}
