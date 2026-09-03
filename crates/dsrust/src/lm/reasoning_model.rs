//! Whether a model name is one of OpenAI's reasoning families — a question dspy answers twice.
//!
//! `clients/openai_format.py` and `clients/lm.py` each define `_is_openai_reasoning_model`, and
//! they are different functions. The wire one strips an `openai/` prefix, refuses any name holding
//! `chat`, and asks whether the rest begins `o1`, `o3`, `o4` or `gpt-5`. The state one takes the
//! last `/`-separated segment and matches a regex: `o1`, `o3`, `o4` or `o5` with an optional
//! `-mini`/`-nano`/`-pro` and an optional release date, or `gpt-5` with any suffix but `-chat`.
//!
//! They disagree on five names in the fixture beside them, in both directions. `o1-preview` and
//! `gpt-5.1` send `max_completion_tokens` and save as ordinary models; `o5`, `azure/o3` and a
//! doubly-prefixed `openrouter/openai/gpt-5` do the reverse. So the two live here as two
//! functions, named for the surface each one decides, because one predicate is wrong somewhere
//! whichever rule it implements.

/// dspy `clients/openai_format.py::_is_openai_reasoning_model`, which decides the chat body's
/// token key and whether a reasoning temperature is refused.
///
/// The prefix is stripped before the name is lowercased, as upstream does it, so `OpenAI/o3` keeps
/// a prefix `startswith` then fails to match — a difference no reader would guess and both a
/// `to_lowercase` first and a case-insensitive strip would erase.
pub fn on_the_wire(model: &str) -> bool {
    let name = model
        .strip_prefix("openai/")
        .unwrap_or(model)
        .to_ascii_lowercase();
    if name.contains("chat") {
        return false;
    }
    ["o1", "o3", "o4", "gpt-5"]
        .iter()
        .any(|family| name.starts_with(family))
}

/// dspy `clients/lm.py::_is_openai_reasoning_model`, which decides what `LM.dump_state` writes.
///
/// Anchored at both ends, so a suffix the pattern has no arm for — `-preview`, or the `.1` of a
/// minor version — is not a reasoning model however the family reads.
pub fn in_saved_state(model: &str) -> bool {
    let name = model
        .rsplit_once('/')
        .map_or(model, |(_, last)| last)
        .to_ascii_lowercase();
    o_series(&name) || gpt_5_series(&name)
}

/// `o[1345]`, then at most one size, then at most one release date, and nothing else.
fn o_series(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('o') else {
        return false;
    };
    let Some(rest) = rest.strip_prefix(|version| matches!(version, '1' | '3' | '4' | '5')) else {
        return false;
    };
    let rest = ["-mini", "-nano", "-pro"]
        .iter()
        .find_map(|size| rest.strip_prefix(size))
        .unwrap_or(rest);
    rest.is_empty() || is_release_date(rest)
}

/// `gpt-5` with any suffix that does not begin `-chat`. The carve-out is a lookahead upstream, so
/// it fires on the suffix alone: `gpt-5.1-chat` has no `-` where the pattern needs one and is not
/// matched at all, rather than being excluded by the `chat` in its name.
fn gpt_5_series(name: &str) -> bool {
    name == "gpt-5" || (name.starts_with("gpt-5-") && !name.starts_with("gpt-5-chat"))
}

/// `-2025-01-31`, upstream's `-\d{4}-\d{2}-\d{2}`.
fn is_release_date(rest: &str) -> bool {
    let digits = rest.as_bytes();
    digits.len() == 11
        && [0, 5, 8].iter().all(|at| digits[*at] == b'-')
        && [1..5, 6..8, 9..11]
            .iter()
            .all(|span| digits[span.clone()].iter().all(u8::is_ascii_digit))
}
