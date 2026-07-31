//! dspy `instruction_proposer`: the seam a caller replaces GEPA's reflection prompt through.
//!
//! Upstream takes a `ProposalFn` and, when one is given, hands it the whole proposal step —
//! `custom_instruction_proposer(candidate=…, reflective_dataset=…, components_to_update=…)`
//! returning the new text per component. The built-in reflection tree is skipped entirely.
//!
//! Here that is a trait, because a Rust caller writing one wants their own state (a template, a
//! budget) and a bare `fn` cannot carry it. The built-in path is unchanged and is what runs when
//! nobody supplies one.
//!
//! The reflection model is handed over rather than left ambient. Upstream runs the custom proposer
//! inside `with dspy.context(lm=reflection_lm)`, so a proposer that builds a `dspy.Predict` reaches
//! the model GEPA was given without being passed it; there is no such ambient here, and a proposer
//! that could not reach it would be strictly less able than upstream's.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::lm::DynChatModel;

use gepa::Reflective;

/// One component's reflective dataset: the examples GEPA gathered for it, each a list of
/// field-name/value pairs in declaration order — dspy's `reflective_dataset[component]`, whose
/// entries are `{"Inputs": …, "Generated Outputs": …, "Feedback": …}`.
///
/// Pairs rather than a map, because the order is the order the reflection prompt renders them in.
pub type ReflectiveDataset = Vec<Vec<(String, Reflective)>>;

/// dspy's `ProposalFn`: rewrite the named components of a candidate, given what their runs produced.
///
/// Return one entry per component you rewrote. A component left out keeps the text it had, which is
/// what upstream's proposer does when it has nothing to say about one.
///
/// ```no_run
/// # use std::collections::BTreeMap;
/// # use std::pin::Pin;
/// # use std::sync::Arc;
/// # use dsrust::lm::DynChatModel;
/// use dsrust::optimize::{InstructionProposer, ReflectiveDataset};
///
/// struct Counting;
///
/// impl InstructionProposer for Counting {
///     fn propose<'a>(
///         &'a self,
///         _reflection: &'a Arc<dyn DynChatModel>,
///         candidate: &'a BTreeMap<String, String>,
///         components: &'a [String],
///         datasets: &'a BTreeMap<String, ReflectiveDataset>,
///     ) -> Pin<Box<dyn Future<Output = BTreeMap<String, String>> + Send + 'a>> {
///         Box::pin(async move {
///             components
///                 .iter()
///                 .map(|name| {
///                     let seen = datasets[name].len();
///                     (name.clone(), format!("{} ({seen} examples seen)", candidate[name]))
///                 })
///                 .collect()
///         })
///     }
/// }
/// ```
///
/// Object-safe by hand — `Pin<Box<dyn Future>>` rather than `impl Future` — for the reason
/// [`DynChatModel`](crate::lm::DynChatModel) is: GEPA stores one behind a pointer, and a trait
/// returning `impl Future` cannot be made into a trait object.
pub trait InstructionProposer: Send + Sync {
    /// The new text for each component named in `components`.
    ///
    /// `reflection` is the model [`GEPA::new`](crate::GEPA::new) was given, so a proposer that asks
    /// a model asks the one the caller configured for reflecting. `datasets` holds one entry per
    /// component, and a component whose runs produced nothing is absent from it — upstream skips
    /// such a component rather than proposing blind.
    fn propose<'a>(
        &'a self,
        reflection: &'a Arc<dyn DynChatModel>,
        candidate: &'a BTreeMap<String, String>,
        components: &'a [String],
        datasets: &'a BTreeMap<String, ReflectiveDataset>,
    ) -> Pin<Box<dyn Future<Output = BTreeMap<String, String>> + Send + 'a>>;
}
