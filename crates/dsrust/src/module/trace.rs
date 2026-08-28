//! What a traced run records: one entry per predictor call, and what that call answered.
//!
//! dspy keeps this as `settings.trace`, a thread-local list of `(predictor, inputs, outputs)`
//! triples an optimizer reads afterwards. Here the trace is passed rather than ambient, and a
//! predictor is named rather than identified by object, so the walk that records and the walk that
//! reads agree by construction.

use crate::example::Example;

/// One predictor call: which predictor ran, what it was asked, and what it answered.
///
/// dspy's `settings.trace` step, a `(predictor, inputs, outputs)` triple appended to a
/// thread-local that an optimizer reads afterwards. Here the trace is passed rather than
/// ambient, and a predictor is identified by the name [`Module::named_predictors`](super::Module::named_predictors) gives it
/// rather than by object identity, so the two walks agree by construction.
#[derive(Clone)]
pub struct TraceStep {
    pub predictor: String,
    pub inputs: Example,
    pub outputs: StepOutputs,
    /// The predictor's signature as it stood for this call, which is dspy's `trace[i][0].signature`
    /// reached through the predictor object the tuple's first slot holds.
    ///
    /// GEPA needs it for two things a name cannot answer: which trace entries belong to the
    /// component being reflected on — upstream matches `signature.equals`, so two predictors
    /// sharing a signature *and* an instruction pool together — and which input is the `History`,
    /// which is a question about the field's annotation rather than its value.
    pub signature: crate::signature::Signature,
}

/// What a predictor answered, or the completion nobody could read.
///
/// An ordinary step holds the parsed fields. The other arm is dspy's `FailedPrediction`, which
/// `bootstrap_trace_data` appends when an `AdapterParseError` ends a forward: the run stops there,
/// but the steps before it and the text that would not parse are worth more to a reflection than
/// an empty trace is.
///
/// An enum rather than an empty [`Example`] beside a flag, because a reader that treats an
/// unparsed step as an answered one with no fields is exactly the mistake: GEPA renders it as a
/// raw-response block and *prefers* it when choosing a step to reflect on, while
/// `BootstrapFewShot` must never turn one into a demo. Both questions have to be asked out loud.
///
/// A program recording its own calls builds the first arm. The second is built by the collector
/// when a forward ends in a parse failure, not by the program:
///
/// ```
/// use dsrust::{example, FailedPrediction, StepOutputs};
///
/// let answered = StepOutputs::Answered(example! { answer: "Paris" });
/// assert_eq!(answered.answered().and_then(|e| e.get("answer")), Some(&"Paris".into()));
/// assert!(answered.failure().is_none());
///
/// let unreadable = StepOutputs::Unparsed(FailedPrediction {
///     completion_text: "I think it is Paris".to_owned(),
///     format_reward: None,
/// });
/// assert!(unreadable.answered().is_none());
/// // No reward recorded, so the caller's `format_failure_score` is what it is worth.
/// assert_eq!(unreadable.failure().map(|f| f.score(-1.0)), Some(-1.0));
/// ```
#[derive(Clone)]
pub enum StepOutputs {
    Answered(Example),
    Unparsed(FailedPrediction),
}

impl StepOutputs {
    /// The parsed fields, or `None` for a completion that would not parse.
    pub fn answered(&self) -> Option<&Example> {
        match self {
            Self::Answered(example) => Some(example),
            Self::Unparsed(_) => None,
        }
    }

    pub fn failure(&self) -> Option<&FailedPrediction> {
        match self {
            Self::Unparsed(failed) => Some(failed),
            Self::Answered(_) => None,
        }
    }
}

/// dspy `FailedPrediction`: a completion no adapter could read, and what that is worth.
///
/// `format_reward` is `None` where dspy's is, and both are read through Python's `or` — a reward of
/// exactly zero is falsy and falls back to the caller's `format_failure_score`, which
/// [`format_reward`](FailedPrediction::score) reproduces.
#[derive(Clone, Debug, PartialEq)]
pub struct FailedPrediction {
    /// The raw text, as it arrived. GEPA shows it to the reflection model verbatim.
    pub completion_text: String,
    pub format_reward: Option<f64>,
}

impl FailedPrediction {
    /// dspy's `prediction.format_reward or format_failure_score`.
    ///
    /// Python's `or`, not a null check: `0.0` is falsy, so a reward of zero takes the fallback.
    /// `unwrap_or` would keep the zero, and the two differ the moment a caller passes a non-zero
    /// `format_failure_score` — which GEPA does not, and `bootstrap_trace_data`'s default does.
    pub fn score(&self, format_failure_score: f64) -> f64 {
        match self.format_reward {
            Some(reward) if reward != 0.0 => reward,
            _ => format_failure_score,
        }
    }
}

/// Rename every step `record` added, which is how a composed module claims its children's calls
/// as its own predictor.
///
/// The `from` mark is taken *before* the child runs, so only what that child added is renamed and a
/// step recorded earlier keeps its own name. An optimizer walks the trace by predictor name, so a
/// composed module that skips this has its children attributed to whatever ran before them:
///
/// ```
/// use dsrust::module::{TraceStep, relabel};
///
/// # fn example(mut trace: Vec<TraceStep>) {
/// let mark = trace.len();
/// // ... a child module runs and records its own steps ...
/// relabel(&mut trace, mark, "summarise");
/// assert!(trace[mark..].iter().all(|step| step.predictor == "summarise"));
/// # }
/// ```
pub fn relabel(trace: &mut [TraceStep], from: usize, name: &str) {
    for step in &mut trace[from..] {
        step.predictor = name.to_owned();
    }
}

/// Ask a module, naming each input where its value goes.
///
/// ```
/// # async fn wrapper(haiku: dsrust::Predict) -> anyhow::Result<()> {
/// let result = dsrust::call!(haiku, subject = "computer science", tone = "wry").await?;
/// # Ok(()) }
/// ```
///
/// Rust has neither named arguments nor a mapping literal, so the two are written here instead:
/// the field name sits where the value does, which is what `subject=` does in Python. Evaluates
/// to the call's future, so the caller writes `.await?` and sees where the model is reached.
///
/// Asks through [`Ask`](super::Ask), so what comes back is whatever the module promised. A module of your
/// own joins in with one line — `dsrust::asks_with_a_prediction!(YourModule);` — which is the same
/// line the modules here use.
#[macro_export]
macro_rules! call {
    ($module:expr, $($field:ident = $value:expr),* $(,)?) => {
        $crate::Ask::ask(
            &$module,
            $crate::input! { $($field: $value),* },
        )
    };
}
