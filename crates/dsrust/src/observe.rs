//! dspy's six callback points, watched two ways at once: a `tracing` span and the [`Callback`] list.
//!
//! Upstream fires a `BaseCallback` at six places, each with a start and an end — module, lm, adapter
//! format, adapter parse, tool, evaluate — and `callbacks` gives each call a uuid and links it
//! to its parent through a context variable. [`crate::callback`] is that, transcribed. A span is the
//! same thing in the shape a Rust caller already has a subscriber for: the identifier, the parent
//! linkage, the start and the end are one object, and it cannot mutate what it sees or break the
//! run — upstream's own two documented worries about callbacks.
//!
//! Both are fired here rather than at the call sites, and each point is one function that opens and
//! closes together. That is what makes them un-forgettable: an end a `?` can skip is an end that
//! will be skipped, and a point split into two calls is a point someone adds half of.
//!
//! Nothing is serialized unless something is listening. `tracing`'s macros check subscriber interest
//! before evaluating a field, [`Watch`] records nothing on a disabled span, the callback list is
//! checked for emptiness before a handler's arguments are built, and an unwatched call carries no
//! parent linkage — upstream's own `if not callbacks: return fn(...)`.
//!
//! **All six points exist**: module, lm, tool, adapter format, adapter parse and evaluate.
//! `tests/observe.rs` and `tests/callback.rs` are what say so, and they can only see what a run
//! actually produced — the ledger entry this replaced claimed these points existed while the tree
//! had none.
//!
//! Six is upstream's count read off its `@callbacks` sites rather than off its handler names:
//! `BaseLM.__call__`/`acall`, `Module.__call__`/`acall`, `Tool.__call__`/`acall`,
//! `Evaluate.__call__`, and `format`/`parse` decorated onto every `Adapter` subclass by
//! `__init_subclass__`. Each pair is one point here, since this crate is async throughout. There is
//! a seventh site — `Retrieve.__call__`, which falls through to the module handlers — and it has no
//! counterpart because retrievers are out of scope (`scripts/check_coverage.py` says so by name).

use std::fmt::Write as _;
use std::future::Future;

use anyhow::Result;
use tracing::{Instrument, Span, field};

use serde_json::Value;

use crate::adapter::Input;
use crate::callback::{self, CallId, Callback, Ends, Rendered, Under};
use crate::evaluate::Pass;
use crate::example::Example;

/// The target every span here carries, so `RUST_LOG=dsrust::observe=info` is the whole of what a
/// caller needs to watch a run — and so a subscriber can select these without matching on names.
///
/// Naming it rather than writing the string is what keeps a filter working when the module moves:
///
/// ```
/// use tracing_subscriber::filter::EnvFilter;
///
/// // Only this crate's run events, at info, and nothing else in the process.
/// let only_dsrust = EnvFilter::new(format!("{}=info", dsrust::observe::TARGET));
/// assert_eq!(only_dsrust.to_string(), "dsrust::observe=info");
/// ```
pub const TARGET: &str = "dsrust::observe";

/// One point, open: the span it entered and the call it identifies.
///
/// Returned by every `*_start`, and consumed by the matching end. The end handler cannot be called
/// without it, which is what stops a point from being started and never finished.
#[must_use = "a point that is opened must be closed, or its end handler never fires"]
pub struct Watch {
    span: Span,
    call: CallId,
    /// The callbacks the instance at this point carries, beyond the process-wide ones. Filled at
    /// the model call and at a module that overrides [`Module::callbacks`](crate::Module::callbacks)
    /// — upstream's two object-level registration paths, `dspy.LM("gpt-4o-mini", callbacks=[…])`
    /// and `dspy.Predict("q -> a", callbacks=[…])`.
    instance: Vec<std::sync::Arc<dyn Callback>>,
}

impl Watch {
    /// Record what dspy's `on_*_start` was shown, on the span.
    ///
    /// Separate from creating the span because a value worth rendering is a value worth not
    /// rendering when nothing is listening, and `Span::record` evaluates its argument either way.
    fn shown(&self, inputs: impl FnOnce() -> String) {
        if self.span.is_disabled() {
            return;
        }
        self.span.record("inputs", inputs().as_str());
    }

    /// Record the outcome on the span: dspy's `on_*_end(outputs=…, exception=…)`, where exactly one
    /// of the two is present.
    ///
    /// A failure renders `{:#}` and not `{}`, because a parse failure keeps its cause in the chain
    /// and the chain is the half naming the field.
    fn finished<T>(
        &self,
        answered: Result<&T, &anyhow::Error>,
        describe: impl FnOnce(&T) -> String,
    ) {
        if self.span.is_disabled() {
            return;
        }
        match answered {
            Ok(outputs) => self.span.record("outputs", describe(outputs).as_str()),
            Err(error) => self.span.record("error", format!("{error:#}").as_str()),
        };
    }
}

/// Open a point: a span with the standard fields, and the call id it is known by.
fn opening(span: Span) -> Watch {
    opening_with(Vec::new(), span)
}

/// The same, for a point whose instance carries callbacks of its own — the model call, and any
/// module that overrides [`Module::callbacks`](crate::Module::callbacks).
fn opening_with(instance: Vec<std::sync::Arc<dyn Callback>>, span: Span) -> Watch {
    Watch {
        span,
        call: CallId::next(),
        instance,
    }
}

/// dspy `on_module_start`: one module's run, with everything it did inside it.
///
/// `kind` is the module's type — `Predict`, `ReAct` — which is what upstream's `instance` is read
/// for. A composed program nests, so the tree is the program's shape.
///
/// Inputs are taken by reference and rendered here rather than at the span's creation, so a
/// `forward` records its inputs and then moves them on: this call followed by a body that consumes
/// `inputs` is two statements and borrows nothing across them.
pub fn module_shown(
    kind: &'static str,
    inputs: &Example,
    instance: &[std::sync::Arc<dyn Callback>],
) -> Watch {
    let watch = opening_with(
        instance.to_vec(),
        tracing::info_span!(
            target: TARGET,
            "module",
            module = kind,
            inputs = field::Empty,
            outputs = field::Empty,
            error = field::Empty,
        ),
    );
    watch.shown(|| as_json(inputs));
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_module_start(&watch.call, kind, inputs)
        });
    }
    watch
}

/// dspy `on_lm_start`: one call to a model, inside whichever module made it.
pub fn lm_shown(
    request: &crate::lm::api::LmRequest,
    instance: &[std::sync::Arc<dyn Callback>],
) -> Watch {
    let mut watch = opening(tracing::info_span!(
        target: TARGET,
        "lm",
        model = request.model.as_str(),
        inputs = field::Empty,
        outputs = field::Empty,
        error = field::Empty,
    ));
    watch.instance = instance.to_vec();
    watch.shown(|| request.watchable());
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_lm_start(&watch.call, request)
        });
    }
    watch
}

/// Run `work` under an open point, telling its `on_*_end` handler what came back.
///
/// One function rather than a start call and an end call for the same reason the points are single
/// functions: every exit records something. [`Ends`] is what says which handler this is — the two
/// asynchronous points answer with different values, and pairing the value with its handler in one
/// impl is what stops a call site from combining a module's span with the model's end handler.
pub async fn watching<T: Ends, Work>(watch: Watch, work: Work) -> Result<T>
where
    Work: Future<Output = Result<T>>,
{
    // Nothing registered means no parent linkage anyone can read, so the work runs unwrapped —
    // upstream's `if not callbacks: return fn(...)`, and what keeps an unwatched call from paying
    // for a boxed future at every point.
    let answered = match callback::watching(&watch.instance) {
        true => {
            Under::new(watch.call, work)
                .instrument(watch.span.clone())
                .await
        }
        false => work.instrument(watch.span.clone()).await,
    };
    watch.finished(answered.as_ref(), T::describe);
    if callback::watching(&watch.instance) {
        T::ended(&watch.call, &watch.instance, answered.as_ref());
    }
    answered
}

/// dspy `on_tool_start`/`on_tool_end`: one tool call an agent made, with its arguments and either
/// what the tool returned or why it refused.
///
/// Synchronous, unlike the other points, because [`Tool::call_value`](crate::Tool::call_value) is —
/// a tool is a Rust closure, not a network call. So this runs the call rather than wrapping a
/// future, and the point opens and closes around it.
///
/// Every agent goes through here rather than through the trait, and that is deliberate:
/// `call_value` is defaulted and two tools in the tree override it, so a point in the default body
/// would miss exactly the tools most worth watching — ReActV2's `submit` and RLM's.
pub fn tool_call(tool: &dyn crate::Tool, args: &Value) -> Result<Value> {
    let watch = opening(tracing::info_span!(
        target: TARGET,
        "tool",
        tool = tool.name(),
        inputs = field::Empty,
        outputs = field::Empty,
        error = field::Empty,
    ));
    let _entered = watch.span.enter();
    watch.shown(|| args.to_string());
    let named = tool.name();
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_tool_start(&watch.call, named, args)
        });
    }

    let _under = callback::entered(&watch.call);
    let answered = tool.call_value(args);
    watch.finished(answered.as_ref(), Value::to_string);
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_tool_end(&watch.call, answered.as_ref())
        });
    }
    answered
}

/// dspy `on_adapter_format_start`/`on_adapter_format_end`: rendering the prompt.
///
/// A free function the callers go through, as [`tool_call`] is, and for the same reason `Module`
/// needed an enumerating test: `Adapter::format` is a required trait method, so an implementor can
/// always write one without the point. Upstream has no such problem — `__init_subclass__` decorates
/// every subclass on its way into existence — so the Rust answer is to watch the caller instead.
pub fn formatting(
    adapter: &dyn crate::Adapter,
    signature: &crate::Signature,
    demos: &[Example],
    inputs: &[Input<'_>],
    rendering: impl FnOnce() -> Result<Vec<crate::lm::api::LmMessage>>,
) -> Result<Vec<crate::lm::api::LmMessage>> {
    let watch = adapter_point("adapter.format", adapter.name());
    let _entered = watch.span.enter();
    let named = adapter.name();
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_adapter_format_start(&watch.call, named, signature, demos, inputs)
        });
    }

    let _under = callback::entered(&watch.call);
    let answered = rendering();
    watch.finished(answered.as_ref(), |messages| {
        format!("{{\"messages\":{}}}", messages.len())
    });
    if callback::watching(&watch.instance) {
        let rendered = answered.as_ref().map(|messages| Rendered {
            messages: messages.as_slice(),
        });
        callback::tell(&watch.instance, |callback| {
            callback.on_adapter_format_end(&watch.call, rendered.as_ref().map_err(|error| *error))
        });
    }
    answered
}

/// dspy `on_adapter_parse_start`/`on_adapter_parse_end`: reading the reply back into fields.
///
/// The raw reply is the input, which is the value a reader opened a trace for: a parse failure is
/// almost always a question about what the model actually said.
pub fn parsing(
    adapter: &dyn crate::Adapter,
    raw: &str,
    reading: impl FnOnce() -> Result<Value>,
) -> Result<Value> {
    let watch = adapter_point("adapter.parse", adapter.name());
    let _entered = watch.span.enter();
    watch.shown(|| raw.to_owned());
    let named = adapter.name();
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_adapter_parse_start(&watch.call, named, raw)
        });
    }

    let _under = callback::entered(&watch.call);
    let answered = reading();
    watch.finished(answered.as_ref(), Value::to_string);
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_adapter_parse_end(&watch.call, answered.as_ref())
        });
    }
    answered
}

fn adapter_point(point: &'static str, adapter: &'static str) -> Watch {
    opening(tracing::info_span!(
        target: TARGET,
        "adapter",
        point = point,
        adapter = adapter,
        inputs = field::Empty,
        outputs = field::Empty,
        error = field::Empty,
    ))
}

/// dspy `on_evaluate_start`: one whole run over a devset, with every module call it made inside it.
///
/// Upstream decorates `Evaluate.__call__`, and this wraps [`Evaluate::run`](crate::Evaluate::run) —
/// the same method under a different name. The outermost point of an optimizer's search, so a reader
/// filtering to `evaluate` sees one line per scoring pass rather than one per row.
///
/// `pass` is dspy's `callback_metadata`, and it is what separates the passes from each other: a
/// search alternates whole-valset scoring with subsamples, and the two mean different things.
///
/// The devset is handed over whole rather than counted, because upstream's is: `with_callbacks`
/// gives a handler the `inputs` dict of `Evaluate.__call__`, `devset` among its keys. A count is
/// what the span records and what a handler can take for itself.
pub fn evaluating(devset: &[Example], threads: usize, pass: Option<Pass>) -> Watch {
    let watch = opening(tracing::info_span!(
        target: TARGET,
        "evaluate",
        rows = devset.len(),
        threads = threads,
        pass = pass.map(|pass| match pass {
            Pass::Full => "full",
            Pass::Minibatch => "minibatch",
        }),
        inputs = field::Empty,
        outputs = field::Empty,
        error = field::Empty,
    ));
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_evaluate_start(&watch.call, devset, threads, pass)
        });
    }
    watch
}

/// Run an evaluation's rows inside the open point, so every module call they make is a child of it.
///
/// Both halves, as [`watching`] does: instrumented on the future rather than entered across an
/// await, which would attribute whatever the runtime polled next to this evaluation, and run under
/// the call id, which is what makes each row's `on_module_start` name this evaluation as its parent.
/// Wrapping only the span left the callbacks reporting every row as an outermost call.
pub async fn evaluated_within<T>(watch: &Watch, rows: impl Future<Output = T>) -> T {
    Under::new(watch.call, rows)
        .instrument(watch.span.clone())
        .await
}

/// dspy `on_evaluate_end`: what an evaluation found.
///
/// Its own function rather than [`watching`] because a run has no error arm: a failing row scores
/// `failure_score` and the run carries on, which is dspy's choice too.
pub fn scored(watch: &Watch, evaluation: &crate::evaluate::Evaluation) {
    if !watch.span.is_disabled() {
        watch.span.record(
            "outputs",
            format!(
                "{{\"score\":{},\"rows\":{},\"failed\":{}}}",
                evaluation.score,
                evaluation.results.len(),
                evaluation.failure_count(),
            )
            .as_str(),
        );
    }
    if callback::watching(&watch.instance) {
        callback::tell(&watch.instance, |callback| {
            callback.on_evaluate_end(&watch.call, evaluation)
        });
    }
}

/// An [`Example`]'s fields as a JSON object, which is the shape dspy's `inputs` dict has.
///
/// A field that will not render is dropped rather than raised: this is only watching the call.
pub fn as_json(example: &Example) -> String {
    let mut rendered = String::from("{");
    for (index, (name, value)) in example.fields().enumerate() {
        if index > 0 {
            rendered.push(',');
        }
        let _ = write!(rendered, "{}:{value}", serde_json::json!(name));
    }
    rendered.push('}');
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_examples_fields_render_as_the_dict_dspy_passes() {
        let example = Example::new([
            ("question", serde_json::json!("capital of France?")),
            ("hops", serde_json::json!(2)),
        ]);
        assert_eq!(
            as_json(&example),
            r#"{"question":"capital of France?","hops":2}"#
        );
        assert_eq!(as_json(&Example::default()), "{}");
    }

    /// A field name with a quote in it still renders as JSON rather than breaking the object.
    #[test]
    fn a_field_name_is_escaped() {
        let example = Example::new([(r#"od"d"#, serde_json::json!(1))]);
        assert_eq!(as_json(&example), r#"{"od\"d":1}"#);
    }

    /// Nothing is recorded on a disabled span, which is every span in a program with no subscriber.
    #[test]
    fn a_disabled_span_records_nothing() {
        let watch = Watch {
            span: Span::none(),
            call: CallId::next(),
            instance: Vec::new(),
        };
        assert!(watch.span.is_disabled());
        watch.shown(|| unreachable!("a disabled span asks for no inputs"));
        watch.finished::<()>(Ok(&()), |_| {
            unreachable!("a disabled span asks for no description")
        });
    }
}

/// How many spans of each name a piece of work opened.
///
/// A span nobody collects is indistinguishable from no span, which is how the ledger came to say
/// two optimizers emitted none while a single MIPROv2 run opened 1772. Reading the source cannot
/// settle that — the points are here, and what reaches them depends on what the run calls — so a
/// claim about what an optimizer emits is held by running it.
///
/// The subscriber is **global and installed once**, not scoped to the calling thread, because
/// `tracing` caches each callsite's interest process-wide: a callsite first reached while no
/// subscriber exists caches as never-interested, and a thread-local subscriber installed afterwards
/// sees nothing through it. Both tests using this were green alone and red in the suite, and the
/// suite's order is what decided which. Rebuilding the cache is not enough either — a test on
/// another thread re-poisons it mid-run. One subscriber that always exists ends the race, and the
/// counts land on whichever thread is recording.
#[cfg(test)]
pub(crate) fn spans_opened_by<T>(
    work: impl Future<Output = T>,
) -> std::collections::BTreeMap<String, usize> {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::sync::OnceLock;
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

    thread_local! {
        static RECORDING: RefCell<Option<BTreeMap<String, usize>>> = const { RefCell::new(None) };
    }

    struct ToWhicheverThreadIsRecording;

    impl<S: tracing::Subscriber> Layer<S> for ToWhicheverThreadIsRecording {
        fn on_new_span(
            &self,
            attributes: &tracing::span::Attributes<'_>,
            _id: &tracing::Id,
            _context: Context<'_, S>,
        ) {
            RECORDING.with_borrow_mut(|recording| {
                if let Some(counts) = recording.as_mut() {
                    *counts
                        .entry(attributes.metadata().name().to_owned())
                        .or_default() += 1;
                }
            });
        }
    }

    static INSTALLED: OnceLock<Result<(), String>> = OnceLock::new();
    INSTALLED
        .get_or_init(|| {
            tracing::subscriber::set_global_default(
                tracing_subscriber::registry().with(ToWhicheverThreadIsRecording),
            )
            .map_err(|error| error.to_string())
        })
        .as_ref()
        .expect("nothing else in this crate installs a global subscriber");

    RECORDING.with_borrow_mut(|recording| *recording = Some(BTreeMap::new()));
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("a runtime")
        .block_on(work);
    RECORDING
        .with_borrow_mut(|recording| recording.take())
        .unwrap_or_default()
}
