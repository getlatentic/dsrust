//! What every module has in common, so a caller can write their own.
//!
//! dspy's `Module` is a base class: `Predict`, `ChainOfThought` and `ReAct` all subclass it,
//! and users write their own the same way. Two things depend on that shape. An evaluator
//! takes any module, not a specific one. And an optimizer walks a program to find the
//! predictors inside it, reads their demos and instructions, and writes back improved ones —
//! `named_predictors` in Python is exactly that walk.
//!
//! The Rust equivalent is a trait. It stays object-safe so a program can hold
//! `Box<dyn Module>`, which is what a composed program needs.

use std::path::Path;

use anyhow::{Result, bail};

use crate::example::{Example, Prediction};
use crate::lm::Sampling;
use crate::signature::Signature;

mod state;
mod trust;

use state::demo_from_fields;
pub use state::{
    DSPY_VERSION, FieldState, FlexState, Metadata, PredictorState, ProgramState, SignatureState,
    SubmoduleState,
};
pub use trust::Trust;

/// One predictor inside a program: its signature and its demos, borrowed for inspection or
/// mutation. An optimizer's whole job is reading these and writing back better ones.
/// One `Flex` within a program, under the name an optimizer keys its component by.
///
/// The counterpart of [`NamedPredictor`] for the other kind of optimizable component. It carries
/// the module itself rather than a field, because a `Flex`'s whole state is its source and
/// `Flex::bind` is what an optimizer writes back through.
pub struct NamedFlex<'a> {
    pub name: String,
    pub flex: &'a mut crate::predict::flex::Flex,
}

pub struct NamedPredictor<'a> {
    pub name: String,
    pub signature: &'a mut Signature,
    pub demos: &'a mut Vec<Example>,
    /// How this predictor asks for its reply to be sampled.
    ///
    /// Unlike the other two this is not something an optimizer *learns* — it is set for the
    /// duration of a round and then set back. It rides the same walk because reaching every
    /// predictor is the same problem, and dspy solves it the same way: `set_lm` on a program
    /// assigns to all of them.
    pub config: &'a mut Sampling,
    /// Advice for this predictor from an earlier attempt, shown to it on the next one.
    ///
    /// `Refine` writes it and `Predict` renders it as one more input field. Per predictor rather
    /// than per program because that is what upstream's advice is: `OfferFeedback` answers with a
    /// map keyed by module name, so the module that went wrong is the one told about it.
    pub hint: &'a mut Option<String>,
    /// The model this predictor was pinned to, which a saved program records and restores.
    ///
    /// dspy's `Predict.lm`, and on the walk for the same reason the other three are: `dump_state`
    /// writes one block per predictor and `load_state` reads them back, and reaching every
    /// predictor is that same problem again. Unset on a predictor that was never pinned, which is
    /// the `null` dspy writes.
    pub lm: &'a mut Option<std::sync::Arc<dyn crate::lm::DynChatModel>>,
}

/// One predictor call: which predictor ran, what it was asked, and what it answered.
///
/// dspy's `settings.trace` step, a `(predictor, inputs, outputs)` triple appended to a
/// thread-local that an optimizer reads afterwards. Here the trace is passed rather than
/// ambient, and a predictor is identified by the name [`Module::named_predictors`] gives it
/// rather than by object identity, so the two walks agree by construction.
#[derive(Clone)]
pub struct TraceStep {
    pub predictor: String,
    pub inputs: Example,
    pub outputs: Example,
    /// The predictor's signature as it stood for this call, which is dspy's `trace[i][0].signature`
    /// reached through the predictor object the tuple's first slot holds.
    ///
    /// GEPA needs it for two things a name cannot answer: which trace entries belong to the
    /// component being reflected on — upstream matches `signature.equals`, so two predictors
    /// sharing a signature *and* an instruction pool together — and which input is the `History`,
    /// which is a question about the field's annotation rather than its value.
    pub signature: crate::signature::Signature,
}

/// What a program of your own has to say: how to run it.
///
/// [`Module`] is what everything else takes, and it is boxed and object-safe so a composed
/// program can hold `Box<dyn Module>`. That shape is right for a caller and wrong for an author,
/// who would be writing `Pin<Box<dyn Future>>` by hand for no reason. Implement this instead and
/// `#[derive(Module)]` writes the rest: the walk an optimizer needs, the trace a compile needs,
/// and the boxing.
///
/// ```ignore
/// #[derive(Module)]
/// struct Outline {
///     plan: Predict,
///     write: Predict,
/// }
///
/// impl Forward for Outline {
///     async fn forward(&self, inputs: Example) -> Result<Prediction> {
///         let angle = self.plan.forward(inputs).await?;
///         self.write
///             .forward(dsrust::input! { angle: angle.get("angle").cloned().unwrap_or_default() })
///             .await
///     }
/// }
/// ```
pub trait Forward: Send + Sync {
    fn forward(&self, inputs: Example) -> impl Future<Output = Result<Prediction>> + Send;
}

/// A callable program. Implement it to add a module of your own.
pub trait Module: Send + Sync {
    /// Run the program over one example's inputs.
    ///
    /// Boxed rather than `async fn` so the trait stays object-safe: a composed program holds
    /// its children as `Box<dyn Module>`, and an evaluator takes any module at all.
    fn forward<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>>;

    /// The callbacks this module carries, beyond the process-wide ones — dspy's third registration
    /// path, `dspy.Predict("q -> a", callbacks=[cb])`.
    ///
    /// Defaulted to none, and that default is the whole reason it is a trait method rather than a
    /// field: upstream stores the list on `Module.__init__` so every subclass inherits one, where a
    /// Rust trait has no shared constructor. A module that wants the third path holds its own field
    /// and overrides this; one that does not pays nothing and writes nothing.
    ///
    /// What a handler could already do without it is filter by *kind* — `on_module_start` is given
    /// the module's type — and by position, through `CallId::parent`. What it could not do is tell
    /// two `Predict`s of the same signature apart, which is exactly what upstream's per-instance
    /// list is for.
    fn callbacks(&self) -> &[std::sync::Arc<dyn crate::callback::Callback>] {
        &[]
    }

    /// Every `Flex` this program contains, named — dspy's `enumerate_flex_submodules`.
    ///
    /// A second walk rather than a wider `named_predictors`, because the two answer different
    /// questions and an optimizer treats their answers differently. A predictor's optimizable
    /// component is its instruction; a [`Flex`](crate::predict::flex::Flex)'s is its *source*, and
    /// GEPA rewrites the two with different proposers. Upstream reaches both through
    /// `named_parameters` and asks `isinstance` which it has; a Rust walk says which by returning
    /// from a different method.
    ///
    /// Defaulted to none, and a composed module must recurse into its children exactly as it does
    /// for `named_predictors` — a `Flex` nested inside a module that does not is invisible to an
    /// optimizer, which is the same cost that walk already carries.
    fn named_flexes(&mut self) -> Vec<NamedFlex<'_>> {
        Vec::new()
    }

    /// Every predictor this program contains, named. dspy's `named_predictors`, and the seam
    /// an optimizer works through: it reads the demos and instructions here and writes back
    /// improved ones. A leaf module returns itself; a composed one returns its children.
    fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
        Vec::new()
    }

    /// Ask every predictor in this program for its reply to be sampled this way.
    ///
    /// dspy's `set_lm`, which assigns one model to a whole program so `lm.copy(rollout_id=n,
    /// temperature=1.0)` reaches every call an attempt makes. Sampling travels on a request here
    /// rather than on a model, so this sets config instead — the effect is the one upstream
    /// relies on: a second attempt at a program differs from the first everywhere, not only at
    /// whichever predictor a caller remembered.
    fn set_config(&mut self, config: Sampling) {
        for predictor in self.named_predictors() {
            *predictor.config = config.clone();
        }
    }

    /// This program's compiled state — every predictor's instructions and demos, keyed by name.
    /// dspy's `program.dump_state`: what an optimizer produced, ready to save and reload. The walk
    /// is [`named_predictors`](Self::named_predictors), so a composed program dumps its children.
    fn dump_state(&mut self) -> ProgramState {
        ProgramState::new(
            self.named_predictors()
                .into_iter()
                .map(|predictor| {
                    let lm = predictor
                        .lm
                        .as_ref()
                        .and_then(|model| model.dump_state_dyn());
                    let state = PredictorState::of(predictor.signature, predictor.demos, lm);
                    (predictor.name, SubmoduleState::Predictor(state))
                })
                .collect(),
        )
    }

    /// Restore a compiled state onto this program, which must be the same program the state was
    /// dumped from. Each demo's input split is re-declared from its signature.
    ///
    /// A predictor this program has and the state does not is an error, as it is upstream — dspy
    /// indexes `state[name]` and raises. Skipping it instead would hand back a program that looks
    /// loaded and is not, which is what loading the wrong saved program would silently produce.
    /// The check runs over every predictor before any is touched, so a state that does not fit
    /// leaves the program as it was rather than half-loaded — dspy gets the same from its
    /// `_apply(self.deepcopy())` trial run.
    fn load_state(&mut self, state: &ProgramState) -> Result<()> {
        self.load_state_with(state, Trust::Default)
    }

    /// [`load_state`](Self::load_state) for a file the caller vouches for.
    ///
    /// dspy's `load_state(state, allow_unsafe_lm_state=True)`. Split into its own method rather
    /// than taking a flag because the safe call is the one nearly everybody makes, and a bare
    /// `false` at every call site says less than the name does. What it widens is described on
    /// [`Trust`].
    fn load_state_trusted(&mut self, state: &ProgramState) -> Result<()> {
        self.load_state_with(state, Trust::File)
    }

    /// The body of both, so neither can drift from the other.
    fn load_state_with(&mut self, state: &ProgramState, trust: Trust) -> Result<()> {
        let missing: Vec<String> = self
            .named_predictors()
            .iter()
            .map(|predictor| predictor.name.clone())
            .filter(|name| state.get(name).is_none())
            .collect();
        if !missing.is_empty() {
            bail!("the saved state has no entry for {missing:?}");
        }

        for predictor in self.named_predictors() {
            let Some(saved) = state.get(&predictor.name) else {
                continue;
            };
            saved.signature.restore(predictor.signature);
            let inputs: Vec<String> = predictor
                .signature
                .inputs
                .iter()
                .map(|field| field.name.clone())
                .collect();
            *predictor.demos = saved
                .demos
                .iter()
                .map(|fields| demo_from_fields(fields, &inputs))
                .collect();

            // dspy sanitises the block and rebuilds a live model from it, so a loaded program asks
            // what it was compiled against rather than whatever this process configured. A block
            // that names a model this crate cannot build fails the load, as upstream's does —
            // answering from somewhere else would be the one outcome nobody could detect.
            if let Some(block) = saved.lm.as_ref().and_then(|lm| lm.as_object()) {
                let block = crate::lm::saved::sanitize(block, trust.allows_redirect());
                let model = crate::lm::saved::rebuild(&block, trust.allows_redirect())?;
                *predictor.lm = Some(std::sync::Arc::new(model));
            }
        }
        Ok(())
    }

    /// Save this program's compiled state to a JSON file, reloaded onto a fresh copy of the same
    /// program with [`load`](Self::load). This is dspy's `program.save`/`dspy.load(path)` — the
    /// reusable artifact an optimizer exists to produce.
    fn save(&mut self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(&self.dump_state())?)?;
        Ok(())
    }

    /// Load a compiled state saved by [`save`](Self::save) onto this program.
    fn load(&mut self, path: &Path) -> Result<()> {
        self.load_state(&serde_json::from_str(&std::fs::read_to_string(path)?)?)?;
        Ok(())
    }

    /// [`load`](Self::load) for a file the caller vouches for — dspy's
    /// `dspy.load(path, allow_unsafe_lm_state=True)`. See [`Trust`].
    fn load_trusted(&mut self, path: &Path) -> Result<()> {
        self.load_state_trusted(&serde_json::from_str(&std::fs::read_to_string(path)?)?)?;
        Ok(())
    }

    /// Run the program, recording which predictor saw what.
    ///
    /// An optimizer needs this to give each predictor demos its own calls earned. A composed
    /// module passes the trace down and relabels what its children recorded, the same way
    /// [`Self::named_predictors`] relabels theirs, so a step's name always matches a predictor's.
    ///
    /// Recording nothing is allowed and means the program cannot be attributed, so every
    /// predictor in it receives the same program-level demo. That costs nothing for a program
    /// with one predictor, where the two are the same list.
    fn forward_traced<'a>(
        &'a self,
        inputs: Example,
        trace: &'a mut Vec<TraceStep>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
        let _ = trace;
        self.forward(inputs)
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
/// Asks through [`Ask`], so what comes back is whatever the module promised. A module of your
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;
    use serde_json::json;

    /// A module written outside the crate's own types, to prove the trait is implementable.
    struct Echo {
        /// This double's pinned model, which a saved program records. Never set — it is here
        /// because a `NamedPredictor` names one and two of them cannot borrow the same slot.
        saved_lm_0: Option<std::sync::Arc<dyn crate::lm::DynChatModel>>,
        signature: Signature,
        demos: Vec<Example>,
        config: Sampling,
        hint: Option<String>,
    }

    impl Module for Echo {
        fn forward<'a>(
            &'a self,
            inputs: Example,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<Prediction>> + Send + 'a>> {
            Box::pin(async move {
                let echoed = inputs.get("request").cloned().unwrap_or(json!(""));
                Ok(Prediction::new(
                    Example::new([("answer", echoed)]),
                    "echoed",
                ))
            })
        }

        fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
            vec![NamedPredictor {
                name: "self".to_owned(),
                signature: &mut self.signature,
                demos: &mut self.demos,
                config: &mut self.config,
                hint: &mut self.hint,
                lm: &mut self.saved_lm_0,
            }]
        }
    }

    fn echo() -> Echo {
        Echo {
            saved_lm_0: None,
            signature: Signature::single_input("Echo the request.", Vec::new()),
            demos: Vec::new(),
            config: Sampling::default(),
            hint: None,
        }
    }

    #[tokio::test]
    async fn a_module_defined_by_a_caller_runs() {
        let module = echo();
        let prediction = module
            .forward(example! { request: "hello" }.with_inputs(["request"]))
            .await
            .expect("echo succeeds");
        assert_eq!(prediction.get("answer"), Some(&json!("hello")));
    }

    #[tokio::test]
    async fn any_module_can_be_held_behind_a_box() {
        // An evaluator and a composed program both need this; a non-object-safe trait would
        // have made both impossible.
        let modules: Vec<Box<dyn Module>> = vec![Box::new(echo())];
        for module in &modules {
            let prediction = module
                .forward(example! { request: "held" }.with_inputs(["request"]))
                .await
                .expect("boxed module runs");
            assert_eq!(prediction.get("answer"), Some(&json!("held")));
        }
    }

    /// A compiled program — an optimizer's product — saves its instructions and demos to a file and
    /// reloads them onto a fresh copy of the same program, each demo's input split re-declared from
    /// the signature. This is the reusable artifact the optimizer family exists to produce.
    #[test]
    fn a_compiled_program_saves_and_reloads_its_state() {
        let mut trained = echo();
        trained.signature.instructions = "Answer in one word.".to_owned();
        trained.demos = vec![example! { request: "hi", answer: "hello" }.with_inputs(["request"])];

        let path = std::env::temp_dir().join(format!(
            "dsrs_program_state_roundtrip-{}.json",
            std::process::id()
        ));
        trained.save(&path).expect("saves");

        let mut fresh = echo();
        assert_eq!(
            fresh.signature.instructions, "Echo the request.",
            "fresh program is unoptimized"
        );
        fresh.load(&path).expect("loads");
        std::fs::remove_file(&path).ok();

        assert_eq!(fresh.signature.instructions, "Answer in one word.");
        assert_eq!(fresh.demos.len(), 1);
        assert_eq!(fresh.demos[0].get("answer"), Some(&json!("hello")));
        assert!(
            fresh.demos[0].is_input("request"),
            "the demo's input split was re-declared from the signature"
        );
        assert!(
            !fresh.demos[0].is_input("answer"),
            "and its output stays a label"
        );
    }

    /// A state that does not name every predictor this program has is refused, and refused before
    /// anything is touched. dspy indexes `state[name]` and raises; skipping instead would hand back
    /// a program that looks loaded and is not, which is what loading the wrong saved program would
    /// quietly produce.
    #[test]
    fn a_state_that_does_not_fit_is_refused_and_changes_nothing() {
        let mut fresh = echo();
        let refused = fresh
            .load_state(&ProgramState::new(Default::default()))
            .expect_err("a state naming no predictor at all does not fit");
        assert!(
            refused.to_string().contains("has no entry for"),
            "got: {refused}"
        );
        assert_eq!(
            fresh.signature.instructions, "Echo the request.",
            "the program is as it was, not half-loaded"
        );
    }

    #[test]
    fn an_optimizer_can_write_demos_back_through_the_walk() {
        let mut module = echo();
        for predictor in module.named_predictors() {
            predictor.demos.push(example! { request: "a", answer: "a" });
            predictor.signature.instructions = "Echo it exactly.".to_owned();
        }
        assert_eq!(module.demos.len(), 1);
        assert_eq!(module.signature.instructions, "Echo it exactly.");
    }
}

/// What asking one module answers with.
///
/// dspy has one call spelling across both ways of declaring a task, because every module there
/// answers with the same dynamic `Prediction`. Rust need not give that up to match: the spelling
/// is shared and the answer stays whatever the module promised, so a derived task still hands
/// back its own outputs struct and `result.answer` still means the field.
pub trait Ask {
    type Answer;

    fn ask<'a>(
        &'a self,
        inputs: Example,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Self::Answer>> + Send + 'a>>;
}

/// Answer with the fields the module parsed, which is what a task declared by its field names
/// has to give.
///
/// Written per module rather than blanket over [`Module`], because a derived task is a `Module`
/// too and answers with something better than a `Prediction`. One blanket impl would make that
/// unreachable.
#[macro_export]
macro_rules! asks_with_a_prediction {
    ($module:ty) => {
        impl $crate::Ask for $module {
            type Answer = $crate::Prediction;

            fn ask<'a>(
                &'a self,
                inputs: $crate::Example,
            ) -> ::std::pin::Pin<
                ::std::boxed::Box<
                    dyn ::std::future::Future<Output = ::anyhow::Result<$crate::Prediction>>
                        + Send
                        + 'a,
                >,
            > {
                $crate::Module::forward(self, inputs)
            }
        }
    };
}
