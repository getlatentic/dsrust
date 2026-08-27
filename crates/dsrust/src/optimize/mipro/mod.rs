//! dspy `MIPROv2` (`teleprompt/mipro_optimizer_v2.py`): propose instruction candidates with a
//! grounded proposer, then search the combinations with a TPE sampler.
//!
//! The search engine is the [`tpe`] crate — optuna's categorical `TPESampler`, reproduced and
//! verified in isolation — and this module wraps it the way dspy wraps optuna. The proposer's
//! prompts live in [`signatures`], byte-verified against dspy.
//!
//! Instructions and demos are both searched, as upstream's defaults do — `max_bootstrapped_demos`
//! and `max_labeled_demos` are dspy's 4 and 4, and setting both to zero is upstream's own zero-shot
//! configuration rather than a separate path. The dataset-summary proposer is not wired; a run here
//! matches dspy with `data_aware_proposer=False`.
//!
//! `auto` is the whole of upstream's `_set_hyperparams_from_run_mode`, which is mostly [`minibatch`]:
//! a preset subsamples the valset off the shared generator before anything else draws, and above
//! fifty examples turns on minibatch trials whose interleaved full evaluations are fed back to the
//! sampler as trials of their own.
//!
//! **The search space is read in two different orders**, which is why the parameters carry dspy's
//! names down to the sampler rather than only their cardinalities. A startup trial is drawn one
//! parameter at a time by `sample_independent`, called from `suggest_categorical` — so in the order
//! upstream suggests them, instruction then demos. A TPE trial is drawn all at once from a search
//! space that came through `IntersectionSearchSpace`, which is `dict(sorted(...))` — so in name
//! order, where `0_predictor_demos` sorts before `0_predictor_instruction`.
//!
//! Measured, after this file asserted the suggest order for both. Every few-shot conformance case
//! had happened to give the two parameters the *same* cardinality, where the two orders draw the
//! same numbers, and no case ran the ten trials it takes to leave the startup phase. `auto` is what
//! exposed it: a preset proposes `n/2` instructions against `n` demo sets and runs long enough to
//! reach TPE.

use std::future::Future;
use std::sync::Arc;

use anyhow::{Result, bail};

use super::Optimizer;
use super::rng::Rng;
use crate::example::Example;
use crate::lm::DynChatModel;
use crate::module::Module;
use crate::signature::Signature;

mod building;
mod dataset_summary;
mod demos;
mod grounded;
pub(crate) mod minibatch;
mod proposer;
mod search;
mod signatures;

#[cfg(test)]
mod conformance;

pub use minibatch::Auto;

use grounded::GroundedProposer;

/// dspy `BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT` / `LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT`: the demo
/// budgets Step 1 uses in the zero-shot case, where the sets ground the proposer rather than being
/// searched.
const ZEROSHOT_BOOTSTRAPPED: usize = 3;
const ZEROSHOT_LABELED: usize = 0;

/// One search trial: the candidate index chosen per predictor, and the score it earned — one
/// entry of dspy's `trial_logs`, as the optuna study records it.
///
/// [`MIPROv2::compile_traced`] answers with these where [`MIPROv2::compile`] answers with nothing,
/// which is how a caller keeps the record of a run that upstream writes to a log directory:
///
/// ```no_run
/// # use dsrust::optimize::Trial;
/// # fn read(trials: Vec<Trial>) {
/// // The trials arrive in the order the study created them, so the best is a fold rather than a
/// // sort — two trials can tie, and the first of a tie is the one the search kept.
/// let best = trials
///     .iter()
///     .max_by(|a, b| a.score.total_cmp(&b.score))
///     .expect("a compiled run ran at least the baseline");
/// println!("{} trials, best {:.1}% on {:?}", trials.len(), best.score, best.params);
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    pub params: Vec<usize>,
    pub score: f64,
}

/// dspy `MIPROv2`: an instruction optimizer that searches proposed instructions with Bayesian
/// optimization.
///
/// `num_candidates` instructions are proposed per predictor and `num_trials` combinations are tried.
/// Proposals are written by `prompt_model`; the program is evaluated on whichever model its
/// predictors already use. `program_code`, when given, turns on dspy's program-aware proposer — the
/// Rust stand-in for its source introspection, which has no equivalent here.
pub struct MIPROv2<M> {
    metric: M,
    prompt_model: Arc<dyn DynChatModel>,
    num_candidates: usize,
    num_trials: usize,
    seed: u64,
    /// dspy `init_temperature`: what instructions are proposed at (default 1.0).
    init_temperature: f64,
    /// dspy `metric_threshold`: the score a bootstrapped trace must beat to be kept in Step 1.
    metric_threshold: Option<f64>,
    /// What a scoring pass is bounded by. See [`Scoring`](super::Scoring).
    scoring: super::Scoring,
    program_code: Option<String>,
    tip_aware: bool,
    /// dspy `task_model`: the model the program is bootstrapped and evaluated on, where that is
    /// not the configured one. See [`task_model`](Self::task_model).
    task_model: Option<Arc<dyn DynChatModel>>,
    /// dspy `max_bootstrapped_demos`: how many demos a bootstrap may put in a set. Zero with
    /// `max_labeled_demos` zero is upstream's zero-shot run.
    max_bootstrapped_demos: usize,
    /// dspy `max_labeled_demos`: how many labelled trainset examples a set may carry unbootstrapped.
    max_labeled_demos: usize,
    /// dspy `auto`: a budget preset, which replaces the two counts and subsamples the valset.
    /// See [`auto`](Self::auto).
    auto: Option<Auto>,
    /// dspy `minibatch`: score each trial on a subsample rather than the whole valset. Read only
    /// when `auto` is unset, which is upstream's precedence and not this crate's.
    minibatch: bool,
    /// dspy `minibatch_size`: how many examples a minibatch trial scores on (default 35).
    minibatch_size: usize,
    /// dspy `minibatch_full_eval_steps`: how often a full evaluation interrupts the minibatch
    /// trials (default 5).
    minibatch_full_eval_steps: usize,
    /// dspy `data_aware_proposer`: show the proposer a summary of the trainset, on by default as
    /// upstream's is. See [`data_aware_proposer`](Self::data_aware_proposer).
    data_aware_proposer: bool,
    /// dspy `view_data_batch_size`: how many examples each summarising call reads (default 10).
    view_data_batch_size: usize,
    /// dspy `fewshot_aware_proposer`: ground each proposal in its own candidate's demos.
    fewshot_aware_proposer: bool,
}

/// What one run's hyperparameters resolve to once `auto` has had its say — dspy's
/// `_set_hyperparams_from_run_mode` return.
struct RunMode {
    num_trials: usize,
    valset: Vec<Example>,
    minibatch: bool,
    instruction_candidates: usize,
    fewshot_candidates: usize,
}

impl<M> MIPROv2<M>
where
    M: crate::evaluate::Metric,
{
    /// dspy `compile(student, trainset=...)`: rewrite the student's instructions in place, leaving it
    /// holding the highest-scoring combination the search found.
    pub async fn compile<S: Module + ?Sized>(
        &self,
        student: &mut S,
        trainset: &[Example],
        valset: Option<&[Example]>,
    ) -> Result<()> {
        self.compile_traced(student, trainset, valset)
            .await
            .map(|_| ())
    }

    /// The same compile, also returning the search's trial sequence — dspy's `trial_logs`, which
    /// upstream attaches to the compiled program. Each entry is the candidate index the trial
    /// chose per predictor and the score it earned; the first entry is the default program's
    /// baseline trial, exactly as the optuna study records it.
    pub async fn compile_traced<S: Module + ?Sized>(
        &self,
        student: &mut S,
        trainset: &[Example],
        valset: Option<&[Example]>,
    ) -> Result<Vec<Trial>> {
        let predictors: Vec<Signature> = student
            .named_predictors()
            .iter()
            .map(|predictor| predictor.signature.clone())
            .collect();
        if predictors.is_empty() {
            return Ok(Vec::new());
        }
        let zeroshot = self.zeroshot();
        let (trainset, valset) = self.datasets(trainset, valset)?;

        let mut rng = Rng::seeded(self.seed);
        // `auto` draws before anything else does — the valset subsample is the first thing off this
        // generator, ahead of Step 1 — so a preset moves every later draw, not only the counts.
        let mode = self.run_mode(predictors.len(), zeroshot, valset, &mut rng);

        // Step 1: bootstrap demo sets. A zero-shot run still builds them — they ground the proposer,
        // and building them advances the shared RNG that Step 2's proposal reads — but it builds them
        // at dspy's in-context constants rather than at the caller's counts, and never searches them.
        let demo_sets = self
            .on_task_model(demos::create_demo_sets(
                student,
                mode.fewshot_candidates,
                &trainset,
                if zeroshot {
                    ZEROSHOT_LABELED
                } else {
                    self.max_labeled_demos
                },
                if zeroshot {
                    ZEROSHOT_BOOTSTRAPPED
                } else {
                    self.max_bootstrapped_demos
                },
                &self.metric,
                self.metric_threshold,
                &mut rng,
            ))
            .await?;

        // Step 2: propose instruction candidates, off the RNG Step 1 advanced.
        // Upstream summarises in `GroundedProposer.__init__`, before any tip is drawn, and takes no
        // draw of its own — so this sits here rather than inside the proposal loop.
        let dataset_summary = match self.data_aware_proposer {
            true => dataset_summary::create_dataset_summary(
                &trainset,
                self.view_data_batch_size,
                &self.prompt_model,
            )
            .await
            .ok(),
            false => None,
        };
        let proposer = GroundedProposer {
            dataset_summary,
            program_code: self.program_code.clone(),
            tip_aware: self.tip_aware,
            prompt_model: self.prompt_model.clone(),
            init_temperature: self.init_temperature,
            fewshot_aware: self.fewshot_aware_proposer,
        };
        let candidates = proposer
            .propose(
                &predictors,
                mode.instruction_candidates,
                (!zeroshot).then_some(demo_sets.as_slice()),
                &mut rng,
            )
            .await?;

        // Step 3: search the combinations. A zero-shot run hands the search no demo sets, which is
        // upstream passing `demo_candidates=None` and suggesting no demo parameter at all.
        let searched = (!zeroshot).then_some(demo_sets.as_slice());
        self.on_task_model(self.search(student, &candidates, searched, &mode, &mut rng))
            .await
    }

    /// dspy `_set_and_validate_datasets`: what a run bootstraps from and what it scores on.
    ///
    /// No valset is not "score on everything": upstream hands the *last* 80% of the trainset to the
    /// valset and keeps the first 20% to bootstrap from, so a caller who passes one set gets a split
    /// rather than an overlap.
    fn datasets(
        &self,
        trainset: &[Example],
        valset: Option<&[Example]>,
    ) -> Result<(Vec<Example>, Vec<Example>)> {
        if trainset.is_empty() {
            bail!("Trainset cannot be empty.");
        }
        let Some(valset) = valset else {
            if trainset.len() < 2 {
                bail!("Trainset must have at least 2 examples if no valset specified.");
            }
            let size = (trainset.len() as f64 * 0.80) as usize;
            let cutoff = trainset.len() - size.clamp(1, 1000);
            return Ok((trainset[..cutoff].to_vec(), trainset[cutoff..].to_vec()));
        };
        if valset.is_empty() {
            bail!("Valset cannot be empty.");
        }
        Ok((trainset.to_vec(), valset.to_vec()))
    }

    /// dspy `_set_hyperparams_from_run_mode`: what `auto` replaces, and what it leaves alone.
    fn run_mode(
        &self,
        predictors: usize,
        zeroshot: bool,
        valset: Vec<Example>,
        rng: &mut Rng,
    ) -> RunMode {
        let Some(auto) = self.auto else {
            return RunMode {
                num_trials: self.num_trials,
                minibatch: self.minibatch,
                valset,
                instruction_candidates: self.num_candidates,
                fewshot_candidates: self.num_candidates,
            };
        };
        let valset = minibatch::subsample(&valset, auto.val_size(), rng);
        RunMode {
            num_trials: trials_for(predictors, zeroshot, auto.candidates()),
            // Recomputed from the subsample, discarding whatever the caller asked for — upstream
            // overwrites the argument here rather than defaulting to it.
            minibatch: valset.len() > minibatch::MIN_MINIBATCH_SIZE,
            valset,
            instruction_candidates: auto.instruction_candidates(zeroshot),
            fewshot_candidates: auto.candidates(),
        }
    }
}

/// dspy `_set_num_trials_from_num_candidates`: how many trials a preset's candidate count is worth.
/// Searching demos doubles the variables, since each predictor then carries two.
fn trials_for(predictors: usize, zeroshot: bool, num_candidates: usize) -> usize {
    let num_vars = if zeroshot { predictors } else { predictors * 2 };
    let by_space = 2.0 * num_vars as f64 * (num_candidates as f64).log2();
    by_space.max(1.5 * num_candidates as f64) as usize
}

/// One slot of the search space: which predictor it belongs to, whether it chooses that
/// predictor's demo set rather than its instruction, and how many choices it has.
#[derive(Clone, Copy)]
pub(super) struct Slot {
    predictor: usize,
    demos: bool,
    cardinality: usize,
}

/// The parameters one trial chooses, under dspy's own names and in the order upstream suggests
/// them: per predictor, the instruction and then — only when demos are searched — the demo set.
///
/// The names travel with them because the sampler draws in suggest order while it is still in its
/// random startup and in *name* order once TPE takes over, and `demos` sorts before `instruction`.
pub(super) fn search_space(
    candidates: &[Vec<String>],
    demo_sets: Option<&[Vec<Vec<Example>>]>,
) -> Vec<(String, Slot)> {
    let mut named: Vec<(String, Slot)> = Vec::with_capacity(candidates.len() * 2);
    for (predictor, instructions) in candidates.iter().enumerate() {
        named.push((
            format!("{predictor}_predictor_instruction"),
            Slot {
                predictor,
                demos: false,
                cardinality: instructions.len(),
            },
        ));
        if let Some(sets) = demo_sets {
            named.push((
                format!("{predictor}_predictor_demos"),
                Slot {
                    predictor,
                    demos: true,
                    cardinality: sets[predictor].len(),
                },
            ));
        }
    }
    named
}

/// Set each predictor to the instruction — and the demo set — the trial chose for it.
pub(super) fn apply<S: Module + ?Sized>(
    student: &mut S,
    candidates: &[Vec<String>],
    demo_sets: Option<&[Vec<Vec<Example>>]>,
    space: &[Slot],
    params: &[usize],
) {
    let mut predictors = student.named_predictors();
    for (slot, &choice) in space.iter().zip(params) {
        let Some(predictor) = predictors.get_mut(slot.predictor) else {
            continue;
        };
        match (slot.demos, demo_sets) {
            (true, Some(sets)) => *predictor.demos = sets[slot.predictor][choice].clone(),
            (true, None) => {}
            (false, _) => {
                predictor.signature.instructions = candidates[slot.predictor][choice].clone();
            }
        }
    }
}

impl Slot {
    pub(super) fn cardinality(self) -> usize {
        self.cardinality
    }
}

impl<M> Optimizer for MIPROv2<M>
where
    M: crate::evaluate::Metric,
{
    fn compile<'a>(
        &'a self,
        student: &'a mut dyn Module,
        teacher: Option<&'a mut dyn Module>,
        trainset: &'a [Example],
        valset: Option<&'a [Example]>,
    ) -> impl Future<Output = Result<()>> + Send + 'a {
        async move {
            if teacher.is_some() {
                bail!(
                    "MIPROv2 proposes instructions from a metric and has no teacher to learn from"
                );
            }
            MIPROv2::compile(self, student, trainset, valset).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Under a preset, `minibatch` is recomputed from the subsampled valset — on only when the
    /// subsample is strictly larger than `MIN_MINIBATCH_SIZE`, which is dspy's own `>`.
    #[test]
    fn a_preset_turns_minibatching_on_only_past_the_minimum() {
        let model: Arc<dyn crate::lm::DynChatModel> = Arc::new(Studio);
        let mipro =
            MIPROv2::new(|_: &crate::Example, _: &crate::Prediction| 1.0, model).auto(Auto::Light);
        let example = |at: usize| {
            crate::Example::new([("question", serde_json::json!(at.to_string()))])
                .with_inputs(["question"])
        };
        let mut rng = Rng::seeded(0);
        let at_minimum: Vec<crate::Example> =
            (0..minibatch::MIN_MINIBATCH_SIZE).map(example).collect();
        let mode = mipro.run_mode(1, false, at_minimum, &mut rng);
        assert!(!mode.minibatch, "at the minimum is not past it");

        let mut rng = Rng::seeded(0);
        let past: Vec<crate::Example> = (0..minibatch::MIN_MINIBATCH_SIZE + 1)
            .map(example)
            .collect();
        let mode = mipro.run_mode(1, false, past, &mut rng);
        assert!(mode.minibatch, "one past the minimum turns it on");
    }

    /// The two refusals of `datasets`, in dspy's words: an empty trainset always, and one example
    /// only when no valset was given. The `<` mutants moved the second boundary in both
    /// directions — `==` admitted a one-example trainset, `<=` refused a two-example one.
    #[test]
    fn datasets_refuses_what_dspy_refuses() {
        let model: Arc<dyn crate::lm::DynChatModel> = Arc::new(Studio);
        let mipro = MIPROv2::new(|_: &crate::Example, _: &crate::Prediction| 1.0, model);
        let example = |q: &str| {
            crate::Example::new([("question", serde_json::json!(q))]).with_inputs(["question"])
        };
        let one = [example("only")];
        let why = mipro
            .datasets(&one, None)
            .expect_err("one example cannot split")
            .to_string();
        assert_eq!(
            why,
            "Trainset must have at least 2 examples if no valset specified."
        );

        let two = [example("a"), example("b")];
        assert!(mipro.datasets(&two, None).is_ok(), "two is the minimum");
        assert!(
            mipro.datasets(&one, Some(&two)).is_ok(),
            "a caller-supplied valset lifts the minimum"
        );
        let none: [crate::Example; 0] = [];
        let why = mipro
            .datasets(&none, Some(&two))
            .expect_err("empty is refused regardless")
            .to_string();
        assert_eq!(why, "Trainset cannot be empty.");
    }
    use crate::evaluate::exact_match;
    use crate::example;
    use crate::lm::{ChatModel, api};
    use crate::predict::Predict;

    /// A model that both proposes and answers. Asked to generate an instruction it returns one
    /// carrying `GOOD`; asked to do the task it answers correctly only when `GOOD` is the instruction
    /// in force. So the search has one candidate that scores and one (the original) that does not.
    struct Studio;

    impl ChatModel for Studio {
        async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
            let system = request.system();
            let content = if system.contains("generate a new instruction that will be used") {
                "[[ ## proposed_instruction ## ]]\nAnswer with GOOD precision.\n\n[[ ## completed ## ]]".to_owned()
            } else {
                let answer = if system.contains("GOOD") {
                    "Paris"
                } else {
                    "London"
                };
                format!("[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]")
            };
            Ok(api::LmResponse::text(content))
        }
    }

    #[tokio::test]
    async fn the_search_keeps_the_instruction_that_scores() {
        let model = Arc::new(Studio);
        let mut student = Predict::parse("question -> answer")
            .expect("parses")
            .set_lm(model.clone());
        let trainset = vec![
            example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
        ];

        MIPROv2::new(exact_match, model.clone())
            .num_candidates(2)
            .num_trials(4)
            .minibatch(false)
            .data_aware_proposer(false)
            .compile(&mut student, &trainset, Some(&trainset))
            .await
            .expect("compiles");

        // Candidate 1 (the proposal) scores 100 against the original's 0, so the search leaves the
        // student holding it.
        assert_eq!(
            student.signature.instructions,
            "Answer with GOOD precision."
        );
    }
}
