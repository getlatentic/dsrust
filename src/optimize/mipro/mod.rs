//! dspy `MIPROv2` (`teleprompt/mipro_optimizer_v2.py`): propose instruction candidates with a
//! grounded proposer, then search the combinations with a TPE sampler.
//!
//! The search engine is the [`tpe`] crate — optuna's categorical `TPESampler`, reproduced and
//! verified in isolation — and this module wraps it the way dspy wraps optuna. The proposer's
//! prompts live in [`signatures`], byte-verified against dspy.
//!
//! This is the zero-shot configuration: instructions are searched, demos are not. dspy's
//! demo-bootstrapping path (`max_bootstrapped_demos > 0`), the dataset-summary proposer, and
//! minibatch evaluation build on the same pieces and are not yet wired; a run here matches dspy
//! with `max_bootstrapped_demos=0`, `max_labeled_demos=0`, `data_aware_proposer=False`,
//! `minibatch=False`.

use std::sync::Arc;

use anyhow::{Result, bail};

use super::Optimizer;
use super::rng::Rng;
use crate::evaluate::Evaluate;
use crate::example::{Example, Prediction};
use crate::lm::DynChatModel;
use crate::module::Module;
use crate::signature::Signature;

mod proposer;
mod signatures;

use proposer::GenerateModuleInstruction;
use signatures::InstructionInputs;

/// dspy GroundedProposer's tips, in declaration order — the order `random.choice` indexes into. The
/// empty `none` tip is a real member: choosing it turns the tip field off for that candidate.
const TIPS: [&str; 6] = [
    "",
    "Don't be afraid to be creative when creating the new instruction!",
    "Keep the instruction clear and concise.",
    "Make sure your instruction is very informative and descriptive.",
    "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "Include a persona that is relevant to the task in the instruction (ie. \"You are a ...\")",
];

/// dspy `round(100 * mean, 2)`: the percentage `Evaluate` reports, which is the number MIPROv2 hands
/// the sampler and compares. Half-to-even at the second decimal, matching Python's `round`.
fn percent(mean: f64) -> f64 {
    (10_000.0 * mean).round_ties_even() / 100.0
}

/// The grounded proposer, in its zero-shot form: propose `num_candidates` instructions per predictor,
/// each optionally program-aware and carrying a randomly chosen tip.
struct GroundedProposer {
    program_code: Option<String>,
    tip_aware: bool,
    prompt_model: Arc<dyn DynChatModel>,
}

impl GroundedProposer {
    /// dspy `propose_instructions_for_program`: for each predictor, `num_candidates` proposals, with
    /// candidate zero forced back to the predictor's current instruction — the baseline the search
    /// starts from. The RNG is CPython's, drawing a tip then a rollout id per proposal, in dspy's order.
    async fn propose(
        &self,
        predictors: &[Signature],
        num_candidates: usize,
        rng: &mut Rng,
    ) -> Result<Vec<Vec<String>>> {
        let mut proposed = Vec::with_capacity(predictors.len());
        for signature in predictors {
            let mut candidates = Vec::with_capacity(num_candidates);
            for _ in 0..num_candidates {
                let tip = self.select_tip(rng);
                // dspy draws a rollout id per proposal to miss the response cache; the draw advances
                // the shared generator whether or not anything is cached, so it is made here too.
                let _rollout = rng.randint(0, 1_000_000_000);
                let inputs = InstructionInputs {
                    dataset_summary: false,
                    program_aware: self.program_code.is_some(),
                    instruct_history: false,
                    tip: tip.is_some(),
                };
                let generator = GenerateModuleInstruction::new(
                    self.program_code.clone(),
                    inputs,
                    self.prompt_model.clone(),
                );
                let instruction = generator
                    .forward(signature, "No task demos provided.", "", "", tip)
                    .await?;
                candidates.push(instruction);
            }
            if !candidates.is_empty() {
                candidates[0] = signature.instructions.clone();
            }
            proposed.push(candidates);
        }
        Ok(proposed)
    }

    /// dspy's `random.choice(list(TIPS.keys()))` when tips are on: a draw is made regardless of which
    /// tip lands, and the empty `none` tip reads as no tip. Off, no draw is made and there is no tip.
    fn select_tip(&self, rng: &mut Rng) -> Option<&'static str> {
        if !self.tip_aware {
            return None;
        }
        let tip = TIPS[rng.choice_index(TIPS.len())];
        (!tip.is_empty()).then_some(tip)
    }
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
    program_code: Option<String>,
    tip_aware: bool,
}

impl<M> MIPROv2<M>
where
    M: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    /// A MIPROv2 proposing with this model. dspy's defaults for the counts are set per auto mode;
    /// here they are explicit — ten instruction candidates and twenty trials is a common medium run.
    pub fn new(metric: M, prompt_model: Arc<dyn DynChatModel>) -> Self {
        Self {
            metric,
            prompt_model,
            num_candidates: 10,
            num_trials: 20,
            seed: 9,
            program_code: None,
            tip_aware: true,
        }
    }

    /// How many instructions to propose per predictor.
    pub fn with_candidates(mut self, num_candidates: usize) -> Self {
        self.num_candidates = num_candidates;
        self
    }

    /// How many instruction combinations the search evaluates.
    pub fn with_trials(mut self, num_trials: usize) -> Self {
        self.num_trials = num_trials;
        self
    }

    /// The seed for the proposer's RNG and the TPE sampler — dspy seeds both from one number.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Turn on the program-aware proposer with this pseudo-code description of the program. dspy
    /// reads it from source; a Rust caller supplies it, which is dspy's own `program_code_string` seam.
    pub fn with_program_code(mut self, program_code: impl Into<String>) -> Self {
        self.program_code = Some(program_code.into());
        self
    }

    /// dspy `compile(student, trainset=...)`: rewrite the student's instructions in place, leaving it
    /// holding the highest-scoring combination the search found.
    pub async fn compile<S: Module + ?Sized>(
        &self,
        student: &mut S,
        trainset: &[Example],
    ) -> Result<()> {
        let predictors: Vec<Signature> = student
            .named_predictors()
            .iter()
            .map(|predictor| predictor.signature.clone())
            .collect();
        if predictors.is_empty() {
            return Ok(());
        }

        let mut rng = Rng::seeded(self.seed);
        let proposer = GroundedProposer {
            program_code: self.program_code.clone(),
            tip_aware: self.tip_aware,
            prompt_model: self.prompt_model.clone(),
        };
        let candidates = proposer.propose(&predictors, self.num_candidates, &mut rng).await?;

        self.search(student, &candidates, trainset).await;
        Ok(())
    }

    /// dspy's Step 3: seed the sampler with the default program as a baseline trial, run `num_trials`
    /// more, and leave the student on the best combination. Candidate zero of every predictor is its
    /// original instruction, so the all-zeros baseline is the default program.
    async fn search<S: Module + ?Sized>(
        &self,
        student: &mut S,
        candidates: &[Vec<String>],
        valset: &[Example],
    ) -> f64 {
        let cardinalities: Vec<usize> = candidates.iter().map(Vec::len).collect();
        let baseline = vec![0usize; candidates.len()];
        let default_score = self.score(student, valset).await;

        let mut sampler = tpe::TpeSampler::new(self.seed as u32, cardinalities);
        sampler.tell(baseline.clone(), default_score);
        let mut best = (default_score, baseline);

        for _ in 0..self.num_trials {
            let params = sampler.ask();
            apply(student, candidates, &params);
            let score = self.score(student, valset).await;
            sampler.tell(params.clone(), score);
            if score > best.0 {
                best = (score, params);
            }
        }

        apply(student, candidates, &best.1);
        best.0
    }

    /// dspy Evaluate's headline for one candidate: the metric's mean over the valset, as a percentage.
    async fn score<S: Module + ?Sized>(&self, student: &S, valset: &[Example]) -> f64 {
        let evaluation = Evaluate::new(
            valset.to_vec(),
            |inputs| student.forward(inputs),
            |example: &Example, prediction: &Prediction| (self.metric)(example, prediction),
        )
        .run()
        .await;
        percent(evaluation.score)
    }
}

/// Set each predictor to the instruction the trial chose for it.
fn apply<S: Module + ?Sized>(student: &mut S, candidates: &[Vec<String>], params: &[usize]) {
    let mut predictors = student.named_predictors();
    for (index, predictor) in predictors.iter_mut().enumerate() {
        if let Some(&choice) = params.get(index) {
            predictor.signature.instructions = candidates[index][choice].clone();
        }
    }
}

impl<M> Optimizer for MIPROv2<M>
where
    M: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    fn compile<'a>(
        &'a self,
        student: &'a mut dyn Module,
        teacher: Option<&'a mut dyn Module>,
        trainset: &'a [Example],
    ) -> impl Future<Output = Result<()>> + Send + 'a {
        async move {
            if teacher.is_some() {
                bail!("MIPROv2 proposes instructions from a metric and has no teacher to learn from");
            }
            MIPROv2::compile(self, student, trainset).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate::exact_match;
    use crate::example;
    use crate::lm::{ChatModel, api};
    use crate::predict::Predict;

    /// A model that both proposes and answers. Asked to generate an instruction it returns one
    /// carrying `GOOD`; asked to do the task it answers correctly only when `GOOD` is the instruction
    /// in force. So the search has one candidate that scores and one (the original) that does not.
    struct Studio;

    impl ChatModel for Studio {
        async fn forward(
            &self,
            _http: &reqwest::Client,
            request: &api::LmRequest,
        ) -> Result<api::LmResponse> {
            let system = request.system();
            let content = if system.contains("generate a new instruction that will be used") {
                "[[ ## proposed_instruction ## ]]\nAnswer with GOOD precision.\n\n[[ ## completed ## ]]".to_owned()
            } else {
                let answer = if system.contains("GOOD") { "Paris" } else { "London" };
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
            .with_lm(model.clone());
        let trainset =
            vec![example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"])];

        MIPROv2::new(exact_match, model.clone())
            .with_candidates(2)
            .with_trials(4)
            .compile(&mut student, &trainset)
            .await
            .expect("compiles");

        // Candidate 1 (the proposal) scores 100 against the original's 0, so the search leaves the
        // student holding it.
        assert_eq!(student.signature.instructions, "Answer with GOOD precision.");
    }
}
