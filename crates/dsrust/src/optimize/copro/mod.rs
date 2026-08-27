//! dspy `COPRO` (`teleprompt/copro_optimizer.py`): rewrite each predictor's instruction by
//! coordinate ascent. Ask a model to propose instructions, score each on the trainset, keep the
//! best, then show the best back to the model and ask for better ones — `depth` times over.
//!
//! What COPRO writes is a predictor's `signature.instructions`, so unlike the few-shot optimizers
//! it needs no demos. It runs on two primitives this crate grew for it: reading several proposals
//! from one call ([`Predict::forward_completions`]) and scoring a program
//! ([`crate::evaluate::Evaluate`]). Its two proposal prompts are byte-verified against dspy; the
//! loop is verified structurally, since what a real model proposes is not reproducible.

use std::sync::Arc;

use anyhow::{Result, bail};
use serde_json::{Value, json};

use super::Optimizer;
use crate::evaluate::Evaluate;
use crate::example::{Example, Prediction};
use crate::lm::{DynChatModel, Sampling};
use crate::module::Module;
use crate::predict::Predict;
use crate::signature::{Signature, infer_prefix};

mod candidates;
mod signatures;
mod stats;

#[cfg(test)]
mod conformance;

use candidates::{Evaluated, Evaluations, Proposal, best_program, stripped};
pub use stats::{CoproStats, DepthScores};

/// dspy `COPRO`: an instruction optimizer driven by a metric.
///
/// `breadth` proposals per round (must exceed one), `depth` rounds, and `init_temperature` for how
/// freely the proposing model samples — upstream's defaults are 10, 3 and 1.4. The proposing model
/// defaults to whichever model the predictors use; [`Self::prompt_model`] sets a stronger one
/// to write the prompts while a cheaper one runs the task, which is dspy's `prompt_model`.
pub struct COPRO<M> {
    metric: M,
    breadth: usize,
    depth: usize,
    init_temperature: f64,
    prompt_model: Option<Arc<dyn DynChatModel>>,
    /// What a scoring pass is bounded by. See [`Scoring`](crate::optimize::Scoring).
    scoring: crate::optimize::Scoring,
}

impl<M> COPRO<M>
where
    M: crate::evaluate::Metric,
{
    /// A COPRO at dspy's defaults: breadth 10, depth 3, temperature 1.4, proposing with the same
    /// model the program already uses.
    pub fn new(metric: M) -> Self {
        Self {
            metric,
            breadth: 10,
            depth: 3,
            init_temperature: 1.4,
            prompt_model: None,
            scoring: crate::optimize::Scoring::default(),
        }
    }

    /// How many instructions to propose each round. dspy rejects one or fewer, since a round has to
    /// leave the current instruction and at least one rival to choose between.
    pub fn breadth(mut self, breadth: usize) -> Self {
        self.breadth = breadth;
        self
    }

    /// How many rounds of propose-score-keep to run.
    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// How freely the proposing model samples. Higher is more varied; dspy's default is 1.4.
    pub fn init_temperature(mut self, temperature: f64) -> Self {
        self.init_temperature = temperature;
        self
    }

    /// Propose instructions with this model rather than the one the program runs on. dspy's
    /// `prompt_model`.
    pub fn prompt_model(mut self, model: Arc<dyn DynChatModel>) -> Self {
        self.prompt_model = Some(model);
        self
    }

    /// dspy `compile(student, trainset=...)`: rewrite the student's instructions in place.
    ///
    /// The student is left holding the single highest-scoring instruction set found across every
    /// predictor and round — dspy's `best_program`, which it returns and this crate writes back.
    pub async fn compile<S: Module + ?Sized>(
        &self,
        student: &mut S,
        trainset: &[Example],
    ) -> Result<()> {
        self.compile_traced(student, trainset).await.map(|_| ())
    }

    /// The same search, handing back what dspy's `track_stats=True` records.
    ///
    /// Upstream hangs `results_best`, `results_latest` and `total_calls` off the compiled program;
    /// a Rust program is the caller's value, so they are returned — the shape
    /// [`MIPROv2::compile_traced`](crate::optimize::MIPROv2::compile_traced) already uses. There is
    /// no flag: the numbers are counted either way and `compile` drops them.
    ///
    /// ```no_run
    /// # use dsrust::optimize::{COPRO, CoproStats, DepthScores};
    /// # async fn wrapper(program: &mut dsrust::Predict, trainset: Vec<dsrust::Example>) -> anyhow::Result<()> {
    /// let stats: CoproStats = COPRO::new(dsrust::exact_match)
    ///     .compile_traced(program, &trainset)
    ///     .await?;
    ///
    /// // One entry per depth, per predictor. Reading `best` shows whether the search was still
    /// // finding better instructions by the last round or had stopped moving.
    /// for (predictor, depths) in stats.best.iter().enumerate() {
    ///     for DepthScores { depth, max, average, .. } in depths {
    ///         println!("predictor {predictor} at depth {depth}: best {max:.1}%, mean {average:.1}%");
    ///     }
    /// }
    /// println!("{} candidates scored", stats.total_calls);
    /// # Ok(()) }
    /// ```
    pub async fn compile_traced<S: Module + ?Sized>(
        &self,
        student: &mut S,
        trainset: &[Example],
    ) -> Result<CoproStats> {
        if self.breadth <= 1 {
            bail!("COPRO breadth must be greater than 1");
        }
        let predictors = student.named_predictors().len();
        if predictors == 0 {
            return Ok(CoproStats::default());
        }

        let originals = originals(student);
        let mut latest = self.seed(&originals).await?;
        let mut all = latest.clone();
        let mut evaluated: Vec<Evaluations> =
            (0..predictors).map(|_| Evaluations::default()).collect();
        let mut current: Vec<String> = originals.iter().map(|o| o.instruction.clone()).collect();

        let mut stats = CoproStats::for_predictors(predictors);
        for round in 0..self.depth {
            for predictor in 0..predictors {
                let pool = if predictors > 1 {
                    &all[predictor]
                } else {
                    &latest[predictor]
                };
                // dspy's `latest_scores`: the tail of the pool, `breadth` long — the candidates
                // this round's proposal produced, rather than every one seen so far.
                let newest = pool.len().saturating_sub(self.breadth);
                let mut latest_scores = Vec::new();
                for (at, candidate) in pool.iter().enumerate() {
                    let outcome = self
                        .try_candidate(student, predictor, candidate, trainset, &mut current)
                        .await?;
                    stats.total_calls += 1;
                    if at >= newest {
                        latest_scores.push(outcome.score);
                    }
                    evaluated[predictor].record(outcome);
                }
                if let Some(summary) = DepthScores::of(round, &latest_scores) {
                    stats.latest[predictor].push(summary);
                }
                let mut seen = evaluated[predictor].scores();
                if let Some(summary) = DepthScores::of(round, CoproStats::top_ten(&mut seen)) {
                    stats.best[predictor].push(summary);
                }
                let best = evaluated[predictor].best().instruction.clone();
                set_instruction(student, predictor, &best);
                current[predictor] = best;
            }
            if round == self.depth - 1 {
                break;
            }
            let next = self.propose_next(&evaluated).await?;
            for (predictor, proposals) in next.iter().enumerate() {
                all[predictor].extend(proposals.iter().cloned());
            }
            latest = next;
        }

        if let Some(program) = best_program(&evaluated) {
            for (predictor, instruction) in program.iter().enumerate() {
                set_instruction(student, predictor, instruction);
            }
        }
        Ok(stats)
    }

    /// Set one predictor to one candidate's instruction, score the whole program, and package the
    /// result — the innermost step of the loop, kept off the loop body so it reads as one action.
    async fn try_candidate<S: Module + ?Sized>(
        &self,
        student: &mut S,
        predictor: usize,
        candidate: &Proposal,
        trainset: &[Example],
        current: &mut [String],
    ) -> Result<Evaluated> {
        let instruction = stripped(&candidate.instruction);
        let prefix = stripped(&candidate.prefix);
        set_instruction(student, predictor, &instruction);
        current[predictor] = instruction.clone();
        let score = self.score(student, trainset).await?;
        Ok(Evaluated {
            instruction,
            prefix,
            score,
            program: current.to_vec(),
        })
    }

    /// The seed round: for each predictor, ask for `breadth - 1` fresh instructions and add its
    /// current one, so the model always keeps the option of not changing it. dspy seeds this way.
    async fn seed(&self, originals: &[Proposal]) -> Result<Vec<Vec<Proposal>>> {
        let mut seeded = Vec::with_capacity(originals.len());
        for original in originals {
            let mut proposals = self
                .propose(
                    signatures::basic_generate_instruction(),
                    self.breadth - 1,
                    [("basic_instruction", json!(original.instruction))],
                )
                .await?;
            proposals.push(original.clone());
            seeded.push(proposals);
        }
        Ok(seeded)
    }

    /// The depth round: show each predictor's best attempts back to the model, worst-first, and ask
    /// for `breadth` more. dspy's `GenerateInstructionGivenAttempts` step.
    async fn propose_next(&self, evaluated: &[Evaluations]) -> Result<Vec<Vec<Proposal>>> {
        let mut next = Vec::with_capacity(evaluated.len());
        for evaluations in evaluated {
            let attempts = evaluations.attempts(self.breadth);
            let proposals = self
                .propose(
                    signatures::generate_instruction_given_attempts(),
                    self.breadth,
                    [("attempted_instructions", json!(attempts))],
                )
                .await?;
            next.push(proposals);
        }
        Ok(next)
    }

    /// Ask the proposing model for `n` instruction candidates from one call, reading every
    /// completion. The signature and its one input are the only things that differ between the
    /// seed round and a depth round.
    async fn propose(
        &self,
        signature: Signature,
        n: usize,
        input: [(&str, Value); 1],
    ) -> Result<Vec<Proposal>> {
        let mut predict = Predict::from_signature(signature).config(Sampling {
            completions: Some(n as u32),
            temperature: Some(self.init_temperature),
            ..Sampling::default()
        });
        if let Some(model) = &self.prompt_model {
            predict = predict.set_lm(model.clone());
        }
        let predictions = predict.forward_completions(Example::new(input)).await?;
        Ok(predictions.iter().map(proposal_of).collect())
    }

    /// dspy Evaluate's headline: the metric's mean over the trainset, scaled to a percentage and
    /// rounded, which is the number COPRO compares and writes into its next prompt.
    async fn score<S: Module + ?Sized>(&self, student: &S, trainset: &[Example]) -> Result<f64> {
        let evaluation = self
            .scoring
            .apply(Evaluate::new(
                trainset.to_vec(),
                |inputs| student.forward(inputs),
                crate::evaluate::MetricRef(&self.metric),
            ))
            .run()
            .await;
        Ok(evaluation?.score)
    }
}

/// Each predictor's instruction and the prefix of its last output field, read before anything is
/// changed. dspy seeds the search from these and keeps each as a candidate.
fn originals<S: Module + ?Sized>(student: &mut S) -> Vec<Proposal> {
    student
        .named_predictors()
        .iter()
        .map(|predictor| Proposal {
            instruction: predictor.signature.instructions.clone(),
            // dspy's default field prefix is `infer_prefix(name) + ":"`; the original prefix seeds
            // the search and is shown back in a depth prompt's attempts, so the colon has to be here.
            prefix: predictor
                .signature
                .outputs
                .last()
                .map(|field| format!("{}:", infer_prefix(&field.name)))
                .unwrap_or_default(),
        })
        .collect()
}

/// Write one predictor's instruction. The seam every optimizer writes through, reached by index
/// because COPRO walks the predictors in the order [`originals`] read them.
fn set_instruction<S: Module + ?Sized>(student: &mut S, index: usize, instruction: &str) {
    let mut predictors = student.named_predictors();
    if let Some(predictor) = predictors.get_mut(index) {
        predictor.signature.instructions = instruction.to_owned();
    }
}

/// The two output fields of a proposal, read off one parsed completion. A field the model omitted
/// reads as empty, which the stripping step then leaves empty.
fn proposal_of(prediction: &Prediction) -> Proposal {
    Proposal {
        instruction: text_field(prediction, "proposed_instruction"),
        prefix: text_field(prediction, "proposed_prefix_for_output_field"),
    }
}

fn text_field(prediction: &Prediction, name: &str) -> String {
    prediction
        .get(name)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

impl<M> Optimizer for COPRO<M>
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
            if valset.is_some() {
                bail!(
                    "COPRO scores on the trainset, as upstream's compile does — it takes no valset"
                );
            }
            if teacher.is_some() {
                bail!(
                    "COPRO optimizes instructions from a metric and has no teacher to learn from"
                );
            }
            COPRO::compile(self, student, trainset).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Proposals are drawn at `init_temperature` — dspy passes it to every BasicGenerateInstruction
    /// call, and the deleted field left the draw at the default with nothing reading the config.
    #[tokio::test]
    async fn proposals_are_drawn_at_init_temperature() {
        let lm = std::sync::Arc::new(crate::DummyLM::new([example! {
            proposed_instruction: "Try harder.",
            proposed_prefix_for_output_field: "Answer:"
        }]));
        let copro = COPRO::new(exact_match)
            .prompt_model(lm.clone())
            .init_temperature(1.2);
        copro
            .propose(
                super::signatures::basic_generate_instruction(),
                1,
                [(
                    "basic_instruction",
                    serde_json::json!("Answer the question."),
                )],
            )
            .await
            .expect("the script answers");
        let asked = lm.asked();
        assert_eq!(asked.len(), 1);
        assert_eq!(asked[0].config.temperature, Some(1.2));
        assert_eq!(asked[0].config.completions, Some(1));
    }
    use crate::evaluate::exact_match;
    use crate::example;
    use crate::lm::{ChatModel, api};

    /// A model that plays both parts COPRO needs from one. Asked to optimize an instruction — its
    /// system message names the optimizer task — it proposes one carrying the token `GOOD`. Asked
    /// to do the task, it answers correctly only when `GOOD` is the instruction in force. So
    /// exactly one candidate scores, and a working coordinate ascent has to end on it.
    struct Coach;

    impl ChatModel for Coach {
        async fn forward(&self, request: &api::LmRequest) -> Result<api::LmResponse> {
            let system = request.system();
            if system.contains("instruction optimizer") {
                let proposal = "[[ ## proposed_instruction ## ]]\nAnswer the question. GOOD\n\n\
                     [[ ## proposed_prefix_for_output_field ## ]]\nAnswer:\n\n[[ ## completed ## ]]";
                return Ok(api::LmResponse::completions([proposal.to_owned()]));
            }
            let answer = if system.contains("GOOD") {
                "Paris"
            } else {
                "London"
            };
            Ok(api::LmResponse::text(format!(
                "[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
            )))
        }
    }

    #[tokio::test]
    async fn coordinate_ascent_keeps_the_instruction_that_scores() {
        let model = Arc::new(Coach);
        let mut student = Predict::parse("question -> answer")
            .expect("parses")
            .set_lm(model.clone());
        let trainset = vec![
            example! { question: "capital of France?", answer: "Paris" }.with_inputs(["question"]),
        ];

        COPRO::new(exact_match)
            .breadth(2)
            .depth(1)
            .prompt_model(model.clone())
            .compile(&mut student, &trainset)
            .await
            .expect("compiles");

        // The seed round offered the model's `GOOD` proposal against the original instruction; only
        // the proposal scores, so it is what the student is left holding.
        assert_eq!(student.signature.instructions, "Answer the question. GOOD");
    }

    #[tokio::test]
    async fn a_teacher_is_refused_since_copro_has_none() {
        let model = Arc::new(Coach);
        let mut student = Predict::parse("question -> answer").expect("parses");
        let mut teacher = Predict::parse("question -> answer").expect("parses");
        let optimizer = COPRO::new(exact_match).prompt_model(model);
        let refused =
            Optimizer::compile(&optimizer, &mut student, Some(&mut teacher), &[], None).await;
        assert!(
            refused.is_err(),
            "a teacher COPRO cannot use is an error, not silently ignored"
        );
    }
}
