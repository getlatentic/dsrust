//! dspy `GroundedProposer`: the loop that proposes every predictor's instruction candidates.
//!
//! One level above [`super::proposer`], which owns the signature a single proposal is asked
//! through. This owns how many are asked for, in what order, and what grounding each one carries —
//! the tip drawn per proposal, the dataset summary drawn once, and the rollout id that makes two
//! proposals from the same prompt differ.

use std::sync::Arc;

use anyhow::Result;

use super::super::rng::Rng;
use super::proposer::GenerateModuleInstruction;
use super::signatures::InstructionInputs;
use crate::lm::DynChatModel;
use crate::signature::Signature;

/// dspy GroundedProposer's tips, in declaration order — the order `random.choice` indexes into. The
/// empty `none` tip is a real member: choosing it turns the tip field off for that candidate.
pub(super) const TIPS: [&str; 6] = [
    "",
    "Don't be afraid to be creative when creating the new instruction!",
    "Keep the instruction clear and concise.",
    "Make sure your instruction is very informative and descriptive.",
    "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "Include a persona that is relevant to the task in the instruction (ie. \"You are a ...\")",
];

/// The grounded proposer, in its zero-shot form: propose `num_candidates` instructions per predictor,
/// each optionally program-aware and carrying a randomly chosen tip.
pub(super) struct GroundedProposer {
    /// dspy's `data_summary`, produced once in `GroundedProposer.__init__` before any candidate is
    /// proposed. `None` is both "not asked for" and "the summarising failed", which is upstream's
    /// own fallback: it catches the exception and runs without the summary rather than losing the
    /// compile.
    pub(super) dataset_summary: Option<String>,
    pub(super) program_code: Option<String>,
    pub(super) tip_aware: bool,
    pub(super) prompt_model: Arc<dyn DynChatModel>,
    /// dspy `init_temperature`, carried from the optimizer to the one call that proposes.
    pub(super) init_temperature: f64,
    /// dspy `fewshot_aware_proposer`: show each proposal the demos its own candidate set carries.
    /// Read together with the demo sets — upstream's `use_task_demos and demo_candidates`, so a
    /// zero-shot run has nothing to show whatever the flag says.
    pub(super) fewshot_aware: bool,
}

impl GroundedProposer {
    /// dspy `propose_instructions_for_program`: for each predictor, `num_candidates` proposals, with
    /// candidate zero forced back to the predictor's current instruction — the baseline the search
    /// starts from. The RNG is CPython's, drawing a tip then a rollout id per proposal, in dspy's order.
    pub(super) async fn propose(
        &self,
        predictors: &[Signature],
        num_candidates: usize,
        demo_sets: Option<&[Vec<Vec<crate::Example>>]>,
        rng: &mut Rng,
    ) -> Result<Vec<Vec<String>>> {
        // dspy walks `range(num_demos)[:min(N, num_demos)]`, so the proposal count is capped by the
        // number of demo sets: `N` is `num_instruct_candidates` and `num_demos` is how many sets
        // Step 1 built. They are the same number on the explicit path and `n/2` against `n` under a
        // preset, where the instruction count is already the smaller — so the cap has never bitten.
        // Reproduced anyway, because without it a caller whose counts disagree gets more candidates
        // than dspy would and an index past the end of the sets.
        let per_predictor = match demo_sets.filter(|_| self.fewshot_aware) {
            Some(sets) => num_candidates.min(sets.first().map_or(num_candidates, Vec::len)),
            None => num_candidates,
        };
        let mut proposed = Vec::with_capacity(predictors.len());
        for (predictor, signature) in predictors.iter().enumerate() {
            let mut candidates = Vec::with_capacity(per_predictor);
            for chosen in 0..per_predictor {
                let tip = self.select_tip(rng);
                // dspy asks for each proposal through `prompt_model.copy(rollout_id=…,
                // temperature=init_temperature)`. The draw also advances the shared generator, which
                // is why it happens whether or not the id is used.
                let rollout = rng.randint(0, 1_000_000_000);
                let sampling = crate::lm::Sampling {
                    temperature: Some(self.init_temperature),
                    ..crate::lm::Sampling::rollout(rollout)
                };
                let inputs = InstructionInputs {
                    dataset_summary: self.dataset_summary.is_some(),
                    program_aware: self.program_code.is_some(),
                    instruct_history: false,
                    tip: tip.is_some(),
                };
                let generator = GenerateModuleInstruction::new(
                    self.program_code.clone(),
                    inputs,
                    self.prompt_model.clone(),
                    sampling,
                );
                // dspy's `demo_set_i` is the candidate index, so candidate k is grounded in demo
                // set k — which is why the two counts are the same number upstream.
                let demos = match (self.fewshot_aware, demo_sets) {
                    (true, Some(sets)) => task_demos(signature, &sets[predictor], chosen),
                    _ => NO_DEMOS.to_owned(),
                };
                let instruction = generator
                    .forward(
                        signature,
                        &demos,
                        self.dataset_summary.as_deref().unwrap_or_default(),
                        "",
                        tip,
                    )
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

/// dspy `num_demos_in_context`: how many demos a proposal is shown, whichever set they come from.
const DEMOS_IN_CONTEXT: usize = 3;

/// dspy's `task_demos` fallback, and what an untouched proposal carries.
const NO_DEMOS: &str = "No task demos provided.";

/// dspy `create_example_string`: one demo as `prefix value` per signature field, newline-joined.
///
/// Every field of the signature, in its own order — a field the demo never recorded prints as
/// Python's `None`, since upstream interpolates `example.get(name)` into an f-string.
fn example_string(signature: &Signature, demo: &crate::Example) -> String {
    let prefixes = signature
        .inputs
        .iter()
        .map(|field| (field.name.as_str(), field.prefix.as_deref()))
        .chain(
            signature
                .outputs
                .iter()
                .map(|field| (field.name.as_str(), field.prefix.as_deref())),
        );
    prefixes
        .map(|(name, prefix)| {
            let prefix = match prefix {
                Some(prefix) => prefix.to_owned(),
                None => format!("{}:", crate::signature::infer_prefix(name)),
            };
            let value = match demo.get(name) {
                Some(serde_json::Value::String(text)) => text.clone(),
                Some(value) => crate::python::repr(value),
                None => "None".to_owned(),
            };
            format!("{prefix} {value}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// dspy's `task_demos` for one candidate: up to three *augmented* demos, read from the demo sets
/// **rotated to start at this candidate's own set**.
///
/// Upstream writes the rotation in three pieces — `[sets[i]] + sets[i+1:] + sets[:i]` — which is
/// `sets[i..]` followed by `sets[..i]`. So candidate k reads set k, then k+1, and wraps round to
/// the ones before it, taking demos until it has three. Deterministic, and the reason it is a
/// rotation rather than plain order is that each candidate should be grounded in its *own* set and
/// borrow from the neighbours only to make up the number.
///
/// Only augmented demos count — `gather_examples_from_sets` tests `"augmented" in example.keys()`,
/// which is the marker a bootstrap puts on a demo the teacher earned. A labelled demo drawn from
/// the trainset is not shown, because it demonstrates nothing about what the program can do.
///
/// Candidate zero is always given the fallback even when demos were gathered, which is upstream's
/// `or demo_set_i == 0` — the baseline proposal is deliberately ungrounded.
fn task_demos(signature: &Signature, sets: &[Vec<crate::Example>], chosen: usize) -> String {
    if chosen == 0 || sets.is_empty() {
        return NO_DEMOS.to_owned();
    }
    let adjacent = sets[chosen..].iter().chain(sets[..chosen].iter());
    let gathered: Vec<String> = adjacent
        .flatten()
        .filter(|demo| demo.get("augmented").is_some())
        .take(DEMOS_IN_CONTEXT)
        .map(|demo| example_string(signature, demo))
        .collect();
    match gathered.is_empty() {
        true => NO_DEMOS.to_owned(),
        false => format!("{}\n\n", gathered.join("\n\n")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every proposal is asked at `init_temperature` under its own rollout id — dspy's
    /// `prompt_model.copy(rollout_id=…, temperature=init_temperature)`. The temperature-field
    /// mutant left the ask at the default and nothing read the config back; DummyLM records it.
    #[tokio::test]
    async fn a_proposal_is_asked_at_init_temperature_with_a_rollout() {
        let lm = std::sync::Arc::new(crate::DummyLM::new([
            example! { proposed_instruction: "Answer better." },
            example! { proposed_instruction: "Answer better still." },
        ]));
        let proposer = GroundedProposer {
            dataset_summary: None,
            program_code: None,
            tip_aware: false,
            prompt_model: lm.clone(),
            init_temperature: 0.7,
            fewshot_aware: false,
        };
        let mut rng = Rng::seeded(0);
        proposer
            .propose(
                &["question -> answer".parse().expect("parses")],
                2,
                None,
                &mut rng,
            )
            .await
            .expect("the script answers");
        let asked = lm.asked();
        assert!(!asked.is_empty());
        for ask in &asked {
            assert_eq!(ask.config.temperature, Some(0.7), "dspy's init_temperature");
            assert!(
                ask.config.rollout_id.is_some(),
                "each proposal draws its own rollout id"
            );
        }
    }

    /// The tip map: with `tip_aware` off no draw is made; on, the drawn tip crosses unless it is
    /// the empty `none` tip, which reads as no tip — never `Some("")`. The deleted `!` inverted
    /// exactly that, answering a tip only when there was none.
    #[test]
    fn select_tip_never_answers_an_empty_tip() {
        let off = GroundedProposer {
            dataset_summary: None,
            program_code: None,
            tip_aware: false,
            prompt_model: std::sync::Arc::new(crate::DummyLM::new([])),
            init_temperature: 0.5,
            fewshot_aware: false,
        };
        assert_eq!(
            off.select_tip(&mut Rng::seeded(0)),
            None,
            "no draw when off"
        );

        let on = GroundedProposer {
            tip_aware: true,
            ..off
        };
        let (mut some, mut none) = (false, false);
        for seed in 0..64 {
            // The parallel draw with the same seed says which tip the call must have seen.
            let drawn = TIPS[Rng::seeded(seed).choice_index(TIPS.len())];
            let answered = on.select_tip(&mut Rng::seeded(seed));
            match drawn.is_empty() {
                true => {
                    assert_eq!(answered, None, "the empty none tip is no tip");
                    none = true;
                }
                false => {
                    assert_eq!(answered, Some(drawn));
                    some = true;
                }
            }
        }
        assert!(some && none, "both branches must occur across 64 seeds");
    }
    use crate::example;

    fn signature() -> Signature {
        "question -> answer".parse().expect("parses")
    }

    fn earned(question: &str, answer: &str) -> crate::Example {
        example! { augmented: true, question: question, answer: answer }
    }

    /// Candidate zero is shown nothing however many demos exist — upstream's `or demo_set_i == 0`,
    /// which keeps the baseline proposal ungrounded.
    #[test]
    fn the_baseline_candidate_is_shown_no_demos() {
        let sets = vec![vec![earned("a?", "a")], vec![earned("b?", "b")]];
        assert_eq!(task_demos(&signature(), &sets, 0), NO_DEMOS);
    }

    /// A candidate reads the sets rotated to start at its own — upstream's
    /// `[sets[i]] + sets[i+1:] + sets[:i]`, not the sets in declaration order. dspy's own recorded
    /// output shows it: with three sets, candidate 4 is shown France, Spain, Germany where
    /// candidate 1 is shown France, Germany, Spain.
    #[test]
    fn a_candidate_reads_its_own_set_first_then_wraps() {
        let sets = vec![
            vec![earned("first?", "1")],
            vec![earned("second?", "2")],
            vec![earned("third?", "3")],
        ];
        let shown = task_demos(&signature(), &sets, 1);
        let order: Vec<&str> = shown
            .lines()
            .filter(|line| line.starts_with("Question:"))
            .collect();
        assert_eq!(
            order,
            ["Question: second?", "Question: third?", "Question: first?"]
        );
    }

    /// The sets a real bootstrap builds are sets this filter can read.
    ///
    /// Every other test here hands `task_demos` demos marked by hand, so all of them pass while
    /// the marker is missing from the demos Step 1 actually produces — and the failure is silent,
    /// because an unmarked set filters down to `NO_DEMOS` and reads as "no demos were gathered"
    /// rather than as a fault. `Solver` records no trace, which is the arm that used to lose it.
    #[tokio::test]
    async fn the_demos_step_one_builds_reach_a_proposal() {
        let mut student =
            crate::optimize::scripted::Solver::new(crate::optimize::scripted::Answers::Correctly);
        let sets = crate::optimize::mipro::demos::create_demo_sets(
            &mut student,
            4,
            &crate::optimize::scripted::trainset(),
            2,
            2,
            &crate::evaluate::exact_match,
            None,
            &mut Rng::seeded(0),
        )
        .await
        .expect("the scripted program bootstraps");

        // Set 2 is the unshuffled bootstrap, the one that earns demos. Reading from candidate 2
        // starts the rotation there, so what comes back is what the bootstrap earned.
        let shown = task_demos(&signature(), &sets[0], 2);
        assert_ne!(
            shown, NO_DEMOS,
            "a bootstrapped set grounds a proposal: {:?}",
            sets[0]
        );
        assert!(
            shown.contains("capital of France?"),
            "grounded in a demo the teacher earned, not a labelled one: {shown}"
        );
    }

    /// Only demos the teacher earned are shown. A labelled one drawn from the trainset carries no
    /// `augmented` marker and demonstrates nothing about what the program can do.
    #[test]
    fn a_labelled_demo_is_not_shown() {
        let labelled = example! { question: "plain?", answer: "plain" };
        let sets = vec![vec![labelled.clone()], vec![labelled]];
        assert_eq!(task_demos(&signature(), &sets, 1), NO_DEMOS);
    }

    /// Three at most, whichever sets they came from — upstream's `num_demos_in_context`.
    #[test]
    fn at_most_three_demos_reach_a_proposal() {
        let sets = vec![
            vec![earned("a?", "a"), earned("b?", "b")],
            vec![earned("c?", "c"), earned("d?", "d")],
        ];
        let shown = task_demos(&signature(), &sets, 1);
        assert_eq!(shown.matches("Question:").count(), DEMOS_IN_CONTEXT);
    }

    /// A field the demo never recorded prints as Python's `None`, since upstream interpolates
    /// `example.get(name)` straight into an f-string.
    #[test]
    fn a_field_the_demo_lacks_prints_as_none() {
        let partial = example! { augmented: true, question: "q?" };
        let shown = task_demos(&signature(), &vec![vec![], vec![partial]], 1);
        assert_eq!(shown, "Question: q?\nAnswer: None\n\n");
    }
}
