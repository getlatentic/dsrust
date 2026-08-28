//! dspy `teleprompt/infer_rules.py`: bootstrap, then ask a model to write the rules down.
//!
//! Every other optimizer here changes a program by changing what it is *shown* — which demos, which
//! instruction, drawn from a search. This one asks a model to read the trainset and say what the
//! task's rules are, appends them to each predictor's instruction, and keeps the candidate that
//! scores best. The variation between candidates comes from nothing but the model: each induction
//! runs at temperature one with a fresh rollout id.
//!
//! The rendering is its own, not an adapter's — `format_examples` writes a shape that appears
//! nowhere else in dspy — and the rollout ids come from one `random.Random(0)` shared across every
//! candidate and predictor, so they are a single stream rather than one per call. Both are held to
//! `optimize/infer_rules.json`.

use anyhow::{Result, bail};
use pyrng::Random;

use super::BootstrapFewShot;
use crate::evaluate::Metric;
use crate::example::Example;
use crate::module::{Module, ProgramState};
use crate::predict::ChainOfThought;
use crate::signature::{InField, OutField, Signature};

/// One candidate program and what it scored, in the order they were built.
///
/// ```
/// # use dsrust::optimize::RuleCandidate;
/// # fn read(candidates: Vec<RuleCandidate>) {
/// // Upstream keeps only the winner; every attempt is here, so a caller can read the rules a
/// // candidate was given as well as the score they earned.
/// if let Some(best) = candidates.iter().max_by(|a, b| a.score.total_cmp(&b.score)) {
///     println!("{} rule set(s), best scored {:.1}%", candidates.len(), best.score);
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct RuleCandidate {
    /// The rules the model wrote, one entry per predictor in walk order.
    pub rules: Vec<String>,
    /// What the program carrying them scored on the validation set.
    pub score: f64,
}

/// dspy `InferRules`.
pub struct InferRules<M> {
    metric: M,
    num_candidates: usize,
    num_rules: usize,
    max_bootstrapped_demos: usize,
    max_labeled_demos: usize,
    scoring: super::Scoring,
}

impl<M> InferRules<M>
where
    M: Metric,
{
    /// dspy's defaults: ten candidates, ten rules each.
    pub fn new(metric: M) -> Self {
        Self {
            metric,
            num_candidates: 10,
            num_rules: 10,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            scoring: super::Scoring::default(),
        }
    }

    /// How many rule sets to induce and score.
    pub fn num_candidates(mut self, candidates: usize) -> Self {
        self.num_candidates = candidates;
        self
    }

    /// How many rules to ask for. It reaches the model as a number inside the instruction, so it
    /// changes the prompt rather than bounding anything this crate counts.
    pub fn num_rules(mut self, rules: usize) -> Self {
        self.num_rules = rules;
        self
    }

    pub fn max_bootstrapped_demos(mut self, demos: usize) -> Self {
        self.max_bootstrapped_demos = demos;
        self
    }

    pub fn max_labeled_demos(mut self, demos: usize) -> Self {
        self.max_labeled_demos = demos;
        self
    }

    pub fn scoring(mut self, scoring: super::Scoring) -> Self {
        self.scoring = scoring;
        self
    }

    /// Bootstrap `student`, then induce and score `num_candidates` rule sets, keeping the best.
    ///
    /// With no validation set the trainset is halved and the **first** half is what the rules are
    /// learned from — `int(0.5 * len)`, so an odd split leaves the larger half for scoring.
    pub async fn compile<S: Module + ?Sized>(
        &self,
        student: &mut S,
        writing_rules: std::sync::Arc<dyn crate::lm::DynChatModel>,
        trainset: &[Example],
        valset: Option<&[Example]>,
    ) -> Result<Vec<RuleCandidate>> {
        let (trainset, valset) = match valset {
            Some(valset) => (trainset, valset),
            None => trainset.split_at(train_size(trainset.len())),
        };
        let mut bootstrap = BootstrapFewShot::new(crate::evaluate::MetricRef(&self.metric));
        bootstrap.max_bootstrapped_demos = self.max_bootstrapped_demos;
        bootstrap.max_labeled_demos = self.max_labeled_demos;
        bootstrap.compile(student, trainset).await?;

        let bootstrapped = student.dump_state();
        let instructions: Vec<String> = student
            .named_predictors()
            .into_iter()
            .map(|predictor| predictor.signature.instructions.clone())
            .collect();
        if instructions.is_empty() {
            bail!("a program with no predictors has no instructions to add rules to");
        }
        // One generator for the whole compile, as upstream holds one on the induction program and
        // reuses it across every candidate and every predictor.
        let mut rng = Random::seeded(0);
        // Upstream builds this itself, as a `ChainOfThought` over a signature whose docstring
        // names the rule count. The model is the caller's, because dspy takes it from ambient
        // settings and there is none here.
        let mut induction =
            ChainOfThought::from_signature(self.induction_signature()).set_lm(writing_rules);

        let mut candidates = Vec::with_capacity(self.num_candidates);
        let mut best: Option<(f64, ProgramState)> = None;
        for _ in 0..self.num_candidates {
            student.load_state(&bootstrapped)?;
            let mut rules = Vec::with_capacity(instructions.len());
            for position in 0..instructions.len() {
                let asked = self.examples_text(student, trainset, position);
                let written = self.induce(&mut induction, &asked, &mut rng).await?;
                let predictor = student.named_predictors().swap_remove(position);
                // Appended to the *original* instruction rather than to whatever the last candidate
                // left, which is what upstream's two resets amount to.
                predictor.signature.instructions = with_rules(&instructions[position], &written);
                rules.push(written);
            }
            let score = self.score(student, valset).await?;
            if best.as_ref().is_none_or(|(seen, _)| score > *seen) {
                best = Some((score, student.dump_state()));
            }
            candidates.push(RuleCandidate { rules, score });
        }
        match best {
            Some((_, state)) => student.load_state(&state)?,
            None => student.load_state(&bootstrapped)?,
        }
        Ok(candidates)
    }

    /// dspy `format_examples` over `get_predictor_demos`: every trainset row, narrowed to the
    /// predictor's own fields and written in the row's order rather than the signature's.
    fn examples_text<S: Module + ?Sized>(
        &self,
        student: &mut S,
        trainset: &[Example],
        position: usize,
    ) -> String {
        let signature = student
            .named_predictors()
            .swap_remove(position)
            .signature
            .clone();
        let mut text = String::new();
        for row in trainset {
            let (mut inputs, mut outputs) = (Vec::new(), Vec::new());
            for (name, value) in row.fields() {
                let rendered = format!(
                    "{name}: {}",
                    crate::adapter::python_json::format_value(value)
                );
                if signature.inputs.iter().any(|f| f.name == name) {
                    inputs.push(rendered);
                } else if signature.outputs.iter().any(|f| f.name == name) {
                    outputs.push(rendered);
                }
            }
            // A `"\n".join` and then two more newlines, which is not the same as ending every line
            // with one: a row missing its output field leaves the block empty, and upstream still
            // writes both newlines after the heading.
            text.push_str(&format!(
                "Input Fields:\n{}\n\n=========\nOutput Fields:\n{}\n\n",
                inputs.join("\n"),
                outputs.join("\n")
            ));
        }
        text
    }

    /// dspy `RulesInductionProgram.forward`: one call at temperature one with a fresh rollout id.
    async fn induce(
        &self,
        induction: &mut ChainOfThought,
        asked: &str,
        rng: &mut Random,
    ) -> Result<String> {
        // `random.Random(0).randint(0, 10**9)` — the same stream every compile, so two runs over
        // the same trainset ask the same questions. Upstream varies it through `lm.copy`, which is
        // a cache key and nothing a provider sees; here it is the predictor's own sampling, which
        // is the same key.
        let rollout = rng.randint(0, 1_000_000_000);
        for predictor in induction.named_predictors() {
            predictor.config.temperature = Some(1.0);
            predictor.config.rollout_id = Some(rollout);
        }
        let prediction = induction
            .forward(Example::new([(
                "examples_text",
                serde_json::Value::String(asked.to_owned()),
            )]))
            .await?;
        Ok(prediction
            .example
            .get("natural_language_rules")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned())
    }

    /// dspy's `CustomRulesInduction`, whose docstring is an f-string over `num_rules`.
    fn induction_signature(&self) -> Signature {
        Signature {
            instructions: format!(
                "Given a set of examples, extract a list of {} concise and non-redundant natural \
                 language rules that provide clear guidance for performing the task. All rules \
                 should be actionable for a well-specified scope of examples of this general kind \
                 of task.",
                self.num_rules
            ),
            inputs: vec![InField {
                name: "examples_text".to_owned(),
                desc: "Text containing examples".to_owned(),
                ..Default::default()
            }],
            outputs: vec![OutField {
                name: "natural_language_rules".to_owned(),
                desc: "Induced natural language rules".to_owned(),
                ..Default::default()
            }],
        }
    }

    async fn score<S: Module + ?Sized>(&self, student: &S, valset: &[Example]) -> Result<f64> {
        Ok(self
            .scoring
            .apply(crate::evaluate::Evaluate::new(
                valset.to_vec(),
                |inputs| student.forward(inputs),
                crate::evaluate::MetricRef(&self.metric),
            ))
            .run()
            .await?
            .score)
    }
}

/// dspy `update_program_instructions`.
pub(crate) fn with_rules(instructions: &str, rules: &str) -> String {
    format!(
        "{instructions}\n\nPlease adhere to the following rules when making your prediction:\n{rules}"
    )
}

/// dspy's `int(0.5 * len(trainset))` — how much of the trainset the rules are learned from when no
/// validation set is given. An odd length leaves the larger half for scoring.
pub(crate) fn train_size(length: usize) -> usize {
    length / 2
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::Value;

    use super::*;
    use crate::example;

    fn golden() -> Value {
        serde_json::from_str(include_str!(
            "../../tests/conformance/optimize/infer_rules.json"
        ))
        .expect("the infer-rules golden is valid JSON")
    }

    /// The trainset rows the golden rendered, in its order — including one missing its output field and
    /// one whose values are not strings.
    fn demos() -> Vec<Example> {
        golden()["demos"]
            .as_array()
            .expect("demos")
            .iter()
            .map(|row| {
                Example::new(
                    row.as_object()
                        .expect("a row")
                        .iter()
                        .map(|(name, value)| (name.clone(), value.clone())),
                )
                .with_inputs(["question", "hint"])
            })
            .collect()
    }

    fn student() -> crate::predict::Predict {
        crate::predict::Predict::parse("question, hint -> answer")
            .expect("parses")
            .set_lm(Arc::new(crate::DummyLM::new([example! { answer: "x" }])))
    }

    /// The rendering, which is `format_examples` over `get_predictor_demos` — every trainset row
    /// narrowed to the predictor's own fields, in the row's order rather than the signature's.
    #[tokio::test]
    async fn the_examples_are_rendered_the_way_dspy_renders_them() {
        let golden = golden();
        let expected = golden["formatted_examples"].as_str().expect("formatted");
        let mut program = student();
        let rendered = InferRules::new(&|_: &Example, _: &crate::Prediction| 1.0).examples_text(
            &mut program,
            &demos(),
            0,
        );
        assert_eq!(rendered, expected, "the examples text");
        // The third row has no `answer`, and upstream still writes both newlines after the heading.
        assert!(
            expected.contains("Output Fields:\n\n\nInput Fields:"),
            "the golden lost the empty-output row that pins the join"
        );
    }

    /// The sentence the rules go under, and the instruction the model is asked through.
    #[test]
    fn the_rules_are_appended_under_dspys_own_sentence() {
        let golden = golden();
        assert_eq!(
            with_rules(
                golden["instruction_before"].as_str().expect("before"),
                "1. Be brief.\n2. Be exact."
            ),
            golden["instruction_after"].as_str().expect("after"),
            "the appended instruction"
        );
        for (count, expected) in golden["induction_instructions"]
            .as_object()
            .expect("induction_instructions")
        {
            let rules: usize = count.parse().expect("a count");
            let signature = InferRules::new(&|_: &Example, _: &crate::Prediction| 1.0)
                .num_rules(rules)
                .induction_signature();
            assert_eq!(
                signature.instructions,
                expected.as_str().expect("an instruction"),
                "the induction instruction for {rules} rules"
            );
        }
    }

    /// One `random.Random(0)` for the whole compile, so the rollout ids are a single stream rather
    /// than one per call.
    ///
    /// Observed on the wire rather than by re-drawing them here: a test that only compared
    /// `Random::seeded(0)` against the golden would pass just as well against a fresh generator per
    /// induction, which is the bug it exists to catch.
    #[tokio::test]
    async fn the_rollout_ids_are_one_stream_from_seed_zero() {
        let expected: Vec<u64> = golden()["rollout_ids"]
            .as_array()
            .expect("rollout_ids")
            .iter()
            .map(|v| v.as_u64().expect("an id"))
            .collect();

        let writing = Arc::new(Rollouts::default());
        let mut program = student();
        let metric = |_: &Example, _: &crate::Prediction| 1.0;
        let candidates = InferRules::new(&metric)
            .num_candidates(3)
            .compile(&mut program, writing.clone(), &demos(), Some(&demos()))
            .await
            .expect("compiles");

        assert_eq!(candidates.len(), 3, "one candidate per rule set");
        assert_eq!(
            writing.seen(),
            expected[..3],
            "the rollout ids the inductions were asked with"
        );
    }

    /// A model that answers with fixed rules and keeps the rollout id each call carried.
    #[derive(Default)]
    struct Rollouts(std::sync::Mutex<Vec<u64>>);

    impl Rollouts {
        fn seen(&self) -> Vec<u64> {
            self.0.lock().expect("not poisoned").clone()
        }
    }

    impl crate::lm::ChatModel for Rollouts {
        async fn forward(
            &self,
            request: &crate::lm::api::LmRequest,
        ) -> anyhow::Result<crate::lm::api::LmResponse> {
            if let Some(crate::lm::api::RolloutId::Number(id)) = request.config.rollout_id() {
                self.0.lock().expect("not poisoned").push(*id as u64);
            }
            Ok(crate::lm::api::LmResponse::completions([
                "[[ ## reasoning ## ]]\nbecause\n\n[[ ## natural_language_rules ## ]]\n                 1. Answer the capital.\n\n[[ ## completed ## ]]"
                    .to_owned(),
            ]))
        }
    }

    /// With no validation set the trainset is halved and the *first* half is what the rules are learned
    /// from, so an odd length leaves the larger half for scoring.
    #[test]
    fn an_odd_trainset_keeps_the_larger_half_for_scoring() {
        for (length, expected) in golden()["train_size_for_length"]
            .as_object()
            .expect("train_size_for_length")
        {
            let length: usize = length.parse().expect("a length");
            assert_eq!(
                train_size(length),
                expected.as_u64().expect("a size") as usize,
                "the training half of {length}"
            );
        }
    }
}
