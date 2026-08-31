//! The dsrust GEPA adapter (dspy's `DspyAdapter`): the bridge between GEPA's engine and a dsrust program.
//! It builds a program from a candidate's instructions, evaluates it on a batch (capturing traces for
//! reflection), assembles the reflective dataset from those traces, and calls the reflection model to
//! rewrite an instruction — the LLM work the [`gepa`] engine drives through [`gepa::GepaAdapter`].

use std::collections::BTreeMap;
use std::sync::Arc;

mod evaluating;

use super::binding::set_instructions;
use super::reflecting::{
    Captured, code_reflective_records, reflective_records, rendered_inputs, rendered_map,
    unparsed_feedback, unparsed_outputs,
};
use gepa::{
    Candidate, EvalBatch, GepaAdapter, Reflective, ReflectiveSample, extract_new_instruction,
    render_prompt,
};

use super::metric::{Feedback, MetricContext};
use crate::example::{Example, Prediction};
use crate::lm::DynChatModel;
use crate::lm::api::{LmMessage, LmPart, LmRequest};
use crate::module::{Module, TraceStep};
use pyrng::Random;

/// The reflection request's model field. A scripted reflection model ignores it; a production model
/// wrapper supplies its own model id, so this is only a label.
const REFLECTION_MODEL: &str = "gepa-reflection";

/// dspy `build_program`: set each named predictor's instruction from the candidate. Shared by the
/// adapter's evaluation and by the optimizer applying the winning candidate.
/// dspy's `DspyAdapter`, over one student program mutated in place per candidate. Generic over the
/// student type so no `dyn` coercion is needed; the engine drives it sequentially, so borrowing the
/// student for each evaluation is sound.
pub(super) struct Adapter<'a, S: Module + ?Sized, M> {
    student: &'a mut S,
    metric: &'a M,
    reflection: Arc<dyn DynChatModel>,
    trainset: &'a [Example],
    valset: &'a [Example],
    failure_score: f64,
    captured: Vec<Captured>,
    /// Whether the student holds a `Flex`, which decides how a metric is called while scoring.
    ///
    /// Read once when the adapter is built rather than per example: the walk is cheap but it takes
    /// `&mut`, and scoring holds the student immutably.
    has_flexes: bool,
    /// dspy `num_threads`: how many examples one evaluation runs at once. One at a time by default,
    /// which is upstream's `num_threads=None`.
    num_threads: usize,
    /// dspy `instruction_proposer`: a caller's own proposal step, replacing the reflection tree
    /// entirely when set. See [`InstructionProposer`](super::InstructionProposer).
    proposer: Option<Arc<dyn super::InstructionProposer>>,
    /// dspy `track_best_outputs`: whether a scoring run keeps each example's prediction so the
    /// engine can report what the best programs answered. Off unless the caller asked, because it
    /// clones every prediction on every valset evaluation.
    track_best_outputs: bool,
    /// dspy `add_format_failure_as_feedback`: whether a step whose completion would not parse may
    /// be reflected on. Off, as upstream's default is — with it off such a step is filtered out of
    /// every reflective dataset, and an example holding nothing else drops out with it.
    add_format_failure_as_feedback: bool,
    /// dspy's `DspyAdapter.rng`, which picks which trace instance a reflection reads. A *second*
    /// generator from the same seed, not the engine's: `gepa.py` builds `random.Random(self.seed)`
    /// for the adapter and passes `optimize(seed=self.seed)` separately, so a draw here cannot
    /// move the engine's stream.
    rng: Random,
}

/// What the caller settles before a run and never changes during it.
///
/// Named rather than passed one by one because the four together are what a reader has to hold to
/// know how a run will behave, and `new`'s other five are the program and its data.
pub(super) struct Settings {
    pub failure_score: f64,
    pub num_threads: usize,
    pub proposer: Option<Arc<dyn super::InstructionProposer>>,
    pub seed: u64,
}

impl<'a, S: Module + ?Sized, M> Adapter<'a, S, M> {
    pub(super) fn new(
        student: &'a mut S,
        metric: &'a M,
        reflection: Arc<dyn DynChatModel>,
        trainset: &'a [Example],
        valset: &'a [Example],
        settings: Settings,
    ) -> Self {
        let Settings {
            failure_score,
            num_threads,
            proposer,
            seed,
        } = settings;
        let has_flexes = !student.named_flexes().is_empty();
        Self {
            student,
            metric,
            reflection,
            trainset,
            valset,
            failure_score,
            captured: Vec::new(),
            has_flexes,
            num_threads,
            proposer,
            track_best_outputs: false,
            add_format_failure_as_feedback: false,
            rng: Random::seeded(seed),
        }
    }

    /// gepa's `add_format_failure_as_feedback`.
    pub(super) fn reflecting_on_format_failures(mut self, add: bool) -> Self {
        self.add_format_failure_as_feedback = add;
        self
    }

    /// Keep each example's prediction while scoring — gepa's `track_best_outputs`.
    ///
    /// A builder rather than another argument to [`new`](Self::new), for the reason
    /// [`Settings`] exists.
    pub(super) fn tracking_outputs(mut self, track: bool) -> Self {
        self.track_best_outputs = track;
        self
    }
}

impl<S, M> Adapter<'_, S, M>
where
    S: Module + ?Sized,
    M: Fn(&Example, &Prediction, &MetricContext<'_>) -> Feedback + Send + Sync,
{
    /// dspy `make_reflective_dataset` for one predictor: an `Inputs`/`Generated Outputs`/`Feedback`
    /// record per captured example whose trace touched this predictor. The predictor's inputs and
    /// outputs render the way dspy's `str(value)` does ([`Example::rendered`]).
    ///
    /// `signature` is the component's own, and the trace is matched against it rather than against
    /// the component's *name*: upstream filters on `t[0].signature.equals(module.signature)`, so
    /// two predictors sharing a signature and an instruction pool their instances together. That is
    /// reachable at the seed candidate, where two identically-declared `Predict`s start from the
    /// same instruction — measured against dspy, every seed then has at least one component
    /// reflecting on a step belonging to the other.
    fn reflective_dataset(
        &mut self,
        predictor: &str,
        signature: &crate::signature::Signature,
    ) -> Vec<ReflectiveSample> {
        let mut samples = Vec::new();
        for captured in &self.captured {
            // dspy draws one of the instances rather than taking the first, and draws even when
            // there is only one — so an example whose trace touched the predictor once still
            // advances the generator, and the next example's draw lands where upstream's does.
            let instances: Vec<&TraceStep> = captured
                .trace
                .iter()
                .filter(|step| step.signature.equals(signature))
                // dspy's `add_format_failure_as_feedback`: with the flag off, a step whose
                // completion would not parse is dropped here, and an example with nothing else
                // for this predictor drops out with it.
                .filter(|step| {
                    self.add_format_failure_as_feedback || step.outputs.answered().is_some()
                })
                .collect();
            if instances.is_empty() {
                continue;
            }
            // A failure is *preferred* and short-circuits the draw — upstream takes the first one
            // and never calls `rng.choice`, so the generator does not advance for this example and
            // the next one's draw lands where upstream's does.
            let failed = instances
                .iter()
                .find(|step| step.outputs.failure().is_some());
            let step = match failed {
                Some(step) => step,
                None => {
                    // An example whose *program* answer failed to parse contributes nothing once
                    // no failing step was selected for this predictor.
                    if captured.unparsed.is_some() {
                        continue;
                    }
                    instances[self.rng.choice_index(instances.len())]
                }
            };
            // dspy calls the metric a second time here, with the predictor it wants feedback for
            // and the instance it just drew, and keeps only the text — the score stays the one
            // scoring produced, which is what upstream's `fb["score"] = module_score` does.
            let reflecting = MetricContext {
                trace: Some(&captured.trace),
                predictor: Some(predictor),
                predictor_step: Some(step),
                // Reflection is not scoring: upstream fills `program_trace` only at scoring time.
                program_trace: None,
            };
            let feedback =
                (self.metric)(&captured.example, &captured.prediction, &reflecting).text();
            let (outputs, feedback) = match step.outputs.failure() {
                // A failure is not scored a second time: upstream replaces the metric's feedback
                // with the structure the model should have followed, and never calls it.
                Some(failed) => (
                    unparsed_outputs(&failed.completion_text),
                    unparsed_feedback(signature),
                ),
                None => (
                    rendered_map(step.outputs.answered().expect("not a failure")),
                    feedback,
                ),
            };
            samples.push(vec![
                (
                    "Inputs".to_owned(),
                    rendered_inputs(&step.inputs, signature),
                ),
                ("Generated Outputs".to_owned(), outputs),
                ("Feedback".to_owned(), Reflective::Text(feedback)),
            ]);
        }
        samples
    }

    /// Ask the code proposer for a new module source for each `Flex` component.
    ///
    /// Its reflective records are the same for every code component — whole-program I/O is not
    /// per-component — so they are built once and shown to each.
    async fn propose_code_for(&mut self, code: &[String], candidate: &Candidate) -> Candidate {
        let records = code_reflective_records(&self.captured);
        if records.is_empty() {
            return Candidate::new();
        }
        let failures: BTreeMap<String, Vec<serde_json::Map<String, serde_json::Value>>> = code
            .iter()
            .map(|name| (name.clone(), reflective_records(&records)))
            .collect();
        let current: BTreeMap<String, String> = candidate
            .iter()
            .map(|(name, text)| (name.clone(), text.clone()))
            .collect();
        let (described, contexts) = crate::predict::flex::proposal::flex_task_context(self.student);
        let proposed = crate::predict::flex::proposal::propose_code(
            code,
            &current,
            &failures,
            &described,
            &contexts,
            crate::predict::flex::proposal::PRIMITIVES_CATALOG,
            self.reflection.clone(),
        )
        .await;
        proposed.into_iter().collect()
    }

    /// Call the reflection model with the rendered prompt and return its raw completion, from which
    /// [`extract_new_instruction`] pulls the fenced instruction.
    async fn reflect(&self, prompt: &str) -> Option<String> {
        let request = LmRequest::new(
            REFLECTION_MODEL,
            vec![LmMessage::user(vec![LmPart::text(prompt)])],
        );
        let response = self.reflection.forward_dyn(&request).await.ok()?;
        Some(response.first_text())
    }
}

impl<S, M> GepaAdapter for Adapter<'_, S, M>
where
    S: Module + ?Sized + Send,
    M: Fn(&Example, &Prediction, &MetricContext<'_>) -> Feedback + Send + Sync,
{
    /// What the student answered, which is what a caller tracking best outputs wants back.
    type Output = Prediction;

    async fn evaluate_minibatch(
        &mut self,
        ids: &[usize],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> EvalBatch<Prediction> {
        let examples: Vec<Example> = ids.iter().map(|&id| self.trainset[id].clone()).collect();
        self.evaluate(&examples, candidate, capture_traces).await
    }

    async fn evaluate_valset(&mut self, candidate: &Candidate) -> EvalBatch<Prediction> {
        let examples = self.valset.to_vec();
        self.evaluate(&examples, candidate, false).await
    }

    async fn evaluate_valset_ids(
        &mut self,
        ids: &[usize],
        candidate: &Candidate,
    ) -> EvalBatch<Prediction> {
        let examples: Vec<Example> = ids.iter().map(|&id| self.valset[id].clone()).collect();
        self.evaluate(&examples, candidate, false).await
    }

    /// dspy's reflective proposer step: for each component, build its reflective dataset, render the
    /// reflection prompt, and rewrite it from the reflection model's reply. A component with no
    /// reflective examples, or whose reflection call fails, is left unchanged (skipped).
    async fn propose_new_texts(
        &mut self,
        candidate: &Candidate,
        components: &[String],
        _captured: &EvalBatch<Prediction>,
    ) -> Candidate {
        // A component whose runs produced nothing is left out of both paths: upstream skips it
        // rather than proposing against an empty dataset, and a caller's proposer is not handed one
        // it cannot use either.
        // dspy's `make_reflective_dataset` opens with `build_program(candidate)`, so the signature
        // a trace is matched against carries *this candidate's* instruction rather than whatever
        // the student last ran — and since `equals` compares instructions first, that is what
        // decides whether two components pool their instances.
        set_instructions(self.student, candidate);
        let signatures: BTreeMap<String, crate::signature::Signature> = self
            .student
            .named_predictors()
            .into_iter()
            .map(|predictor| (predictor.name.clone(), predictor.signature.clone()))
            .collect();
        // A `Flex`'s component is source, not an instruction, and the two are proposed differently:
        // a predictor reflects on one drawn step and is asked for an instruction, a Flex reflects on
        // whole-program I/O and is asked for a whole module.
        //
        // A code component drops out of the map below on its own — it has no signature, because it
        // is not a predictor — so what was missing was never a filter but the *branch*: nothing
        // proposed for the components that fell out. A control confirmed the filter was inert.
        let code: Vec<String> = self
            .student
            .named_flexes()
            .into_iter()
            .map(|named| named.name)
            .filter(|name| components.contains(name))
            .collect();
        let datasets: BTreeMap<String, super::ReflectiveDataset> = components
            .iter()
            .filter_map(|name| {
                let signature = signatures.get(name)?;
                Some((name.clone(), self.reflective_dataset(name, signature)))
            })
            .filter(|(_, dataset)| !dataset.is_empty())
            .collect();

        // dspy: when a custom proposer is given it "overrides everything" — the reflection tree is
        // not consulted at all, only the components it was asked about.
        if let Some(proposer) = &self.proposer {
            let asked: Vec<String> = datasets.keys().cloned().collect();
            return proposer
                .propose(&self.reflection, candidate, &asked, &datasets)
                .await;
        }

        let mut new_texts = Candidate::new();
        for (name, dataset) in &datasets {
            let prompt = render_prompt(&candidate[name], dataset, None);
            if let Some(raw) = self.reflect(&prompt).await {
                new_texts.insert(name.clone(), extract_new_instruction(&raw));
            }
        }
        if !code.is_empty() {
            new_texts.extend(self.propose_code_for(&code, candidate).await);
        }
        new_texts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example;

    /// The reflective dataset is built from the captured traces, one record per example whose
    /// trace touched the predictor — not a default-shaped placeholder, which is what the
    /// replacement mutant handed the reflection prompt while every test stayed green.
    #[test]
    fn the_reflective_dataset_carries_the_captured_trace() {
        let mut student =
            crate::optimize::scripted::Solver::new(crate::optimize::scripted::Answers::Correctly);
        let metric = |_: &Example, _: &crate::Prediction, _: &MetricContext<'_>| {
            super::super::metric::Feedback::new(1.0, "right city")
        };
        let trainset = crate::optimize::scripted::trainset();
        let mut adapter = Adapter::new(
            &mut student,
            &metric,
            std::sync::Arc::new(crate::DummyLM::new([])),
            &trainset,
            &trainset,
            Settings {
                failure_score: 0.0,
                num_threads: 1,
                proposer: None,
                seed: 0,
            },
        );
        adapter.captured = vec![Captured {
            example: example! { question: "capital of France?" },
            prediction: crate::Prediction::new(example! { answer: "Paris" }, "raw"),
            trace: vec![crate::TraceStep {
                predictor: "self".to_owned(),
                inputs: example! { question: "capital of France?" },
                outputs: crate::StepOutputs::Answered(example! { answer: "Paris" }),
                signature: qa_signature("Answer."),
            }],
            scored: Feedback::score_only(1.0),
            unparsed: None,
        }];

        let dataset = adapter.reflective_dataset("self", &qa_signature("Answer."));
        assert_eq!(dataset.len(), 1, "one captured example touched `self`");
        let sample = &dataset[0];
        let rendered = render_prompt("", std::slice::from_ref(sample), None);
        assert!(rendered.contains("capital of France?"), "{rendered}");
        assert!(rendered.contains("right city"), "{rendered}");
        assert!(
            adapter
                .reflective_dataset("someone_else", &qa_signature("Something else."))
                .is_empty(),
            "a signature nothing traced gets no records"
        );
    }

    /// `make_reflective_dataset` against dspy's own, over a trace that repeats a predictor.
    ///
    /// The golden is generated by driving the real `DspyAdapter`
    /// (`scripts/generate_gepa_reflective_fixture.py`), which nothing did before: the engine
    /// fixture drives a *stub* adapter returning a canned dataset, because what it measures is the
    /// engine around this one.
    ///
    /// What the repeat buys: dspy picks the instance to reflect on with
    /// `rng.choice(trace_instances)`, so a port taking the first agrees only at the seeds where the
    /// draw happens to land there. It lands on the second hop at seed 0 and the first at 1-3 — a
    /// fixture at one seed would have agreed with a port that never draws, three times in four.
    ///
    /// `Feedback` is compared too, which is what the metric's context buys: upstream's text names
    /// the hop that was drawn, so it is the strictest witness that the *same* instance was picked —
    /// two records can share inputs and still disagree about which step the feedback describes.
    #[test]
    fn reflects_on_the_instance_dspy_draws() {
        let golden = golden();
        let answers = &golden["answers"];
        let questions: Vec<&str> = golden["questions"]
            .as_array()
            .expect("questions")
            .iter()
            .map(|q| q.as_str().expect("a question"))
            .collect();
        let cases = golden["cases"].as_array().expect("cases");
        let drawn: std::collections::BTreeSet<String> = cases
            .iter()
            .map(|case| case["records"][0]["Inputs"]["question"].to_string())
            .collect();
        assert!(
            drawn.len() >= 2,
            "every seed reflected on the same instance, so the draw is not exercised: {drawn:?}"
        );

        for case in cases {
            let seed = case["seed"].as_u64().expect("a seed");
            let mut student = crate::optimize::scripted::Solver::new(
                crate::optimize::scripted::Answers::Correctly,
            );
            // dspy's fixture metric: feedback naming the instance GEPA drew, which is exactly
            // what the five-argument call makes writable.
            let metric = |_: &Example, _: &crate::Prediction, ctx: &MetricContext<'_>| {
                let seen = ctx
                    .predictor_step
                    .and_then(|step| step.inputs.get("question"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("-");
                super::super::metric::Feedback::new(1.0, format!("reflected on {seen}"))
            };
            let trainset = crate::optimize::scripted::trainset();
            let mut adapter = Adapter::new(
                &mut student,
                &metric,
                std::sync::Arc::new(crate::DummyLM::new([])),
                &trainset,
                &trainset,
                Settings {
                    failure_score: 0.0,
                    num_threads: 1,
                    proposer: None,
                    seed,
                },
            );
            let instruction = golden["instruction"].as_str().expect("the instruction");
            adapter.captured = questions
                .iter()
                .map(|q| two_hops(q, answers, instruction))
                .collect();

            let dataset = adapter.reflective_dataset("step", &qa_signature(instruction));
            let records = case["records"].as_array().expect("records");
            assert_eq!(dataset.len(), records.len(), "seed {seed}: record count");
            for (sample, record) in dataset.iter().zip(records) {
                for section in ["Inputs", "Generated Outputs"] {
                    assert_eq!(
                        rendered_section(sample, section),
                        golden_section(record, section),
                        "seed {seed}: {section} — a different trace instance was reflected on"
                    );
                }
                assert_eq!(
                    text_section(sample, "Feedback"),
                    record["Feedback"].as_str().expect("upstream's feedback"),
                    "seed {seed}: Feedback"
                );
            }
        }
    }

    fn golden() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../tests/conformance/optimize/gepa_reflective.json"
        ))
        .expect("the reflective golden parses")
    }

    /// The student a fixture-driven test hands the adapter: only its `named_predictors` matter,
    /// and these tests set `captured` directly rather than running it.
    fn student() -> crate::optimize::scripted::Solver {
        crate::optimize::scripted::Solver::new(crate::optimize::scripted::Answers::Correctly)
    }

    /// dspy's fixture metric: feedback naming the instance GEPA drew, which is what the
    /// five-argument call makes writable and what makes the draw visible in the record.
    fn metric_naming_the_drawn_step()
    -> impl Fn(&Example, &crate::Prediction, &MetricContext<'_>) -> super::super::metric::Feedback
    {
        |_: &Example, _: &crate::Prediction, ctx: &MetricContext<'_>| {
            let seen = ctx
                .predictor_step
                .and_then(|step| step.inputs.get("question"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("-");
            super::super::metric::Feedback::new(1.0, format!("reflected on {seen}"))
        }
    }

    fn adapter_over<'a, M>(
        student: &'a mut crate::optimize::scripted::Solver,
        metric: &'a M,
        seed: u64,
    ) -> Adapter<'a, crate::optimize::scripted::Solver, M> {
        Adapter::new(
            student,
            metric,
            std::sync::Arc::new(crate::DummyLM::new([])),
            &[],
            &[],
            Settings {
                failure_score: 0.0,
                num_threads: 1,
                proposer: None,
                seed,
            },
        )
    }

    /// `question -> answer` under one instruction, which is the shape both fixture programs
    /// declare. Built once so a step's signature and the one it is matched against agree for the
    /// same reason dspy's do — they came from the same declaration.
    fn qa_signature(instruction: &str) -> crate::Signature {
        crate::Signature {
            instructions: instruction.to_owned(),
            inputs: vec![crate::signature::InField {
                name: "question".to_owned(),
                ..Default::default()
            }],
            outputs: vec![crate::signature::OutField {
                name: "answer".to_owned(),
                ..Default::default()
            }],
        }
    }

    /// The two hops one example produces, both named `step` — dspy's `TwoHop`, whose single
    /// `Predict` is called twice, so one component owns two trace instances.
    fn two_hops(question: &str, answers: &serde_json::Value, instruction: &str) -> Captured {
        let answer_to = |q: &str| {
            answers[q]
                .as_str()
                .unwrap_or_else(|| panic!("the fixture's table answers {q:?}"))
                .to_owned()
        };
        let first = answer_to(question);
        let second = answer_to(&first);
        Captured {
            example: example! { question: question },
            prediction: crate::Prediction::new(example! { answer: second.clone() }, "raw"),
            trace: vec![
                crate::TraceStep {
                    predictor: "step".to_owned(),
                    inputs: example! { question: question },
                    outputs: crate::StepOutputs::Answered(example! { answer: first.clone() }),
                    signature: qa_signature(instruction),
                },
                crate::TraceStep {
                    predictor: "step".to_owned(),
                    inputs: example! { question: first },
                    outputs: crate::StepOutputs::Answered(example! { answer: second }),
                    signature: qa_signature(instruction),
                },
            ],
            scored: Feedback::score_only(1.0),
            unparsed: None,
        }
    }

    /// A `History` input is hoisted out of the field map into a fenced `Context` block, and its
    /// own key does not appear — dspy's, whose fence says json around Python dict reprs.
    #[test]
    fn a_history_input_is_hoisted_into_context() {
        let golden = golden();
        let history = &golden["history"];
        let messages: Vec<serde_json::Value> =
            history["messages"].as_array().expect("messages").to_vec();
        let question = history["question"].as_str().expect("a question");
        let answer = history["answer"].as_str().expect("an answer");

        let mut signature = qa_signature(golden["instruction"].as_str().expect("instruction"));
        signature.inputs.push(crate::signature::InField {
            name: "history".to_owned(),
            kind: crate::signature::FieldKind::Json(crate::signature::JsonType::plain("History")),
            ..Default::default()
        });

        let mut inputs = example! { question: question };
        inputs.set("history", serde_json::json!({ "messages": messages }));
        let (mut student, metric) = (student(), metric_naming_the_drawn_step());
        let mut adapter = adapter_over(&mut student, &metric, 0);
        adapter.captured = vec![Captured {
            example: inputs.clone(),
            prediction: crate::Prediction::new(example! { answer: answer }, "raw"),
            trace: vec![crate::TraceStep {
                predictor: "step".to_owned(),
                inputs,
                outputs: crate::StepOutputs::Answered(example! { answer: answer }),
                signature: signature.clone(),
            }],
            scored: Feedback::score_only(1.0),
            unparsed: None,
        }];

        let dataset = adapter.reflective_dataset("step", &signature);
        let records = history["records"].as_array().expect("records");
        assert_eq!(dataset.len(), records.len());
        assert_eq!(
            rendered_section(&dataset[0], "Inputs"),
            golden_section(&records[0], "Inputs"),
            "the history field is dropped and Context takes its place, first"
        );
    }

    /// Two predictors declaring the same signature pool their trace instances once their
    /// instructions agree, which is what the seed candidate makes true — so a component can be
    /// handed a step belonging to the other. Matching by the predictor's *name* never does that.
    #[test]
    fn predictors_sharing_a_signature_pool_their_instances() {
        let golden = golden();
        let shared = &golden["shared_signature"];
        let question = shared["question"].as_str().expect("a question");
        let answers = &shared["answers"];
        let instruction = golden["instruction"].as_str().expect("instruction");
        let signature = qa_signature(instruction);
        let middle = answers[question].as_str().expect("the first answer");
        let end = answers[middle].as_str().expect("the second answer");

        for case in shared["cases"].as_array().expect("cases") {
            let seed = case["seed"].as_u64().expect("a seed");
            let (mut student, metric) = (student(), metric_naming_the_drawn_step());
            let mut adapter = adapter_over(&mut student, &metric, seed);
            adapter.captured = vec![Captured {
                example: example! { question: question },
                prediction: crate::Prediction::new(example! { answer: end }, "raw"),
                trace: vec![
                    crate::TraceStep {
                        predictor: "alpha".to_owned(),
                        inputs: example! { question: question },
                        outputs: crate::StepOutputs::Answered(example! { answer: middle }),
                        signature: signature.clone(),
                    },
                    crate::TraceStep {
                        predictor: "beta".to_owned(),
                        inputs: example! { question: middle },
                        outputs: crate::StepOutputs::Answered(example! { answer: end }),
                        signature: signature.clone(),
                    },
                ],
                scored: Feedback::score_only(1.0),
                unparsed: None,
            }];

            // Upstream walks `components_to_update` in order off one generator, so the order the
            // two datasets are built in is part of what is being compared.
            for name in ["alpha", "beta"] {
                let dataset = adapter.reflective_dataset(name, &signature);
                let expected = case["components"][name].as_array().expect("records");
                assert_eq!(dataset.len(), expected.len(), "seed {seed}: {name} count");
                for (sample, record) in dataset.iter().zip(expected) {
                    assert_eq!(
                        rendered_section(sample, "Inputs"),
                        golden_section(record, "Inputs"),
                        "seed {seed}: {name} reflected on a different instance"
                    );
                }
            }
        }
    }

    fn rendered_section(sample: &ReflectiveSample, section: &str) -> Vec<(String, String)> {
        let (_, value) = sample
            .iter()
            .find(|(name, _)| name == section)
            .unwrap_or_else(|| panic!("the record has a {section} section"));
        let Reflective::Map(entries) = value else {
            panic!("{section} is not a map")
        };
        entries
            .iter()
            .map(|(name, value)| match value {
                Reflective::Text(text) => (name.clone(), text.clone()),
                _ => panic!("{section}.{name} is not text"),
            })
            .collect()
    }

    fn text_section<'a>(sample: &'a ReflectiveSample, section: &str) -> &'a str {
        let (_, value) = sample
            .iter()
            .find(|(name, _)| name == section)
            .unwrap_or_else(|| panic!("the record has a {section} section"));
        match value {
            Reflective::Text(text) => text,
            _ => panic!("{section} is not text"),
        }
    }

    fn golden_section(record: &serde_json::Value, section: &str) -> Vec<(String, String)> {
        record[section]
            .as_object()
            .unwrap_or_else(|| panic!("the golden record has a {section} object"))
            .iter()
            .map(|(name, value)| {
                (
                    name.clone(),
                    value.as_str().expect("a rendered value").to_owned(),
                )
            })
            .collect()
    }
}

/// What GEPA does with a completion no adapter could read, against dspy's own run.
///
/// `optimize/failed_parse.json` was generated by calling `bootstrap_trace_data` — the collector
/// upstream's GEPA reaches for — and recording both of its arms. They differ in more than a
/// number: one keeps the example as a `FailedPrediction`, the other drops it from the batch.
#[cfg(test)]
mod failed_parse_tests {
    use super::*;
    use crate::example;
    use crate::optimize::scripted::Unparsed;

    fn golden() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../tests/conformance/optimize/failed_parse.json"
        ))
        .expect("the failed-parse golden is valid JSON")
    }

    fn batch(size: usize) -> Vec<Example> {
        (0..size)
            .map(|i| example! { question: format!("q{i}") }.with_inputs(["question"]))
            .collect()
    }

    fn scoring() -> impl Fn(&Example, &Prediction, &MetricContext<'_>) -> Feedback {
        |_: &Example, _: &Prediction, _: &MetricContext<'_>| Feedback::score_only(1.0)
    }

    async fn traced<M>(
        student: &mut Unparsed,
        metric: &M,
        size: usize,
        failure_score: f64,
    ) -> Vec<f64>
    where
        M: Fn(&Example, &Prediction, &MetricContext<'_>) -> Feedback + Send + Sync,
    {
        let examples = batch(size);
        let mut adapter = Adapter::new(
            student,
            metric,
            std::sync::Arc::new(crate::DummyLM::new([])),
            &[],
            &[],
            Settings {
                failure_score,
                num_threads: 1,
                proposer: None,
                seed: 0,
            },
        );
        adapter
            .evaluate(&examples, &Candidate::new(), true)
            .await
            .scores
    }

    /// Every example whose completion parsed *nothing* survives, each scoring the format reward.
    #[tokio::test]
    async fn a_completion_reading_as_nothing_stays_in_the_batch() {
        let arm = golden()["arms"]["no_declared_field_parsed"].clone();
        let expected: Vec<f64> = arm["trajectories"]
            .as_array()
            .expect("trajectories")
            .iter()
            .map(|row| row["score"].as_f64().expect("a score"))
            .collect();
        assert_eq!(
            arm["kept"], arm["batch_size"],
            "dspy kept every example here"
        );

        let mut student = Unparsed::reading_nothing("there is nothing structured here");
        let metric = scoring();
        let scores = traced(
            &mut student,
            &metric,
            arm["batch_size"].as_u64().expect("a size") as usize,
            arm["format_failure_score"].as_f64().expect("a score"),
        )
        .await;
        assert_eq!(
            scores, expected,
            "one score per example, all the format reward"
        );
    }

    /// And every example whose completion parsed *something* is dropped, because upstream's graded
    /// arm raises before it can build a reward.
    #[tokio::test]
    async fn a_completion_reading_as_something_is_dropped() {
        let arm = golden()["arms"]["some_declared_field_parsed"].clone();
        assert_eq!(arm["kept"].as_u64(), Some(0), "dspy kept none of them");

        let mut student = Unparsed::reading_one_field(r#"{"answer": "half"}"#);
        let metric = scoring();
        let scores = traced(
            &mut student,
            &metric,
            arm["batch_size"].as_u64().expect("a size") as usize,
            arm["format_failure_score"].as_f64().expect("a score"),
        )
        .await;
        assert!(
            scores.is_empty(),
            "the batch came back with {} score(s), not none",
            scores.len()
        );
    }

    /// Python's `or`, not a null check: a reward of exactly zero is falsy and takes the fallback.
    #[test]
    fn a_zero_format_reward_falls_back_to_the_constant() {
        let arm = golden()["arms"]["a_zero_reward_falls_back"].clone();
        let failed = crate::FailedPrediction {
            completion_text: "unreadable".to_owned(),
            format_reward: Some(0.0),
        };
        assert_eq!(
            failed.score(arm["format_failure_score"].as_f64().expect("a score")),
            arm["trajectories"][0]["score"].as_f64().expect("a score"),
            "a zero reward is discarded for `format_failure_score`"
        );
        assert_eq!(
            failed.score(0.75),
            0.75,
            "and `unwrap_or` would have kept the zero"
        );
    }

    /// The record a reflection model reads: the raw completion, and the structure it should have
    /// followed. Both byte-for-byte, because both are prompt.
    #[tokio::test]
    async fn the_reflective_record_shows_the_raw_completion_and_the_structure() {
        let recorded = golden()["reflective"]
            .as_array()
            .expect("reflective")
            .iter()
            .find(|run| run["add_format_failure_as_feedback"] == serde_json::Value::Bool(true))
            .expect("the flag-on run")
            .clone();
        let record = &recorded["records"][0];

        let dataset = dataset_over(true).await;
        assert_eq!(dataset.len(), 2, "both failing examples produced a record");
        let sample = &dataset[0];
        assert_eq!(
            text_of(sample, "Generated Outputs"),
            record["Generated Outputs"].as_str().expect("a string"),
            "the raw-response block"
        );
        assert_eq!(
            text_of(sample, "Feedback"),
            record["Feedback"].as_str().expect("a string"),
            "the structure instruction"
        );
    }

    /// With the flag off every failing step is filtered out, and nothing is left to reflect on —
    /// which upstream refuses rather than proposing from nothing.
    #[tokio::test]
    async fn with_the_flag_off_no_record_survives() {
        let recorded = golden()["reflective"]
            .as_array()
            .expect("reflective")
            .iter()
            .find(|run| run["add_format_failure_as_feedback"] == serde_json::Value::Bool(false))
            .expect("the flag-off run")
            .clone();
        assert!(
            recorded["records"].is_null(),
            "dspy produced no records at all"
        );
        assert_eq!(
            recorded["raises"]["message"].as_str(),
            Some("No valid predictions found for any module."),
            "and refused rather than guessing"
        );
        assert!(
            dataset_over(false).await.is_empty(),
            "the flag is what lets a failure through"
        );
    }

    async fn dataset_over(add_format_failure_as_feedback: bool) -> Vec<ReflectiveSample> {
        let mut student = Unparsed::reading_nothing("there is nothing structured here");
        let metric = scoring();
        let examples = batch(2);
        let signature = student.named_predictors()[0].signature.clone();
        let mut adapter = Adapter::new(
            &mut student,
            &metric,
            std::sync::Arc::new(crate::DummyLM::new([])),
            &[],
            &[],
            Settings {
                failure_score: 0.0,
                num_threads: 1,
                proposer: None,
                seed: 0,
            },
        )
        .reflecting_on_format_failures(add_format_failure_as_feedback);
        adapter.evaluate(&examples, &Candidate::new(), true).await;
        adapter.reflective_dataset("p", &signature)
    }

    fn text_of(sample: &ReflectiveSample, key: &str) -> String {
        match sample.iter().find(|(name, _)| name == key).map(|(_, v)| v) {
            Some(Reflective::Text(text)) => text.clone(),
            _ => panic!("{key} is not rendered as text"),
        }
    }
}

/// GEPA on a program that implements no `forward_traced` of its own.
///
/// The reflective dataset is built by filtering a run's trace to the predictor being reflected on,
/// so a program that recorded nothing produced an empty dataset for every predictor — GEPA would
/// run the whole trainset, propose nothing, and report a search that improved nothing. Nothing said
/// why, and nothing failed.
#[cfg(test)]
mod untraced_student {
    use super::*;

    use crate::signature::{FieldKind, InField, OutField, Signature};
    use crate::{DummyLM, NamedPredictor, Predict, Prediction, example};

    fn signature(input: &str, output: &str, instructions: &str) -> Signature {
        Signature {
            instructions: instructions.to_owned(),
            inputs: vec![InField {
                name: input.to_owned(),
                kind: FieldKind::Str,
                ..Default::default()
            }],
            outputs: vec![OutField {
                name: output.to_owned(),
                kind: FieldKind::Str,
                ..Default::default()
            }],
        }
    }

    /// Two predictors, reached through plain `forward` — what `#[derive(Module)]` produces.
    struct Pipeline {
        drafting: Predict,
        polishing: Predict,
    }

    impl Module for Pipeline {
        fn forward<'a>(
            &'a self,
            inputs: Example,
        ) -> std::pin::Pin<Box<dyn Future<Output = anyhow::Result<Prediction>> + Send + 'a>>
        {
            Box::pin(async move {
                let drafted = self.drafting.forward(inputs).await?;
                let mut next = Example::default();
                next.set("note", drafted.get("note").cloned().unwrap_or_default());
                self.polishing.forward(next).await
            })
        }

        fn named_predictors(&mut self) -> Vec<NamedPredictor<'_>> {
            let mut all = Vec::new();
            for (name, predictor) in [
                ("drafting", &mut self.drafting),
                ("polishing", &mut self.polishing),
            ] {
                for mut inner in predictor.named_predictors() {
                    inner.name = name.to_owned();
                    all.push(inner);
                }
            }
            all
        }
    }

    #[tokio::test]
    async fn each_predictor_has_something_to_reflect_on() {
        let answering = |field: &'static str, value: &'static str| {
            std::sync::Arc::new(DummyLM::new(std::iter::repeat_n(
                Example::new([(field, serde_json::json!(value))]),
                4,
            ))) as Arc<dyn DynChatModel>
        };
        let mut student = Pipeline {
            drafting: Predict::from_signature(signature("question", "note", "Draft."))
                .set_lm(answering("note", "a note")),
            polishing: Predict::from_signature(signature("note", "answer", "Polish."))
                .set_lm(answering("answer", "Paris")),
        };
        let metric =
            |_: &Example, _: &crate::Prediction, _: &MetricContext<'_>| Feedback::score_only(1.0);
        let examples = [example! { question: "capital of France?" }.with_inputs(["question"])];
        let mut adapter = Adapter::new(
            &mut student,
            &metric,
            Arc::new(DummyLM::new([])),
            &examples,
            &[],
            Settings {
                failure_score: 0.0,
                num_threads: 1,
                proposer: None,
                seed: 0,
            },
        );

        let candidate: Candidate = [
            ("drafting".to_owned(), "Draft.".to_owned()),
            ("polishing".to_owned(), "Polish.".to_owned()),
        ]
        .into_iter()
        .collect();
        adapter.evaluate(&examples, &candidate, true).await;

        for (name, input, output, instructions) in [
            ("drafting", "question", "note", "Draft."),
            ("polishing", "note", "answer", "Polish."),
        ] {
            let dataset = adapter.reflective_dataset(name, &signature(input, output, instructions));
            assert!(
                !dataset.is_empty(),
                "{name} had nothing to reflect on, so GEPA would propose nothing for it"
            );
        }
    }
}
