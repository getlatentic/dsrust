//! The dsrust GEPA adapter (dspy's `DspyAdapter`): the bridge between GEPA's engine and a dsrust program.
//! It builds a program from a candidate's instructions, evaluates it on a batch (capturing traces for
//! reflection), assembles the reflective dataset from those traces, and calls the reflection model to
//! rewrite an instruction — the LLM work the [`gepa`] engine drives through [`gepa::GepaAdapter`].

use futures_util::StreamExt;
use std::collections::BTreeMap;
use std::sync::Arc;

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
pub(super) fn set_instructions<S: Module + ?Sized>(student: &mut S, candidate: &Candidate) {
    for predictor in student.named_predictors() {
        if let Some(instruction) = candidate.get(&predictor.name) {
            predictor.signature.instructions = instruction.clone();
        }
    }
}

/// One example's captured run, kept from a `capture_traces=true` evaluation so [`Adapter::propose_new_texts`]
/// can build the reflective dataset — dspy's `eval_batch.trajectories`.
///
/// The example and its prediction travel with the trace because the feedback text is *not* computed
/// here: dspy calls the metric again at reflection time, once per record, with the predictor it
/// drew. So what scoring keeps is the run, not a sentence about it.
struct Captured {
    example: Example,
    prediction: Prediction,
    trace: Vec<TraceStep>,
}

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
    /// dspy `num_threads`: how many examples one evaluation runs at once. One at a time by default,
    /// which is upstream's `num_threads=None`.
    num_threads: usize,
    /// dspy `instruction_proposer`: a caller's own proposal step, replacing the reflection tree
    /// entirely when set. See [`InstructionProposer`](super::InstructionProposer).
    proposer: Option<Arc<dyn super::InstructionProposer>>,
    /// dspy's `DspyAdapter.rng`, which picks which trace instance a reflection reads. A *second*
    /// generator from the same seed, not the engine's: `gepa.py` builds `random.Random(self.seed)`
    /// for the adapter and passes `optimize(seed=self.seed)` separately, so a draw here cannot
    /// move the engine's stream.
    rng: Random,
}

impl<'a, S: Module + ?Sized, M> Adapter<'a, S, M> {
    pub(super) fn new(
        student: &'a mut S,
        metric: &'a M,
        reflection: Arc<dyn DynChatModel>,
        trainset: &'a [Example],
        valset: &'a [Example],
        failure_score: f64,
        num_threads: usize,
        proposer: Option<Arc<dyn super::InstructionProposer>>,
        seed: u64,
    ) -> Self {
        Self {
            student,
            metric,
            reflection,
            trainset,
            valset,
            failure_score,
            captured: Vec::new(),
            num_threads,
            proposer,
            rng: Random::seeded(seed),
        }
    }
}

impl<S, M> Adapter<'_, S, M>
where
    S: Module + ?Sized,
    M: Fn(&Example, &Prediction, &MetricContext<'_>) -> Feedback + Send + Sync,
{
    /// Run the candidate program over `examples`, scoring each with the metric. When capturing, the
    /// per-example runs are stashed for the reflection step. dspy never raises for one example's
    /// failure — a failed run scores `failure_score` and contributes no trace.
    async fn evaluate(
        &mut self,
        examples: &[Example],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> EvalBatch {
        set_instructions(self.student, candidate);
        if capture_traces {
            self.captured.clear();
        }
        // Order-preserving, so a trace still lines up with the example that produced it and the
        // reflection reads the same dataset whatever the thread count. `buffered` and not
        // `buffer_unordered`, for the same reason `Evaluate` uses it.
        // Built in a plain loop rather than through `map`: a closure returning a future that
        // borrows both its argument and `self` needs a higher-ranked bound the compiler will not
        // infer, and there is no closure here to need one.
        let mut running = Vec::with_capacity(examples.len());
        for example in examples {
            running.push(self.run_one(example, capture_traces));
        }
        let ran: Vec<(f64, Option<Captured>)> = futures_util::stream::iter(running)
            .buffered(self.num_threads.max(1))
            .collect()
            .await;
        let mut scores = Vec::with_capacity(examples.len());
        for (score, captured) in ran {
            scores.push(score);
            if let Some(captured) = captured {
                self.captured.push(captured);
            }
        }
        if capture_traces {
            EvalBatch::traced(scores)
        } else {
            EvalBatch::scored(scores)
        }
    }

    /// One example: run the (already-built) program with tracing, then score it. Returns the score
    /// and, when capturing, the run itself for reflection.
    async fn run_one(&self, example: &Example, capture_traces: bool) -> (f64, Option<Captured>) {
        let inputs = example.inputs().expect("a dataset row declares its inputs");
        let mut trace = Vec::new();
        let Ok(prediction) = self.student.forward_traced(inputs, &mut trace).await else {
            let captured = capture_traces.then(|| Captured {
                example: example.clone(),
                prediction: Prediction::new(Example::default(), String::new()),
                trace: Vec::new(),
            });
            return (self.failure_score, captured);
        };
        // dspy's scoring call is the ordinary metric call — `Evaluate` and `bootstrap_trace_data`
        // pass no predictor, and a trace only while capturing.
        let scoring = MetricContext::scoring(capture_traces.then_some(trace.as_slice()));
        let feedback = (self.metric)(example, &prediction, &scoring);
        let captured = capture_traces.then(|| Captured {
            example: example.clone(),
            prediction: prediction.clone(),
            trace,
        });
        (feedback.score, captured)
    }

    /// dspy `make_reflective_dataset` for one predictor: an `Inputs`/`Generated Outputs`/`Feedback`
    /// record per captured example whose trace touched this predictor. The predictor's inputs and
    /// outputs render the way dspy's `str(value)` does ([`Example::rendered`]).
    fn reflective_dataset(&mut self, predictor: &str) -> Vec<ReflectiveSample> {
        let mut samples = Vec::new();
        for captured in &self.captured {
            // dspy draws one of the instances rather than taking the first, and draws even when
            // there is only one — so an example whose trace touched the predictor once still
            // advances the generator, and the next example's draw lands where upstream's does.
            let instances: Vec<&TraceStep> = captured
                .trace
                .iter()
                .filter(|step| step.predictor == predictor)
                .collect();
            if instances.is_empty() {
                continue;
            }
            let step = instances[self.rng.choice_index(instances.len())];
            // dspy calls the metric a second time here, with the predictor it wants feedback for
            // and the instance it just drew, and keeps only the text — the score stays the one
            // scoring produced, which is what upstream's `fb["score"] = module_score` does.
            let reflecting = MetricContext {
                trace: Some(&captured.trace),
                predictor: Some(predictor),
                predictor_step: Some(step),
            };
            let feedback =
                (self.metric)(&captured.example, &captured.prediction, &reflecting).text();
            samples.push(vec![
                ("Inputs".to_owned(), rendered_map(&step.inputs)),
                ("Generated Outputs".to_owned(), rendered_map(&step.outputs)),
                ("Feedback".to_owned(), Reflective::Text(feedback)),
            ]);
        }
        samples
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
    async fn evaluate_minibatch(
        &mut self,
        ids: &[usize],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> EvalBatch {
        let examples: Vec<Example> = ids.iter().map(|&id| self.trainset[id].clone()).collect();
        self.evaluate(&examples, candidate, capture_traces).await
    }

    async fn evaluate_valset(&mut self, candidate: &Candidate) -> EvalBatch {
        let examples = self.valset.to_vec();
        self.evaluate(&examples, candidate, false).await
    }

    async fn evaluate_valset_ids(&mut self, ids: &[usize], candidate: &Candidate) -> EvalBatch {
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
        _captured: &EvalBatch,
    ) -> Candidate {
        // A component whose runs produced nothing is left out of both paths: upstream skips it
        // rather than proposing against an empty dataset, and a caller's proposer is not handed one
        // it cannot use either.
        let datasets: BTreeMap<String, super::ReflectiveDataset> = components
            .iter()
            .map(|name| (name.clone(), self.reflective_dataset(name)))
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
        new_texts
    }
}

/// An example's fields as a GEPA reflective map: field name → its rendered value, in declaration
/// order (dspy's `{k: str(v) for k, v in inputs.items()}`).
fn rendered_map(example: &Example) -> Reflective {
    Reflective::Map(
        example
            .rendered()
            .into_iter()
            .map(|(name, value)| (name, Reflective::Text(value)))
            .collect(),
    )
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
            0.0,
            1,
            None,
            0,
        );
        adapter.captured = vec![Captured {
            example: example! { question: "capital of France?" },
            prediction: crate::Prediction::new(example! { answer: "Paris" }, "raw"),
            trace: vec![crate::TraceStep {
                predictor: "self".to_owned(),
                inputs: example! { question: "capital of France?" },
                outputs: example! { answer: "Paris" },
            }],
        }];

        let dataset = adapter.reflective_dataset("self");
        assert_eq!(dataset.len(), 1, "one captured example touched `self`");
        let sample = &dataset[0];
        let rendered = render_prompt("", std::slice::from_ref(sample), None);
        assert!(rendered.contains("capital of France?"), "{rendered}");
        assert!(rendered.contains("right city"), "{rendered}");
        assert!(
            adapter.reflective_dataset("someone_else").is_empty(),
            "a predictor nothing traced gets no records"
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
        let golden: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/conformance/optimize/gepa_reflective.json"
        ))
        .expect("the reflective golden parses");
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
                0.0,
                1,
                None,
                seed,
            );
            adapter.captured = questions.iter().map(|q| two_hops(q, answers)).collect();

            let dataset = adapter.reflective_dataset("step");
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

    /// The two hops one example produces, both named `step` — dspy's `TwoHop`, whose single
    /// `Predict` is called twice, so one component owns two trace instances.
    fn two_hops(question: &str, answers: &serde_json::Value) -> Captured {
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
                    outputs: example! { answer: first.clone() },
                },
                crate::TraceStep {
                    predictor: "step".to_owned(),
                    inputs: example! { question: first },
                    outputs: example! { answer: second },
                },
            ],
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
