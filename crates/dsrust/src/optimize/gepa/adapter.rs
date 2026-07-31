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

use super::metric::Feedback;
use crate::example::{Example, Prediction};
use crate::lm::DynChatModel;
use crate::lm::api::{LmMessage, LmPart, LmRequest};
use crate::module::{Module, TraceStep};

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
struct Captured {
    trace: Vec<TraceStep>,
    feedback: String,
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
        }
    }
}

impl<S, M> Adapter<'_, S, M>
where
    S: Module + ?Sized,
    M: Fn(&Example, &Prediction) -> Feedback + Send + Sync,
{
    /// Run the candidate program over `examples`, scoring each with the metric. When capturing, the
    /// per-example traces and feedback are stashed for the reflection step. dspy never raises for one
    /// example's failure — a failed run scores `failure_score` and contributes no trace.
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

    /// One example: run the (already-built) program with tracing, then score it. Returns the score and,
    /// when capturing, the trace and feedback for reflection.
    async fn run_one(&self, example: &Example, capture_traces: bool) -> (f64, Option<Captured>) {
        let inputs = example.inputs().expect("a dataset row declares its inputs");
        let mut trace = Vec::new();
        let Ok(prediction) = self.student.forward_traced(inputs, &mut trace).await else {
            let captured = capture_traces.then(|| Captured {
                trace: Vec::new(),
                feedback: String::new(),
            });
            return (self.failure_score, captured);
        };
        let feedback = (self.metric)(example, &prediction);
        let captured = capture_traces.then(|| Captured {
            trace,
            feedback: feedback.text(),
        });
        (feedback.score, captured)
    }

    /// dspy `make_reflective_dataset` for one predictor: an `Inputs`/`Generated Outputs`/`Feedback`
    /// record per captured example whose trace touched this predictor. The predictor's inputs and
    /// outputs render the way dspy's `str(value)` does ([`Example::rendered`]).
    fn reflective_dataset(&self, predictor: &str) -> Vec<ReflectiveSample> {
        let mut samples = Vec::new();
        for captured in &self.captured {
            let Some(step) = captured
                .trace
                .iter()
                .find(|step| step.predictor == predictor)
            else {
                continue;
            };
            samples.push(vec![
                ("Inputs".to_owned(), rendered_map(&step.inputs)),
                ("Generated Outputs".to_owned(), rendered_map(&step.outputs)),
                (
                    "Feedback".to_owned(),
                    Reflective::Text(captured.feedback.clone()),
                ),
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
    M: Fn(&Example, &Prediction) -> Feedback + Send + Sync,
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
