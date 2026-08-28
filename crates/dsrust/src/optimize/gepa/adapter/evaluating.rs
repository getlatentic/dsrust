//! Running a candidate over a batch: one score per example that survived, and the traces a
//! reflection reads afterwards.
//!
//! A child module rather than a sibling, so it can reach the adapter's own fields — the walk is
//! part of the adapter, split out only because the file it lived in outgrew its budget.

use futures_util::StreamExt as _;

use super::super::binding::set_instructions;
use super::super::failing::{Capture, did_not_parse};
use super::super::metric::{Feedback, MetricContext};
use super::super::reflecting::Captured;
use super::Adapter;
use crate::example::{Example, Prediction};
use crate::module::Module;
use gepa::{Candidate, EvalBatch};

impl<S, M> Adapter<'_, S, M>
where
    S: Module + ?Sized,
    M: Fn(&Example, &Prediction, &MetricContext<'_>) -> Feedback + Send + Sync,
{
    /// Run the candidate program over `examples`, scoring each with the metric. When capturing, the
    /// per-example runs are stashed for the reflection step. dspy never raises for one example's
    /// failure — a failed run scores `failure_score` and contributes no trace.
    pub(super) async fn evaluate(
        &mut self,
        examples: &[Example],
        candidate: &Candidate,
        capture_traces: bool,
    ) -> EvalBatch<Prediction> {
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
        // Taken once, before the runs: matching a parse failure to the predictor that raised it
        // needs the walk, and the walk needs `&mut`, which a concurrent run cannot hold.
        let predictors: Vec<(String, crate::signature::Signature)> = self
            .student
            .named_predictors()
            .into_iter()
            .map(|predictor| (predictor.name.clone(), predictor.signature.clone()))
            .collect();
        let mut running = Vec::with_capacity(examples.len());
        for example in examples {
            running.push(self.run_one(example, &predictors, capture_traces));
        }
        let ran: Vec<(Option<f64>, Option<Captured>, Option<Prediction>)> =
            futures_util::stream::iter(running)
                .buffered(self.num_threads.max(1))
                .collect()
                .await;
        let mut scores = Vec::with_capacity(examples.len());
        let mut answered = Vec::with_capacity(examples.len());
        let mut kept = 0;
        for (score, captured, prediction) in ran {
            // `None` is an example dspy dropped: it contributes no score, no trajectory and no
            // output, so all three lists come back shorter than the batch. See `did_not_parse`.
            let Some(score) = score else { continue };
            kept += 1;
            scores.push(score);
            if let Some(captured) = captured {
                self.captured.push(captured);
            }
            if let Some(prediction) = prediction {
                answered.push(prediction);
            }
        }
        let mut batch = match capture_traces {
            true => EvalBatch::traced(scores),
            false => EvalBatch::scored(scores),
        };
        // Only when every run reported one, so a partial list can never be read positionally
        // against the scores it is supposed to line up with.
        if self.track_best_outputs && answered.len() == kept {
            batch.outputs = Some(answered);
        }
        batch
    }

    /// One example: run the (already-built) program with tracing, then score it. Returns the score
    /// and, when capturing, the run itself for reflection.
    async fn run_one(
        &self,
        example: &Example,
        predictors: &[(String, crate::signature::Signature)],
        capture_traces: bool,
    ) -> (Option<f64>, Option<Captured>, Option<Prediction>) {
        let inputs = example.inputs().expect("a dataset row declares its inputs");
        let mut trace = Vec::new();
        let ran = self
            .student
            .forward_traced(inputs.clone(), &mut trace)
            .await;
        let prediction = match ran {
            Ok(prediction) => prediction,
            Err(error) => {
                return did_not_parse(
                    example,
                    inputs,
                    trace,
                    &error,
                    predictors,
                    self.failure_score,
                    Capture {
                        traces: capture_traces,
                        track_best_outputs: self.track_best_outputs,
                    },
                );
            }
        };
        // dspy's scoring call is the ordinary metric call — `Evaluate` and `bootstrap_trace_data`
        // pass no predictor, and a trace only while capturing. A program holding a `Flex` scores
        // through `evaluate_with_trace` instead, which hands the run over as `program_trace` and
        // leaves `trace` empty, so a metric written for ordinary scoring reads the same thing in
        // both regimes.
        let scoring = match self.has_flexes {
            true => MetricContext::scoring_a_program(&trace),
            false => MetricContext::scoring(capture_traces.then_some(trace.as_slice())),
        };
        let feedback = (self.metric)(example, &prediction, &scoring);
        let captured = capture_traces.then(|| Captured {
            example: example.clone(),
            prediction: prediction.clone(),
            trace,
            scored: feedback.clone(),
            unparsed: None,
        });
        let answered = self.track_best_outputs.then(|| prediction.clone());
        (Some(feedback.score), captured, answered)
    }
}
