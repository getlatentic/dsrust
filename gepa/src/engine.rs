//! GEPA's optimization loop (`core/engine.py` + `proposer/reflective_mutation/reflective_mutation.py`),
//! restricted to the defaults dspy drives it with: reflective mutation only (no merge), the Pareto
//! candidate selector, the epoch-shuffled batch sampler, round-robin component selection, full-valset
//! evaluation, and a `max_metric_calls` budget.
//!
//! One `pyrng::Random` is shared across the whole run (dspy's `api.py`: a single `random.Random(seed)`
//! handed to the selector and the sampler). Each iteration draws in a fixed order — the candidate
//! selection `choice`, then the batch sampler's `shuffle` when the epoch rolls over — so the stream
//! is reproduced only if the engine consumes it in exactly that order.

use pyrng::Random;

use crate::adapter::{Candidate, GepaAdapter};
use crate::batch::BatchSampler;
use crate::pareto::select_candidate;
use crate::state::GepaState;

/// A reflective-mutation proposal: a mutated candidate and the minibatch scores of its parent and of
/// itself, whose sums the engine compares to decide acceptance.
struct Proposal {
    candidate: Candidate,
    parent: usize,
    scores_before: Vec<f64>,
    scores_after: Vec<f64>,
}

/// What a completed run reports — the fields of dspy's `GEPAResult` the engine determines.
pub struct GepaOutcome {
    pub candidates: Vec<Candidate>,
    pub parents: Vec<Vec<usize>>,
    pub val_aggregate_scores: Vec<f64>,
    pub best_idx: usize,
    pub best: Candidate,
    pub total_num_evals: usize,
    pub num_full_ds_evals: usize,
    pub num_metric_calls_by_discovery: Vec<usize>,
    pub iterations: i64,
}

/// dspy `GEPAEngine` under its default configuration. The adapter supplies the evaluation and
/// reflection; the engine owns the loop, the budget, and the shared generator.
pub struct GepaEngine<A: GepaAdapter> {
    pub adapter: A,
    pub trainset_size: usize,
    pub valset_size: usize,
    pub minibatch_size: usize,
    pub max_metric_calls: usize,
    pub perfect_score: f64,
    pub skip_perfect_score: bool,
    pub seed: u64,
}

impl<A: GepaAdapter> GepaEngine<A> {
    /// Run the loop from `seed_candidate` until the metric-call budget is spent, returning the best
    /// candidate found. dspy checks the budget only at the top of each iteration, so the iteration in
    /// flight always runs to completion even when it pushes the total past `max_metric_calls`.
    pub fn optimize(mut self, seed_candidate: Candidate) -> GepaOutcome {
        let base = self.adapter.evaluate_valset(&seed_candidate);
        let mut state = GepaState::new(seed_candidate, base.scores);
        let mut rng = Random::seeded(self.seed);
        let mut sampler = BatchSampler::new(self.minibatch_size);

        while state.total_num_evals < self.max_metric_calls {
            state.i += 1;
            let Some(proposal) = self.propose(&mut state, &mut rng, &mut sampler) else { continue };
            let before: f64 = proposal.scores_before.iter().sum();
            let after: f64 = proposal.scores_after.iter().sum();
            if after <= before {
                continue;
            }
            self.accept(&mut state, proposal);
        }
        self.finish(state)
    }

    /// dspy `ReflectiveMutationProposer.propose`: select a candidate, sample a minibatch, evaluate it
    /// with traces, reflect on a round-robin component to mutate it, and evaluate the mutant on the
    /// same minibatch. Returns `None` on the skip paths (no traces, or an already-perfect minibatch),
    /// each of which still spends the parent's minibatch evaluation.
    fn propose(&mut self, state: &mut GepaState, rng: &mut Random, sampler: &mut BatchSampler) -> Option<Proposal> {
        let parent = select_candidate(state.fronts(), &state.mean_scores(), rng);
        let subsample = sampler.next_minibatch_ids(self.trainset_size, state.i as usize, rng);
        let parent_candidate = state.candidates[parent].clone();

        let eval_parent = self.adapter.evaluate_minibatch(&subsample, &parent_candidate, true);
        state.total_num_evals += subsample.len();
        if !eval_parent.captured_traces {
            return None;
        }
        if self.skip_perfect_score && eval_parent.scores.iter().all(|&s| s >= self.perfect_score) {
            return None;
        }

        let components = vec![state.select_component(parent)];
        let new_texts = self.adapter.propose_new_texts(&parent_candidate, &components, &eval_parent);
        let mut candidate = parent_candidate;
        candidate.extend(new_texts);

        let eval_new = self.adapter.evaluate_minibatch(&subsample, &candidate, false);
        state.total_num_evals += subsample.len();

        Some(Proposal { candidate, parent, scores_before: eval_parent.scores, scores_after: eval_new.scores })
    }

    /// dspy `_run_full_eval_and_add`: an accepted proposal is re-scored on the whole valset (recording
    /// the eval total at discovery first) and folded into the state.
    fn accept(&mut self, state: &mut GepaState, proposal: Proposal) {
        let discovered_at = state.total_num_evals;
        let eval = self.adapter.evaluate_valset(&proposal.candidate);
        state.total_num_evals += self.valset_size;
        state.num_full_ds_evals += 1;
        state.add_program(&[proposal.parent], proposal.candidate, eval.scores, discovered_at);
    }

    /// Assemble the outcome: the best program is the highest mean valset score (dspy's `GEPAResult`).
    fn finish(self, state: GepaState) -> GepaOutcome {
        let best_idx = state.best_program();
        GepaOutcome {
            best: state.candidates[best_idx].clone(),
            val_aggregate_scores: state.mean_scores(),
            candidates: state.candidates,
            parents: state.parents,
            best_idx,
            total_num_evals: state.total_num_evals,
            num_full_ds_evals: state.num_full_ds_evals,
            num_metric_calls_by_discovery: state.num_metric_calls_by_discovery,
            iterations: state.i,
        }
    }
}
