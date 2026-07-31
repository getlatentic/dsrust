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
use crate::merge::{MergesPerformed, sample_and_attempt_merge, select_eval_subsample};
use crate::pareto::{find_dominator_programs, select_candidate};

/// gepa's `idxmax`: the first index holding the maximum. `lst.index(max(lst))` in Python, so a tie
/// resolves to the earliest — which for candidates means the one found first.
fn idxmax(scores: &[f64]) -> usize {
    let mut best = 0;
    for (index, score) in scores.iter().enumerate() {
        if score > &scores[best] {
            best = index;
        }
    }
    best
}
use crate::state::GepaState;

/// The engine's merge bookkeeping, dspy's counters on the `MergeProposer`: how many merges are due,
/// how many have been accepted, whether the last iteration added a program (a merge is only tried
/// right after one), and the record of merges already attempted.
#[derive(Default)]
struct MergeSchedule {
    due: usize,
    total_tested: usize,
    last_iter_found_new_program: bool,
    performed: MergesPerformed,
}

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
    /// dspy `use_merge`: whether to attempt merges between reflective mutations. On by default
    /// upstream; a run that leaves it off is the reflective-mutation-only engine.
    pub use_merge: bool,
    /// dspy `max_merge_invocations`: the cap on accepted merges over a run (default 5).
    pub max_merge_invocations: usize,
    /// Which candidate each iteration reflects on. See [`CandidateSelection`].
    pub candidate_selection_strategy: CandidateSelection,
    /// Which of a candidate's components a reflection rewrites. See [`ComponentSelection`].
    pub component_selector: ComponentSelection,
}

/// gepa's `CandidateSelector`: which candidate an iteration mutates.
///
/// The two differ in more than their choice. [`Pareto`](Self::Pareto) draws from the shared
/// generator and [`CurrentBest`](Self::CurrentBest) does not, so switching moves every later draw in
/// the run — the batch sample, the merge attempt, the next selection. It is not a preference applied
/// to the same sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CandidateSelection {
    /// `ParetoCandidateSelector`: the survivor set of the per-testcase fronts, then a
    /// frequency-weighted draw. gepa's default, and dspy's.
    #[default]
    Pareto,
    /// `CurrentBestCandidateSelector`: `idxmax` over the aggregate valset scores — the first index
    /// holding the maximum, ties going to the earliest candidate found.
    CurrentBest,
}

/// gepa's `ReflectionComponentSelector`: which components one reflection rewrites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComponentSelection {
    /// `RoundRobinReflectionComponentSelector`: one component per iteration, the parent's cursor
    /// advancing so a family cycles through its components across generations. gepa's default.
    #[default]
    RoundRobin,
    /// `AllReflectionComponentSelector`: every component of the candidate, in the seed candidate's
    /// key order. One reflection rewrites the whole program rather than a predictor of it.
    All,
}

/// The number of validation ids a merged candidate is scored on before the full re-evaluation, and
/// the floor of shared support below which two candidates are not compared — dspy's
/// `num_subsample_ids` and `val_overlap_floor`.
const MERGE_SUBSAMPLE: usize = 5;
const VAL_OVERLAP_FLOOR: usize = 5;

/// What a scheduled merge attempt did this iteration.
enum MergeOutcome {
    /// A merged candidate beat both parents on the subsample and was added.
    Accepted,
    /// A merge was produced but lost to a parent; the iteration ends without reflective mutation.
    Rejected,
    /// No mergeable pair was found; the iteration falls through to reflective mutation.
    NoMerge,
}

impl<A: GepaAdapter + Send> GepaEngine<A> {
    /// Run the loop from `seed_candidate` until the metric-call budget is spent, returning the best
    /// candidate found. dspy checks the budget only at the top of each iteration, so the iteration in
    /// flight always runs to completion even when it pushes the total past `max_metric_calls`.
    pub async fn optimize(mut self, seed_candidate: Candidate) -> GepaOutcome {
        let base = self.adapter.evaluate_valset(&seed_candidate).await;
        let mut state = GepaState::new(seed_candidate, base.scores);
        let mut rng = Random::seeded(self.seed);
        let mut sampler = BatchSampler::new(self.minibatch_size);
        let mut merge = MergeSchedule::default();

        while state.total_num_evals < self.max_metric_calls {
            state.i += 1;

            // dspy attempts a merge before the reflective step, but only right after an iteration
            // that added a program, and only while merges are due. The attempt draws from the
            // shared generator whether or not it finds a pair, so it runs before the reflective
            // selection either way.
            if self.use_merge && merge.due > 0 && merge.last_iter_found_new_program {
                merge.last_iter_found_new_program = false;
                match self
                    .try_merge(&mut state, &mut rng, &mut merge.performed)
                    .await
                {
                    MergeOutcome::Accepted => {
                        merge.due -= 1;
                        merge.total_tested += 1;
                        continue;
                    }
                    MergeOutcome::Rejected => continue,
                    MergeOutcome::NoMerge => {}
                }
            }
            merge.last_iter_found_new_program = false;

            let Some(proposal) = self.propose(&mut state, &mut rng, &mut sampler).await else {
                continue;
            };
            let before: f64 = proposal.scores_before.iter().sum();
            let after: f64 = proposal.scores_after.iter().sum();
            if after <= before {
                continue;
            }
            self.accept(&mut state, proposal).await;

            // A program was added, so a merge becomes due — capped at the run's invocation budget.
            if self.use_merge {
                merge.last_iter_found_new_program = true;
                if merge.total_tested < self.max_merge_invocations {
                    merge.due += 1;
                }
            }
        }
        self.finish(state)
    }

    /// dspy `MergeProposer.propose` plus the engine's accept: find a mergeable pair, score the
    /// merged candidate on a validation subsample, and — if it beats both parents there — re-score
    /// it on the whole valset and fold it in with both parents. Every RNG draw here lands in the
    /// shared stream ahead of the iteration's reflective step.
    async fn try_merge(
        &mut self,
        state: &mut GepaState,
        rng: &mut Random,
        performed: &mut MergesPerformed,
    ) -> MergeOutcome {
        let agg = state.mean_scores();
        let merge_candidates = find_dominator_programs(state.fronts(), &agg);
        let overlap = |_: usize, _: usize| self.valset_size >= VAL_OVERLAP_FLOOR;
        let Some(attempt) = sample_and_attempt_merge(
            rng,
            &agg,
            &merge_candidates,
            performed,
            &state.candidates,
            &state.parents,
            overlap,
            10,
        ) else {
            return MergeOutcome::NoMerge;
        };
        performed.record_triple(attempt.id1, attempt.id2, attempt.ancestor);

        let common_ids: Vec<usize> = (0..self.valset_size).collect();
        let subsample = select_eval_subsample(
            state.subscores(attempt.id1),
            state.subscores(attempt.id2),
            &common_ids,
            rng,
            MERGE_SUBSAMPLE,
        );
        if subsample.is_empty() {
            return MergeOutcome::NoMerge;
        }

        let eval = self
            .adapter
            .evaluate_valset_ids(&subsample, &attempt.candidate)
            .await;
        state.total_num_evals += subsample.len();

        // dspy compares the merged candidate's subsample sum against the better parent's over the
        // same ids: it is accepted only if it is at least as good as both.
        let parent_sum = |id: usize| -> f64 {
            subsample
                .iter()
                .map(|&val_id| state.subscores(id)[val_id])
                .sum()
        };
        let best_parent = parent_sum(attempt.id1).max(parent_sum(attempt.id2));
        if eval.scores.iter().sum::<f64>() < best_parent {
            return MergeOutcome::Rejected;
        }

        let discovered_at = state.total_num_evals;
        let full = self.adapter.evaluate_valset(&attempt.candidate).await;
        state.total_num_evals += self.valset_size;
        state.num_full_ds_evals += 1;
        state.add_program(
            &[attempt.id1, attempt.id2],
            attempt.candidate,
            full.scores,
            discovered_at,
        );
        MergeOutcome::Accepted
    }

    /// dspy `ReflectiveMutationProposer.propose`: select a candidate, sample a minibatch, evaluate it
    /// with traces, reflect on a round-robin component to mutate it, and evaluate the mutant on the
    /// same minibatch. Returns `None` on the skip paths (no traces, or an already-perfect minibatch),
    /// each of which still spends the parent's minibatch evaluation.
    async fn propose(
        &mut self,
        state: &mut GepaState,
        rng: &mut Random,
        sampler: &mut BatchSampler,
    ) -> Option<Proposal> {
        let parent = match self.candidate_selection_strategy {
            CandidateSelection::Pareto => {
                select_candidate(state.fronts(), &state.mean_scores(), rng)
            }
            // gepa's `idxmax`: `lst.index(max(lst))`, so a tie goes to the earliest candidate. No
            // draw from the generator, unlike the Pareto arm.
            CandidateSelection::CurrentBest => idxmax(&state.mean_scores()),
        };
        let subsample = sampler.next_minibatch_ids(self.trainset_size, state.i as usize, rng);
        let parent_candidate = state.candidates[parent].clone();

        let eval_parent = self
            .adapter
            .evaluate_minibatch(&subsample, &parent_candidate, true)
            .await;
        state.total_num_evals += subsample.len();
        if !eval_parent.captured_traces {
            return None;
        }
        if self.skip_perfect_score && eval_parent.scores.iter().all(|&s| s >= self.perfect_score) {
            return None;
        }

        let components = match self.component_selector {
            ComponentSelection::RoundRobin => vec![state.select_component(parent)],
            ComponentSelection::All => parent_candidate.keys().cloned().collect(),
        };
        let new_texts = self
            .adapter
            .propose_new_texts(&parent_candidate, &components, &eval_parent)
            .await;
        let mut candidate = parent_candidate;
        candidate.extend(new_texts);

        let eval_new = self
            .adapter
            .evaluate_minibatch(&subsample, &candidate, false)
            .await;
        state.total_num_evals += subsample.len();

        Some(Proposal {
            candidate,
            parent,
            scores_before: eval_parent.scores,
            scores_after: eval_new.scores,
        })
    }

    /// dspy `_run_full_eval_and_add`: an accepted proposal is re-scored on the whole valset (recording
    /// the eval total at discovery first) and folded into the state.
    async fn accept(&mut self, state: &mut GepaState, proposal: Proposal) {
        let discovered_at = state.total_num_evals;
        let eval = self.adapter.evaluate_valset(&proposal.candidate).await;
        state.total_num_evals += self.valset_size;
        state.num_full_ds_evals += 1;
        state.add_program(
            &[proposal.parent],
            proposal.candidate,
            eval.scores,
            discovered_at,
        );
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
