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
use crate::progress::{Event, Progress};
use crate::pyset::PyIntSet;

/// Which candidate a strategy picks — gepa's `CandidateSelector.select_candidate_idx`.
///
/// Its own function rather than a `match` inside the loop, so a conformance test can drive each arm
/// against the package directly. What has to agree is the pick *and* where the generator is left:
/// the four arms take zero, one or two draws, and a round that advances it differently diverges on
/// the round after rather than on this one.
pub fn select_with(
    selection: CandidateSelection,
    fronts: &[PyIntSet],
    scores: &[f64],
    rng: &mut Random,
) -> usize {
    match selection {
        CandidateSelection::Pareto => select_candidate(fronts, scores, rng),
        // gepa's `idxmax`: `lst.index(max(lst))`, so a tie goes to the earliest candidate. No draw
        // from the generator, unlike the Pareto arm.
        CandidateSelection::CurrentBest => idxmax(scores),
        CandidateSelection::EpsilonGreedy { epsilon } => {
            // The coin is drawn whatever it decides, which is what keeps the two branches from
            // disagreeing about how far the generator has advanced.
            match rng.random() < epsilon {
                true => rng.randint(0, scores.len().saturating_sub(1) as u64) as usize,
                false => idxmax(scores),
            }
        }
        CandidateSelection::TopKPareto { k } => top_k_pareto(fronts, scores, k, rng),
    }
}

/// gepa's `TopKParetoCandidateSelector`: the Pareto draw over only the `k` best candidates.
///
/// The top-k comes from `sorted(range(n), key=scores.__getitem__, reverse=True)[:k]`, and Python's
/// sort is stable — so a tie keeps index order, and reversing a stable descending sort is not the
/// same as sorting ascending and reversing the result. The filtered fronts are set *intersections*,
/// which is why they go through [`PyIntSet`]: their iteration order reaches the sampling list and
/// therefore the draw.
///
/// An empty filtered mapping falls back to `idxmax` and draws nothing at all — a branch that moves
/// every later draw by not taking one.
fn top_k_pareto(fronts: &[PyIntSet], scores: &[f64], k: usize, rng: &mut Random) -> usize {
    let mut ranked: Vec<usize> = (0..scores.len()).collect();
    // `sorted(..., reverse=True)` on a stable sort: equal scores keep their index order.
    ranked.sort_by(|left, right| {
        scores[*right]
            .partial_cmp(&scores[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top = PyIntSet::from_keys(ranked.into_iter().take(k));

    let filtered: Vec<PyIntSet> = fronts
        .iter()
        .map(|front| front.intersection(&top))
        .filter(|front| !front.is_empty())
        .collect();
    match filtered.is_empty() {
        true => idxmax(scores),
        false => select_candidate(&filtered, scores, rng),
    }
}

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
pub struct GepaOutcome<O> {
    pub candidates: Vec<Candidate>,
    pub parents: Vec<Vec<usize>>,
    pub val_aggregate_scores: Vec<f64>,
    pub best_idx: usize,
    pub best: Candidate,
    pub total_num_evals: usize,
    pub num_full_ds_evals: usize,
    pub num_metric_calls_by_discovery: Vec<usize>,
    pub iterations: i64,
    /// gepa's `prog_candidate_val_subscores`, dspy's `val_subscores`: every candidate's score on
    /// every validation example, in candidate order. `val_aggregate_scores` is the mean of each.
    pub val_subscores: Vec<Vec<f64>>,
    /// gepa's `program_at_pareto_front_valset`, dspy's `per_val_instance_best_candidates`: per
    /// validation example, which candidates achieve its best score. The Pareto front the search
    /// selects from, reported so a caller can see which candidate won where.
    pub per_val_instance_best_candidates: Vec<Vec<usize>>,
    /// gepa's `best_outputs_valset`: per validation example, every program on its Pareto front and
    /// what that program answered. `None` unless the engine was asked to track them — gepa's
    /// `track_best_outputs`, which exists for using GEPA as a batch inference-time search, where
    /// the answers *are* the result rather than the candidate that produced them.
    pub best_outputs_valset: Option<Vec<Vec<(usize, O)>>>,
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
    /// gepa's `track_best_outputs`: keep what each front's programs answered, reported on
    /// [`GepaOutcome::best_outputs_valset`]. Off by default, as upstream's is — an adapter pays to
    /// carry the outputs and nothing reads them otherwise.
    pub track_best_outputs: bool,
    /// Where this run reports its decisions — gepa's `logger`. [`Silent`](crate::progress::Silent) by default, so a run
    /// nobody is watching costs nothing.
    pub progress: std::sync::Arc<dyn Progress>,
}

/// gepa's `CandidateSelector`: which candidate an iteration mutates.
///
/// The two differ in more than their choice. [`Pareto`](Self::Pareto) draws from the shared
/// generator and [`CurrentBest`](Self::CurrentBest) does not, so switching moves every later draw in
/// the run — the batch sample, the merge attempt, the next selection. It is not a preference applied
/// to the same sequence.
// No `Eq`: epsilon is a float, and gepa compares strategies by name rather than by value.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CandidateSelection {
    /// `ParetoCandidateSelector`: the survivor set of the per-testcase fronts, then a
    /// frequency-weighted draw. gepa's default, and dspy's.
    #[default]
    Pareto,
    /// `CurrentBestCandidateSelector`: `idxmax` over the aggregate valset scores — the first index
    /// holding the maximum, ties going to the earliest candidate found.
    CurrentBest,
    /// `EpsilonGreedyCandidateSelector`: with probability `epsilon` a uniform candidate, otherwise
    /// `idxmax`. gepa reaches it with `epsilon=0.1` and dspy passes only the strategy name, so the
    /// constant is fixed rather than a knob.
    ///
    /// Two draws or one: the coin is always drawn, and the uniform index only when it comes up
    /// below `epsilon`. Both come off the shared generator, so which branch a round takes moves
    /// every draw after it.
    EpsilonGreedy { epsilon: f64 },
    /// `TopKParetoCandidateSelector`: the Pareto draw restricted to the `k` best candidates by
    /// aggregate score. gepa reaches it with `k=5`.
    TopKPareto { k: usize },
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
    pub async fn optimize(mut self, seed_candidate: Candidate) -> GepaOutcome<A::Output> {
        let base = self.adapter.evaluate_valset(&seed_candidate).await;
        // The seed's own answers start the front, as gepa's initialisation does — a run whose seed
        // is never beaten on some example still reports what it answered there.
        let mut state = match (self.track_best_outputs, base.outputs) {
            (true, Some(outputs)) => {
                GepaState::tracking_outputs(seed_candidate, base.scores, &outputs)
            }
            _ => GepaState::new(seed_candidate, base.scores),
        };
        let mut rng = Random::seeded(self.seed);
        let mut sampler = BatchSampler::new(self.minibatch_size);
        let mut merge = MergeSchedule::default();

        // The loop ends when the budget is spent, so it ends only if every iteration spends some
        // of it. Nothing enforced that. `propose` counts a minibatch before each of its two early
        // returns, so the one way through without spending is an *empty* minibatch — which an
        // empty trainset produces, and which then spins here forever on a real run rather than on
        // a mutated one. Upstream asserts the sampler has enough ids; this checks the effect
        // instead, which also covers an adapter that reports no evaluations.
        //
        // **Both mutants of the `state.i > 0` below are unreachable, not untested.** Every path
        // through an iteration spends: `next_minibatch_ids` refuses an empty trainset outright
        // (`batch.rs`, and `an_empty_trainset_is_refused_rather_than_sampled` pins it), so a
        // subsample is never empty; `try_merge` counts its subsample before every `Rejected`, and
        // `NoMerge` returns before spending but falls through to `propose`, which spends. So the
        // equality can never hold and neither spelling of the guard can be told from the other.
        // It stays anyway, for the reason `pyset`'s probe bound stays: it turns a hang into a
        // break if the assert above it is ever softened, and a bound that cannot fire is still a
        // bound. Recorded here so a mutation run does not keep offering it as work.
        let mut spent_last_iteration = state.total_num_evals;
        while state.total_num_evals < self.max_metric_calls {
            if state.i > 0 && state.total_num_evals == spent_last_iteration {
                break;
            }
            spent_last_iteration = state.total_num_evals;
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
                self.progress.report(Event::ProposedNothing {
                    iteration: state.i + 1,
                });
                continue;
            };
            let before: f64 = proposal.scores_before.iter().sum();
            let after: f64 = proposal.scores_after.iter().sum();
            if after <= before {
                self.progress.report(Event::Rejected {
                    iteration: state.i + 1,
                    before,
                    after,
                });
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
        state: &mut GepaState<A::Output>,
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
        state: &mut GepaState<A::Output>,
        rng: &mut Random,
        sampler: &mut BatchSampler,
    ) -> Option<Proposal> {
        // Through `select_with`, which is the function `tests/selectors.rs` drives against the
        // gepa package. This match was written out again here, so the arm production took was a
        // copy of the arm the conformance test checked — two bodies that agree until one is
        // edited, with a golden that would keep passing either way.
        let parent = select_with(
            self.candidate_selection_strategy,
            state.fronts(),
            &state.mean_scores(),
            rng,
        );
        let subsample = sampler.next_minibatch_ids(self.trainset_size, state.i as usize, rng);
        let parent_candidate = state.candidates[parent].clone();

        let eval_parent = self
            .adapter
            .evaluate_minibatch(&subsample, &parent_candidate, true)
            .await;
        state.total_num_evals += subsample.len();
        if !eval_parent.captured_traces {
            self.progress.report(Event::NoTrajectories {
                iteration: state.i + 1,
            });
            return None;
        }
        if self.skip_perfect_score && eval_parent.scores.iter().all(|&s| s >= self.perfect_score) {
            self.progress.report(Event::NothingToLearnFrom {
                iteration: state.i + 1,
            });
            return None;
        }

        let components = match self.component_selector {
            ComponentSelection::RoundRobin => vec![state.select_component(parent)],
            ComponentSelection::All => parent_candidate.keys().cloned().collect(),
        };
        // `None` is upstream raising out of `make_reflective_dataset`, which gepa catches: the
        // iteration ends here rather than scoring a candidate identical to its parent, and the
        // minibatch evaluation below is the one that is not spent.
        let Some(new_texts) = self
            .adapter
            .propose_new_texts(&parent_candidate, &components, &eval_parent)
            .await
        else {
            self.progress.report(Event::ReflectionFailed {
                iteration: state.i + 1,
                error: "No valid predictions found for any module.",
            });
            return None;
        };
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
    async fn accept(&mut self, state: &mut GepaState<A::Output>, proposal: Proposal) {
        let discovered_at = state.total_num_evals;
        let eval = self.adapter.evaluate_valset(&proposal.candidate).await;
        state.total_num_evals += self.valset_size;
        state.num_full_ds_evals += 1;
        // The score gepa's "Found a better program" line prints, read before the state moves on.
        let score = match eval.scores.is_empty() {
            true => 0.0,
            false => eval.scores.iter().sum::<f64>() / eval.scores.len() as f64,
        };
        let iteration = state.i + 1;
        state.add_program(
            &[proposal.parent],
            proposal.candidate,
            eval.scores,
            discovered_at,
        );
        let candidate = state.candidates.len() - 1;
        self.progress.report(Event::Accepted {
            iteration,
            candidate,
            score,
            is_best: state.best_program() == candidate,
        });
    }

    /// Assemble the outcome: the best program is the highest mean valset score (dspy's `GEPAResult`).
    fn finish(self, state: GepaState<A::Output>) -> GepaOutcome<A::Output> {
        let best_idx = state.best_program();
        let best_outputs_valset = state.best_outputs().map(<[_]>::to_vec);
        let val_subscores = state.all_subscores().to_vec();
        let per_val_instance_best_candidates: Vec<Vec<usize>> =
            state.fronts().iter().map(PyIntSet::to_vec).collect();
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
            val_subscores,
            per_val_instance_best_candidates,
            best_outputs_valset,
        }
    }
}
