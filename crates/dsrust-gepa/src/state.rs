//! GEPA's optimizer state (`core/state.py`): the growing candidate pool, each candidate's per-valset
//! subscores, the Pareto front over them, and the round-robin cursor that picks which component to
//! reflect on next. The engine reads the per-program mean subscore to select and to rank the best.

use crate::adapter::Candidate;
use crate::pareto::ParetoFront;

/// dspy `GEPAState`, restricted to the default `instance` frontier and `FullEvaluationPolicy` (every
/// candidate is scored on the whole valset, so every subscore vector has the same length).
pub struct GepaState<O> {
    pub candidates: Vec<Candidate>,
    pub parents: Vec<Vec<usize>>,
    pub num_metric_calls_by_discovery: Vec<usize>,
    /// dspy `i`, the iteration counter — starts at -1 so the first loop turn makes it 0.
    pub i: i64,
    pub total_num_evals: usize,
    pub num_full_ds_evals: usize,
    subscores: Vec<Vec<f64>>,
    front: ParetoFront<O>,
    /// dspy `named_predictor_id_to_update_next_for_program_candidate`.
    next_predictor: Vec<usize>,
    /// dspy `list_of_named_predictors` — the seed candidate's component names.
    components: Vec<String>,
}

impl<O: Clone> GepaState<O> {
    /// dspy `initialize_gepa_state`: the seed is candidate 0, scored on the full valset, and its score
    /// count is the starting eval total (`num_full_ds_evals = 1`).
    pub fn new(seed_candidate: Candidate, seed_scores: Vec<f64>) -> Self {
        let components: Vec<String> = seed_candidate.keys().cloned().collect();
        let total = seed_scores.len();
        Self {
            front: ParetoFront::seeded(&seed_scores),
            subscores: vec![seed_scores],
            candidates: vec![seed_candidate],
            parents: vec![Vec::new()],
            num_metric_calls_by_discovery: vec![0],
            next_predictor: vec![0],
            components,
            i: -1,
            total_num_evals: total,
            num_full_ds_evals: 1,
        }
    }

    /// The same, keeping what the seed and every later candidate answered — gepa's
    /// `track_best_outputs=True`, which it documents as requiring `track_stats` because the
    /// outputs are reported on the result.
    pub fn tracking_outputs(
        seed_candidate: Candidate,
        seed_scores: Vec<f64>,
        seed_outputs: &[O],
    ) -> Self {
        let mut state = Self::new(seed_candidate, seed_scores.clone());
        state.front = ParetoFront::seeded_tracking(&seed_scores, seed_outputs);
        state
    }

    /// gepa's `best_outputs_valset`: per validation example, every program on its front and what
    /// that program answered. `None` unless the run was started with
    /// [`tracking_outputs`](Self::tracking_outputs).
    pub fn best_outputs(&self) -> Option<&[Vec<(usize, O)>]> {
        self.front.best_outputs()
    }

    /// The per-testcase Pareto front, as [`crate::pareto::select_candidate`] reads it.
    pub fn fronts(&self) -> &[crate::pyset::PyIntSet] {
        self.front.fronts()
    }

    /// dspy `program_full_scores_val_set`: each candidate's mean valset subscore. Selection weights the
    /// domination sweep by this, and it ranks the best program.
    pub fn mean_scores(&self) -> Vec<f64> {
        self.subscores.iter().map(|scores| mean(scores)).collect()
    }

    /// One candidate's per-testcase valset subscores — `prog_candidate_val_subscores[id]`, which
    /// merge reads to bucket the evaluation subsample and to sum a parent's score over it.
    pub fn subscores(&self, candidate: usize) -> &[f64] {
        &self.subscores[candidate]
    }

    /// gepa's `prog_candidate_val_subscores`: every candidate's per-example scores, in candidate
    /// order. What [`subscores`](Self::subscores) reads one row of.
    pub fn all_subscores(&self) -> &[Vec<f64>] {
        &self.subscores
    }

    /// dspy `RoundRobinReflectionComponentSelector`: return the parent's next component to reflect on
    /// and advance that parent's cursor. A new candidate inherits the advanced cursor (see
    /// [`Self::add_program`]), so the family cycles through its components across generations.
    /// A state carrying only what a component selector reads — the component list and one
    /// candidate's cursor. For holding the round-robin walk to gepa's without building a run.
    pub fn for_components(components: Vec<String>, cursor: usize) -> Self {
        let mut state = Self::new(
            components
                .iter()
                .map(|name| (name.clone(), String::new()))
                .collect(),
            Vec::new(),
        );
        state.next_predictor = vec![cursor];
        state
    }

    /// Which component this candidate would reflect on next — gepa's
    /// `named_predictor_id_to_update_next_for_program_candidate`, which a new candidate inherits.
    pub fn next_component_for(&self, candidate_idx: usize) -> usize {
        self.next_predictor[candidate_idx]
    }

    pub fn select_component(&mut self, candidate_idx: usize) -> String {
        let pid = self.next_predictor[candidate_idx];
        self.next_predictor[candidate_idx] = (pid + 1) % self.components.len();
        self.components[pid].clone()
    }

    /// dspy `update_state_with_new_program`: append the accepted candidate, record when it was found,
    /// inherit the parents' furthest round-robin cursor, and fold its valset scores into the front.
    pub fn add_program(
        &mut self,
        parents: &[usize],
        candidate: Candidate,
        val_scores: Vec<f64>,
        discovered_at: usize,
    ) -> usize {
        self.add_program_with_outputs(parents, candidate, val_scores, discovered_at, None)
    }

    /// The same, carrying what this candidate answered on each validation example so the front can
    /// keep the outputs of whichever programs are currently best — gepa's `track_best_outputs`.
    pub fn add_program_with_outputs(
        &mut self,
        parents: &[usize],
        candidate: Candidate,
        val_scores: Vec<f64>,
        discovered_at: usize,
        outputs: Option<Vec<O>>,
    ) -> usize {
        let new_idx = self.candidates.len();
        let inherited = parents
            .iter()
            .map(|&p| self.next_predictor[p])
            .max()
            .unwrap_or(0);
        self.front
            .add_program_with_outputs(new_idx, &val_scores, outputs.as_deref());
        self.candidates.push(candidate);
        self.parents.push(parents.to_vec());
        self.num_metric_calls_by_discovery.push(discovered_at);
        self.next_predictor.push(inherited);
        self.subscores.push(val_scores);
        new_idx
    }

    /// dspy `FullEvaluationPolicy.get_best_program`: the highest mean valset score, ties broken toward
    /// wider coverage and then — coverage being equal under full evaluation — the earliest candidate.
    pub fn best_program(&self) -> usize {
        let mut best = (0usize, f64::NEG_INFINITY, 0usize);
        for (idx, scores) in self.subscores.iter().enumerate() {
            let coverage = scores.len();
            let avg = if coverage == 0 {
                f64::NEG_INFINITY
            } else {
                mean(scores)
            };
            if avg > best.1 || (avg == best.1 && coverage > best.2) {
                best = (idx, avg, coverage);
            }
        }
        best.0
    }
}

fn mean(scores: &[f64]) -> f64 {
    scores.iter().sum::<f64>() / scores.len() as f64
}

#[cfg(test)]
mod best_program_tests {
    use super::*;

    fn candidate(text: &str) -> Candidate {
        Candidate::from_iter([("instructions".to_owned(), text.to_owned())])
    }

    /// A state whose programs carry these valset subscores.
    ///
    /// Written onto the field rather than through `add_program`, because that folds each program
    /// into the pareto front and the front is sized by the seed's valset — so a program evaluated
    /// on a *different* number of examples cannot be added that way at all. Coverage varies only
    /// under a partial evaluation policy, which is the case the tie clause exists for and the case
    /// no test had. `best_program` reads nothing but this field.
    fn with_scores(all: &[&[f64]]) -> GepaState<()> {
        let mut state = GepaState::new(candidate("seed"), vec![0.0]);
        state.subscores = all.iter().map(|scores| scores.to_vec()).collect();
        state
    }

    /// The highest mean wins, which is the rule's whole first clause.
    #[test]
    fn the_highest_mean_wins() {
        assert_eq!(
            with_scores(&[&[0.2, 0.2], &[0.9, 0.9], &[0.5, 0.5]]).best_program(),
            1
        );
    }

    /// An exact tie in mean goes to the wider coverage — upstream's second clause.
    ///
    /// Both `==` and `>` in that clause survived mutation: nothing had two programs with the same
    /// mean and different coverage, so breaking the tie the other way, or on the wrong comparison
    /// entirely, changed no test. This is the case that separates them.
    #[test]
    fn an_exact_tie_goes_to_the_wider_coverage() {
        // Same mean, 0.5; the second is evaluated on four examples rather than two.
        let state = with_scores(&[&[0.5, 0.5], &[0.5, 0.5, 0.5, 0.5]]);
        assert_eq!(state.best_program(), 1);

        // And the other order, so passing cannot be an artifact of which came first.
        let state = with_scores(&[&[0.5, 0.5, 0.5, 0.5], &[0.5, 0.5]]);
        assert_eq!(state.best_program(), 0);
    }

    /// Wider coverage does *not* beat a better mean — the tie clause is a tie-break, not a rank.
    ///
    /// The `==` mutation reads as `avg != best && coverage > best_coverage`, which lets a program
    /// with a *worse* mean win on coverage alone. Only a case with both a lower mean and wider
    /// coverage tells the two apart.
    #[test]
    fn wider_coverage_does_not_beat_a_better_mean() {
        let state = with_scores(&[&[0.9, 0.9], &[0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]);
        assert_eq!(state.best_program(), 0);
    }

    /// An unevaluated program scores negative infinity rather than dividing by zero, so it never
    /// wins against one that was evaluated — upstream's `avg = ... if coverage else float("-inf")`.
    #[test]
    fn an_unevaluated_program_never_wins() {
        let state = with_scores(&[&[], &[0.1]]);
        assert_eq!(state.best_program(), 1);
    }
}
