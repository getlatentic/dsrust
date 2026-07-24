//! GEPA's optimizer state (`core/state.py`): the growing candidate pool, each candidate's per-valset
//! subscores, the Pareto front over them, and the round-robin cursor that picks which component to
//! reflect on next. The engine reads the per-program mean subscore to select and to rank the best.

use crate::adapter::Candidate;
use crate::pareto::ParetoFront;

/// dspy `GEPAState`, restricted to the default `instance` frontier and `FullEvaluationPolicy` (every
/// candidate is scored on the whole valset, so every subscore vector has the same length).
pub struct GepaState {
    pub candidates: Vec<Candidate>,
    pub parents: Vec<Vec<usize>>,
    pub num_metric_calls_by_discovery: Vec<usize>,
    /// dspy `i`, the iteration counter — starts at -1 so the first loop turn makes it 0.
    pub i: i64,
    pub total_num_evals: usize,
    pub num_full_ds_evals: usize,
    subscores: Vec<Vec<f64>>,
    front: ParetoFront,
    /// dspy `named_predictor_id_to_update_next_for_program_candidate`.
    next_predictor: Vec<usize>,
    /// dspy `list_of_named_predictors` — the seed candidate's component names.
    components: Vec<String>,
}

impl GepaState {
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

    /// dspy `RoundRobinReflectionComponentSelector`: return the parent's next component to reflect on
    /// and advance that parent's cursor. A new candidate inherits the advanced cursor (see
    /// [`Self::add_program`]), so the family cycles through its components across generations.
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
        let new_idx = self.candidates.len();
        let inherited = parents.iter().map(|&p| self.next_predictor[p]).max().unwrap_or(0);
        self.front.add_program(new_idx, &val_scores);
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
            let avg = if coverage == 0 { f64::NEG_INFINITY } else { mean(scores) };
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
