//! The bookkeeping COPRO's coordinate ascent runs on: the instruction/prefix pairs a model
//! proposed, the score each earned, and the two orderings dspy reads them back in — highest score
//! first to pick a winner, and worst-first to show the next round what has been tried.

use serde_json::json;

/// One instruction/prefix pair a model proposed. dspy keeps the raw strings and strips surrounding
/// quotes and whitespace only where it uses them, so both are held raw and cleaned on the way in.
#[derive(Clone)]
pub(super) struct Proposal {
    pub instruction: String,
    pub prefix: String,
}

/// dspy `c.proposed_instruction.strip('"').strip()`: drop surrounding quote marks, then surrounding
/// whitespace — in that order, since a quote outside the whitespace would otherwise survive.
pub(super) fn stripped(raw: &str) -> String {
    raw.trim_matches('"').trim().to_owned()
}

/// One evaluated candidate: the cleaned instruction and prefix, the score they earned, and the
/// whole program's instructions at that moment — dspy stores a deep copy of the module so the
/// highest-scoring one can be handed back verbatim.
#[derive(Clone)]
pub(super) struct Evaluated {
    pub instruction: String,
    pub prefix: String,
    pub score: f64,
    /// One instruction per predictor, in predictor order, as they stood when this was scored.
    pub program: Vec<String>,
}

/// A score as Python prints it: `serde_json`'s float form already keeps the trailing `.0` an
/// integral score carries and agrees with Python's `repr` across the 0..100 range a score lives in.
fn score_text(score: f64) -> String {
    json!(score).to_string()
}

/// Every candidate evaluated for one predictor, in the order they were first tried — dspy keys
/// these by `(instruction, prefix)` in an insertion-ordered dict, which both its "keep the best"
/// and its "show the newest" reads depend on.
#[derive(Default)]
pub(super) struct Evaluations(Vec<Evaluated>);

impl Evaluations {
    /// dspy's replace rule: a repeat `(instruction, prefix)` overwrites the stored entry only when
    /// its score is strictly higher (`existing >= new` keeps the old one, and its position). A pair
    /// not seen before is appended, which is what makes the order an insertion order.
    pub fn record(&mut self, candidate: Evaluated) {
        let existing = self
            .0
            .iter_mut()
            .find(|e| e.instruction == candidate.instruction && e.prefix == candidate.prefix);
        match existing {
            Some(entry) if candidate.score > entry.score => *entry = candidate,
            Some(_) => {}
            None => self.0.push(candidate),
        }
    }

    /// Every candidate's score, in the insertion order this holds them in.
    ///
    /// dspy reads `[x["score"] for x in evaluated_candidates[id(p)].values()]` and sorts a copy, so
    /// the order here is the one it sorts *from* — which decides nothing for `max`/`min` but does
    /// decide which of two equal scores a stable sort keeps.
    pub fn scores(&self) -> Vec<f64> {
        self.0.iter().map(|candidate| candidate.score).collect()
    }

    /// dspy `max(values(), key=score)`: the highest-scoring candidate, and the earliest one on a
    /// tie — the coordinate-ascent winner this predictor is set to before the next is scored.
    pub fn best(&self) -> &Evaluated {
        self.0
            .iter()
            .reduce(|best, next| if next.score > best.score { next } else { best })
            .expect("a predictor is only read after it has been scored at least once")
    }

    /// dspy's few-shot block for the next round: the top `breadth` candidates by score, laid out
    /// worst-first and numbered from one, each as its instruction, prefix and score. Feeds
    /// `GenerateInstructionGivenAttempts`, whose signature promises increasing order.
    pub fn attempts(&self, breadth: usize) -> Vec<String> {
        let ranked = self.ranked();
        let shown = breadth.min(ranked.len());
        let mut lines = Vec::with_capacity(shown * 3);
        for rank in (0..shown).rev() {
            let number = shown - rank;
            let candidate = ranked[rank];
            lines.push(format!("Instruction #{number}: {}", candidate.instruction));
            lines.push(format!("Prefix #{number}: {}", candidate.prefix));
            lines.push(format!(
                "Resulting Score #{number}: {}",
                score_text(candidate.score)
            ));
        }
        lines
    }

    /// This predictor's candidates, highest score first, ties left in insertion order — the stable
    /// descending sort both `attempts` and the final pick read.
    fn ranked(&self) -> Vec<&Evaluated> {
        let mut ranked: Vec<&Evaluated> = self.0.iter().collect();
        ranked.sort_by(|a, b| b.score.total_cmp(&a.score));
        ranked
    }
}

/// dspy's closing pick: flatten every predictor's candidates in predictor order, take the
/// highest-scoring across all of them, and the earliest on a tie. Its stored `program` is the
/// instruction set the student is compiled to.
pub(super) fn best_program(predictors: &[Evaluations]) -> Option<Vec<String>> {
    let mut all: Vec<&Evaluated> = predictors.iter().flat_map(|e| e.0.iter()).collect();
    all.sort_by(|a, b| b.score.total_cmp(&a.score));
    all.first().map(|winner| winner.program.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluated(instruction: &str, score: f64) -> Evaluated {
        Evaluated {
            instruction: instruction.to_owned(),
            prefix: String::new(),
            score,
            program: Vec::new(),
        }
    }

    /// A re-evaluation of the same candidate keeps the higher score and only the higher score —
    /// dspy's dedup by (instruction, prefix), where the guard's mutants either never updated or
    /// preferred the worse run.
    #[test]
    fn a_repeat_keeps_the_higher_score_only() {
        let mut seen = Evaluations::default();
        seen.record(evaluated("answer well", 0.4));
        seen.record(evaluated("answer well", 0.7));
        assert_eq!(seen.best().score, 0.7, "the better run replaced the worse");
        seen.record(evaluated("answer well", 0.5));
        assert_eq!(seen.best().score, 0.7, "the worse run replaced nothing");
    }

    /// `max(values(), key=score)` keeps the *earliest* on a tie, which is the coordinate-ascent
    /// winner the next round builds on.
    #[test]
    fn best_takes_the_earliest_of_a_tie() {
        let mut seen = Evaluations::default();
        seen.record(evaluated("first", 0.9));
        seen.record(evaluated("second", 0.9));
        assert_eq!(seen.best().instruction, "first");
        seen.record(evaluated("third", 1.0));
        assert_eq!(seen.best().instruction, "third");
    }
}
