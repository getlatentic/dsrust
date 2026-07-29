//! dspy `teleprompt/random_search.py`: bootstrap several times and keep the best.
//!
//! [`BootstrapFewShot`] produces one demo set from one walk of the trainset. This runs that walk
//! nineteen times over — each with the trainset shuffled differently and a different number of
//! demos asked for — scores each result on a validation set, and keeps the winner. It is the
//! cheapest real optimizer in dspy: no proposal model, no search algorithm, just more attempts.
//!
//! The first three attempts are not random at all, and that is the point of the negative seeds:
//! zero-shot, labels only, and one unshuffled bootstrap. A random search that never tried the
//! obvious baselines could report an improvement over nothing.

use anyhow::Result;
use pyrng::Random;

use super::{BootstrapFewShot, LabeledFewShot};
use crate::example::{Example, Prediction};
use crate::module::{Module, ProgramState};

/// One attempt and what it scored, in the order dspy records them.
#[derive(Debug, Clone)]
pub struct Attempt {
    /// dspy's `seed`: `-3` zero-shot, `-2` labels only, `-1` an unshuffled bootstrap, then the
    /// shuffled ones from zero up.
    pub seed: i64,
    pub score: f64,
    /// The program this attempt produced, as the state that would restore it.
    pub state: ProgramState,
}

/// dspy `BootstrapFewShotWithRandomSearch`: bootstrap under several shuffles, keep the best.
pub struct BootstrapRandomSearch<M> {
    metric: M,
    max_bootstrapped_demos: usize,
    max_labeled_demos: usize,
    max_rounds: usize,
    num_candidate_programs: usize,
    stop_at_score: Option<f64>,
    metric_threshold: Option<f64>,
    /// dspy's `min_num_samples`, fixed at 1 on the instance and never exposed as a constructor
    /// argument — kept here because the size draw reads it.
    min_num_samples: usize,
}

impl<M> BootstrapRandomSearch<M>
where
    M: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    /// dspy's defaults: four bootstrapped demos, sixteen labelled, sixteen candidate sets.
    pub fn new(metric: M) -> Self {
        Self {
            metric,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            max_rounds: 1,
            num_candidate_programs: 16,
            stop_at_score: None,
            metric_threshold: None,
            min_num_samples: 1,
        }
    }

    /// The upper end of the per-attempt demo draw — dspy's `max_bootstrapped_demos`, which it
    /// stores as `max_num_samples`.
    pub fn with_max_bootstrapped_demos(mut self, demos: usize) -> Self {
        self.max_bootstrapped_demos = demos;
        self
    }

    pub fn with_max_labeled_demos(mut self, demos: usize) -> Self {
        self.max_labeled_demos = demos;
        self
    }

    pub fn with_max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }

    /// How many *shuffled* attempts to make. The three baselines run regardless, so a search of
    /// `n` candidate programs makes `n + 3` attempts — as upstream's `range(-3, n)` does.
    pub fn with_num_candidate_programs(mut self, programs: usize) -> Self {
        self.num_candidate_programs = programs;
        self
    }

    /// Stop as soon as an attempt scores at least this. dspy compares with `>=`.
    pub fn with_stop_at_score(mut self, score: f64) -> Self {
        self.stop_at_score = Some(score);
        self
    }

    /// Read the bootstrap metric as a number against this bar rather than as a yes/no. See
    /// [`BootstrapFewShot::metric_threshold`].
    pub fn with_metric_threshold(mut self, threshold: f64) -> Self {
        self.metric_threshold = Some(threshold);
        self
    }

    /// dspy's seed sequence: the three baselines, then one per candidate program.
    fn seeds(&self) -> impl Iterator<Item = i64> + '_ {
        -3..self.num_candidate_programs as i64
    }

    /// The trainset this seed bootstraps from, and how many demos it asks for.
    ///
    /// Two separately seeded generators, which is upstream's `random.Random(seed).shuffle(...)`
    /// followed by `random.Random(seed).randint(...)`. Reusing one would draw the size from a
    /// stream the shuffle had already advanced, and every attempt would ask for a different number
    /// of demos than dspy's.
    fn shuffled(&self, seed: i64, trainset: &[Example]) -> (Vec<Example>, usize) {
        let mut shuffled = trainset.to_vec();
        Random::seeded(seed as u64).shuffle(&mut shuffled);
        let size = Random::seeded(seed as u64).randint(
            self.min_num_samples as u64,
            self.max_bootstrapped_demos as u64,
        ) as usize;
        (shuffled, size)
    }

    fn bootstrap(&self, demos: usize) -> BootstrapFewShot<&M> {
        let mut optimizer = BootstrapFewShot::new(&self.metric);
        optimizer.max_bootstrapped_demos = demos;
        optimizer.max_labeled_demos = self.max_labeled_demos;
        optimizer.max_rounds = self.max_rounds;
        optimizer.metric_threshold = self.metric_threshold;
        optimizer
    }

    /// Run every attempt, score each on `valset`, and leave `student` holding the best.
    ///
    /// Answers with every attempt in decreasing score, which is dspy's `candidate_programs`
    /// attached to the returned program. dspy defaults `valset` to the trainset; pass the trainset
    /// for both to match that.
    pub async fn compile<S: Module + ?Sized>(
        &self,
        student: &mut S,
        trainset: &[Example],
        valset: &[Example],
    ) -> Result<Vec<Attempt>> {
        // The state the student arrived in, so every attempt starts from the same program rather
        // than from whatever the previous one left behind — dspy gets that from `reset_copy`.
        let start = student.dump_state();
        let mut attempts: Vec<Attempt> = Vec::new();

        for seed in self.seeds() {
            student.load_state(&start)?;
            match seed {
                // Zero-shot: the student as it came, with no demos at all.
                -3 => {}
                -2 => LabeledFewShot::new(self.max_labeled_demos).compile(student, trainset),
                // The unshuffled bootstrap, asking for the full demo budget rather than a draw.
                -1 => {
                    self.bootstrap(self.max_bootstrapped_demos)
                        .compile(student, trainset)
                        .await?;
                }
                _ => {
                    let (shuffled, demos) = self.shuffled(seed, trainset);
                    self.bootstrap(demos).compile(student, &shuffled).await?;
                }
            }

            let state = student.dump_state();
            let scored = crate::evaluate::Evaluate::new(
                valset.to_vec(),
                |example| student.forward(example),
                &self.metric,
            )
            .run()
            .await
            .score;
            attempts.push(Attempt {
                seed,
                score: scored,
                state,
            });
            if self.stop_at_score.is_some_and(|bar| scored >= bar) {
                break;
            }
        }

        let best = attempts
            .iter()
            .enumerate()
            // dspy keeps the *first* attempt to beat every earlier one — `score > max(scores)` —
            // so a later tie does not displace an earlier winner.
            .fold(None::<(usize, f64)>, |best, (at, attempt)| match best {
                Some((_, high)) if attempt.score <= high => best,
                _ => Some((at, attempt.score)),
            });
        if let Some((at, _)) = best {
            student.load_state(&attempts[at].state)?;
        }

        let mut ranked = attempts;
        // dspy sorts by score descending; a stable sort keeps ties in the order they were tried.
        ranked.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(ranked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seed sequence is three baselines then one per candidate program, which is what makes a
    /// "16 candidate" search 19 attempts.
    #[test]
    fn the_seeds_are_three_baselines_then_the_candidates() {
        let search = BootstrapRandomSearch::new(|_: &Example, _: &Prediction| 0.0);
        assert_eq!(search.seeds().collect::<Vec<_>>()[..4], [-3, -2, -1, 0]);
        assert_eq!(
            search.seeds().count(),
            19,
            "dspy's default 16 candidates plus 3 baselines"
        );
        assert_eq!(
            search
                .with_num_candidate_programs(2)
                .seeds()
                .collect::<Vec<_>>(),
            [-3, -2, -1, 0, 1]
        );
    }

    /// The size draw is a *fresh* generator, not the one the shuffle advanced. Sharing one would
    /// ask for a different number of demos on every seed than dspy asks for.
    #[test]
    fn the_size_is_drawn_from_a_fresh_generator() {
        let search = BootstrapRandomSearch::new(|_: &Example, _: &Prediction| 0.0)
            .with_max_bootstrapped_demos(8);
        let trainset: Vec<Example> = (0..20)
            .map(|n| Example::new([("q", serde_json::json!(n))]))
            .collect();

        for seed in 0..5 {
            let (_, drawn) = search.shuffled(seed, &trainset);
            let expected = Random::seeded(seed as u64).randint(1, 8) as usize;
            assert_eq!(
                drawn, expected,
                "seed {seed} draws from a generator of its own"
            );
            assert!((1..=8).contains(&drawn), "and stays within the range");
        }
    }

    /// The same seed shuffles the same way, and different seeds differently — the whole reason the
    /// search is worth running.
    #[test]
    fn a_seed_shuffles_reproducibly() {
        let search = BootstrapRandomSearch::new(|_: &Example, _: &Prediction| 0.0);
        let trainset: Vec<Example> = (0..10)
            .map(|n| Example::new([("q", serde_json::json!(n))]))
            .collect();
        let order = |seed| {
            search
                .shuffled(seed, &trainset)
                .0
                .iter()
                .map(|example| example.get("q").cloned().unwrap_or_default())
                .collect::<Vec<_>>()
        };
        assert_eq!(order(1), order(1));
        assert_ne!(order(1), order(2));
        assert_ne!(
            order(1),
            order(-1_i64.abs()),
            "and none of them is the original order"
        );
    }
}
