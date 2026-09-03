//! dspy `predict/knn.py::KNN`: the k training examples nearest an input, by embedding.
//!
//! Each training example's inputs are rendered `key: value | key: value` and embedded once; a call
//! renders its own inputs the same way, scores every training vector by dot product, and answers
//! with the top k in descending order — `np.argsort(scores)[-k:][::-1]`, numpy's permutation.

use std::sync::Arc;

use anyhow::Result;

use crate::example::Example;
use crate::lm::embedding::Embedder;
use crate::numpy::{argsort_f32, dot_f32};
use crate::python::text;

pub struct Knn {
    k: usize,
    trainset: Vec<Example>,
    embedder: Arc<Embedder>,
    trainset_vectors: Vec<Vec<f32>>,
}

impl Knn {
    /// `KNN(k, trainset, vectorizer)`: the training set embedded now.
    pub async fn build(k: usize, trainset: Vec<Example>, embedder: Arc<Embedder>) -> Result<Self> {
        let texts: Vec<String> = trainset.iter().map(example_text).collect::<Result<_>>()?;
        let trainset_vectors = embedder.call(&texts).await?;
        Ok(Self {
            k,
            trainset,
            embedder,
            trainset_vectors,
        })
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn trainset(&self) -> &[Example] {
        &self.trainset
    }

    pub fn trainset_vectors(&self) -> &[Vec<f32>] {
        &self.trainset_vectors
    }

    /// `knn(**kwargs)`: the k nearest training examples to these inputs, nearest first.
    pub async fn call(&self, inputs: &Example) -> Result<Vec<Example>> {
        let query = self.embedder.call_one(&query_text(inputs)).await?;
        Ok(nearest_indices(&self.trainset_vectors, &query, self.k)
            .into_iter()
            .map(|at| self.trainset[at].clone())
            .collect())
    }
}

/// A training example as it is embedded: its inputs, `key: value | key: value`. An example whose
/// inputs were never marked is refused, as upstream's `key in None` raises.
pub fn example_text(example: &Example) -> Result<String> {
    let inputs = example.inputs()?;
    Ok(rendered(
        inputs
            .fields()
            .filter(|(name, _)| !name.starts_with("dspy_")),
    ))
}

/// A call's inputs as they are embedded: every field given, in the order given.
pub fn query_text(inputs: &Example) -> String {
    rendered(inputs.fields())
}

/// The `k` nearest rows to `query` by dot product, nearest first: `np.argsort(scores)[-k:][::-1]`.
pub fn nearest_indices(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let scores: Vec<f32> = vectors.iter().map(|row| dot_f32(row, query)).collect();
    let order = argsort_f32(&scores);
    let nearest = order.len().saturating_sub(k);
    order[nearest..].iter().rev().copied().collect()
}

/// Upstream's `" | ".join(f"{key}: {value}" ...)`, each value as Python's `str` prints it.
fn rendered<'a>(fields: impl Iterator<Item = (&'a str, &'a serde_json::Value)>) -> String {
    fields
        .map(|(name, value)| format!("{name}: {}", text(value)))
        .collect::<Vec<_>>()
        .join(" | ")
}
