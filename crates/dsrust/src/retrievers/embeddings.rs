//! dspy `retrievers/embeddings.py`: the top-k passages nearest a query, by embedding.
//!
//! The corpus is embedded once — normalised unless told not to — and a query is scored against
//! every passage by dot product, then ordered by `np.argsort(-scores)`. Upstream builds a FAISS
//! index past `brute_force_threshold` when faiss is installed, an approximate search whose
//! candidates it re-ranks exactly; here every corpus is searched exactly, which agrees wherever
//! the index recalls the true top-k. `save` and `load` speak upstream's files: its config file and
//! numpy's matrix file.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde_json::json;

use super::npy;
use crate::example::{Example, Prediction};
use crate::lm::embedding::Embedder;
use crate::numpy::{argsort_f32, dot_f32, l2_norm_f32};

/// What a search answers: the passages, their corpus indices, and their scores.
#[derive(Debug, Clone, PartialEq)]
pub struct Retrieved {
    pub passages: Vec<String>,
    pub indices: Vec<usize>,
    pub scores: Vec<f32>,
}

pub struct Embeddings {
    corpus: Vec<String>,
    embedder: Arc<Embedder>,
    k: usize,
    normalize: bool,
    corpus_embeddings: Vec<Vec<f32>>,
}

impl Embeddings {
    /// `Embeddings(corpus, embedder, k=5, normalize=True)`, the corpus embedded now.
    pub async fn build(
        corpus: Vec<String>,
        embedder: Arc<Embedder>,
        k: usize,
        normalize: bool,
    ) -> Result<Self> {
        let mut corpus_embeddings = embedder.call(&corpus).await?;
        if normalize {
            normalize_rows(&mut corpus_embeddings);
        }
        Ok(Self {
            corpus,
            embedder,
            k,
            normalize,
            corpus_embeddings,
        })
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn normalize(&self) -> bool {
        self.normalize
    }

    pub fn corpus(&self) -> &[String] {
        &self.corpus
    }

    pub fn corpus_embeddings(&self) -> &[Vec<f32>] {
        &self.corpus_embeddings
    }

    /// `retriever(query)`: `Prediction(passages=..., indices=...)`.
    pub async fn forward(&self, query: &str) -> Result<Prediction> {
        let found = self.search(query).await?;
        Ok(Prediction::new(
            Example::new([
                ("passages", json!(found.passages)),
                ("indices", json!(found.indices)),
            ]),
            "",
        ))
    }

    /// The search itself, scores included.
    pub async fn search(&self, query: &str) -> Result<Retrieved> {
        let mut query_embedding = self.embedder.call_one(query).await?;
        if self.normalize {
            let mut rows = vec![query_embedding];
            normalize_rows(&mut rows);
            query_embedding = rows.pop().unwrap_or_default();
        }
        let scores: Vec<f32> = self
            .corpus_embeddings
            .iter()
            .map(|row| dot_f32(&query_embedding, row))
            .collect();
        let negated: Vec<f32> = scores.iter().map(|score| -score).collect();
        let indices: Vec<usize> = argsort_f32(&negated).into_iter().take(self.k).collect();
        Ok(Retrieved {
            passages: indices.iter().map(|at| self.corpus[*at].clone()).collect(),
            scores: indices.iter().map(|at| scores[*at]).collect(),
            indices,
        })
    }

    /// `save(path)`: upstream's config file and numpy's matrix file in `path`, as upstream writes them.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))?;
        let config = json!({
            "k": self.k,
            "normalize": self.normalize,
            "corpus": self.corpus,
            "has_faiss_index": false,
        });
        std::fs::write(
            path.join("config.json"),
            serde_json::to_string_pretty(&config)?,
        )?;
        std::fs::write(
            path.join("corpus_embeddings.npy"),
            npy::encode_f32(&self.corpus_embeddings),
        )?;
        Ok(())
    }

    /// `load(path, embedder)`: the saved corpus, configuration and embeddings replace this one's.
    pub fn load(&mut self, path: impl AsRef<Path>, embedder: Arc<Embedder>) -> Result<&mut Self> {
        let path = path.as_ref();
        let config_path = path.join("config.json");
        if !config_path.exists() {
            bail!("No config.json found at {}", config_path.display());
        }
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        self.k = config["k"].as_u64().context("config.json: `k`")? as usize;
        self.normalize = config["normalize"]
            .as_bool()
            .context("config.json: `normalize`")?;
        self.corpus = config["corpus"]
            .as_array()
            .context("config.json: `corpus`")?
            .iter()
            .filter_map(|passage| passage.as_str().map(str::to_owned))
            .collect();
        self.corpus_embeddings =
            npy::decode_f32(&std::fs::read(path.join("corpus_embeddings.npy"))?)?;
        self.embedder = embedder;
        Ok(self)
    }

    /// `Embeddings.from_saved(path, embedder)`.
    pub fn from_saved(path: impl AsRef<Path>, embedder: Arc<Embedder>) -> Result<Self> {
        let mut loaded = Self {
            corpus: Vec::new(),
            embedder: Arc::clone(&embedder),
            k: 5,
            normalize: true,
            corpus_embeddings: Vec::new(),
        };
        loaded.load(path, embedder)?;
        Ok(loaded)
    }
}

/// dspy `EmbeddingsWithScores`: the same search, answering with the scores as well.
pub struct EmbeddingsWithScores(pub Embeddings);

impl EmbeddingsWithScores {
    pub async fn build(
        corpus: Vec<String>,
        embedder: Arc<Embedder>,
        k: usize,
        normalize: bool,
    ) -> Result<Self> {
        Ok(Self(
            Embeddings::build(corpus, embedder, k, normalize).await?,
        ))
    }

    pub fn from_saved(path: impl AsRef<Path>, embedder: Arc<Embedder>) -> Result<Self> {
        Ok(Self(Embeddings::from_saved(path, embedder)?))
    }

    /// `Prediction(passages=..., indices=..., scores=...)`.
    pub async fn forward(&self, query: &str) -> Result<Prediction> {
        let found = self.0.search(query).await?;
        Ok(Prediction::new(
            Example::new([
                ("passages", json!(found.passages)),
                ("indices", json!(found.indices)),
                ("scores", json!(found.scores)),
            ]),
            "",
        ))
    }
}

/// Upstream's `_normalize`: each row divided by its norm, or by `1e-10` where the norm is smaller.
fn normalize_rows(rows: &mut [Vec<f32>]) {
    for row in rows {
        let norm = l2_norm_f32(row).max(1e-10f32);
        for value in row.iter_mut() {
            *value /= norm;
        }
    }
}
