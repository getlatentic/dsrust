//! dspy `retrievers/`: what fetches passages for a program to read.

pub mod embeddings;
pub mod npy;

pub use embeddings::{Embeddings, EmbeddingsWithScores, Retrieved};
