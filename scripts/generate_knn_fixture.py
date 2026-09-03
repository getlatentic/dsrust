"""What dspy's KNN family computes, recorded: the dummy vectorizer's float32 vectors, `KNN`'s
selections, the `Embeddings` retriever's answers, and the bytes numpy writes for a saved index.

`DummyVectorizer` seeds Python's `random` and hashes character n-grams, then centres and
normalises in float32 — arithmetic whose every rounding is observable in a ranking, so the vectors
are recorded exactly. `KNN` scores by a float32 dot product and orders with `np.argsort`, whose
tie-breaking is numpy's own. `Embeddings` normalises (or not), scores, and takes `argsort(-scores)`.
`np.save` writes a header whose padding and dict spelling a reader must accept and a writer must
reproduce for dspy to read the file back.

    .venv/bin/python scripts/generate_knn_fixture.py
"""

from __future__ import annotations

import base64
import io
import json
import pathlib
import sys

import dspy
import numpy as np
from dspy.predict import KNN
from dspy.retrievers.embeddings import Embeddings, EmbeddingsWithScores
from dspy.utils import DummyVectorizer

OUT = pathlib.Path(__file__).parent.parent / "crates" / "dsrust" / "tests" / "conformance" / "knn"
PINNED = (pathlib.Path(__file__).parent / "DSPY_VERSION").read_text().strip()

TEXTS = [
    "question: What is the capital of France?",
    "question: What is the largest ocean?",
    "question: What is 2+2?",
    "question: What is 3+3?",
    "question: What is the capital of Germany?",
    "",
    "ab",
    "héllo wörld — ünïcode",
]


def mock_example(question, answer):
    return dspy.Example(question=question, answer=answer).with_inputs("question")


def dummy_embedder(texts):
    embeddings = []
    for text in texts:
        if "cat" in text:
            embeddings.append(np.array([1, 0, 0], dtype=np.float32))
        elif "dog" in text:
            embeddings.append(np.array([0, 1, 0], dtype=np.float32))
        else:
            embeddings.append(np.array([0, 0, 1], dtype=np.float32))
    return np.stack(embeddings)


def npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    vectorizer = DummyVectorizer()
    vectors = vectorizer(TEXTS)
    coeffs = list(vectorizer.coeffs)

    trainset = [
        mock_example("What is the capital of France?", "Paris"),
        mock_example("What is the largest ocean?", "Pacific"),
        mock_example("What is 2+2?", "4"),
    ]
    knn = KNN(k=2, trainset=trainset, vectorizer=dspy.Embedder(DummyVectorizer()))
    queries = ["What is 3+3?", "What is the capital of Germany?", "What is the largest ocean?"]
    selections = {}
    for query in queries:
        nearest = knn(question=query)
        selections[query] = [trainset.index(example) for example in nearest]
    trainset_vectors = knn.trainset_vectors.tolist()

    corpus = ["The cat sat on the mat.", "The dog barked at the mailman.", "Birds fly in the sky."]
    retriever = EmbeddingsWithScores(corpus=corpus, embedder=dummy_embedder, k=2)
    unnormalized = EmbeddingsWithScores(corpus=corpus, embedder=dummy_embedder, k=3, normalize=False)
    searches = {}
    for query in ["A dog is barking.", "cat sitting", "nothing at all"]:
        got = retriever(query)
        raw = unnormalized(query)
        searches[query] = {
            "normalized": {"passages": got.passages, "indices": got.indices, "scores": got.scores},
            "unnormalized": {"passages": raw.passages, "indices": raw.indices, "scores": raw.scores},
        }
    saved = {
        "config": {"k": retriever.k, "normalize": retriever.normalize, "corpus": retriever.corpus, "has_faiss_index": retriever.index is not None},
        "corpus_embeddings_npy": npy_bytes(retriever.corpus_embeddings),
        "corpus_embeddings": retriever.corpus_embeddings.tolist(),
        "unnormalized_npy": npy_bytes(unnormalized.corpus_embeddings),
    }
    # A second array shape, so the header's padding rule is held beyond one case.
    wide = np.arange(2 * 50, dtype=np.float32).reshape(2, 50) / 7
    # numpy's float32 reductions over rows of every length the pairwise sum treats differently —
    # under eight, up to the 128-element block, and past it — and its argsort over ties.
    rng = np.random.default_rng(7)
    lengths = [3, 8, 9, 100, 128, 129, 300]
    numpy_rows = [rng.standard_normal(n).astype(np.float32) for n in lengths]
    pairs = [(rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)) for n in (3, 100, 129)]
    tied = np.array([0.5, -0.0, 1.0, 0.5, 0.0, 1.0, -1.0, 0.5, 2.0, 0.5, 1.0, 0.5, 0.0, 0.5, 0.5, 1.0, 0.5, 0.25], dtype=np.float32)
    numpy_facts = {
        "rows": [row.tolist() for row in numpy_rows],
        "means": [float(np.mean(row)) for row in numpy_rows],
        # Along an axis, as the vectorizer and the retriever take it: `sqrt(add.reduce(x * x))`, the
        # pairwise sum. A one-dimensional `norm(row)` goes through BLAS's dot instead and can differ
        # in its last bit.
        "norms": [float(np.linalg.norm(row[None, :], axis=1)[0]) for row in numpy_rows],
        "dots": [{"a": a.tolist(), "b": b.tolist(), "dot": float(np.dot(a, b))} for a, b in pairs],
        "tied_scores": tied.tolist(),
        "tied_argsort": np.argsort(tied).tolist(),
    }
    fixture = {
        "source": f"generated from dspy=={PINNED} via scripts/generate_knn_fixture.py",
        "numpy": numpy_facts,
        "dspy_version": PINNED,
        "numpy_version": np.__version__,
        "vectorizer": {"max_length": 100, "n_gram": 2, "coeffs": coeffs, "texts": TEXTS, "vectors": vectors.tolist()},
        "knn": {
            "trainset": [{"question": e.question, "answer": e.answer} for e in trainset],
            "trainset_vectors": trainset_vectors,
            "k": 2,
            "selections": selections,
        },
        "embeddings": {"corpus": corpus, "searches": searches, "saved": saved},
        "npy": {"wide": npy_bytes(wide), "wide_values": wide.tolist()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "knn.json"
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(OUT.parents[2])}", file=sys.stderr)
    print(f"    coeffs {coeffs}; selections {selections}", file=sys.stderr)
    for query, got in searches.items():
        print(f"    {query!r}: {got['normalized']['indices']} {got['normalized']['scores']}", file=sys.stderr)


if __name__ == "__main__":
    main()
