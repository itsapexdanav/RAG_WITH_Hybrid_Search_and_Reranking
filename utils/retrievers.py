"""
Polished versions of the BM25 and dense retrievers built in 2-bm25.py and
3-embed.py. The later files import from here so each one can focus on the new
idea it introduces (fusion, reranking, evaluation) instead of re-loading state.

This is the same pattern Dave uses in agents/agentic-search: numbered files
teach the construction, then the cleaned-up version graduates to a helper.
"""

from pathlib import Path
import bm25s
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "fiqa"
BM25_DIR = ROOT / "indexes" / "bm25"
DENSE_DIR = ROOT / "indexes" / "dense"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_corpus() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "corpus.parquet")


# --------------------------------------------------------------
# BM25
# --------------------------------------------------------------


class BM25Retriever:
    def __init__(self) -> None:
        self._retriever = bm25s.BM25.load(str(BM25_DIR))
        self._doc_ids = (BM25_DIR / "doc_ids.txt").read_text().splitlines()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([query], stopwords="en")
        indices, scores = self._retriever.retrieve(tokens, k=k)
        return [(self._doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())]


# --------------------------------------------------------------
# Dense
# --------------------------------------------------------------


class DenseRetriever:
    def __init__(self) -> None:
        corpus = load_corpus()
        self._doc_ids = corpus["_id"].tolist()
        raw = np.load(DENSE_DIR / "embeddings.npy")
        self._embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        self._model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False
        )[0].astype(np.float32)
        return vec / np.linalg.norm(vec)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self._embeddings @ self._embed_query(query)
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]