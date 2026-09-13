"""
Polished version of the Hugging Face cross-encoder rerank built in 5-rerank.py.

6-evaluate.py imports from here so the eval script can focus on the metric.
"""

import pandas as pd

from sentence_transformers import CrossEncoder

from utils.fusion import hybrid_candidates
from utils.retrievers import BM25Retriever, DenseRetriever


RERANK_MODEL = "BAAI/bge-reranker-base"

reranker = CrossEncoder(
    RERANK_MODEL,
    device="cuda"
)


def rerank_with_huggingface(
    query: str,
    candidate_ids: list[str],
    corpus_by_id: pd.DataFrame,
    k: int = 10,
) -> list[tuple[str, float]]:

    documents = [
        corpus_by_id.loc[d, "text"]
        for d in candidate_ids
    ]

    pairs = [
        [query, document]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        (doc_id, float(score))
        for doc_id, score in ranked[:k]
    ]


def search_reranked(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:

    candidates = hybrid_candidates(
        query,
        bm25,
        dense,
        candidate_k=candidate_k
    )

    candidate_ids = [
        doc_id
        for doc_id, _ in candidates
    ]

    return rerank_with_huggingface(
        query,
        candidate_ids,
        corpus_by_id,
        k=k
    )