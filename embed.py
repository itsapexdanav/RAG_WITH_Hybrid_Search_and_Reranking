from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data" / "fiqa"
INDEX_DIR = Path(__file__).parent / "indexes" / "dense"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL, device="cuda")
print("Using:", model.device)


def embed_batch(texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=256,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(np.float32)


def build_index(
    doc_texts: list[str],
    batch_size: int = 256
) -> np.ndarray:

    embeddings = []

    for i in tqdm(
        range(0, len(doc_texts), batch_size),
        desc="Embedding"
    ):
        batch = doc_texts[i:i + batch_size]
        embeddings.append(embed_batch(batch))

    return np.vstack(embeddings)


corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
doc_ids = corpus["_id"].tolist()

doc_texts = [
    t.strip() or "[empty document]"
    for t in corpus["text"].tolist()
]

embeddings_path = INDEX_DIR / "embeddings.npy"

if embeddings_path.exists():
    print(f"Loading cached embeddings from {embeddings_path}")
    doc_embeddings = np.load(embeddings_path)

else:
    print(f"Embedding {len(doc_texts)} documents...")
    doc_embeddings = build_index(doc_texts)
    np.save(embeddings_path, doc_embeddings)

doc_embeddings_normed = (
    doc_embeddings /
    np.linalg.norm(
        doc_embeddings,
        axis=1,
        keepdims=True
    )
)


def search_dense(
    query: str,
    k: int = 10
) -> list[tuple[str, float]]:

    query_vec = embed_batch([query])[0]

    query_vec /= np.linalg.norm(query_vec)

    scores = doc_embeddings_normed @ query_vec

    top_k = np.argsort(-scores)[:k]

    return [
        (doc_ids[i], float(scores[i]))
        for i in top_k
    ]


if __name__ == "__main__":

    query = "Where should I park my rainy-day fund?"

    print(f"\nQuery: {query}\n")

    for i, (doc_id, score) in enumerate(
        search_dense(query, k=5),
        1
    ):
        text = corpus.loc[
            corpus["_id"] == doc_id,
            "text"
        ].iloc[0]

        print(
            f"{i}. [{score:.3f}] "
            f"{doc_id} {text}\n"
        )