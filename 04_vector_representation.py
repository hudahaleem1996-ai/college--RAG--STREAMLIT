import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"


def load_embedding_model(model_id: str = DEFAULT_EMBED_MODEL) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_id, device=device)


def embed_passages(model: SentenceTransformer, chunks: list[str]) -> np.ndarray:
    passages = [f"passage: {c}" for c in chunks]
    embeddings = model.encode(
        passages,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    embedding = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return np.asarray(embedding, dtype=np.float32)
