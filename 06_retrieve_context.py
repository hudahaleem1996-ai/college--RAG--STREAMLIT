import importlib.util
from pathlib import Path
from sentence_transformers import SentenceTransformer


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_vec_mod = _load_module_from_path(
    "vector_rep", str(Path(__file__).with_name("04_vector_representation.py"))
)
embed_query = _vec_mod.embed_query


def retrieve_context(
    query: str,
    collection,
    embed_model: SentenceTransformer,
    top_k: int = 3,
) -> list[dict]:
    query_vec = embed_query(embed_model, query).tolist()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []
    for content, meta, dist in zip(docs, metas, distances):
        similarity = max(0.0, min(1.0, 1.0 - float(dist)))
        retrieved.append({
            "content": content,
            "source": (meta or {}).get("source", "College Rules"),
            "similarity_score": similarity,
        })

    return retrieved
