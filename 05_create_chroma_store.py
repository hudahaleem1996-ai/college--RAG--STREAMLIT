import chromadb
import numpy as np

DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_COLLECTION_NAME = "college_rules_collection"


def create_or_get_chroma_collection(
    persist_directory: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=persist_directory)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_chroma_store(
    chunk_records: list[dict],
    embeddings: np.ndarray,
    persist_directory: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=persist_directory)
    existing_names = [col.name for col in client.list_collections()]
    if collection_name in existing_names:
        client.delete_collection(name=collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[rec["id"] for rec in chunk_records],
        documents=[rec["content"] for rec in chunk_records],
        embeddings=embeddings.tolist(),
        metadatas=[{"source": rec.get("source", "College Rules")} for rec in chunk_records],
    )
    return collection
