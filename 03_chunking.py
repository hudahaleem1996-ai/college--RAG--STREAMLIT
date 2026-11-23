def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[dict]:
    chunked_records = []
    for doc_idx, doc in enumerate(documents):
        text_chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for chunk_idx, chunk_str in enumerate(text_chunks):
            chunked_records.append({
                "id": f"doc_{doc_idx}_chunk_{chunk_idx}",
                "content": chunk_str,
                "source": doc.get("source", f"Document {doc_idx + 1}"),
            })
    return chunked_records
