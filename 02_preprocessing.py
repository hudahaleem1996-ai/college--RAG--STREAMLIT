import re


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_documents(documents: list[dict]) -> list[dict]:
    seen_texts = set()
    cleaned_docs = []

    for doc in documents:
        cleaned = clean_text(doc["text"])
        if not cleaned or cleaned in seen_texts:
            continue
        seen_texts.add(cleaned)
        cleaned_docs.append({
            "source": doc.get("source", "unknown_source"),
            "text": cleaned,
        })

    return cleaned_docs
