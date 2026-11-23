import os
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a helpful academic assistant for college students. "
    "Answer the student's question based ONLY on the provided college rules below. "
    "Respond in the same language as the student's question (Arabic or English). "
    "Clearly mention the source label(s) of the context used in your answer. "
    "If the answer is not explicitly stated in the provided context, reply strictly with: "
    '"عذراً، هذه المعلومة غير مذكورة في لائحة الكلية المتاحة." '
    "Do not guess, assume, or add external information."
)


def build_messages(question: str, retrieved_docs: list[dict]) -> list[dict]:
    formatted_parts = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        formatted_parts.append(
            f"[Source {idx}: {doc.get('source', 'College Rules')}]\n{doc['content']}"
        )
    context_text = "\n\n---\n\n".join(formatted_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"College Rules Context:\n{context_text}\n\nStudent Question: {question}",
        },
    ]


def call_openrouter_chat(
    messages: list[dict],
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    timeout: int = 45,
) -> str:
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Configure it in Streamlit Secrets in TOML format."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_grounded_answer(
    question: str,
    retrieved_docs: list[dict],
    api_key: str = "",
    model: str = "openai/gpt-4o-mini",
    min_similarity: float = 0.68,
) -> dict:
    if not retrieved_docs or retrieved_docs[0]["similarity_score"] < min_similarity:
        return {
            "question": question,
            "answer": "عذراً، هذه المعلومة غير مذكورة في لائحة الكلية المتاحة.",
            "sources": retrieved_docs,
            "found": False,
        }

    messages = build_messages(question, retrieved_docs)
    answer = call_openrouter_chat(
        messages=messages,
        api_key=api_key or OPENROUTER_API_KEY,
        model=model or OPENROUTER_MODEL,
    )

    return {
        "question": question,
        "answer": answer,
        "sources": retrieved_docs,
        "found": True,
    }
