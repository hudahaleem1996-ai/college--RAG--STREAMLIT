import os
from datasets import load_dataset
from pypdf import PdfReader


def load_squad_documents(
    dataset_name: str = "rajpurkar/squad_v2",
    split: str = "validation[:500]",
) -> list[dict]:
    dataset = load_dataset(dataset_name, split=split)
    return [
        {
            "source": f"{dataset_name} [{split}] - row {idx}",
            "text": item["context"],
        }
        for idx, item in enumerate(dataset)
    ]


def load_local_document(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    docs = []
    base_name = os.path.basename(file_path)

    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text()
            if extracted and extracted.strip():
                docs.append({
                    "source": f"{base_name} - Page {page_num}",
                    "text": extracted,
                })
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if content.strip():
            docs.append({
                "source": base_name,
                "text": content,
            })
    else:
        raise ValueError("Unsupported file format. Please provide a .pdf or .txt file.")

    return docs
