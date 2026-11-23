import importlib.util
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="College Rules RAG Assistant", layout="wide")


def _load(name: str, filename: str):
    path = str(Path(__file__).with_name(filename))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


docs_mod = _load("m01", "01_documents.py")
prep_mod = _load("m02", "02_preprocessing.py")
chunk_mod = _load("m03", "03_chunking.py")
vec_mod = _load("m04", "04_vector_representation.py")
chroma_mod = _load("m05", "05_create_chroma_store.py")
retr_mod = _load("m06", "06_retrieve_context.py")
prompt_mod = _load("m07", "07_prompting.py")

try:
    if not prompt_mod.OPENROUTER_API_KEY:
        prompt_mod.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    prompt_mod.OPENROUTER_MODEL = st.secrets.get(
        "OPENROUTER_MODEL", prompt_mod.OPENROUTER_MODEL
    )
except Exception:
    pass


@st.cache_resource
def init_retriever(chunk_size: int = 150, overlap: int = 30):
    raw_docs = docs_mod.load_squad_documents()
    clean_docs = prep_mod.preprocess_documents(raw_docs)
    chunk_records = chunk_mod.chunk_documents(
        clean_docs, chunk_size=chunk_size, overlap=overlap
    )
    embed_model = vec_mod.load_embedding_model()
    embeddings = vec_mod.embed_passages(
        embed_model, [r["content"] for r in chunk_records]
    )
    collection = chroma_mod.build_chroma_store(chunk_records, embeddings)
    return collection, embed_model


st.title("مساعد لائحة وقواعد الكلية (RAG)")
st.caption("إجابات مبنية على السياق المسترجع عبر OpenRouter مع ذكر المصادر.")

with st.sidebar:
    st.header("إعدادات الاسترجاع")
    top_k = st.slider("عدد المقاطع المسترجعة (top_k)", min_value=1, max_value=5, value=3)
    min_similarity = st.slider(
        "حد التشابه الأدنى (min_similarity)",
        min_value=0.10,
        max_value=0.95,
        value=0.68,
        step=0.02,
    )

with st.spinner("جاري تجهيز قاعدة المعرفة ومخزن المتجهات..."):
    collection, embed_model = init_retriever()

question = st.text_input("اكتبي سؤالك حول اللائحة أو القواعد:")

if st.button("إرسال السؤال") and question.strip():
    try:
        with st.spinner("جاري استرجاع السياق وتوليد الإجابة..."):
            retrieved_docs = retr_mod.retrieve_context(
                query=question.strip(),
                collection=collection,
                embed_model=embed_model,
                top_k=top_k,
            )
            result = prompt_mod.generate_grounded_answer(
                question=question.strip(),
                retrieved_docs=retrieved_docs,
                api_key=prompt_mod.OPENROUTER_API_KEY,
                model=prompt_mod.OPENROUTER_MODEL,
                min_similarity=min_similarity,
            )

        st.subheader("الإجابة")
        st.write(result["answer"])

        st.subheader("المصادر المسترجعة")
        for idx, src in enumerate(result["sources"], start=1):
            with st.expander(
                f"مصدر {idx}: {src['source']} | التشابه: {src['similarity_score']:.4f}"
            ):
                st.write(src["content"])
    except Exception as exc:
        st.error(f"حدث خطأ أثناء توليد الإجابة: {exc}")
