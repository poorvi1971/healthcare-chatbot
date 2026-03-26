import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ---------------- UI ----------------
st.set_page_config(page_title="Healthcare AI Assistant", layout="centered")

st.title("🩺 Healthcare AI Assistant")
st.write("Upload medical PDFs and ask intelligent questions")

st.warning("⚠️ AI-generated content. Always consult a doctor.")

# ---------------- Upload ----------------
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

# ---------------- Extract Text ----------------
def extract_text(files):
    text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# ---------------- Chunking ----------------
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1  # CPU
    )

    return embed_model, llm

embed_model, llm = load_models()

# ---------------- FAISS Index ----------------
def create_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, embeddings

# ---------------- Ask LLM ----------------
def get_answer(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    result = llm(prompt, max_length=200, do_sample=True)
    return result[0]['generated_text']

# ---------------- MAIN ----------------
if uploaded_files:
    text = extract_text(uploaded_files)
    st.success(f"Loaded documents ({len(text)} characters)")

    chunks = chunk_text(text)
    st.info(f"Created {len(chunks)} chunks")

    index, embeddings = create_faiss_index(chunks)

    query = st.text_input("Ask a question")

    if query:
        query_vec = embed_model.encode([query])
        D, I = index.search(np.array(query_vec), k=3)

        context = " ".join([chunks[i] for i in I[0]])

        answer = get_answer(query, context)

        st.subheader("Answer:")
        st.write(answer)