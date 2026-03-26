import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

# 🔑 SET YOUR OPENAI KEY HERE
openai.api_key = "YOUR_API_KEY_HERE"

# ---------------- UI ----------------
st.set_page_config(page_title="Healthcare AI Assistant", layout="centered")

st.title("🩺 Healthcare AI Assistant")
st.write("Upload medical PDFs and ask questions")
st.warning("⚠️ AI-generated content. Always consult a doctor.")

# ---------------- Upload ----------------
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

# ---------------- Extract Text ----------------
def extract_text(files):
    text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# ---------------- Chunking ----------------
def split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ---------------- Embedding ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Ask OpenAI ----------------
def ask_openai(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer based on the medical document context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )
    return response["choices"][0]["message"]["content"]

# ---------------- Main Logic ----------------
if uploaded_files:
    text = extract_text(uploaded_files)
    st.success(f"Loaded {len(text)} characters")

    chunks = split_text(text)
    st.info(f"Created {len(chunks)} chunks")

    model = load_model()
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    query = st.text_input("Ask a question:")

    if query:
        query_vec = model.encode([query])
        D, I = index.search(np.array(query_vec), k=3)

        context = " ".join([chunks[i] for i in I[0]])

        answer = ask_openai(query, context)
        st.write("### 💬 Answer:")
        st.write(answer)