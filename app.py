import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------- UI ----------------
st.set_page_config(page_title="Healthcare AI Assistant", layout="centered")

st.title("🩺 Healthcare AI Assistant")
st.write("Upload a medical PDF and ask questions")

st.warning("⚠️ AI-generated content. Verify with a doctor.")

# ---------------- Upload ----------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ---------------- Extract Text ----------------
def extract_text(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# ---------------- Chunking ----------------
def split_text(text, chunk_size=300):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ---------------- Load Embedding Model ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Main ----------------
if uploaded_file:

    text = extract_text(uploaded_file)
    st.success("PDF processed successfully!")

    chunks = split_text(text)

    model = load_model()
    embeddings = model.encode(chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    query = st.text_input("Ask a question:")

    if query:
        query_vec = model.encode([query])
        D, I = index.search(np.array(query_vec), k=3)

        # Get relevant chunks
        context = " ".join([chunks[i] for i in I[0]])

        # ---------------- SIMPLE ANSWER (NO REPEATING ISSUE) ----------------
        answer = context.strip()

        st.subheader("Answer:")
        st.write(answer[:800])  # limit to avoid repetition