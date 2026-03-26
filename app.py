import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

# ---------------- Process ----------------
if uploaded_files:
    raw_text = extract_text(uploaded_files)

    # Split into chunks
    chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings
    embeddings = model.encode(chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.success("PDF processed successfully!")

    # ---------------- Ask Question ----------------
    query = st.text_input("Ask a question:")

    if query:
        # Encode query
        query_vec = model.encode([query])
        D, I = index.search(np.array(query_vec), k=3)

        # Get relevant chunks
        context = " ".join([chunks[i] for i in I[0]])

        # Split into sentences
        sentences = context.split(".")
        question_words = query.lower().split()

        best_sentences = []

        for sentence in sentences:
            score = 0
            for word in question_words:
                if word in sentence.lower():
                    score += 1

            if score > 0:
                best_sentences.append(sentence.strip())

        # Remove duplicates
        best_sentences = list(dict.fromkeys(best_sentences))

        # Take top 3 sentences
        final_answer = ". ".join(best_sentences[:3])

        # Display answer
        st.subheader("Answer:")
        st.write(final_answer if final_answer else "No relevant answer found.")