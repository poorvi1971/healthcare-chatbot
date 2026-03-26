import streamlit as st
from pypdf import PdfReader
from transformers import pipeline

# ---------------- UI ----------------
st.set_page_config(page_title="Healthcare AI Assistant", layout="centered")

st.title("🩺 Healthcare AI Assistant")
st.write("Upload medical PDFs and ask intelligent questions")

st.warning("⚠️ AI-generated content. Always consult a doctor.")

# ---------------- Upload ----------------
uploaded_files = st.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True
)

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

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

# ---------------- Main Logic ----------------
if uploaded_files:
    text_data = extract_text(uploaded_files)

    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question about your document:")

    if question:
        pipe = load_model()

        prompt = f"Context: {text_data[:2000]}\n\nQuestion: {question}"

        response = pipe(prompt, max_length=200)

        answer = response[0]['generated_text']

        st.subheader("Answer:")
        st.write(answer)