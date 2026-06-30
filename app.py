"""
app.py

Single Streamlit app: upload a PDF, it gets processed automatically,
then you can ask questions. No terminal commands needed after setup.

Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st

from data_loader import load_pdf_elements
from image_summarizer import summarize_images
from retriever import add_documents_to_index, load_hybrid_retriever
from chain import answer_question

st.title("Multimodal PDF Q&A")

# Keeps track of which files we've already processed in this session,
# so re-running the app doesn't make you re-upload to re-ask questions.
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


def process_pdf(uploaded_file):
    """Runs the full pipeline on one uploaded PDF and adds it to the index."""
    # Save the uploaded file to a temp path since our parser needs a file path, not bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    status = st.status(f"Processing {uploaded_file.name}...", expanded=True)

    status.write("Parsing PDF (text, tables, images)...")
    status.write("Building searchable chunks...")
    documents= load_pdf_elements(tmp_path)


    status.write("Adding to index...")
    total_docs = add_documents_to_index(documents)

    os.remove(tmp_path)
    status.update(label=f"Done! {uploaded_file.name} added. Index now has {total_docs} chunks.", state="complete")


# --- Upload section ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None and uploaded_file.name not in st.session_state.processed_files:
    process_pdf(uploaded_file)
    st.session_state.processed_files.append(uploaded_file.name)
    # Clear cached retriever so it picks up the newly added documents
    st.cache_resource.clear()

# --- Q&A section ---
st.divider()

index_exists = os.path.exists("data/faiss_index")

if not index_exists:
    st.info("Upload a PDF above to get started.")
else:
    @st.cache_resource
    def get_retriever():
        return load_hybrid_retriever(k=5)

    retriever = get_retriever()

    question = st.text_input("Ask a question about your uploaded document(s):")

    if question:
        with st.spinner("Searching and generating answer..."):
            answer = answer_question(question, retriever)
        st.write(answer)
