"""
main.py

FastAPI wrapper around the existing multimodal RAG pipeline.
Each chat has its OWN document index -- uploading a PDF in one chat does
not affect any other chat's context. This mirrors ChatGPT's per-conversation
file scoping.

Run with: uvicorn main:app --reload
"""

import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_loader import load_pdf_elements
from retriever import add_documents_to_index, load_hybrid_retriever
from chain import answer_question
import chats

app = FastAPI(title="Multimodal PDF Q&A")

# Allows a frontend running on a different port/origin (React dev server, etc.)
# to call this API from the browser. Tighten allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chats.init_db()


def chat_paths(chat_id: int):
    """Every chat gets its own folder, so its documents never mix with another chat's."""
    base = f"data/chats/{chat_id}"
    return {
        "index_dir": f"{base}/faiss_index",
        "docs_path": f"{base}/documents.pkl",
    }


# Cached retrievers, keyed by chat_id, same idea as st.cache_resource in app.py.
# A chat's entry is cleared whenever a new PDF is uploaded to THAT chat, so the
# next question in that chat loads a fresh retriever -- other chats are untouched.
_retrievers = {}


def get_retriever(chat_id: int):
    if chat_id not in _retrievers:
        paths = chat_paths(chat_id)
        _retrievers[chat_id] = load_hybrid_retriever(k=5, **paths)
    return _retrievers[chat_id]


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Chat session endpoints ---

@app.post("/chats")
def create_chat():
    chat_id = chats.create_chat()
    return {"id": chat_id, "title": "New chat"}


@app.get("/chats")
def list_chats():
    return chats.list_chats()


@app.get("/chats/{chat_id}/messages")
def get_chat_messages(chat_id: int):
    if not chats.chat_exists(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    return chats.get_messages(chat_id)


@app.delete("/chats/{chat_id}")
def remove_chat(chat_id: int):
    if not chats.chat_exists(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    chats.delete_chat(chat_id)
    _retrievers.pop(chat_id, None)
    return {"deleted": chat_id}


# --- Per-chat document upload ---

@app.post("/chats/{chat_id}/upload")
async def upload_pdf(chat_id: int, file: UploadFile = File(...)):
    if not chats.chat_exists(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file to a temp path since load_pdf_elements needs a file path, not bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        documents = load_pdf_elements(tmp_path)
        paths = chat_paths(chat_id)
        total_docs = add_documents_to_index(documents, **paths)
    finally:
        os.remove(tmp_path)

    # Force the next question in THIS chat to reload the retriever with the new documents.
    # Other chats' cached retrievers are untouched.
    _retrievers.pop(chat_id, None)

    return {
        "filename": file.filename,
        "chunks_added": len(documents),
        "total_chunks": total_docs,
    }


@app.post("/chats/{chat_id}/query")
def query(chat_id: int, request: QuestionRequest):
    if not chats.chat_exists(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")

    paths = chat_paths(chat_id)
    if not os.path.exists(paths["index_dir"]):
        raise HTTPException(status_code=400, detail="No documents uploaded in this chat yet.")

    retriever = get_retriever(chat_id)
    answer = answer_question(request.question, retriever)

    chats.add_message(chat_id, "user", request.question)
    chats.add_message(chat_id, "assistant", answer)
    chats.set_title_if_default(chat_id, request.question)

    return {"question": request.question, "answer": answer}