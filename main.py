"""
main.py

FastAPI backend for MERGE — a multimodal RAG pipeline for chatting with your
documents. Each chat belongs to the user who created it and has its OWN
document index -- uploading a PDF in one chat does not affect any other
chat's context, and no user can see or touch another user's chats.

Auth: a simple username/password dictionary (DEFAULT_USERS below, or
override via the MERGE_USERS env var).

Run with: uvicorn main:app --reload
"""

import os
import json
import secrets
import tempfile
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_loader import load_pdf_elements
from retriever import add_documents_to_index, load_hybrid_retriever
from chain import answer_question
import chats

logger = logging.getLogger("merge")

app = FastAPI(title="MERGE — Multimodal PDF Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chats.init_db()


# ============================================================
# Auth
# ============================================================
#
# Simple id/password dictionary. Change these defaults before deploying,
# or set MERGE_USERS to a JSON object, e.g.:
#   export MERGE_USERS='{"admin": "correct-horse-battery-staple", "sam": "hunter2"}'
DEFAULT_USERS = {
    "admin": "admin123",
    "anshu": "anshu123"
}

def load_users() -> dict:
    raw = os.environ.get("MERGE_USERS")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("MERGE_USERS is not valid JSON, falling back to defaults.")
    return DEFAULT_USERS

USERS = load_users()

# In-memory session tokens: token -> username. Fine for a single-process app;
# swap for a real session store / JWT if you deploy this behind multiple workers.
_sessions: dict[str, str] = {}


class LoginRequest(BaseModel):
    username: str
    password: str


def require_auth(authorization: str | None = Header(default=None)) -> str:
    """Dependency that protects every chat/document route. Expects
    'Authorization: Bearer <token>' from a prior /login call. Returns the
    logged-in username, which every route below uses to scope data."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    token = authorization.removeprefix("Bearer ").strip()
    username = _sessions.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="Session expired or invalid. Please sign in again.")
    return username


@app.post("/login")
def login(request: LoginRequest):
    expected = USERS.get(request.username)
    success = expected is not None and secrets.compare_digest(expected, request.password)

    if not success:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = secrets.token_urlsafe(32)
    _sessions[token] = request.username
    return {"token": token, "username": request.username}


@app.post("/logout")
def logout(username: str = Depends(require_auth), authorization: str = Header(...)):
    token = authorization.removeprefix("Bearer ").strip()
    _sessions.pop(token, None)
    return {"logged_out": True}


# ============================================================
# Chat / document routes (all require auth, all scoped to the caller)
# ============================================================

def chat_paths(username: str, chat_id: int):
    """Every chat gets its own folder, nested under the owning user, so
    its documents never mix with another chat's -- and never leak across
    users even if a chat_id were ever guessed.

    This lives under the same out-of-project DATA_DIR as chats.db (see
    chats.py) and NOT under a "data/" folder next to main.py. Every PDF
    upload writes new index files here, and if that path sat inside the
    directory `uvicorn --reload` watches, each upload would trigger a
    full server restart -- wiping in-memory sessions and bouncing users
    back to the login screen, which looks like the page refreshing.
    """
    base = chats.DATA_DIR / "chats" / username / str(chat_id)
    return {
        "index_dir": str(base / "faiss_index"),
        "docs_path": str(base / "documents.pkl"),
    }


# Cached retrievers, keyed by chat_id (chat_ids are globally unique, and
# every route below already verifies ownership before a chat_id reaches
# this cache). Cleared for a chat whenever a new PDF is uploaded to it, so
# the next question in that chat loads a fresh retriever -- other chats
# are untouched.
_retrievers = {}


def get_retriever(username: str, chat_id: int):
    if chat_id not in _retrievers:
        paths = chat_paths(username, chat_id)
        _retrievers[chat_id] = load_hybrid_retriever(k=5, **paths)
    return _retrievers[chat_id]


def require_own_chat(chat_id: int, username: str):
    """Raises 404 if the chat doesn't exist OR belongs to someone else --
    same response either way, so a user can't tell the difference between
    'no such chat' and 'not yours'."""
    if not chats.chat_belongs_to(chat_id, username):
        raise HTTPException(status_code=404, detail="Chat not found.")


class QuestionRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


# --- Chat session endpoints ---

@app.post("/chats")
def create_chat(username: str = Depends(require_auth)):
    chat_id = chats.create_chat(username)
    return {"id": chat_id, "title": "New chat"}


@app.get("/chats")
def list_chats(username: str = Depends(require_auth)):
    return chats.list_chats(username)


@app.get("/chats/{chat_id}/messages")
def get_chat_messages(chat_id: int, username: str = Depends(require_auth)):
    require_own_chat(chat_id, username)
    return chats.get_messages(chat_id)


@app.delete("/chats/{chat_id}")
def remove_chat(chat_id: int, username: str = Depends(require_auth)):
    require_own_chat(chat_id, username)
    chats.delete_chat(chat_id)
    _retrievers.pop(chat_id, None)
    return {"deleted": chat_id}


# --- Per-chat document upload ---

@app.get("/chats/{chat_id}/documents")
def get_chat_documents(chat_id: int, username: str = Depends(require_auth)):
    """List every PDF that has been indexed into this chat so far. The
    frontend calls this whenever a chat is opened, so uploaded files keep
    showing up at the top of the chat even after a reload or after
    switching to another chat and back -- they're no longer only tracked
    in browser memory."""
    require_own_chat(chat_id, username)
    return chats.get_documents(chat_id)


@app.post("/chats/{chat_id}/upload")
async def upload_pdf(chat_id: int, file: UploadFile = File(...), username: str = Depends(require_auth)):
    require_own_chat(chat_id, username)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file to a temp path since load_pdf_elements needs a file path, not bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        documents = load_pdf_elements(tmp_path)
        paths = chat_paths(username, chat_id)
        total_docs = add_documents_to_index(documents, **paths)
    finally:
        os.remove(tmp_path)

    # Force the next question in THIS chat to reload the retriever with the new documents.
    # Other chats' cached retrievers are untouched.
    _retrievers.pop(chat_id, None)

    # Persist the upload so it survives a reload / chat switch -- previously
    # this list only lived in the browser's in-memory filesByChat map.
    chats.add_document(chat_id, file.filename, len(documents))

    return {
        "filename": file.filename,
        "chunks_added": len(documents),
        "total_chunks": total_docs,
    }


@app.post("/chats/{chat_id}/query")
def query(chat_id: int, request: QuestionRequest, username: str = Depends(require_auth)):
    require_own_chat(chat_id, username)

    paths = chat_paths(username, chat_id)
    if not os.path.exists(paths["index_dir"]):
        raise HTTPException(status_code=400, detail="No documents uploaded in this chat yet.")

    retriever = get_retriever(username, chat_id)
    answer = answer_question(request.question, retriever)

    chats.add_message(chat_id, "user", request.question)
    chats.add_message(chat_id, "assistant", answer)
    chats.set_title_if_default(chat_id, request.question)

    return {"question": request.question, "answer": answer}