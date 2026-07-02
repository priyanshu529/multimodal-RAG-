"""
chats.py

Job: Persist chats, their messages, and their uploaded documents to SQLite,
so past conversations AND the list of PDFs attached to them survive a
server restart or a chat switch -- same idea as ChatGPT's chat history
sidebar, plus a per-chat "attached files" list.

Each chat belongs to a single username, so users only ever see and touch
their own chats and documents.

Kept as plain functions over sqlite3 (stdlib), no ORM, to match the rest
of the project's style.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime, timezone

# IMPORTANT: this lives OUTSIDE the project folder on purpose. If it lived
# inside the project (e.g. "data/chats.db" relative to main.py), every
# write here -- which happens on every single /query call -- touches a
# file that `uvicorn --reload` is watching. That triggers a full server
# restart, which wipes the in-memory session/retriever caches and makes
# the frontend bounce back to the login screen. From the browser that
# looks exactly like "the whole page refreshed" after sending a message.
# Override with MERGE_DATA_DIR if you want it somewhere else.
DATA_DIR = Path(os.environ.get("MERGE_DATA_DIR", Path.home() / ".merge_data"))
DB_PATH = str(DATA_DIR / "chats.db")


def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
    """)
    # Every PDF successfully indexed into a chat gets a row here, so the
    # frontend can show "attached files" for a chat even after a reload or
    # after switching away and back -- previously this only lived in the
    # browser's in-memory filesByChat map and vanished on navigation.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            chunks_added INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
    """)

    # Migration: if chats.db already exists from before the username column
    # was added, add it so old installs don't crash on startup. Existing
    # rows get username = "" (they predate per-user scoping and won't
    # belong to anyone under the new scheme).
    existing_cols = [row["name"] for row in conn.execute("PRAGMA table_info(chats)")]
    if "username" not in existing_cols:
        conn.execute("ALTER TABLE chats ADD COLUMN username TEXT NOT NULL DEFAULT ''")

    conn.commit()
    conn.close()


def create_chat(username: str, title: str = "New chat") -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO chats (username, title, created_at) VALUES (?, ?, ?)",
        (username, title, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    chat_id = cur.lastrowid
    conn.close()
    return chat_id


def list_chats(username: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, title, created_at FROM chats WHERE username = ? ORDER BY id DESC",
        (username,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_messages(chat_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_message(chat_id: int, role: str, content: str):
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def set_title_if_default(chat_id: int, first_question: str):
    """Auto-titles a chat from its first question, same as ChatGPT does."""
    conn = _connect()
    row = conn.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if row and row["title"] == "New chat":
        title = first_question.strip()[:50]
        conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))
        conn.commit()
    conn.close()


def delete_chat(chat_id: int):
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM documents WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()


def chat_belongs_to(chat_id: int, username: str) -> bool:
    """Ownership check -- use this instead of chat_exists on every route
    that touches a specific chat, so one user can never read, query, or
    delete another user's chat by guessing/incrementing an id."""
    conn = _connect()
    row = conn.execute(
        "SELECT id FROM chats WHERE id = ? AND username = ?",
        (chat_id, username),
    ).fetchone()
    conn.close()
    return row is not None


# ---------------- Documents ----------------

def add_document(chat_id: int, filename: str, chunks_added: int) -> dict:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO documents (chat_id, filename, chunks_added, created_at) VALUES (?, ?, ?, ?)",
        (chat_id, filename, chunks_added, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    doc_id = cur.lastrowid
    conn.close()
    return {"id": doc_id, "chat_id": chat_id, "filename": filename, "chunks_added": chunks_added}


def get_documents(chat_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, filename, chunks_added, created_at FROM documents WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]