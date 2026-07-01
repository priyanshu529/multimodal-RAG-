"""
chats.py

Job: Persist chats and their messages to SQLite, so past conversations
survive a server restart -- same idea as ChatGPT's chat history sidebar.

Kept as plain functions over sqlite3 (stdlib), no ORM, to match the rest
of the project's style.
"""

import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = "data/chats.db"


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    conn.commit()
    conn.close()


def create_chat(title: str = "New chat") -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO chats (title, created_at) VALUES (?, ?)",
        (title, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    chat_id = cur.lastrowid
    conn.close()
    return chat_id


def list_chats() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, title, created_at FROM chats ORDER BY id DESC"
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
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()


def chat_exists(chat_id: int) -> bool:
    conn = _connect()
    row = conn.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
    conn.close()
    return row is not None