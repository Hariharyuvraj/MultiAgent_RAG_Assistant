import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SQLiteManager:
    def __init__(self, db_path: str = "./storage/history.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id         TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id         TEXT PRIMARY KEY,
                    user_id    TEXT NOT NULL,
                    title      TEXT DEFAULT 'New Chat',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         TEXT NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    sources         TEXT DEFAULT '[]',
                    grounding_score REAL DEFAULT 0.0,
                    timestamp       TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename     TEXT NOT NULL UNIQUE,
                    chunks_count INTEGER DEFAULT 0,
                    indexed_by   TEXT,
                    indexed_at   TEXT DEFAULT CURRENT_TIMESTAMP
                );

                INSERT OR IGNORE INTO users (id, name) VALUES
                    ('user1', 'User 1'),
                    ('user2', 'User 2'),
                    ('user3', 'User 3');
            """)
            self._migrate(conn)
        logger.info("SQLite database ready at %s", self.db_path)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
        if "session_id" not in existing:
            conn.execute("ALTER TABLE conversations ADD COLUMN session_id TEXT")
            logger.info("Migration: added session_id column to conversations")
        if "sources" not in existing:
            conn.execute("ALTER TABLE conversations ADD COLUMN sources TEXT DEFAULT '[]'")
            logger.info("Migration: added sources column to conversations")
        if "grounding_score" not in existing:
            conn.execute("ALTER TABLE conversations ADD COLUMN grounding_score REAL DEFAULT 0.0")
            logger.info("Migration: added grounding_score column to conversations")

    # ── Users ────────────────────────────────────────────────────────────────

    def get_users(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, name FROM users ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    # ── Sessions ─────────────────────────────────────────────────────────────

    def create_session(self, user_id: str) -> str:
        sid = str(uuid.uuid4())[:12]
        ts  = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, user_id, title, created_at) VALUES (?,?,?,?)",
                (sid, user_id, "New Chat", ts),
            )
        return sid

    def update_session_title(self, session_id: str, title: str) -> None:
        short = title[:45] + "..." if len(title) > 45 else title
        with self._conn() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title=? WHERE id=?", (short, session_id)
            )

    def get_sessions(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at FROM chat_sessions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Messages ─────────────────────────────────────────────────────────────

    def save_message(
        self,
        user_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        sources: List[str] = None,
        grounding_score: float = 0.0,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO conversations
                    (session_id, user_id, role, content, sources, grounding_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, role, content,
                 json.dumps(sources or []), grounding_score, ts),
            )

    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content, sources, grounding_score, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {
                "role":            r["role"],
                "content":         r["content"],
                "sources":         json.loads(r["sources"] or "[]"),
                "grounding_score": r["grounding_score"],
                "timestamp":       r["timestamp"],
            }
            for r in rows
        ]

    def context_pairs(self, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        msgs = self.get_session_messages(session_id, limit=limit * 2)
        pairs: List[Dict[str, str]] = []
        i = 0
        while i < len(msgs) - 1:
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                pairs.append({"user": msgs[i]["content"], "assistant": msgs[i + 1]["content"]})
                i += 2
            else:
                i += 1
        return pairs[-limit:]

    # ── Documents ─────────────────────────────────────────────────────────────

    def get_documents(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT filename, chunks_count, indexed_by, indexed_at FROM documents ORDER BY indexed_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def add_document(self, filename: str, chunks_count: int, indexed_by: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents (filename, chunks_count, indexed_by, indexed_at) VALUES (?,?,?,?)",
                (filename, chunks_count, indexed_by, ts),
            )

    def document_exists(self, filename: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE filename = ?", (filename,)
            ).fetchone()
        return row is not None

    def delete_document(self, filename: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        logger.info("Removed document record: %s", filename)
