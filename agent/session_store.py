"""会话元数据存储。

Checkpointer 只存消息快照，不存「这轮对话叫什么、什么时候动的」。
这里用同一套 SQLite 文件另建一张 sessions 表，只负责列表展示与删除。

删除会话要同时清三处：sessions、checkpoints、writes，
只删 sessions 会出现「列表没了但历史还能靠 tid 访问」的脏数据。
"""

import sqlite3
from datetime import datetime

from utils.logger_handler import logger


class SessionStore:
    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id  TEXT PRIMARY KEY,
                title      TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        self._conn.commit()
        self._backfill()

    def _backfill(self) -> None:
        """把已存在但没登记的会话补进列表（例如接入前留下的旧 thread）。"""
        try:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO sessions (thread_id, title, created_at, updated_at)
                SELECT thread_id, '（历史会话）', '', '' FROM checkpoints GROUP BY thread_id
                """
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            # checkpoints 表尚未创建（首次启动），忽略即可
            pass

    def touch(self, thread_id: str, question: str) -> None:
        """记录/更新会话；标题取首条提问的前 24 字。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        title = (question or "新对话").strip().replace("\n", " ")[:24]
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO sessions (thread_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (thread_id, title, now, now),
            )

    def rename(self, thread_id: str, title: str) -> bool:
        """自定义会话名。后续对话不会覆盖（touch 的 upsert 只更新 updated_at）。"""
        title = (title or "").strip().replace("\n", " ")[:40]
        if not title:
            return False
        with self._conn:
            cur = self._conn.execute(
                "UPDATE sessions SET title = ? WHERE thread_id = ?", (title, thread_id)
            )
        return cur.rowcount > 0

    def list_all(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT thread_id, title, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [{"thread_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]

    def delete(self, thread_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
            self._conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            self._conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        logger.info("会话已删除 | thread=%s", thread_id)
