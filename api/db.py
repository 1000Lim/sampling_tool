import os
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_PATH = os.environ.get("TASKS_DB_PATH", "/app/api/tasks.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                params_json TEXT NOT NULL,
                export_dir TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                finished_at REAL,
                stats TEXT,
                progress TEXT
            )
            """
        )
        conn.commit()

        # Migrate existing database: add stats and progress columns if they don't exist
        try:
            conn.execute("ALTER TABLE tasks ADD COLUMN stats TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass

        try:
            conn.execute("ALTER TABLE tasks ADD COLUMN progress TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass


def create_task(task_id: str, status: str, params: Dict[str, Any], created_at: float, export_dir: Optional[str] = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO tasks (id, status, params_json, export_dir, created_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, status, json.dumps(params), export_dir, created_at),
        )
        conn.commit()


def update_task(task_id: str, status: Optional[str] = None, export_dir: Optional[str] = None, error: Optional[str] = None, finished_at: Optional[float] = None, stats: Optional[Dict[str, Any]] = None, progress: Optional[Dict[str, Any]] = None) -> None:
    sets: List[str] = []
    values: List[Any] = []
    if status is not None:
        sets.append("status = ?")
        values.append(status)
    if export_dir is not None:
        sets.append("export_dir = ?")
        values.append(export_dir)
    if error is not None:
        sets.append("error = ?")
        values.append(error)
    if finished_at is not None:
        sets.append("finished_at = ?")
        values.append(finished_at)
    if stats is not None:
        sets.append("stats = ?")
        values.append(json.dumps(stats))
    if progress is not None:
        sets.append("progress = ?")
        values.append(json.dumps(progress))
    if not sets:
        return
    values.append(task_id)
    with _connect() as conn:
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", values)
        conn.commit()


def delete_task(task_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_tasks() -> List[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


def cleanup_running_tasks() -> int:
    """Mark all running tasks as failed (orphaned from server restart)."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE tasks SET status = ?, error = ?, finished_at = ? WHERE status = ?",
            ("failed", "Job was interrupted (server restarted). Click 'Restart' to resume.", time.time(), "running")
        )
        conn.commit()
        return cur.rowcount


def cleanup_completed_task_errors() -> int:
    """Clear error messages from successfully completed tasks."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE tasks SET error = NULL WHERE status = ? AND error IS NOT NULL",
            ("completed",)
        )
        conn.commit()
        return cur.rowcount


