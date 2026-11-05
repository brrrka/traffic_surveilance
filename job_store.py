"""Persistent job & saved-case storage (SQLite).

Replaces the old in-memory `jobs = {}` dict so analysis results survive a
backend restart, and gives saved cases a server-side home instead of only
living in one browser's localStorage (which doesn't survive a cleared
profile and isn't shared across devices).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "anchor_vi.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
_conn.execute("PRAGMA journal_mode=WAL;")
_conn.execute(
    """CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )"""
)
_conn.execute(
    """CREATE TABLE IF NOT EXISTS cases (
        id TEXT PRIMARY KEY,
        job_id TEXT,
        created_at TEXT NOT NULL,
        payload TEXT NOT NULL
    )"""
)
_conn.commit()


def create_job(job_id: str, data: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO jobs (job_id, created_at, updated_at, payload) VALUES (?, ?, ?, ?)",
            (job_id, now, now, json.dumps(data)),
        )
        _conn.commit()


def update_job(job_id: str, **changes) -> None:
    with _lock:
        row = _conn.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return
        data = json.loads(row[0])
        data.update(changes)
        _conn.execute(
            "UPDATE jobs SET payload = ?, updated_at = ? WHERE job_id = ?",
            (json.dumps(data), datetime.now(timezone.utc).isoformat(), job_id),
        )
        _conn.commit()


def get_job(job_id: str) -> dict | None:
    with _lock:
        row = _conn.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return json.loads(row[0]) if row else None


def create_case(case: dict) -> dict:
    if not isinstance(case, dict):
        raise ValueError("Payload kasus harus berupa object JSON.")
    case_id = str(case.get("id") or "").strip()
    if not case_id:
        raise ValueError("Kasus harus memiliki field 'id'.")
    now = datetime.now(timezone.utc).isoformat()
    record = {**case, "savedAt": now}
    with _lock:
        _conn.execute(
            "INSERT OR REPLACE INTO cases (id, job_id, created_at, payload) VALUES (?, ?, ?, ?)",
            (case_id, case.get("jobId"), now, json.dumps(record)),
        )
        _conn.commit()
    return record


def list_cases() -> list[dict]:
    with _lock:
        rows = _conn.execute("SELECT payload FROM cases ORDER BY created_at DESC").fetchall()
    return [json.loads(row[0]) for row in rows]


def get_case(case_id: str) -> dict | None:
    with _lock:
        row = _conn.execute("SELECT payload FROM cases WHERE id = ?", (case_id,)).fetchone()
    return json.loads(row[0]) if row else None


def delete_case(case_id: str) -> bool:
    with _lock:
        cur = _conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))
        _conn.commit()
    return cur.rowcount > 0
