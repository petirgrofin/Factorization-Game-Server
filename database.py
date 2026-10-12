"""SQLite persistence for player sessions and gameplay telemetry."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent.parent / "factorization.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def database_path() -> Path:
    return Path(os.getenv("FACTORIZATION_DB_PATH", str(DEFAULT_DATABASE_PATH)))


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA journal_mode = WAL")
    try:
        yield db
        db.commit()
    finally:
        db.close()


def initialize_database() -> None:
    with connection() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                player_id TEXT NOT NULL REFERENCES players(player_id),
                started_at TEXT NOT NULL,
                ended_at TEXT,
                duration_seconds REAL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS telemetry_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL REFERENCES players(player_id),
                session_id TEXT REFERENCES sessions(session_id),
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                endpoint TEXT,
                method TEXT,
                difficulty TEXT,
                previous_difficulty TEXT,
                action TEXT,
                polynomial TEXT,
                question_text TEXT,
                player_input_json TEXT,
                correct INTEGER,
                elapsed_seconds REAL,
                payload_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_events_player_time
                ON telemetry_events(player_id, occurred_at, event_id);
            CREATE INDEX IF NOT EXISTS idx_events_session
                ON telemetry_events(session_id, event_id);
            CREATE INDEX IF NOT EXISTS idx_events_type
                ON telemetry_events(event_type);
            """
        )


def ensure_player(player_id: str, metadata: dict[str, Any] | None = None) -> None:
    now = utc_now()
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
    with connection() as db:
        if metadata is None:
            db.execute(
                """INSERT INTO players(player_id, created_at, last_seen_at, metadata_json)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(player_id) DO UPDATE SET last_seen_at = excluded.last_seen_at""",
                (player_id, now, now, metadata_json),
            )
        else:
            db.execute(
                """INSERT INTO players(player_id, created_at, last_seen_at, metadata_json)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(player_id) DO UPDATE SET
                       last_seen_at = excluded.last_seen_at,
                       metadata_json = excluded.metadata_json""",
                (player_id, now, now, metadata_json),
            )


def start_session(
    player_id: str,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    ensure_player(player_id)
    resolved_id = session_id or str(uuid4())
    with connection() as db:
        existing = db.execute(
            "SELECT player_id FROM sessions WHERE session_id = ?", (resolved_id,)
        ).fetchone()
        if existing and existing["player_id"] != player_id:
            raise ValueError("The session belongs to a different player.")
        db.execute(
            """INSERT OR IGNORE INTO sessions
               (session_id, player_id, started_at, metadata_json)
               VALUES (?, ?, ?, ?)""",
            (
                resolved_id,
                player_id,
                utc_now(),
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
    return resolved_id


def end_session(player_id: str, session_id: str, duration_seconds: float | None) -> bool:
    with connection() as db:
        cursor = db.execute(
            """UPDATE sessions SET ended_at = ?, duration_seconds = ?
               WHERE session_id = ? AND player_id = ? AND ended_at IS NULL""",
            (utc_now(), duration_seconds, session_id, player_id),
        )
        return cursor.rowcount > 0


def record_event(
    player_id: str,
    event_type: str,
    *,
    session_id: str | None = None,
    direction: str = "client_to_server",
    occurred_at: str | None = None,
    endpoint: str | None = None,
    method: str | None = None,
    difficulty: str | None = None,
    previous_difficulty: str | None = None,
    action: str | None = None,
    polynomial: str | None = None,
    question_text: str | None = None,
    player_input: Any = None,
    correct: bool | None = None,
    elapsed_seconds: float | None = None,
    payload: dict[str, Any] | list[Any] | None = None,
) -> int:
    ensure_player(player_id)
    if session_id:
        start_session(player_id, session_id)
    with connection() as db:
        cursor = db.execute(
            """INSERT INTO telemetry_events (
                   player_id, session_id, occurred_at, event_type, direction,
                   endpoint, method, difficulty, previous_difficulty, action,
                   polynomial, question_text, player_input_json, correct,
                   elapsed_seconds, payload_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                player_id, session_id, occurred_at or utc_now(), event_type,
                direction, endpoint, method, difficulty, previous_difficulty,
                action, polynomial, question_text,
                None if player_input is None else json.dumps(player_input, ensure_ascii=False),
                None if correct is None else int(correct), elapsed_seconds,
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )
        return int(cursor.lastrowid)


def _decode_event(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["payload"] = json.loads(result.pop("payload_json"))
    raw_input = result.pop("player_input_json")
    result["player_input"] = json.loads(raw_input) if raw_input is not None else None
    if result["correct"] is not None:
        result["correct"] = bool(result["correct"])
    return result


def get_timeline(player_id: str, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
    with connection() as db:
        rows = db.execute(
            """SELECT * FROM telemetry_events WHERE player_id = ?
               ORDER BY occurred_at ASC, event_id ASC LIMIT ? OFFSET ?""",
            (player_id, limit, offset),
        ).fetchall()
    return [_decode_event(row) for row in rows]


def get_player_summary(player_id: str) -> dict[str, Any] | None:
    with connection() as db:
        player = db.execute(
            "SELECT * FROM players WHERE player_id = ?", (player_id,)
        ).fetchone()
        if not player:
            return None
        sessions = db.execute(
            """SELECT COUNT(*) AS count, COALESCE(SUM(duration_seconds), 0) AS seconds
               FROM sessions WHERE player_id = ?""", (player_id,)
        ).fetchone()
        event_counts = db.execute(
            """SELECT event_type, COUNT(*) AS count FROM telemetry_events
               WHERE player_id = ? GROUP BY event_type""", (player_id,)
        ).fetchall()
        accuracy = db.execute(
            """SELECT SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct_count,
                      COUNT(correct) AS answered_count
               FROM telemetry_events WHERE player_id = ?""", (player_id,)
        ).fetchone()
    return {
        "player_id": player_id,
        "created_at": player["created_at"],
        "last_seen_at": player["last_seen_at"],
        "number_of_sessions": sessions["count"],
        "time_spent_seconds": sessions["seconds"],
        "event_counts": {row["event_type"]: row["count"] for row in event_counts},
        "answered_count": accuracy["answered_count"],
        "correct_count": accuracy["correct_count"] or 0,
    }
