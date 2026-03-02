"""SQLite-backed immutable telemetry for runs, samples, steps, and tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def hash_tool_input(tool_input: dict[str, Any]) -> str:
    """Return a stable short hash for a tool input payload."""
    payload = json.dumps(tool_input, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class LearningTelemetry:
    """Write append-only telemetry events to SQLite."""

    def __init__(self, db_path: str | Path = "learning.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = asyncio.Lock()
        self._setup()

    def _setup(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                finished_at REAL,
                task_name TEXT,
                csv_path TEXT,
                config_json TEXT,
                status TEXT,
                error_code TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS run_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                ts REAL NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                ts REAL NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS step_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                ts REAL NOT NULL,
                result_type TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_code TEXT,
                latency_ms INTEGER,
                payload_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                ts REAL NOT NULL,
                tool_name TEXT NOT NULL,
                tool_input_hash TEXT NOT NULL,
                result_type TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_code TEXT,
                latency_ms INTEGER,
                payload_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                ts REAL NOT NULL,
                score INTEGER NOT NULL,
                schema_pass_rate REAL NOT NULL,
                evidence_pass_rate REAL NOT NULL,
                screenshot_validity REAL NOT NULL,
                retries INTEGER NOT NULL,
                completion_time_sec REAL NOT NULL,
                payload_json TEXT
            )
            """
        )

        cur.execute("CREATE INDEX IF NOT EXISTS idx_tool_events_run_sample ON tool_events(run_id, sample_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_step_events_run_sample ON step_events(run_id, sample_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sample_scores_run_sample ON sample_scores(run_id, sample_id)")
        self._conn.commit()

    async def close(self) -> None:
        async with self._lock:
            self._conn.close()

    async def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        async with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            self._conn.commit()

    async def start_run(
        self,
        run_id: str,
        task_name: str,
        csv_path: str,
        config: dict[str, Any],
    ) -> None:
        now = time.time()
        await self._execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, started_at, finished_at, task_name, csv_path, config_json, status, error_code)
            VALUES (?, ?, NULL, ?, ?, ?, ?, NULL)
            """,
            (run_id, now, task_name, csv_path, json.dumps(config, default=str), "running"),
        )
        await self.log_run_event(run_id, "run_started", {"task_name": task_name})

    async def finish_run(self, run_id: str, status: str, error_code: str | None = None) -> None:
        now = time.time()
        await self._execute(
            """
            UPDATE runs
            SET finished_at = ?, status = ?, error_code = ?
            WHERE run_id = ?
            """,
            (now, status, error_code, run_id),
        )
        await self.log_run_event(run_id, "run_finished", {"status": status, "error_code": error_code})

    async def log_run_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self._execute(
            """
            INSERT INTO run_events (run_id, ts, event_type, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, time.time(), event_type, json.dumps(payload or {}, default=str)),
        )

    async def log_sample_event(
        self,
        run_id: str,
        sample_id: str,
        phase: str,
        event_type: str,
        attempt: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self._execute(
            """
            INSERT INTO sample_events (run_id, sample_id, phase, attempt, ts, event_type, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                sample_id,
                phase,
                attempt,
                time.time(),
                event_type,
                json.dumps(payload or {}, default=str),
            ),
        )

    async def log_step_event(
        self,
        run_id: str,
        sample_id: str,
        phase: str,
        iteration: int,
        result_type: str,
        success: bool,
        error_code: str | None,
        latency_ms: int | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self._execute(
            """
            INSERT INTO step_events
            (run_id, sample_id, phase, iteration, ts, result_type, success, error_code, latency_ms, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                sample_id,
                phase,
                iteration,
                time.time(),
                result_type,
                1 if success else 0,
                error_code,
                latency_ms,
                json.dumps(payload or {}, default=str),
            ),
        )

    async def log_tool_event(
        self,
        run_id: str,
        sample_id: str,
        phase: str,
        iteration: int,
        tool_name: str,
        tool_input_hash: str,
        result_type: str,
        success: bool,
        error_code: str | None,
        latency_ms: int | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self._execute(
            """
            INSERT INTO tool_events
            (run_id, sample_id, phase, iteration, ts, tool_name, tool_input_hash, result_type, success, error_code, latency_ms, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                sample_id,
                phase,
                iteration,
                time.time(),
                tool_name,
                tool_input_hash,
                result_type,
                1 if success else 0,
                error_code,
                latency_ms,
                json.dumps(payload or {}, default=str),
            ),
        )

    async def log_sample_score(
        self,
        run_id: str,
        sample_id: str,
        phase: str,
        score_data: dict[str, Any],
    ) -> None:
        await self._execute(
            """
            INSERT INTO sample_scores
            (run_id, sample_id, phase, ts, score, schema_pass_rate, evidence_pass_rate, screenshot_validity, retries, completion_time_sec, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                sample_id,
                phase,
                time.time(),
                int(score_data.get("score", 0)),
                float(score_data.get("schema_pass_rate", 0.0)),
                float(score_data.get("required_evidence_present", 0.0)),
                float(score_data.get("screenshot_validity", 0.0)),
                int(score_data.get("retries", 0)),
                float(score_data.get("completion_time_sec", 0.0)),
                json.dumps(score_data, default=str),
            ),
        )
