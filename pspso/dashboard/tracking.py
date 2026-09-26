"""SQLite-backed tracking for experiments, runs, and run events."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psutil

from pspso.config import OptimizationResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_workspace() -> Path:
    """Return the isolated PSPSO 1.0 workspace directory."""

    configured = os.environ.get("PSPSO_HOME")
    root = Path(configured).expanduser() if configured else Path.cwd() / ".pspso"
    return root / "v1"


class TrackingRepository:
    """Persist dashboard activity so runs survive page refreshes and restarts."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or default_workspace() / "tracking.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace = self.db_path.parent
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            try:
                connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    is_ad_hoc INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    result_json TEXT,
                    best_params_json TEXT,
                    best_metric REAL,
                    best_cost REAL,
                    duration REAL,
                    n_trials INTEGER NOT NULL DEFAULT 0,
                    n_failures INTEGER NOT NULL DEFAULT 0,
                    strategy TEXT,
                    task TEXT,
                    metric TEXT,
                    estimator TEXT
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    sequence_number INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence_number)
                );

                CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id);

                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_path TEXT,
                    fingerprint TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    parent_run_id TEXT REFERENCES runs(run_id),
                    status TEXT NOT NULL,
                    worker_pid INTEGER,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_run_attempts_run_id ON run_attempts(run_id);

                CREATE TABLE IF NOT EXISTS service_leases (
                    service_id TEXT PRIMARY KEY,
                    process_id INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                """
            )
            # A write transaction serializes migrations across CLI and service startup.
            connection.execute("BEGIN IMMEDIATE")
            applied = {
                row[0] for row in connection.execute("SELECT version FROM schema_migrations")
            }
            migrations = {
                1: [
                    ("runs", "cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
                    ("runs", "artifact_json", "TEXT"),
                    ("runs", "schema_version", "INTEGER NOT NULL DEFAULT 1"),
                    ("experiments", "result_layout_json", "TEXT"),
                    ("runs", "parent_run_id", "TEXT"),
                    ("run_attempts", "service_id", "TEXT"),
                    ("run_attempts", "heartbeat_at", "TEXT"),
                    ("run_attempts", "timeout_seconds", "REAL"),
                    ("run_attempts", "termination_requested", "INTEGER NOT NULL DEFAULT 0"),
                ],
                2: [
                    ("run_attempts", "termination_reason", "TEXT"),
                    ("run_attempts", "worker_created_at", "REAL"),
                ],
            }
            for version, columns in migrations.items():
                if version in applied:
                    continue
                for table, column, definition in columns:
                    self._ensure_column(connection, table, column, definition)
                connection.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, utc_now()),
                )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection, table: str, column: str, definition: str
    ) -> None:
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        *,
        is_ad_hoc: bool = False,
    ) -> dict[str, Any]:
        experiment_id = str(uuid.uuid4())
        created_at = utc_now()
        tags = tags or []
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments (
                    experiment_id, name, description, tags_json, is_ad_hoc, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    name,
                    description,
                    json.dumps(tags),
                    int(is_ad_hoc),
                    created_at,
                ),
            )
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise RuntimeError("Experiment could not be loaded after creation.")
        return experiment

    def create_ad_hoc_experiment(self) -> dict[str, Any]:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.create_experiment(f"Ad hoc run {stamp}", is_ad_hoc=True)

    def experiment_exists(self, experiment_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return row is not None

    def list_experiments(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    e.experiment_id,
                    e.name,
                    e.description,
                    e.tags_json,
                    e.is_ad_hoc,
                    e.created_at,
                    e.result_layout_json,
                    COUNT(r.run_id) AS run_count,
                    MAX(r.updated_at) AS latest_updated_at,
                    (
                        SELECT r2.status FROM runs r2
                        WHERE r2.experiment_id = e.experiment_id
                        ORDER BY r2.updated_at DESC
                        LIMIT 1
                    ) AS latest_status,
                    (
                        SELECT r3.run_id FROM runs r3
                        WHERE r3.experiment_id = e.experiment_id
                        ORDER BY r3.updated_at DESC
                        LIMIT 1
                    ) AS latest_run_id
                FROM experiments e
                LEFT JOIN runs r ON r.experiment_id = e.experiment_id
                GROUP BY e.experiment_id
                ORDER BY latest_updated_at DESC NULLS LAST, e.created_at DESC
                """
            ).fetchall()
        return [self._experiment_summary(row) for row in rows]

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    e.experiment_id,
                    e.name,
                    e.description,
                    e.tags_json,
                    e.is_ad_hoc,
                    e.created_at,
                    e.result_layout_json,
                    COUNT(r.run_id) AS run_count,
                    MAX(r.updated_at) AS latest_updated_at,
                    (
                        SELECT r2.status FROM runs r2
                        WHERE r2.experiment_id = e.experiment_id
                        ORDER BY r2.updated_at DESC
                        LIMIT 1
                    ) AS latest_status,
                    (
                        SELECT r3.run_id FROM runs r3
                        WHERE r3.experiment_id = e.experiment_id
                        ORDER BY r3.updated_at DESC
                        LIMIT 1
                    ) AS latest_run_id
                FROM experiments e
                LEFT JOIN runs r ON r.experiment_id = e.experiment_id
                WHERE e.experiment_id = ?
                GROUP BY e.experiment_id
                """,
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        detail = self._experiment_summary(row)
        detail["runs"] = self.list_runs_for_experiment(experiment_id)
        return detail

    def save_result_layout(self, experiment_id: str, tools: list[str]) -> dict[str, Any] | None:
        """Persist the chosen Results dashboard tools for an experiment."""

        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE experiments SET result_layout_json = ? WHERE experiment_id = ?",
                (json.dumps(tools), experiment_id),
            )
        return self.get_experiment(experiment_id)

    def create_run(
        self,
        request_payload: dict[str, Any],
        *,
        experiment_id: str,
        parent_run_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        created_at = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id,
                    experiment_id,
                    request_json,
                    status,
                    created_at,
                    updated_at,
                    strategy,
                    task,
                    metric,
                    estimator,
                    parent_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_id,
                    json.dumps(request_payload),
                    "queued",
                    created_at,
                    created_at,
                    request_payload.get("strategy"),
                    request_payload.get("task"),
                    request_payload.get("metric"),
                    request_payload.get("estimator"),
                    parent_run_id,
                ),
            )
        snapshot = self.get_run(run_id)
        if snapshot is None:
            raise RuntimeError("Run could not be loaded after creation.")
        return snapshot

    def create_attempt(
        self,
        run_id: str,
        parent_run_id: str | None = None,
        *,
        service_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        attempt_id = str(uuid.uuid4())
        created_at = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM run_attempts WHERE run_id = ? AND status IN ('queued', 'running')",
                (run_id,),
            ).fetchone():
                raise ValueError("This run already has an active attempt.")
            connection.execute(
                """
                INSERT INTO run_attempts (
                    attempt_id, run_id, parent_run_id, status, created_at,
                    service_id, heartbeat_at, timeout_seconds
                )
                VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    run_id,
                    parent_run_id,
                    created_at,
                    service_id,
                    created_at,
                    timeout_seconds,
                ),
            )
        return {
            "attempt_id": attempt_id,
            "run_id": run_id,
            "status": "queued",
            "created_at": created_at,
            "service_id": service_id,
            "timeout_seconds": timeout_seconds,
        }

    def set_attempt_running(
        self,
        attempt_id: str,
        worker_pid: int,
        service_id: str | None = None,
        worker_created_at: float | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE run_attempts
                SET status = 'running', worker_pid = ?, started_at = ?,
                    heartbeat_at = ?, service_id = COALESCE(?, service_id), worker_created_at = ?
                WHERE attempt_id = ? AND status = 'queued'
                """,
                (worker_pid, utc_now(), utc_now(), service_id, worker_created_at, attempt_id),
            )

    def finish_attempt(self, attempt_id: str, status: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE run_attempts SET status = ?, finished_at = ? WHERE attempt_id = ? "
                "AND status IN ('queued', 'running')",
                (status, utc_now(), attempt_id),
            )

    def request_termination(self, attempt_id: str, reason: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE run_attempts SET termination_requested = 1, termination_reason = ? "
                "WHERE attempt_id = ? AND status IN ('queued', 'running') "
                "AND termination_requested = 0",
                (reason, attempt_id),
            )

    def termination_reason(self, attempt_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT termination_reason FROM run_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
        return row[0] if row else None

    def heartbeat_attempt(self, attempt_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE run_attempts SET heartbeat_at = ?
                WHERE attempt_id = ? AND status = 'running'
                """,
                (utc_now(), attempt_id),
            )

    def list_attempts(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_attempts WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def queued_attempts(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT a.* FROM run_attempts a
                JOIN runs r ON r.run_id = a.run_id
                WHERE a.status = 'queued' AND r.status = 'queued'
                ORDER BY a.created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def register_service(self, service_id: str, process_id: int) -> None:
        now = utc_now()
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            leases = connection.execute(
                "SELECT service_id, process_id, heartbeat_at FROM service_leases "
                "WHERE service_id != ?",
                (service_id,),
            ).fetchall()
            for lease in leases:
                alive = self._process_exists(int(lease["process_id"]))
                if alive and lease["heartbeat_at"] >= cutoff:
                    raise RuntimeError(
                        "Another PSPSO API service already owns this workspace. "
                        "Stop it or use a different PSPSO_HOME."
                    )
                connection.execute(
                    "DELETE FROM service_leases WHERE service_id = ?", (lease["service_id"],)
                )
            connection.execute(
                """
                INSERT INTO service_leases (service_id, process_id, started_at, heartbeat_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(service_id) DO UPDATE SET
                    process_id = excluded.process_id,
                    started_at = excluded.started_at,
                    heartbeat_at = excluded.heartbeat_at
                """,
                (service_id, process_id, now, now),
            )

    @staticmethod
    def _process_exists(process_id: int) -> bool:
        # os.kill(pid, 0) calls TerminateProcess on Windows; never use it as a probe.
        return process_id > 0 and psutil.pid_exists(process_id)

    def heartbeat_service(self, service_id: str) -> bool:
        with self._lock, self._connect() as connection:
            updated = connection.execute(
                "UPDATE service_leases SET heartbeat_at = ? WHERE service_id = ?",
                (utc_now(), service_id),
            )
        return updated.rowcount == 1

    def release_service(self, service_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM service_leases WHERE service_id = ?", (service_id,))

    def interrupt_stale_attempts(
        self, *, service_id: str, stale_after_seconds: float = 10.0
    ) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)).isoformat()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT a.*, r.request_json, r.status AS run_status
                FROM run_attempts a
                JOIN runs r ON r.run_id = a.run_id
                LEFT JOIN service_leases s ON s.service_id = a.service_id
                WHERE a.status = 'running'
                  AND (a.service_id IS NULL OR a.service_id != ?)
                  AND (s.heartbeat_at IS NULL OR s.heartbeat_at < ?)
                """,
                (service_id, cutoff),
            ).fetchall()
        for row in rows:
            if row["run_status"] in {"completed", "failed", "cancelled", "interrupted"}:
                self.finish_attempt(row["attempt_id"], row["run_status"])
                continue
            pid = row["worker_pid"]
            source = json.loads(row["request_json"]).get("source")
            if source == "python" and pid and self._process_exists(pid):
                # Notebook callers own their foreground execution, not the API lease.
                continue
            if pid and row["worker_created_at"] is not None:
                from pspso.dashboard.processes import ProcessTree

                try:
                    process = psutil.Process(pid)
                    if process.create_time() == row["worker_created_at"]:
                        ProcessTree(pid, contain=False).close()
                except psutil.NoSuchProcess:
                    pass
            event = self.record_event(
                row["run_id"],
                {
                    "type": "run_interrupted",
                    "timestamp": utc_now(),
                    "payload": {"reason": "The owning worker service stopped sending heartbeats."},
                },
            )
            self.finish_attempt(row["attempt_id"], event["type"].removeprefix("run_"))
        return len(rows)

    def request_cancel(self, run_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE runs SET cancel_requested = 1, updated_at = ? WHERE run_id = ?",
                (utc_now(), run_id),
            )

    def clear_cancel_request(self, run_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("UPDATE runs SET cancel_requested = 0 WHERE run_id = ?", (run_id,))

    def cancel_requested(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def save_artifacts(self, run_id: str, artifacts: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE runs SET artifact_json = ?, updated_at = ? WHERE run_id = ?",
                (json.dumps(artifacts), utc_now(), run_id),
            )

    def create_dataset(
        self,
        name: str,
        source_kind: str,
        source_path: str | None,
        fingerprint: str,
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        dataset_id = str(uuid.uuid4())
        created_at = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO datasets (
                    dataset_id, name, source_kind, source_path,
                    fingerprint, profile_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    name,
                    source_kind,
                    source_path,
                    fingerprint,
                    json.dumps(profile),
                    created_at,
                ),
            )
        dataset = self.get_dataset(dataset_id)
        if dataset is None:
            raise RuntimeError("Dataset could not be loaded after creation.")
        return dataset

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [self._dataset_snapshot(row) for row in rows]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        return self._dataset_snapshot(row) if row is not None else None

    def record_event(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        event = dict(event)
        payload = dict(event["payload"])
        with self._lock, self._connect() as connection:
            # The API and worker are separate processes. Acquire a SQLite write
            # lock before assigning the next sequence number so timeline order
            # cannot race under simultaneous log and optimizer events.
            connection.execute("BEGIN IMMEDIATE")
            terminal_types = {"run_completed", "run_failed", "run_cancelled", "run_interrupted"}
            if event["type"] in terminal_types:
                attempt = connection.execute(
                    "SELECT termination_reason FROM run_attempts WHERE run_id = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
                if attempt is not None and attempt[0]:
                    reason = attempt[0]
                    event["type"] = {
                        "timeout": "run_failed",
                        "cancelled": "run_cancelled",
                        "interrupted": "run_interrupted",
                    }[reason]
                    payload["reason"] = reason
                    if reason == "timeout":
                        payload["error"] = "Run exceeded its configured timeout."
                current = connection.execute(
                    "SELECT status FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if current is not None and current["status"] in {
                    "completed",
                    "failed",
                    "cancelled",
                    "interrupted",
                }:
                    existing = connection.execute(
                        """
                        SELECT sequence_number, event_type, timestamp, payload_json
                        FROM run_events WHERE run_id = ? AND event_type IN (
                            'run_completed', 'run_failed', 'run_cancelled', 'run_interrupted'
                        ) ORDER BY sequence_number DESC LIMIT 1
                        """,
                        (run_id,),
                    ).fetchone()
                    if existing is not None:
                        return {
                            "sequence_number": existing["sequence_number"],
                            "type": existing["event_type"],
                            "timestamp": existing["timestamp"],
                            "payload": json.loads(existing["payload_json"]),
                        }
            sequence_number = connection.execute(
                "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            persisted = {
                "sequence_number": int(sequence_number),
                "type": event["type"],
                "timestamp": event["timestamp"],
                "payload": payload,
            }
            connection.execute(
                """
                INSERT INTO run_events (
                    run_id, sequence_number, event_type, timestamp, payload_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sequence_number,
                    event["type"],
                    event["timestamp"],
                    json.dumps(payload),
                ),
            )
            self._refresh_run_state(connection, run_id)
        return persisted

    def record_log_messages(self, run_id: str, messages: list[str]) -> list[dict[str, Any]]:
        """Persist worker output in one transaction so noisy logs cannot delay cleanup."""

        if not messages:
            return []
        timestamp = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            first_sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM run_events WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
            )
            persisted = [
                {
                    "sequence_number": first_sequence + index,
                    "type": "run_log",
                    "timestamp": timestamp,
                    "payload": {"stream": "worker", "message": message},
                }
                for index, message in enumerate(messages)
            ]
            connection.executemany(
                """
                INSERT INTO run_events (
                    run_id, sequence_number, event_type, timestamp, payload_json
                ) VALUES (?, ?, 'run_log', ?, ?)
                """,
                [
                    (
                        run_id,
                        event["sequence_number"],
                        timestamp,
                        json.dumps(event["payload"]),
                    )
                    for event in persisted
                ],
            )
            self._refresh_run_state(connection, run_id)
        return persisted

    def save_result(self, run_id: str, result: OptimizationResult) -> None:
        payload = result.to_dict()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET result_json = ?,
                    best_params_json = ?,
                    best_metric = ?,
                    best_cost = ?,
                    duration = ?,
                    n_trials = ?,
                    n_failures = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    json.dumps(payload),
                    json.dumps(payload["best_params"])
                    if payload["best_params"] is not None
                    else None,
                    payload["best_metric"],
                    payload["best_cost"],
                    payload["duration"],
                    len(payload["trials"]),
                    len(payload["failures"]),
                    utc_now(),
                    run_id,
                ),
            )

    def save_error(self, run_id: str, error: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE runs SET error = ?, updated_at = ? WHERE run_id = ?",
                (error, utc_now(), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    r.*,
                    e.name AS experiment_name,
                    e.description AS experiment_description,
                    (
                        SELECT COUNT(*) FROM run_events re WHERE re.run_id = r.run_id
                    ) AS event_count
                FROM runs r
                JOIN experiments e ON e.experiment_id = r.experiment_id
                WHERE r.run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._run_snapshot(row)

    def get_run_result(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None or row["result_json"] is None:
            return None
        return json.loads(row["result_json"])

    def get_run_history(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence_number, event_type, timestamp, payload_json
                FROM run_events
                WHERE run_id = ? AND sequence_number > ?
                ORDER BY sequence_number
                """,
                (run_id, after),
            ).fetchall()
        return [
            {
                "sequence_number": row["sequence_number"],
                "type": row["event_type"],
                "timestamp": row["timestamp"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def list_runs_for_experiment(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    r.*,
                    e.name AS experiment_name,
                    e.description AS experiment_description,
                    (
                        SELECT COUNT(*) FROM run_events re WHERE re.run_id = r.run_id
                    ) AS event_count
                FROM runs r
                JOIN experiments e ON e.experiment_id = r.experiment_id
                WHERE r.experiment_id = ?
                ORDER BY r.updated_at DESC, r.created_at DESC
                """,
                (experiment_id,),
            ).fetchall()
        return [
            self._run_snapshot(row, include_result=False, include_request=False) for row in rows
        ]

    def list_runs(self) -> list[dict[str, Any]]:
        """Return all saved runs, newest first, for the dashboard history page."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    r.*,
                    e.name AS experiment_name,
                    e.description AS experiment_description,
                    (
                        SELECT COUNT(*) FROM run_events re WHERE re.run_id = r.run_id
                    ) AS event_count
                FROM runs r
                JOIN experiments e ON e.experiment_id = r.experiment_id
                ORDER BY r.updated_at DESC, r.created_at DESC
                """
            ).fetchall()
        return [self._run_snapshot(row, include_result=False, include_request=True) for row in rows]

    def get_artifacts(self, run_id: str) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        experiment = self.get_experiment(run["experiment_id"])
        return {
            "experiment": experiment,
            "run": run,
            "events": self.get_run_history(run_id),
        }

    def _refresh_run_state(self, connection: sqlite3.Connection, run_id: str) -> None:
        rows = connection.execute(
            """
            SELECT event_type, timestamp, payload_json
            FROM run_events
            WHERE run_id = ?
            ORDER BY sequence_number
            """,
            (run_id,),
        ).fetchall()
        status = "queued"
        updated_at = None
        best_metric = None
        best_cost = None
        best_params = None
        duration = None
        n_trials = 0
        n_failures = 0
        error = None
        for row in rows:
            event_type = row["event_type"]
            payload = json.loads(row["payload_json"])
            updated_at = row["timestamp"]
            if event_type in {
                "dataset_prepared",
                "worker_started",
                "run_started",
                "trial_started",
                "trial_completed",
                "trial_failed",
                "iteration_completed",
                "best_updated",
                "run_warning",
            }:
                status = "running"
            if event_type == "run_completed":
                status = "completed"
            if event_type in {"run_failed", "validation_failed"}:
                status = "failed"
            if event_type == "run_cancelled":
                status = "cancelled"
            if event_type == "run_interrupted":
                status = "interrupted"
            if event_type in {"trial_completed", "trial_failed"}:
                n_trials = max(n_trials, int(payload.get("trial_id", n_trials)))
            if event_type == "trial_failed":
                n_failures += 1
            if event_type == "best_updated":
                best_metric = payload.get("best_metric")
                best_cost = payload.get("best_cost")
                best_params = payload.get("best_params")
            if event_type in {"run_completed", "run_failed"}:
                duration = payload.get("duration", duration)
                n_trials = int(payload.get("n_trials", n_trials))
                n_failures = int(payload.get("n_failures", n_failures))
                best_metric = payload.get("best_metric", best_metric)
                best_cost = payload.get("best_cost", best_cost)
                best_params = payload.get("best_params", best_params)
                error = payload.get("error", error)
            if event_type == "run_failed":
                error = payload.get("error", error)
            # Trailing buffered logs/progress must never resurrect a finished run.
            if event_type in {"run_completed", "run_failed", "run_cancelled", "run_interrupted"}:
                break
        connection.execute(
            """
            UPDATE runs
            SET status = ?,
                updated_at = COALESCE(?, updated_at),
                best_metric = ?,
                best_cost = ?,
                best_params_json = ?,
                duration = ?,
                n_trials = ?,
                n_failures = ?,
                error = ?
            WHERE run_id = ?
            """,
            (
                status,
                updated_at,
                best_metric,
                best_cost,
                json.dumps(best_params) if best_params is not None else None,
                duration,
                n_trials,
                n_failures,
                error,
                run_id,
            ),
        )

    @staticmethod
    def _experiment_summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "experiment_id": row["experiment_id"],
            "name": row["name"],
            "description": row["description"],
            "tags": json.loads(row["tags_json"] or "[]"),
            "is_ad_hoc": bool(row["is_ad_hoc"]),
            "created_at": row["created_at"],
            "run_count": row["run_count"],
            "latest_status": row["latest_status"],
            "latest_run_id": row["latest_run_id"],
            "latest_updated_at": row["latest_updated_at"],
            "result_layout": json.loads(row["result_layout_json"] or "[]"),
        }

    def _run_snapshot(
        self,
        row: sqlite3.Row,
        *,
        include_result: bool = True,
        include_request: bool = True,
    ) -> dict[str, Any]:
        best_params = json.loads(row["best_params_json"]) if row["best_params_json"] else None
        result = json.loads(row["result_json"]) if include_result and row["result_json"] else None
        request = json.loads(row["request_json"]) if include_request else None
        attempts = self.list_attempts(row["run_id"])
        queued_attempt = next(
            (attempt for attempt in reversed(attempts) if attempt["status"] == "queued"), None
        )
        queue_position = None
        if queued_attempt is not None:
            with self._connect() as connection:
                queue_position = connection.execute(
                    """
                    SELECT COUNT(*) FROM run_attempts
                    WHERE status = 'queued' AND created_at <= ?
                    """,
                    (queued_attempt["created_at"],),
                ).fetchone()[0]
        return {
            "run_id": row["run_id"],
            "experiment_id": row["experiment_id"],
            "experiment_name": row["experiment_name"],
            "experiment_description": row["experiment_description"],
            "status": row["status"],
            "event_count": row["event_count"],
            "best": (
                {
                    "best_metric": row["best_metric"],
                    "best_cost": row["best_cost"],
                    "best_params": best_params,
                }
                if best_params is not None
                or row["best_metric"] is not None
                or row["best_cost"] is not None
                else None
            ),
            "result": result,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "duration": row["duration"],
            "n_trials": row["n_trials"],
            "n_failures": row["n_failures"],
            "strategy": row["strategy"],
            "task": row["task"],
            "metric": row["metric"],
            "estimator": row["estimator"],
            "request": request,
            "cancel_requested": bool(row["cancel_requested"])
            if "cancel_requested" in row.keys()
            else False,
            "artifacts": json.loads(row["artifact_json"])
            if "artifact_json" in row.keys() and row["artifact_json"]
            else {},
            "parent_run_id": row["parent_run_id"] if "parent_run_id" in row.keys() else None,
            "attempts": attempts,
            "queue_position": queue_position,
        }

    @staticmethod
    def _dataset_snapshot(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "dataset_id": row["dataset_id"],
            "name": row["name"],
            "source_kind": row["source_kind"],
            "source_path": row["source_path"],
            "fingerprint": row["fingerprint"],
            "profile": json.loads(row["profile_json"]),
            "created_at": row["created_at"],
        }
