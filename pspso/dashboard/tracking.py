"""SQLite-backed tracking for experiments, runs, and run events."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pspso.config import OptimizationResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrackingRepository:
    """Persist dashboard activity so runs survive page refreshes and restarts."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or Path.cwd() / ".pspso" / "tracking.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
                """
            )

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
                INSERT INTO experiments (experiment_id, name, description, tags_json, is_ad_hoc, created_at)
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
        return self.get_experiment(experiment_id)

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

    def create_run(
        self,
        request_payload: dict[str, Any],
        *,
        experiment_id: str,
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
                    estimator
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
        snapshot = self.get_run(run_id)
        if snapshot is None:
            raise RuntimeError("Run could not be loaded after creation.")
        return snapshot

    def record_event(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        payload = dict(event["payload"])
        with self._lock, self._connect() as connection:
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
                INSERT INTO run_events (run_id, sequence_number, event_type, timestamp, payload_json)
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
                    json.dumps(payload["best_params"]) if payload["best_params"] is not None else None,
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

    def get_run_history(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence_number, event_type, timestamp, payload_json
                FROM run_events
                WHERE run_id = ?
                ORDER BY sequence_number
                """,
                (run_id,),
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
        return [self._run_snapshot(row, include_result=False, include_request=False) for row in rows]

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
        }

    @staticmethod
    def _run_snapshot(
        row: sqlite3.Row,
        *,
        include_result: bool = True,
        include_request: bool = True,
    ) -> dict[str, Any]:
        best_params = json.loads(row["best_params_json"]) if row["best_params_json"] else None
        result = json.loads(row["result_json"]) if include_result and row["result_json"] else None
        request = json.loads(row["request_json"]) if include_request else None
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
                if best_params is not None or row["best_metric"] is not None or row["best_cost"] is not None
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
        }
