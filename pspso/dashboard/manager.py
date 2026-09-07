"""Durable local subprocess execution owned by one foreground service."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pspso.config import ProgressEvent
from pspso.dashboard.processes import ProcessTree
from pspso.dashboard.tracking import TrackingRepository

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}


@dataclass
class ActiveWorker:
    run_id: str
    attempt_id: str
    process: subprocess.Popen[bytes]
    tree: ProcessTree
    timeout_seconds: float | None
    started_at: float
    log_path: Path
    termination_reason: str | None = None
    termination_started_at: float | None = None
    log_offset: int = 0


class LocalRunManager:
    """Supervise recorded workers and recover queued attempts from SQLite."""

    def __init__(
        self,
        repository: TrackingRepository,
        max_workers: int | None = None,
        *,
        start_monitor: bool = True,
        grace_seconds: float = 3,
    ) -> None:
        self.repository = repository
        self.service_id = str(uuid.uuid4())
        self.max_workers = max(1, max_workers or int(os.environ.get("PSPSO_MAX_WORKERS", "1")))
        self.grace_seconds = max(0, grace_seconds)
        self._active: dict[str, ActiveWorker] = {}
        self._queue: deque[tuple[str, str, float | None]] = deque()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._shutdown = False
        self.last_error: str | None = None
        self.repository.register_service(self.service_id, os.getpid())
        try:
            self.repository.interrupt_stale_attempts(service_id=self.service_id)
            for attempt in self.repository.queued_attempts():
                self._queue.append(
                    (attempt["run_id"], attempt["attempt_id"], attempt.get("timeout_seconds"))
                )
        except BaseException:
            self.repository.release_service(self.service_id)
            raise
        self._monitor = threading.Thread(
            target=self._monitor_workers, daemon=True, name="pspso-worker-monitor"
        )
        if start_monitor:
            self._monitor.start()

    def submit(
        self, run_id: str, timeout_seconds: float | None = None, parent_run_id: str | None = None
    ) -> str:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("The worker service is shutting down.")
            attempt = self.repository.create_attempt(
                run_id,
                parent_run_id=parent_run_id,
                service_id=self.service_id,
                timeout_seconds=timeout_seconds,
            )
            self.repository.clear_cancel_request(run_id)
            self._event(run_id, "run_queued", {"attempt_id": attempt["attempt_id"]})
            self._queue.append((run_id, attempt["attempt_id"], timeout_seconds))
            self._dispatch_locked()
            return attempt["attempt_id"]

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            snapshot = self.repository.get_run(run_id)
            if snapshot is None or snapshot["status"] in TERMINAL_STATUSES:
                return False
            for item in list(self._queue):
                if item[0] == run_id:
                    self.repository.request_termination(item[1], "cancelled")
                    self.repository.request_cancel(run_id)
                    self._queue.remove(item)
                    self._event(run_id, "run_cancelled", {"reason": "cancelled while queued"})
                    self.repository.finish_attempt(item[1], "cancelled")
                    return True
            worker = self._active.get(run_id)
            if worker is None:
                return False
            if worker.termination_reason is None:
                self._request_stop(worker, "cancelled")
                self.repository.request_cancel(run_id)
                self._event(run_id, "run_cancel_requested", {})
            return True

    def _request_stop(self, worker: ActiveWorker, reason: str) -> None:
        if worker.termination_reason is None:
            self.repository.request_termination(worker.attempt_id, reason)
            worker.termination_reason = reason
            worker.termination_started_at = time.monotonic()

    def shutdown(self) -> None:
        self._stop.set()
        if self._monitor.is_alive() and threading.current_thread() is not self._monitor:
            self._monitor.join(timeout=10)
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            workers = list(self._active.values())
            for worker in workers:
                self._request_stop(worker, "interrupted")
            deadline = time.monotonic() + self.grace_seconds
            for worker in workers:
                try:
                    worker.process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
                self._finish_worker(worker)
            self.repository.release_service(self.service_id)

    def _dispatch_locked(self) -> None:
        while not self._shutdown and len(self._active) < self.max_workers and self._queue:
            run_id, attempt_id, timeout_seconds = self._queue.popleft()
            process = None
            tree = None
            try:
                log_path = self.repository.workspace / "artifacts" / "runs" / run_id / "worker.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("ab") as log_file:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "pspso.dashboard.worker",
                            "--database",
                            str(self.repository.db_path.resolve()),
                            "--run-id",
                            run_id,
                            "--attempt-id",
                            attempt_id,
                        ],
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        start_new_session=os.name != "nt",
                        env={**os.environ, "PSPSO_WORKER": "1", "PYTHONUNBUFFERED": "1"},
                    )
                tree = ProcessTree(process.pid)
                worker = ActiveWorker(
                    run_id,
                    attempt_id,
                    process,
                    tree,
                    timeout_seconds,
                    time.monotonic(),
                    log_path,
                )
                self.repository.set_attempt_running(
                    attempt_id,
                    process.pid,
                    self.service_id,
                    tree.created_at,
                )
                self._active[run_id] = worker
                self._event(
                    run_id,
                    "worker_started",
                    {
                        "attempt_id": attempt_id,
                        "pid": process.pid,
                        "runner": "local",
                    },
                )
            except Exception as exc:
                if tree is not None:
                    tree.close()
                if process is not None:
                    process.kill()
                    process.wait(timeout=5)
                self._event(run_id, "run_failed", {"error": str(exc), "reason": "process_start"})
                self.repository.finish_attempt(attempt_id, "failed")

    def _capture_output(self, worker: ActiveWorker) -> None:
        with worker.log_path.open("rb") as log_file:
            log_file.seek(worker.log_offset)
            output = log_file.read()
            worker.log_offset = log_file.tell()
        messages = [line for line in output.decode("utf-8", errors="replace").splitlines() if line]
        if messages:
            self.repository.record_log_messages(worker.run_id, messages)

    def _tick(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            if not self.repository.heartbeat_service(self.service_id):
                raise RuntimeError("This service no longer owns the workspace lease.")
            now = time.monotonic()
            for worker in list(self._active.values()):
                worker.tree.refresh()
                self._capture_output(worker)
                self.repository.heartbeat_attempt(worker.attempt_id)
                if worker.process.poll() is None:
                    if (
                        worker.timeout_seconds is not None
                        and now - worker.started_at >= worker.timeout_seconds
                    ):
                        self._request_stop(worker, "timeout")
                    if (
                        worker.termination_started_at is None
                        or now - worker.termination_started_at < self.grace_seconds
                    ):
                        continue
                self._finish_worker(worker)
            self._dispatch_locked()

    def _finish_worker(self, worker: ActiveWorker) -> None:
        # Contain descendants even when the parent already returned successfully.
        worker.tree.close()
        code = worker.process.wait(timeout=5)
        self._capture_output(worker)
        snapshot = self.repository.get_run(worker.run_id)
        status = snapshot["status"] if snapshot else "failed"
        if status not in TERMINAL_STATUSES:
            reason = worker.termination_reason or "process_exit"
            event_type = {"cancelled": "run_cancelled", "interrupted": "run_interrupted"}.get(
                reason, "run_failed"
            )
            event = self._event(
                worker.run_id,
                event_type,
                {
                    "reason": reason,
                    "error": f"Worker exited with code {code}.",
                },
            )
            status = event["type"].removeprefix("run_")
        self.repository.finish_attempt(worker.attempt_id, status)
        self._active.pop(worker.run_id, None)

    def _monitor_workers(self) -> None:
        try:
            while not self._stop.wait(0.2):
                self._tick()
        except Exception as exc:
            self.last_error = str(exc)
            self.shutdown()

    def _event(self, run_id: str, event_type: str, payload: dict) -> dict[str, Any]:
        return self.repository.record_event(run_id, ProgressEvent(event_type, payload).to_dict())
