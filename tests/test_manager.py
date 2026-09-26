import os
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

import pspso.dashboard.manager as manager_module
from pspso.config import ProgressEvent
from pspso.dashboard import cli
from pspso.dashboard.manager import LocalRunManager
from pspso.dashboard.tracking import TrackingRepository


def _recorded_run(repository: TrackingRepository) -> str:
    experiment = repository.create_experiment("Manager checks")
    run = repository.create_run(
        {
            "schema_version": 1,
            "strategy": "random",
            "task": "regression",
            "metric": "rmse",
            "estimator": "linear_regression",
        },
        experiment_id=experiment["experiment_id"],
    )
    return run["run_id"]


def test_only_one_manager_can_own_a_workspace(tmp_path):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    manager = LocalRunManager(repository)
    try:
        with pytest.raises(RuntimeError, match="already owns this workspace"):
            LocalRunManager(TrackingRepository(repository.db_path))
    finally:
        manager.shutdown()


def test_terminal_event_transition_is_idempotent(tmp_path):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    run_id = _recorded_run(repository)

    first = repository.record_event(
        run_id, ProgressEvent("run_failed", {"error": "timeout"}).to_dict()
    )
    second = repository.record_event(
        run_id, ProgressEvent("run_failed", {"error": "process exit"}).to_dict()
    )

    assert first["sequence_number"] == second["sequence_number"]
    assert len(repository.get_run_history(run_id)) == 1
    assert repository.get_run(run_id)["status"] == "failed"


def test_repository_only_cli_store_does_not_claim_a_service(monkeypatch, tmp_path):
    monkeypatch.setenv("PSPSO_HOME", str(tmp_path))

    store = cli._store()

    assert store.manager is None
    with sqlite3.connect(store.repository.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM service_leases").fetchone()[0] == 0


@pytest.fixture
def workers(monkeypatch):
    launched = []

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 400000 + len(launched)
            self.returncode = None
            self.closed = 0
            launched.append(self)
            kwargs["stdout"].write(b"worker output\n")

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("worker", timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9

    class Tree:
        def __init__(self, pid):
            self.process = next(process for process in launched if process.pid == pid)
            self.created_at = 100.0

        def refresh(self):
            pass

        def close(self):
            self.process.closed += 1
            if self.process.returncode is None:
                self.process.kill()

    monkeypatch.setattr(manager_module.subprocess, "Popen", Process)
    monkeypatch.setattr(manager_module, "ProcessTree", Tree)
    return launched


def test_queue_recovery_preserves_attempts_and_only_interrupts_running_work(tmp_path, workers):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    first, queued = _recorded_run(repository), _recorded_run(repository)
    manager = LocalRunManager(repository, start_monitor=False, grace_seconds=0)
    manager.submit(first)
    attempt_id = manager.submit(queued, timeout_seconds=30)
    assert repository.get_run(queued)["queue_position"] == 1
    manager.shutdown()
    assert repository.get_run(first)["status"] == "interrupted"
    assert repository.get_run(queued)["status"] == "queued"
    restarted = LocalRunManager(repository, start_monitor=False, grace_seconds=0)
    try:
        restarted._tick()
        attempt = repository.list_attempts(queued)[0]
        assert attempt["attempt_id"] == attempt_id
        assert attempt["service_id"] == restarted.service_id
        assert attempt["timeout_seconds"] == 30
        repository.record_event(queued, ProgressEvent("run_completed", {}).to_dict())
        workers[-1].returncode = 0
        restarted._tick()
        assert repository.list_attempts(queued)[0]["status"] == "completed"
        assert workers[-1].closed == 1
    finally:
        restarted.shutdown()


@pytest.mark.parametrize("reason", ["cancelled", "timeout", "interrupted"])
def test_termination_has_one_outcome_and_waits_for_process_cleanup(tmp_path, workers, reason):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    run_id = _recorded_run(repository)
    manager = LocalRunManager(repository, start_monitor=False, grace_seconds=0)
    manager.submit(run_id, timeout_seconds=0 if reason == "timeout" else None)
    if reason == "cancelled":
        assert manager.cancel(run_id)
        started = manager._active[run_id].termination_started_at
        assert manager.cancel(run_id)
        assert manager._active[run_id].termination_started_at == started
    if reason == "interrupted":
        manager.shutdown()
    else:
        manager._tick()
        manager.shutdown()
    repository.record_event(run_id, ProgressEvent("trial_completed", {"trial_id": 1}).to_dict())
    repository.record_event(run_id, ProgressEvent("run_completed", {}).to_dict())
    terminal = [
        event
        for event in repository.get_run_history(run_id)
        if event["type"] in {"run_completed", "run_failed", "run_cancelled", "run_interrupted"}
    ]
    assert len(terminal) == 1
    assert terminal[0]["payload"]["reason"] == reason
    expected = "failed" if reason == "timeout" else reason
    assert repository.get_run(run_id)["status"] == expected
    attempt = repository.list_attempts(run_id)[0]
    assert attempt["status"] == expected
    assert attempt["termination_requested"] == 1
    assert attempt["termination_reason"] == reason
    assert attempt["finished_at"]
    assert workers[0].closed == 1
    assert workers[0].poll() is not None


def test_queued_cancel_and_completed_cancel_are_idempotent(tmp_path, workers):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    first, queued = _recorded_run(repository), _recorded_run(repository)
    manager = LocalRunManager(repository, start_monitor=False, grace_seconds=0)
    try:
        manager.submit(first)
        manager.submit(queued)
        assert manager.cancel(queued)
        assert not manager.cancel(queued)
        assert not manager.cancel("missing")
        assert repository.list_attempts(queued)[0]["status"] == "cancelled"
        repository.record_event(first, ProgressEvent("run_completed", {}).to_dict())
        assert not manager.cancel(first)
        assert not repository.cancel_requested(first)
    finally:
        manager.shutdown()
    assert repository.list_attempts(first)[0]["status"] == "completed"


def test_spawn_failure_does_not_break_the_queue(tmp_path, monkeypatch):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    manager = LocalRunManager(repository, start_monitor=False, grace_seconds=0)

    def fail(*args, **kwargs):
        raise OSError("cannot launch worker")

    monkeypatch.setattr(manager_module.subprocess, "Popen", fail)
    try:
        run_id = _recorded_run(repository)
        manager.submit(run_id)
        assert repository.get_run(run_id)["error"] == "cannot launch worker"
        assert repository.list_attempts(run_id)[0]["status"] == "failed"
    finally:
        manager.shutdown()
    with pytest.raises(RuntimeError, match="shutting down"):
        manager.submit(_recorded_run(repository))


def test_unexpected_process_exit_records_log_and_failure(tmp_path, workers):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    manager = LocalRunManager(repository, start_monitor=False, grace_seconds=0)
    try:
        run_id = _recorded_run(repository)
        manager.submit(run_id)
        workers[0].returncode = 17
        manager._tick()
        snapshot = repository.get_run(run_id)
        assert snapshot["status"] == "failed"
        assert "17" in snapshot["error"]
        logs = [e for e in repository.get_run_history(run_id) if e["type"] == "run_log"]
        assert [e["payload"]["message"] for e in logs] == ["worker output"]
        assert snapshot["attempts"][0]["status"] == "failed"
    finally:
        manager.shutdown()


def test_worker_output_is_persisted_as_one_database_batch(tmp_path, workers, monkeypatch):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    batches = []
    original = repository.record_log_messages

    def record_batch(run_id, messages):
        batches.append(list(messages))
        return original(run_id, messages)

    monkeypatch.setattr(repository, "record_log_messages", record_batch)
    manager = LocalRunManager(repository, start_monitor=False, grace_seconds=0)
    try:
        run_id = _recorded_run(repository)
        manager.submit(run_id)
        with manager._active[run_id].log_path.open("ab") as log_file:
            for index in range(500):
                log_file.write(f"warning {index}\n".encode())
        workers[0].returncode = 0

        manager._tick()

        assert len(batches) == 1
        assert len(batches[0]) == 501
        logs = [event for event in repository.get_run_history(run_id) if event["type"] == "run_log"]
        assert len(logs) == 501
        assert logs[-1]["payload"]["message"] == "warning 499"
    finally:
        manager.shutdown()


def test_concurrent_service_claim_is_atomic(tmp_path):
    database = tmp_path / "tracking.sqlite3"
    TrackingRepository(database)

    def claim(number):
        try:
            TrackingRepository(database).register_service(str(number), os.getpid())
            return True
        except RuntimeError:
            return False

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert sum(executor.map(claim, range(4))) == 1


def test_repository_cli_reads_leave_active_service_and_run_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("PSPSO_HOME", str(tmp_path))
    repository = TrackingRepository()
    repository.register_service("live-service", os.getpid())
    run_id = _recorded_run(repository)
    attempt = repository.create_attempt(run_id, service_id="live-service")
    repository.set_attempt_running(attempt["attempt_id"], os.getpid(), "live-service")
    repository.record_event(run_id, ProgressEvent("worker_started", {}).to_dict())
    before = repository.get_run(run_id)
    store = cli._store()
    assert store.get_snapshot(run_id) == before
    assert store.repository.list_runs()[0]["status"] == "running"
    assert repository.get_run(run_id) == before


def test_lost_service_lease_fences_manager(tmp_path, workers):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    manager = LocalRunManager(repository, start_monitor=False, grace_seconds=0)
    repository.release_service(manager.service_id)
    try:
        with pytest.raises(RuntimeError, match="no longer owns"):
            manager._tick()
    finally:
        manager.shutdown()


def test_recovery_preserves_fresh_leases_and_completes_stale_attempts(tmp_path):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    live, stale, completed = [_recorded_run(repository) for _ in range(3)]
    repository.register_service("healthy", os.getpid())
    for run_id, service in [(live, "healthy"), (stale, "gone"), (completed, "gone")]:
        attempt = repository.create_attempt(run_id, service_id=service)
        repository.set_attempt_running(attempt["attempt_id"], 9999999, service)
        repository.record_event(run_id, ProgressEvent("worker_started", {}).to_dict())
    repository.record_event(completed, ProgressEvent("run_completed", {}).to_dict())
    assert repository.interrupt_stale_attempts(service_id="new") == 2
    assert repository.get_run(live)["status"] == "running"
    assert repository.get_run(stale)["status"] == "interrupted"
    assert repository.list_attempts(stale)[0]["status"] == "interrupted"
    assert repository.get_run(completed)["status"] == "completed"
    assert repository.list_attempts(completed)[0]["status"] == "completed"


def test_worker_attempt_rejects_duplicate_submission(tmp_path):
    repository = TrackingRepository(tmp_path / "tracking.sqlite3")
    run_id = _recorded_run(repository)
    repository.create_attempt(run_id)
    with pytest.raises(ValueError, match="active attempt"):
        repository.create_attempt(run_id)
    assert len(repository.list_attempts(run_id)) == 1


def test_terminal_event_is_atomic_between_independent_repositories(tmp_path):
    database = tmp_path / "tracking.sqlite3"
    repository = TrackingRepository(database)
    run_id = _recorded_run(repository)

    def finish(number):
        other = TrackingRepository(database)
        return other.record_event(
            run_id, ProgressEvent("run_failed", {"error": str(number)}).to_dict()
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(finish, range(12)))
    assert len({event["sequence_number"] for event in results}) == 1
    assert len(repository.get_run_history(run_id)) == 1


def test_app_factory_does_not_start_a_manager_and_lifespan_releases_it(tmp_path):
    from fastapi.testclient import TestClient

    from pspso.dashboard.app import create_app

    app = create_app(tmp_path / "tracking.sqlite3")
    assert app.state.store.manager is None
    with TestClient(app) as client:
        assert app.state.store.manager is not None
        assert client.get("/api/v1/runs").status_code == 200
    assert app.state.store.manager is None
    with sqlite3.connect(tmp_path / "tracking.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM service_leases").fetchone()[0] == 0
