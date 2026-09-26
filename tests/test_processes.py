import os
import subprocess
import sys
import time

import psutil
import pytest

from pspso.dashboard.processes import ProcessTree
from pspso.dashboard.tracking import TrackingRepository


def wait_until(predicate, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("Condition did not become true before the deadline.")


def exited(pid):
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_process_probe_does_not_signal_or_terminate_the_target():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert TrackingRepository._process_exists(process.pid)
        assert process.poll() is None
        assert not TrackingRepository._process_exists(-1)
    finally:
        process.kill()
        process.wait(timeout=5)
    assert not TrackingRepository._process_exists(process.pid)


@pytest.mark.parametrize("parent_exits_first", [False, True])
def test_worker_tree_cleanup_includes_children_after_parent_exit(tmp_path, parent_exits_first):
    gate = tmp_path / "go"
    child_file = tmp_path / "child"
    code = (
        "import pathlib, subprocess, sys, time; "
        "gate, child_file = map(pathlib.Path, sys.argv[1:3]); "
        "exec('while not gate.exists(): time.sleep(0.01)'); "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "child_file.write_text(str(child.pid)); "
        "time.sleep(0 if sys.argv[3] == 'exit' else 60)"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(gate),
            str(child_file),
            "exit" if parent_exits_first else "wait",
        ],
        start_new_session=os.name != "nt",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    tree = ProcessTree(process.pid)
    try:
        gate.touch()
        wait_until(lambda: child_file.exists() and child_file.read_text())
        child_pid = int(child_file.read_text())
        if parent_exits_first:
            process.wait(timeout=5)
        else:
            tree.refresh()
        tree.close()
        process.wait(timeout=5)
        wait_until(lambda: exited(child_pid))
    finally:
        tree.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows job containment")
def test_windows_job_kills_workers_when_service_crashes(tmp_path):
    worker_file = tmp_path / "worker"
    code = (
        "import pathlib, subprocess, sys, time; "
        "from pspso.dashboard.processes import ProcessTree; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "tree = ProcessTree(p.pid); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); "
        "time.sleep(60)"
    )
    service = subprocess.Popen(
        [sys.executable, "-c", code, str(worker_file)],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        wait_until(lambda: worker_file.exists() and worker_file.read_text())
        worker_pid = int(worker_file.read_text())
        assert psutil.pid_exists(worker_pid)
        service.kill()
        service.wait(timeout=5)
        wait_until(lambda: not psutil.pid_exists(worker_pid))
    finally:
        service.kill()
        service.wait(timeout=5)
