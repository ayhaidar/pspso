import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pspso.dashboard import cli
from pspso.dashboard.tracking import TrackingRepository

ROOT = Path(__file__).parents[1]


@pytest.fixture
def live_service(tmp_path):
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    environment = {**os.environ, "PSPSO_HOME": str(tmp_path / "workspace")}
    url = f"http://127.0.0.1:{port}"
    with (tmp_path / "service.log").open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from pspso.dashboard.cli import dashboard_main; dashboard_main()",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail((tmp_path / "service.log").read_text())
            try:
                cli._api_request(url, "GET", "/api/v1/runs")
                break
            except RuntimeError:
                time.sleep(0.05)
        else:
            pytest.fail("Service did not become ready within 30 seconds.")
        yield url, environment
    finally:
        process.terminate()
        process.wait(timeout=10)


def spec(**updates):
    return {
        "dataset": {"source": "example", "name": "diabetes"},
        "task": "regression",
        "metric": "rmse",
        "estimator": "linear_regression",
        "search_space": {"fit_intercept": {"type": "choice", "values": [True]}},
        "runtime": {"max_trials": 1},
        **updates,
    }


def invoke(monkeypatch, capsys, home, *args):
    monkeypatch.setenv("PSPSO_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["pspso", *map(str, args)])
    code = 0
    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_version_and_command_help(monkeypatch, capsys, tmp_path):
    code, output, error = invoke(monkeypatch, capsys, tmp_path, "--version")
    assert code == 0
    assert output.strip() == "pspso 1.0.0"
    assert not error

    help_text = cli.build_parser().format_help()
    assert "dataset" in help_text
    assert "experiment" in help_text
    assert "run" in help_text
    run_action = next(
        action
        for action in cli.build_parser()._actions
        if isinstance(action, cli.argparse._SubParsersAction)
    )
    run_help = run_action.choices["run"].format_help()
    for command in ("validate", "start", "show", "cancel", "retry", "list"):
        assert command in run_help

    monkeypatch.setattr(sys, "argv", ["pspso-dashboard", "--version"])
    with pytest.raises(SystemExit) as exc:
        cli.dashboard_main()
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "pspso-dashboard 1.0.0"


def test_dashboard_warns_when_bound_beyond_loopback(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", ["pspso-dashboard", "--host", "0.0.0.0"])
    cli.dashboard_main()
    assert calls[0][1]["host"] == "0.0.0.0"
    assert "no built-in authentication" in capsys.readouterr().err


def test_cli_rejects_non_http_api_urls():
    with pytest.raises(RuntimeError, match="http:// or https://"):
        cli._api_request("file:///tmp/pspso", "GET", "/api/v1/runs")


def test_cli_start_list_show_cancel_retry_against_live_service(
    live_service, tmp_path, monkeypatch, capsys
):
    url, environment = live_service
    monkeypatch.setenv("PSPSO_API_URL", url)
    home = Path(environment["PSPSO_HOME"])
    path = tmp_path / "run.json"
    # A substantial first candidate keeps the real worker active during read-only CLI commands.
    path.write_text(
        json.dumps(
            spec(
                estimator="random_forest",
                search_space={"n_estimators": {"type": "choice", "values": [50000]}},
                runtime={"max_trials": 1, "timeout_seconds": 20},
            )
        )
    )
    code, output, error = invoke(monkeypatch, capsys, home, "run", "start", path)
    assert code == 0, error
    run_id = json.loads(output)["run_id"]
    before = cli._api_request(url, "GET", f"/api/v1/runs/{run_id}")
    for arguments in [("list",), ("show", run_id)]:
        code, output, error = invoke(monkeypatch, capsys, home, "run", *arguments)
        assert code == 0, error
        assert output
    after = cli._api_request(url, "GET", f"/api/v1/runs/{run_id}")
    assert before["status"] in {"queued", "running"}
    assert after["status"] in {"queued", "running"}
    assert after["attempts"][0]["service_id"] == before["attempts"][0]["service_id"]
    code, _, error = invoke(monkeypatch, capsys, home, "--api-url", url, "run", "cancel", run_id)
    assert code == 0, error
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        after = cli._api_request(url, "GET", f"/api/v1/runs/{run_id}")
        if after["attempts"][0]["status"] == "cancelled":
            break
        time.sleep(0.05)
    assert after["status"] == "cancelled"
    assert after["attempts"][0]["status"] == "cancelled"
    code, output, error = invoke(monkeypatch, capsys, home, "run", "retry", run_id)
    assert code == 0, error
    retried = json.loads(output)
    assert retried["parent_run_id"] == run_id
    assert retried["run_id"] != run_id
    assert retried["attempts"][0]["parent_run_id"] == run_id
    invoke(monkeypatch, capsys, home, "run", "cancel", retried["run_id"])


def test_cli_standalone_wait_runs_without_an_api(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(spec()))
    code, output, error = invoke(
        monkeypatch, capsys, tmp_path, "run", "start", path, "--standalone", "--wait"
    )
    assert code == 0, error
    run = json.loads(output)
    assert run["status"] == "completed", run
    assert run["attempts"][0]["worker_pid"] > 0
    assert run["attempts"][0]["status"] == "completed"
    assert Path(run["artifacts"]["model"]).is_file()


def test_cli_validation_failures_have_nonzero_exit_codes(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(spec(metric="roc_auc")))
    code, output, _ = invoke(monkeypatch, capsys, tmp_path, "run", "validate", path)
    assert code == 1
    assert json.loads(output)["valid"] is False
    path.write_text(json.dumps(spec()))
    code, _, error = invoke(monkeypatch, capsys, tmp_path, "run", "start", path, "--standalone")
    assert code == 2
    assert "--wait" in error
    assert "Traceback" not in error


def test_cli_dataset_and_experiment_commands(tmp_path, monkeypatch, capsys):
    path = tmp_path / "data.csv"
    path.write_text("feature,target\n1,0\n2,1\n3,0\n4,1\n")
    code, output, error = invoke(monkeypatch, capsys, tmp_path, "dataset", "import", path)
    assert code == 0, error
    dataset = json.loads(output)
    code, output, _ = invoke(monkeypatch, capsys, tmp_path, "dataset", "list")
    assert json.loads(output)[0]["dataset_id"] == dataset["dataset_id"]
    code, output, error = invoke(monkeypatch, capsys, tmp_path, "experiment", "create", "CLI test")
    assert code == 0, error
    assert json.loads(output)["name"] == "CLI test"
    assert TrackingRepository(tmp_path / "v1" / "tracking.sqlite3").list_runs() == []


def test_cli_yaml_and_unreachable_service(tmp_path, monkeypatch, capsys):
    path = tmp_path / "spec.yaml"
    path.write_text("task: regression\nmetric: rmse\nestimator: linear_regression\n")
    assert cli._read_spec(path)["task"] == "regression"
    monkeypatch.setattr(
        cli.urlrequest,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(cli.urlerror.URLError("offline")),
    )
    with pytest.raises(RuntimeError, match="not reachable"):
        cli._api_request("http://127.0.0.1:1", "GET", "/api/v1/runs")
