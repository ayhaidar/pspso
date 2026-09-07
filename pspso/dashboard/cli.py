"""Console entrypoints for the local pspso experiment workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import uvicorn
from fastapi import HTTPException

from pspso.config import ProgressEvent
from pspso.dashboard.app import RunStore, _validate_request, create_app
from pspso.dashboard.data import preview_csv
from pspso.dashboard.manager import LocalRunManager
from pspso.dashboard.schemas import ExperimentCreateRequest, ExperimentSpec
from pspso.dashboard.tracking import TrackingRepository


def build_dashboard_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pspso-dashboard",
        description="Start the PSPSO dashboard and local experiment service.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {package_version('pspso')}"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind (default: 127.0.0.1).")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind (default: 8000).")
    parser.add_argument(
        "--reload", action="store_true", help="Restart automatically when source files change."
    )
    return parser


def dashboard_main() -> None:
    parser = build_dashboard_parser()
    args = parser.parse_args()
    if args.reload:
        uvicorn.run(
            "pspso.dashboard.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
    else:
        uvicorn.run(create_app(), host=args.host, port=args.port)


def _read_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML specs require PyYAML. Run `uv sync` to install project dependencies."
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _store() -> RunStore:
    return RunStore(TrackingRepository(), start_manager=False)


def _submit_standalone(store: RunStore, request: ExperimentSpec) -> dict[str, Any]:
    errors = _validate_request(request, store.repository)
    if any(errors.values()):
        raise ValueError(json.dumps({"valid": False, "errors": errors}, indent=2))
    manager = LocalRunManager(store.repository)
    run = None
    try:
        run = store.create(request)
        store.record_event(run.run_id, ProgressEvent("validation_passed", {"source": "cli"}))
        manager.submit(run.run_id, request.runtime.timeout_seconds)
        while True:
            snapshot = store.get_snapshot(run.run_id)
            if snapshot["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                break
            if manager.last_error:
                raise RuntimeError(manager.last_error)
            time.sleep(0.1)
    except KeyboardInterrupt:
        if run is not None:
            manager.cancel(run.run_id)
        raise
    finally:
        manager.shutdown()
    return store.get_snapshot(run.run_id)


def _api_request(
    api_url: str, method: str, path: str, payload: dict[str, Any] | None = None
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(
        f"{api_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlrequest.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"PSPSO service returned HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(
            "The PSPSO service is not reachable. Start `pspso-dashboard` or use "
            "`pspso run start --standalone --wait`."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pspso", description="Manage PSPSO datasets, experiments, and runs."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {package_version('pspso')}"
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("PSPSO_API_URL", "http://127.0.0.1:8000"),
        help="Local PSPSO service URL for start, cancel, and retry operations.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset", help="Import and inspect saved datasets.")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_import = dataset_commands.add_parser(
        "import", help="Store a CSV dataset in the local workspace."
    )
    dataset_import.add_argument("path", type=Path, help="Path to a UTF-8 CSV file.")
    dataset_import.add_argument("--name", help="Display name (default: CSV filename).")
    dataset_commands.add_parser("list", help="List saved datasets.")
    experiment = commands.add_parser("experiment", help="Create experiments.")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    experiment_create = experiment_commands.add_parser("create", help="Create a named experiment.")
    experiment_create.add_argument("name", help="Experiment name.")
    experiment_create.add_argument("--description", default="", help="Optional description.")
    experiment_create.add_argument(
        "--tag", action="append", default=[], help="Tag to attach; repeat for multiple tags."
    )
    experiment_commands.add_parser("list", help="List saved experiments.")
    run = commands.add_parser("run", help="Validate, start, and inspect runs.")
    run_commands = run.add_subparsers(dest="run_command", required=True)
    run_validate = run_commands.add_parser("validate", help="Validate a JSON or YAML run spec.")
    run_validate.add_argument("spec", type=Path, help="Path to the run specification.")
    run_start = run_commands.add_parser("start", help="Submit a run specification.")
    run_start.add_argument("spec", type=Path, help="Path to the run specification.")
    run_start.add_argument(
        "--wait", action="store_true", help="Wait for completion and return a useful exit code."
    )
    run_start.add_argument(
        "--standalone",
        action="store_true",
        help="Supervise a foreground worker without the FastAPI service.",
    )
    run_show = run_commands.add_parser("show", help="Show one run and its attempts.")
    run_show.add_argument("run_id", help="Run identifier from start or list.")
    run_cancel = run_commands.add_parser("cancel", help="Request cancellation of a managed run.")
    run_cancel.add_argument("run_id", help="Run identifier to cancel.")
    run_retry = run_commands.add_parser("retry", help="Create a retry linked to an earlier run.")
    run_retry.add_argument("run_id", help="Run identifier to retry.")
    run_retry.add_argument(
        "--wait", action="store_true", help="Wait for the retried run to finish."
    )
    run_commands.add_parser("list", help="List runs in the current workspace.")
    return parser


def _main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    store = _store()
    if args.command == "dataset":
        if args.dataset_command == "list":
            print(json.dumps(store.repository.list_datasets(), indent=2))
            return
        csv_text = args.path.read_text(encoding="utf-8")
        profile = preview_csv(csv_text, rows=8)
        fingerprint = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
        destination = store.repository.workspace / "datasets" / f"{fingerprint}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_text(csv_text, encoding="utf-8")
        created = store.repository.create_dataset(
            args.name or args.path.stem, "csv", str(destination), fingerprint, profile
        )
        print(json.dumps(created, indent=2))
        return
    if args.command == "experiment":
        if args.experiment_command == "list":
            print(json.dumps(store.repository.list_experiments(), indent=2))
            return
        created = store.create_experiment(
            ExperimentCreateRequest(name=args.name, description=args.description, tags=args.tag)
        )
        print(json.dumps(created, indent=2))
        return
    if args.run_command == "validate":
        request = ExperimentSpec(**_read_spec(args.spec))
        errors = _validate_request(request, store.repository)
        print(json.dumps({"valid": not any(errors.values()), "errors": errors}, indent=2))
        if any(errors.values()):
            raise SystemExit(1)
        return
    if args.run_command == "start":
        spec = ExperimentSpec(**_read_spec(args.spec))
        if args.standalone:
            if not args.wait:
                raise ValueError("--standalone requires --wait so the worker remains supervised.")
            snapshot = _submit_standalone(store, spec)
        else:
            snapshot = _api_request(args.api_url, "POST", "/api/v1/runs", spec.model_dump())
        if args.wait and not args.standalone:
            while snapshot["status"] not in {"completed", "failed", "cancelled", "interrupted"}:
                time.sleep(0.15)
                snapshot = _api_request(args.api_url, "GET", f"/api/v1/runs/{snapshot['run_id']}")
        print(json.dumps(snapshot, indent=2, default=str))
        if args.wait and snapshot["status"] != "completed":
            raise SystemExit(1)
        return
    if args.run_command == "list":
        print(json.dumps(store.repository.list_runs(), indent=2, default=str))
        return
    if args.run_command == "show":
        print(json.dumps(store.get_snapshot(args.run_id), indent=2, default=str))
        return
    if args.run_command == "cancel":
        print(
            json.dumps(
                _api_request(args.api_url, "POST", f"/api/v1/runs/{args.run_id}/cancel"), indent=2
            )
        )
        return
    retry = _api_request(args.api_url, "POST", f"/api/v1/runs/{args.run_id}/retry")
    if args.wait:
        while retry["status"] not in {"completed", "failed", "cancelled", "interrupted"}:
            time.sleep(0.15)
            retry = _api_request(args.api_url, "GET", f"/api/v1/runs/{retry['run_id']}")
    print(json.dumps(retry, indent=2))
    if args.wait and retry["status"] != "completed":
        raise SystemExit(1)


def main() -> None:
    try:
        _main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (ValueError, RuntimeError, OSError, HTTPException) as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        print(f"pspso: {detail}", file=sys.stderr)
        raise SystemExit(2) from None


__all__ = ["build_dashboard_parser", "build_parser", "dashboard_main", "main"]
