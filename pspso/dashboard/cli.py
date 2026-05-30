"""Console entrypoints for pspso."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn

from pspso.dashboard.app import app


def dashboard_main() -> None:
    parser = argparse.ArgumentParser(description="Start the pspso dashboard API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "pspso.dashboard.app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def run_main() -> None:
    parser = argparse.ArgumentParser(description="Run a pspso optimization from JSON config.")
    parser.add_argument("config", type=Path, help="Path to a JSON RunRequest payload.")
    args = parser.parse_args()
    from pspso.dashboard.app import RunRequest, _build_optimizer_inputs

    payload: dict[str, Any] = json.loads(args.config.read_text(encoding="utf-8"))
    request = RunRequest(**payload)
    X_train, y_train, X_val, y_val, optimizer = _build_optimizer_inputs(
        request,
        validate_dataset=False,
    )
    result = optimizer.optimize(X_train, y_train, X_val, y_val)
    print(json.dumps(result.to_dict(), indent=2, default=str))


__all__ = ["dashboard_main", "run_main"]
