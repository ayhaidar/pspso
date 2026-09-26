"""Dashboard package without eager API startup in worker processes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_app(db_path: str | Path | None = None, *, start_manager: bool = True) -> Any:
    """Create the FastAPI application without importing it at package discovery time."""

    from .app import create_app as factory

    return factory(db_path, start_manager=start_manager)


__all__ = ["create_app"]
