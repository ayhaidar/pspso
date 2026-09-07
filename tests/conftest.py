from contextlib import ExitStack
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pspso.dashboard.app import create_app


@pytest.fixture
def client_factory():
    """Exercise real application startup/shutdown and close every worker service."""
    contexts: dict[Path, ExitStack] = {}

    def create(database: Path) -> TestClient:
        if database in contexts:
            contexts[database].close()
        stack = contexts[database] = ExitStack()
        return stack.enter_context(TestClient(create_app(database)))

    yield create
    for context in contexts.values():
        context.close()
