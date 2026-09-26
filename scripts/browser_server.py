"""Run the packaged dashboard in an isolated temporary workspace for browser tests."""

import os
import tempfile
from pathlib import Path

import uvicorn

from pspso.dashboard.app import create_app


def main():
    with tempfile.TemporaryDirectory(prefix="pspso-browser-") as workspace:
        os.environ["PSPSO_HOME"] = workspace
        uvicorn.run(
            create_app(Path(workspace) / "tracking.sqlite3"),
            host="127.0.0.1",
            port=8371,
            log_level="warning",
        )


if __name__ == "__main__":
    main()
