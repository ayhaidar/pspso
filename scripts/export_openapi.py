"""Export the API contract without starting a service or touching a user's workspace."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pspso.dashboard.app import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="pspso-openapi-") as temporary:
        app = create_app(Path(temporary) / "tracking.sqlite3", start_manager=False)
        schema = app.openapi()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
