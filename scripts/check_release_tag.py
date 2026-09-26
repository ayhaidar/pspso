"""Check that a release tag and all package surfaces use the project version."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def project_version(root: Path) -> str:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?ms)^\[project\]\s*$.*?^version\s*=\s*"([^"]+)"', pyproject)
    if match is None:
        raise SystemExit("Could not read [project].version from pyproject.toml.")
    return match.group(1)


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("Usage: check_release_tag.py v<version>")
    root = Path(__file__).resolve().parents[1]
    version = project_version(root)
    expected_tag = f"v{version}"
    tag = arguments[0]
    if tag != expected_tag:
        raise SystemExit(f"Release tag {tag!r} must exactly match {expected_tag!r}.")

    frontend_version = json.loads((root / "frontend/package.json").read_text(encoding="utf-8"))[
        "version"
    ]
    lock_version = json.loads((root / "frontend/package-lock.json").read_text(encoding="utf-8"))[
        "packages"
    ][""]["version"]
    mismatches = {
        name: value
        for name, value in {
            "frontend/package.json": frontend_version,
            "frontend/package-lock.json": lock_version,
        }.items()
        if value != version
    }
    if mismatches:
        details = ", ".join(f"{name}={value}" for name, value in mismatches.items())
        raise SystemExit(f"Version mismatch: pyproject.toml={version}, {details}.")
    print(f"Release tag and package versions match: {tag}")


if __name__ == "__main__":
    main()
