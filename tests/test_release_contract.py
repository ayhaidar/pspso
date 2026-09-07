import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_packaging_exposes_only_modern_console_scripts():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "pspso-run" not in pyproject
    assert not (ROOT / "pspso" / "pspso.py").exists()


def test_frontend_uses_only_the_versioned_api_root():
    api_source = (ROOT / "frontend" / "src" / "api.ts").read_text(encoding="utf-8")
    workflow_source = (ROOT / "frontend" / "src" / "workflow.tsx").read_text(encoding="utf-8")
    assert 'const API = "/api/v1"' in api_source
    assert "new EventSource(`/api/v1/" in workflow_source


def test_current_documentation_does_not_advertise_removed_interfaces():
    excluded = {ROOT / "docs" / "migration" / "1.0.md", ROOT / "docs" / "release-notes" / "1.0.md"}
    forbidden = (
        "from pspso import pspso",
        "pspso-run",
        "SearchSpace.from_legacy",
        'task="binary classification"',
        'task="multiclass classification"',
        "/api/runs",
    )
    files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    violations = []
    for path in files:
        if path in excluded:
            continue
        text = path.read_text(encoding="utf-8")
        for value in forbidden:
            if value in text:
                violations.append(f"{path.relative_to(ROOT)}: {value}")
    assert not violations, "\n".join(violations)


def test_public_installation_assumes_the_pypi_package():
    files = [
        ROOT / "README.md",
        ROOT / "docs" / "index.md",
        ROOT / "docs" / "getting-started.md",
        ROOT / "docs" / "guides" / "cli.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for forbidden in (
        "being prepared for PyPI",
        "unreleased update",
        "uv init",
        "pspso-1.0.0-py3-none-any.whl",
        "/blob/dev/",
    ):
        assert forbidden not in combined
    assert "uv tool install pspso" in combined
    assert "uv add pspso" in combined


def test_release_tag_must_match_all_package_versions():
    script = ROOT / "scripts" / "check_release_tag.py"
    accepted = subprocess.run(
        [sys.executable, str(script), "v1.0.0"], capture_output=True, text=True, check=False
    )
    rejected = subprocess.run(
        [sys.executable, str(script), "v1.0.1"], capture_output=True, text=True, check=False
    )
    assert accepted.returncode == 0 and "versions match" in accepted.stdout
    assert rejected.returncode != 0 and "must exactly match" in rejected.stderr
