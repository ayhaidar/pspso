"""Clean generated package directories, build PSPSO, and verify their contents."""

import json
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def _project_version(root: Path) -> str:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?ms)^\[project\]\s*$.*?^version\s*=\s*"([^"]+)"', pyproject)
    if match is None:
        raise SystemExit("Could not read [project].version from pyproject.toml.")
    return match.group(1)


def main():
    root = Path(__file__).resolve().parents[1]
    version = _project_version(root)
    if not (root / "pspso/dashboard/static/index.html").is_file():
        raise SystemExit("Build the dashboard with npm run build in frontend first.")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    (root / "pspso/build-info.json").write_text(
        json.dumps(
            {
                "source_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                "source_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ("build", "dist"):
        target = (root / name).resolve()
        if target.parent != root or target.name != name or target.is_symlink():
            raise SystemExit(f"Refusing to clean an unexpected build directory: {target}")
        if target.exists():
            shutil.rmtree(target)
    subprocess.run([sys.executable, "-m", "build"], cwd=root, check=True)
    with tarfile.open(root / f"dist/pspso-{version}.tar.gz") as archive:
        source_files = set(archive.getnames())
        for name in (
            "tests/conftest.py",
            "frontend/package-lock.json",
            "frontend/src/App.tsx",
            "frontend/e2e/workflow.spec.ts",
            "scripts/check_release_tag.py",
            "scripts/smoke_wheel.py",
            "pspso/datasets/banknote_authentication.csv",
            "pspso/datasets/auto_mpg.csv",
            "pspso/datasets/palmer_penguins.csv",
            "pspso/datasets/README.md",
            ".github/workflows/docs.yml",
            ".github/workflows/publish.yml",
            ".github/workflows/verify.yml",
        ):
            assert f"pspso-{version}/{name}" in source_files, f"Source archive is missing {name}"
    wheels = list((root / "dist").glob("*.whl"))
    assert len(wheels) == 1 and wheels[0].name == f"pspso-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheels[0]) as wheel:
        files = wheel.namelist()
        assert "pspso/py.typed" in files
        assert "pspso/dashboard/static/index.html" in files
        for dataset in (
            "banknote_authentication.csv",
            "auto_mpg.csv",
            "palmer_penguins.csv",
        ):
            assert f"pspso/datasets/{dataset}" in files
        assert any(name.startswith("pspso/dashboard/static/assets/") for name in files)
        assert not any("legacy" in name or "pspso/pspso.py" == name for name in files)
        entries = wheel.read(f"pspso-{version}.dist-info/entry_points.txt").decode()
        assert "pspso-run" not in entries
        assert "pspso-dashboard" in entries and "pspso =" in entries
    print(
        json.dumps(
            {"version": version, "wheel": str(wheels[0]), "files": len(files), "verified": True}
        )
    )


if __name__ == "__main__":
    main()
