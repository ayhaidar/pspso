"""Exercise an installed wheel using only its runtime dependencies and the stdlib."""

import importlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from urllib.request import urlopen


def _expected_version() -> str:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?ms)^\[project\]\s*$.*?^version\s*=\s*"([^"]+)"', pyproject)
    if match is None:
        raise AssertionError("Could not read [project].version from pyproject.toml.")
    return match.group(1)


def main():
    import pspso
    from pspso import OptimizationConfig, PSPSOOptimizer, SearchSpace, optimize
    from pspso.dashboard.data import DatasetSelection, load_dataset

    assert all((OptimizationConfig, PSPSOOptimizer, SearchSpace, optimize))
    package = Path(pspso.__file__).resolve().parent
    assert "site-packages" in package.parts, f"Expected installed wheel, got {package}"
    expected_version = _expected_version()
    assert metadata.version("pspso") == expected_version
    assert (package / "py.typed").is_file()
    for name, target, rows in (
        ("banknote_authentication", "class", 1372),
        ("auto_mpg", "mpg", 398),
        ("palmer_penguins", "species", 344),
    ):
        frame = load_dataset(DatasetSelection(source="example", name=name))
        assert len(frame) == rows and target in frame
    for name in ("pspso.pspso", "pspso.PSPSO", "pspso.legacy"):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError(f"Legacy import unexpectedly succeeded: {name}")
    entries = {entry.name for entry in metadata.distribution("pspso").entry_points}
    assert {"pspso", "pspso-dashboard"} <= entries and "pspso-run" not in entries
    for name in ("pspso", "pspso-dashboard"):
        executable = Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else ""))
        subprocess.run([str(executable), "--help"], check=True, capture_output=True)
        version_output = subprocess.run(
            [str(executable), "--version"], check=True, capture_output=True, text=True
        ).stdout
        assert version_output.strip() == f"{name} {expected_version}"
    with tempfile.TemporaryDirectory(prefix="pspso-wheel-") as workspace:
        environment = {**os.environ, "PSPSO_HOME": workspace, "PYTHONPATH": ""}
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [
            sys.executable,
            "-I",
            "-c",
            "from pspso.dashboard.cli import dashboard_main; dashboard_main()",
            "--port",
            str(port),
        ]
        with open(Path(workspace) / "service.log", "w", encoding="utf-8") as log:
            service = subprocess.Popen(
                command, cwd=workspace, env=environment, stdout=log, stderr=log
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        with urlopen(
                            f"http://127.0.0.1:{port}/api/v1/estimators", timeout=2
                        ) as response:
                            assert "estimators" in json.load(response)
                        break
                    except OSError:
                        if service.poll() is not None or time.monotonic() > deadline:
                            raise AssertionError("Installed dashboard failed to start") from None
                        time.sleep(0.1)
                for route, expected in (
                    ("/", b'"root"'),
                    ("/history", b'"root"'),
                    ("/api/v1/docs", b"swagger"),
                    ("/api/v1/openapi.json", b"ExperimentSpec"),
                ):
                    with urlopen(f"http://127.0.0.1:{port}{route}", timeout=5) as response:
                        assert response.status == 200 and expected in response.read()
                with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                    html = response.read().decode("utf-8")
                assets = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html)
                assert assets and any(asset.endswith(".js") for asset in assets)
                for asset in assets:
                    with urlopen(f"http://127.0.0.1:{port}{asset}", timeout=5) as response:
                        assert response.status == 200 and len(response.read()) > 100
            finally:
                service.terminate()
                service.wait(timeout=15)
        from sklearn.datasets import load_diabetes

        X, y = load_diabetes(return_X_y=True)
        result = optimize(
            X,
            y,
            estimator="linear_regression",
            config=OptimizationConfig(strategy="random", max_trials=1, random_state=42),
        )
        assert result.best_metric is not None
    print(
        f"Fresh {expected_version} wheel: dashboard, SPA routes, API docs, CLI entries and "
        "bundled datasets and notebook training passed; no Node runtime used."
    )


if __name__ == "__main__":
    main()
