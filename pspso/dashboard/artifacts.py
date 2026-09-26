"""Local artifact storage for reproducible dashboard and CLI runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib

from pspso.config import OptimizationResult


class ArtifactStore:
    """Store run outputs alongside the local tracking database."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.runs_path = self.workspace / "artifacts" / "runs"
        self.runs_path.mkdir(parents=True, exist_ok=True)
        self.last_warnings: list[str] = []

    def run_path(self, run_id: str) -> Path:
        path = self.runs_path / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_run(
        self,
        run_id: str,
        request: dict[str, Any],
        result: OptimizationResult,
        transformer: Any | None = None,
        analysis: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        dataset: Any | None = None,
        selection_model: Any | None = None,
    ) -> dict[str, str]:
        path = self.run_path(run_id)
        self.last_warnings = []
        self._write_json(path / "spec.json", request)
        environment = self._environment_manifest(request)
        if provenance:
            environment.update(provenance)
        self._write_json(
            path / "environment.json",
            environment,
        )
        artifacts = {
            "spec": str(path / "spec.json"),
            "result": str(path / "result.json"),
            "environment": str(path / "environment.json"),
        }
        if dataset is not None:
            dataset_path = path / "dataset.joblib"
            joblib.dump(dataset, dataset_path)
            artifacts["dataset"] = str(dataset_path)
        if result.model is not None:
            model_path = path / "model.joblib"
            try:
                joblib.dump(result.model, model_path)
                artifacts["model"] = str(model_path)
            except Exception as exc:
                self.last_warnings.append(f"Model serialization failed: {exc}")
        if selection_model is not None:
            selection_path = path / "selection-model.joblib"
            try:
                joblib.dump(selection_model, selection_path)
                artifacts["selection_model"] = str(selection_path)
            except Exception as exc:
                self.last_warnings.append(f"Selection model serialization failed: {exc}")
        if transformer is not None:
            transformer_path = path / "preprocessor.joblib"
            try:
                joblib.dump(transformer, transformer_path)
                artifacts["preprocessor"] = str(transformer_path)
            except Exception as exc:
                self.last_warnings.append(f"Preprocessor serialization failed: {exc}")
        if analysis is not None:
            analysis_path = path / "analysis.json"
            self._write_json(analysis_path, analysis)
            artifacts["analysis"] = str(analysis_path)
        split_indices = result.optimizer_state.get("split_indices") or result.optimizer_state.get(
            "fold_indices"
        )
        if split_indices:
            split_path = path / "split-indices.json"
            self._write_json(split_path, split_indices)
            artifacts["split_indices"] = str(split_path)
        if self.last_warnings:
            warnings_path = path / "artifact-warnings.json"
            self._write_json(warnings_path, self.last_warnings)
            artifacts["warnings"] = str(warnings_path)
        worker_log = path / "worker.log"
        if worker_log.exists():
            artifacts["log"] = str(worker_log)
        artifacts["manifest"] = str(path / "manifest.json")
        result.artifacts = dict(artifacts)
        self.save_result(run_id, result)
        self._write_json(path / "manifest.json", artifacts)
        return artifacts

    def save_result(self, run_id: str, result: OptimizationResult) -> None:
        """Rewrite the portable result payload after provenance changes."""

        self._write_json(self.run_path(run_id) / "result.json", result.to_dict())

    def get_run_artifacts(self, run_id: str) -> dict[str, Any]:
        path = self.runs_path / run_id
        manifest = path / "manifest.json"
        if not manifest.exists():
            return {"paths": {}, "available": False}
        return {"paths": json.loads(manifest.read_text(encoding="utf-8")), "available": True}

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(_portable_json(payload), indent=2, default=str, allow_nan=False),
            encoding="utf-8",
        )

    @staticmethod
    def _environment_manifest(request: dict[str, Any]) -> dict[str, Any]:
        packages: dict[str, str | None] = {}
        for name in ("pspso", "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "torch"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
                cwd=Path(__file__).resolve().parents[2],
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = None
        package = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for source in sorted(package.rglob("*.py")):
            digest.update(source.relative_to(package).as_posix().encode())
            digest.update(source.read_bytes())
        build_info = package / "build-info.json"
        built = json.loads(build_info.read_text(encoding="utf-8")) if build_info.exists() else {}
        if revision is None:
            revision = built.get("source_revision")
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "implementation": platform.python_implementation(),
            "cpu_count": os.cpu_count(),
            "device": "cpu",
            "packages": packages,
            "source_revision": revision,
            "source_sha256": digest.hexdigest(),
            "build": built,
            "random_seed": request.get("split", {}).get("random_state"),
        }


def _portable_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _portable_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_json(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value
