"""Start a dashboard API run from Python."""

import json
import time
from pathlib import Path

import httpx


payload = json.loads(
    Path("examples/scenarios/breast_cancer_svm_random.json").read_text(encoding="utf-8")
)

with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0) as client:
    validation = client.post("/api/runs/validate", json=payload)
    validation.raise_for_status()
    created = client.post("/api/runs", json=payload)
    created.raise_for_status()
    run_id = created.json()["run_id"]

    while True:
        snapshot = client.get(f"/api/runs/{run_id}").json()
        if snapshot["status"] in {"completed", "failed"}:
            print(snapshot)
            break
        time.sleep(0.5)
