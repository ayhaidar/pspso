import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from pspso.dashboard.app import create_app


SCENARIO_DIR = Path("examples/scenarios")


def test_all_scenarios_validate():
    client = TestClient(create_app())
    for path in SCENARIO_DIR.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        response = client.post("/api/runs/validate", json=payload)
        assert response.status_code == 200, path.name
        body = response.json()
        assert body["valid"] is True, (path.name, body["errors"])


def test_lightweight_sklearn_scenario_runs_to_completion():
    client = TestClient(create_app())
    payload = json.loads(
        (SCENARIO_DIR / "breast_cancer_svm_random.json").read_text(encoding="utf-8")
    )
    payload["runtime"]["max_trials"] = 1

    created = client.post("/api/runs", json=payload)

    assert created.status_code == 200
    run_id = created.json()["run_id"]
    snapshot = {"status": "queued"}
    for _ in range(50):
        snapshot = client.get(f"/api/runs/{run_id}").json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert snapshot["status"] == "completed"
