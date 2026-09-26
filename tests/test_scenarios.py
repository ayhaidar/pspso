import json
import time
from pathlib import Path

SCENARIO_DIR = Path("examples/scenarios")


def test_all_scenarios_validate(tmp_path, client_factory):
    client = client_factory(tmp_path / "tracking.sqlite3")
    for path in SCENARIO_DIR.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        response = client.post("/api/v1/runs/validate", json=payload)
        assert response.status_code == 200, path.name
        body = response.json()
        assert body["valid"] is True, (path.name, body["errors"])


def test_lightweight_sklearn_scenario_runs_to_completion(tmp_path, client_factory):
    client = client_factory(tmp_path / "tracking.sqlite3")
    payload = json.loads(
        (SCENARIO_DIR / "breast_cancer_svm_random.json").read_text(encoding="utf-8")
    )
    payload["runtime"]["max_trials"] = 1

    created = client.post("/api/v1/runs", json=payload)

    assert created.status_code == 200
    run_id = created.json()["run_id"]
    snapshot = {"status": "queued"}
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/v1/runs/{run_id}").json()
        if snapshot["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            break
        time.sleep(0.1)
    assert snapshot["status"] == "completed"
