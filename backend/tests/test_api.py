"""
API endpoint tests using FastAPI test client.
"""

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Stage 1" in data["stage"]


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "provider" in data
    assert "model" in data
    assert "evaluator_threshold" in data


def test_run_endpoint_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")

    payload = {"task": "Compare solar vs wind power feasibility."}
    response = client.post("/api/run", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "run_id" in data
    assert len(data["subtasks"]) > 0
    assert len(data["research_results"]) > 0
    assert "analysis" in data
    assert "evaluation" in data
    assert "final_answer" in data
    assert len(data["execution_trace"]) == 5


def test_recent_runs_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    # Make a run
    client.post("/api/run", json={"task": "Sample observability test."})

    response = client.get("/api/runs")
    assert response.status_code == 200
    runs = response.json()
    assert isinstance(runs, list)
    assert len(runs) >= 1
