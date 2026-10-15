import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

def test_get_models_list():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "default_model" in data
    assert len(data["models"]) >= 4
    
    # Check Nemotron 3 is present and recommended
    nemotron = next((m for m in data["models"] if m["id"] == "nvidia/nemotron-3-super-120b-a12b"), None)
    assert nemotron is not None
    assert nemotron["provider"] == "nvidia"
    assert nemotron["recommended"] is True

def test_mock_model_execution():
    response = client.post(
        "/api/run",
        json={
            "task": "Test mock pipeline execution",
            "model": "mock-model",
            "async_run": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "final_answer" in data
    assert "run_id" in data
