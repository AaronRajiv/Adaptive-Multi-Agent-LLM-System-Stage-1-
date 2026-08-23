"""
Tests for Error Handling and Failure Modes.
"""

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)


def test_missing_api_key_error(monkeypatch):
    # Set to Gemini without API key
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "llm_api_key", "")

    response = client.post("/api/run", json={"task": "Explain quantum computing."})
    assert response.status_code == 503
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_type"] == "ConfigurationError"
    assert "Gemini API key is not configured" in data["detail"]["message"]


def test_empty_or_too_short_task():
    response = client.post("/api/run", json={"task": "hi"})
    assert response.status_code == 422


def test_unsupported_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "unknown_provider")
    monkeypatch.setattr(settings, "llm_api_key", "some-dummy-key")

    response = client.post("/api/run", json={"task": "Valid task length."})
    assert response.status_code == 503
    data = response.json()
    assert data["detail"]["error_type"] == "ConfigurationError"
