"""
Phase 9 Tests: Reliability (Timeouts, Cancellation, Concurrency Throttling)
and Persistence (InMemoryRunRepository, PostgresRunRepository).
"""

import asyncio
import os
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.models.schemas import RunResponse
from app.persistence.run_repository import (
    InMemoryRunRepository,
    PostgresRunRepository,
    get_run_repository,
    set_run_repository,
)
from app.telemetry.metrics import CallMetric, RunMetrics


# ==============================================================================
# 1. Repository Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_in_memory_run_repository_lifecycle():
    """Verify save_run, get_run, list_runs, save_metrics, get_metrics in memory."""
    repo = InMemoryRunRepository(max_runs=5)

    from app.models.schemas import EvaluationResult

    # Mock RunResponse
    run = RunResponse(
        run_id="run-repo-test-1",
        user_task="Test task for repository persistence",
        execution_strategy="DIRECT",
        final_answer="Direct answer content.",
        subtasks=[],
        research_results=[],
        analysis="Quick analysis",
        evaluation=EvaluationResult(
            status="PASS",
            feedback="Constructive analysis and well grounded.",
            score=90,
            groundedness_score=23,
            coverage_score=22,
            distinction_score=23,
            honesty_score=22,
        ),
        execution_trace=[],
        created_at="2026-10-10T14:00:00Z",
    )

    await repo.save_run(run)
    retrieved = await repo.get_run("run-repo-test-1")
    assert retrieved is not None
    assert retrieved.run_id == "run-repo-test-1"
    assert retrieved.execution_strategy == "DIRECT"

    # List runs
    all_runs = await repo.list_runs()
    assert len(all_runs) == 1
    assert all_runs[0].run_id == "run-repo-test-1"

    # Metrics
    metrics = RunMetrics(
        run_id="run-repo-test-1",
        strategy="DIRECT",
        total_tokens=150,
        total_llm_calls=1,
        wall_time_ms=120.0,
    )
    await repo.save_metrics(metrics)
    retrieved_metrics = await repo.get_metrics("run-repo-test-1")
    assert retrieved_metrics is not None
    assert retrieved_metrics.total_tokens == 150
    assert retrieved_metrics.total_llm_calls == 1

    # Delete
    deleted = await repo.delete_run("run-repo-test-1")
    assert deleted is True
    assert await repo.get_run("run-repo-test-1") is None
    assert await repo.get_metrics("run-repo-test-1") is None


@pytest.mark.asyncio
async def test_postgres_run_repository_graceful_connect_failure():
    """Verify PostgresRunRepository cleanly errors if connection URL is invalid."""
    repo = PostgresRunRepository(database_url="postgresql://invalid_user:invalid_pwd@localhost:5433/nonexistent_db")
    with pytest.raises(Exception):
        await repo.initialize_schema()


# ==============================================================================
# 2. Concurrency Throttling Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_concurrency_limit_enforcement(monkeypatch):
    """When ACTIVE_RUN_TASKS reaches max_concurrent_runs, subsequent calls receive HTTP 429."""
    from app.api.routes import ACTIVE_RUN_TASKS

    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "max_concurrent_runs", 2)

    # Create dummy tasks to simulate at-capacity server
    async def dummy_worker():
        await asyncio.sleep(10)

    t1 = asyncio.create_task(dummy_worker())
    t2 = asyncio.create_task(dummy_worker())
    ACTIVE_RUN_TASKS["dummy-1"] = t1
    ACTIVE_RUN_TASKS["dummy-2"] = t2

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post("/api/run", json={"task": "Over-capacity task"})
            assert res.status_code == 429
            assert "maximum concurrent runs" in res.json()["detail"].lower()
    finally:
        t1.cancel()
        t2.cancel()
        ACTIVE_RUN_TASKS.pop("dummy-1", None)
        ACTIVE_RUN_TASKS.pop("dummy-2", None)


# ==============================================================================
# 3. Execution Timeout Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_execution_timeout_enforcement(monkeypatch):
    """When pipeline execution exceeds run_timeout_seconds, returns HTTP 504."""
    from app.pipeline.orchestrated_pipeline import OrchestratedPipeline

    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "run_timeout_seconds", 0.05)

    # Mock pipeline execute to sleep longer than timeout
    async def mock_slow_execute(*args, **kwargs):
        await asyncio.sleep(0.5)

    monkeypatch.setattr(OrchestratedPipeline, "execute", mock_slow_execute)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/run", json={"task": "Slow task that times out"})
        assert res.status_code == 504
        assert "timed out" in res.json()["detail"]["message"].lower()


# ==============================================================================
# 4. Cancellation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_async_run_cancellation(monkeypatch):
    """Starting an async run and calling POST /api/runs/{id}/cancel aborts execution."""
    from app.pipeline.orchestrated_pipeline import OrchestratedPipeline

    monkeypatch.setattr(settings, "llm_provider", "mock")

    async def mock_long_pipeline(*args, **kwargs):
        await asyncio.sleep(5.0)

    monkeypatch.setattr(OrchestratedPipeline, "execute", mock_long_pipeline)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Start async run
        start_res = await client.post(
            "/api/run",
            json={"task": "Task to be cancelled", "async_run": True},
        )
        assert start_res.status_code == 200
        run_id = start_res.json()["run_id"]

        # Small pause to ensure worker is running
        await asyncio.sleep(0.05)

        # Cancel run
        cancel_res = await client.post(f"/api/runs/{run_id}/cancel")
        assert cancel_res.status_code == 200
        assert cancel_res.json()["status"] == "cancelled"

        # Check status endpoint
        status_res = await client.get(f"/api/runs/{run_id}")
        assert status_res.status_code == 200
        assert status_res.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_nonexistent_run_returns_404():
    """Cancelling a non-existent or completed run returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/runs/non-existent-run-id-9999/cancel")
        assert res.status_code == 404
