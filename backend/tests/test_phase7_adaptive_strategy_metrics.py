"""
Phase 7 Tests: Adaptive Strategy Selector, Resource Budgets, and Telemetry Instrumentation.

Verifies:
1. Strategy Selector Classifier:
   - Evaluates task length/structure, domains, document attachments, and cross-document reasoning.
   - Selects DIRECT, SINGLE_AGENT, SINGLE_AGENT_RAG, or MULTI_AGENT.
   - Emits STRATEGY_DECIDED and STRATEGY_SELECTED events with rationale and metadata.
2. Resource Budgets:
   - Allocates strategy-specific resource budgets (max LLM calls, max tokens, max wall time).
   - Strictly enforces limits and raises ResourceBudgetExceededError when breached.
   - Emits RESOURCE_BUDGET_EXCEEDED event on exhaustion.
3. Telemetry Instrumentation:
   - Instruments every LLM call: tokens_in, tokens_out, latency_ms, estimated_cost, model, run_id, stage.
   - Instruments embedding calls: tokens, latency_ms, cost.
   - Persists per-run metrics and attaches them to RunResponse.
4. Metrics API Endpoint:
   - Exposes GET /api/runs/{id}/metrics returning full run telemetry.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.events import ExecutionEvent, ExecutionEventType, RunEventBus
from app.llm.base import LLMProvider
from app.llm.provider import MockLLMProvider
from app.main import app
from app.models.schemas import DocumentPayload, PlannerOutput, RunResponse
from app.models.task_graph import TaskNode, TaskType
from app.orchestration.resource_budget import (
    BudgetTracker,
    ResourceBudget,
    ResourceBudgetExceededError,
    get_budget_for_strategy,
)
from app.orchestration.strategy import ExecutionStrategy, StrategyDecision, StrategySelector
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline
from app.telemetry.instrumented_provider import InstrumentedLLMProvider, estimate_tokens
from app.telemetry.metrics import (
    CallMetric,
    MetricsCollector,
    compute_estimated_cost,
    global_metrics_collector,
)


# ==============================================================================
# 1. Strategy Selector Classifier Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_classifier_direct_for_conversational_greetings():
    """Rule & heuristic fast-path classifies simple pleasantries as DIRECT."""
    selector = StrategySelector()
    decision = await selector.select_strategy("Hello there, how are you?")
    assert decision.strategy == ExecutionStrategy.DIRECT
    assert decision.requires_rag is False
    assert decision.cross_document_reasoning is False
    assert len(decision.domains) == 0


@pytest.mark.asyncio
async def test_classifier_single_agent_rag_for_single_document_query():
    """Attached document with focused single-topic query selects SINGLE_AGENT_RAG."""
    selector = StrategySelector()
    query = "According to the uploaded document, what is the maximum wing span allowed?"
    decision = await selector.select_strategy(query, has_documents=True)
    assert decision.strategy == ExecutionStrategy.SINGLE_AGENT_RAG
    assert decision.requires_rag is True
    assert decision.cross_document_reasoning is False


@pytest.mark.asyncio
async def test_classifier_multi_agent_for_multiple_domains():
    """Query touching 2+ distinct domains (aerodynamics, suspension, cost) selects MULTI_AGENT."""
    selector = StrategySelector()
    query = "Analyze the trade-offs between aerodynamics, suspension stiffness, and financial cost."
    decision = await selector.select_strategy(query, has_documents=False)
    assert decision.strategy == ExecutionStrategy.MULTI_AGENT
    assert len(decision.domains) >= 2
    assert any("aero" in d for d in decision.domains)
    assert any("suspension" in d for d in decision.domains) or any("cost" in d for d in decision.domains)


@pytest.mark.asyncio
async def test_classifier_multi_agent_for_cross_document_reasoning():
    """Query requiring cross-referencing multiple regulations/sections selects MULTI_AGENT."""
    selector = StrategySelector()
    query = "Cross-reference the technical regulations and compare documents across articles."
    decision = await selector.select_strategy(query, has_documents=True)
    assert decision.strategy == ExecutionStrategy.MULTI_AGENT
    assert decision.cross_document_reasoning is True
    assert decision.requires_rag is True


@pytest.mark.asyncio
async def test_pipeline_emits_strategy_decided_and_selected_events():
    """Pipeline logs decision and rationale as typed events with budget metadata."""
    llm = MockLLMProvider()
    settings = Settings(llm_provider="mock")
    bus = RunEventBus(run_id="run-strat-events")
    queue = await bus.subscribe()
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings, event_bus=bus, run_id="run-strat-events")

    response = await pipeline.execute("Hello assistant!")

    events: List[ExecutionEvent] = []
    while not queue.empty():
        evt = queue.get_nowait()
        if evt:
            events.append(evt)

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.STRATEGY_DECIDED in event_types
    assert ExecutionEventType.STRATEGY_SELECTED in event_types

    decided_evt = next(e for e in events if e.event_type == ExecutionEventType.STRATEGY_DECIDED)
    assert decided_evt.metadata["strategy"] == "DIRECT"
    assert "budget" in decided_evt.metadata
    assert decided_evt.metadata["budget"]["max_llm_calls"] >= 1


# ==============================================================================
# 2. Resource Budgets and Enforcement Tests
# ==============================================================================

def test_resource_budget_defaults_and_settings_overrides():
    """Verify strategy budgets match expected thresholds and respect settings."""
    settings = Settings(
        budget_direct_max_llm_calls=2,
        budget_direct_max_tokens=4000,
        budget_direct_max_wall_time_seconds=20.0,
    )
    direct_budget = get_budget_for_strategy(ExecutionStrategy.DIRECT, settings)
    assert direct_budget.max_llm_calls == 2
    assert direct_budget.max_tokens == 4000
    assert direct_budget.max_wall_time_seconds == 20.0

    multi_budget = get_budget_for_strategy(ExecutionStrategy.MULTI_AGENT, settings)
    assert multi_budget.max_llm_calls == 20
    assert multi_budget.max_tokens == 80000


def test_budget_tracker_enforces_max_llm_calls():
    """BudgetTracker raises ResourceBudgetExceededError when max LLM calls exceeded."""
    budget = ResourceBudget(
        strategy=ExecutionStrategy.DIRECT,
        max_llm_calls=1,
        max_tokens=5000,
        max_wall_time_seconds=10.0,
    )
    tracker = BudgetTracker(budget)
    tracker.record_llm_call(tokens_in=10, tokens_out=10)
    assert tracker.llm_calls == 1

    with pytest.raises(ResourceBudgetExceededError) as exc_info:
        tracker.record_llm_call(tokens_in=10, tokens_out=10)
    assert exc_info.value.limit_type == "llm_calls"
    assert "LLM calls (2) exceeded maximum allowance (1)" in str(exc_info.value)


def test_budget_tracker_enforces_max_tokens():
    """BudgetTracker raises ResourceBudgetExceededError when max tokens exceeded."""
    budget = ResourceBudget(
        strategy=ExecutionStrategy.SINGLE_AGENT,
        max_llm_calls=5,
        max_tokens=100,
        max_wall_time_seconds=10.0,
    )
    tracker = BudgetTracker(budget)
    with pytest.raises(ResourceBudgetExceededError) as exc_info:
        tracker.record_llm_call(tokens_in=60, tokens_out=50)
    assert exc_info.value.limit_type == "tokens"
    assert "Total tokens (110) exceeded maximum allowance (100)" in str(exc_info.value)


def test_budget_tracker_enforces_max_wall_time():
    """BudgetTracker raises ResourceBudgetExceededError when execution duration exceeds limit."""
    budget = ResourceBudget(
        strategy=ExecutionStrategy.DIRECT,
        max_llm_calls=5,
        max_tokens=5000,
        max_wall_time_seconds=0.01,
    )
    tracker = BudgetTracker(budget)
    time.sleep(0.02)
    with pytest.raises(ResourceBudgetExceededError) as exc_info:
        tracker.check_limits()
    assert exc_info.value.limit_type == "wall_time"


@pytest.mark.asyncio
async def test_pipeline_catches_budget_breach_and_emits_event():
    """Pipeline catches ResourceBudgetExceededError, emits RESOURCE_BUDGET_EXCEEDED, and fails cleanly."""
    settings = Settings(
        llm_provider="mock",
        budget_direct_max_tokens=10,  # Extremely low limit to trigger breach immediately
    )
    llm = MockLLMProvider()
    bus = RunEventBus(run_id="run-budget-breach")
    queue = await bus.subscribe()
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings, event_bus=bus, run_id="run-budget-breach")

    with pytest.raises(ResourceBudgetExceededError):
        await pipeline.execute("Hello there!")

    events: List[ExecutionEvent] = []
    while not queue.empty():
        evt = queue.get_nowait()
        if evt:
            events.append(evt)

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.RESOURCE_BUDGET_EXCEEDED in event_types
    assert ExecutionEventType.RUN_FAILED in event_types


# ==============================================================================
# 3. Telemetry Instrumentation & Cost Tracking Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_instrumented_provider_records_tokens_latency_and_cost():
    """InstrumentedLLMProvider records stage-attributed CallMetric with valid cost."""
    collector = MetricsCollector()
    base_llm = MockLLMProvider()
    instrumented = InstrumentedLLMProvider(
        base_provider=base_llm,
        run_id="run-telemetry-1",
        metrics_collector=collector,
        model_name="gemini-3.1-flash-lite",
    )

    instrumented.set_stage("Planning")
    result = await instrumented.generate("Decompose the user task into subtasks.")
    assert len(result) > 0

    run_metrics = collector.get_metrics("run-telemetry-1")
    assert run_metrics is not None
    assert run_metrics.total_llm_calls == 1
    assert run_metrics.total_tokens_in > 0
    assert run_metrics.total_tokens_out > 0
    assert run_metrics.total_tokens == run_metrics.total_tokens_in + run_metrics.total_tokens_out
    assert run_metrics.total_latency_ms >= 0.0
    assert run_metrics.total_estimated_cost >= 0.0

    call = run_metrics.calls[0]
    assert call.stage == "Planning"
    assert call.model == "gemini-3.1-flash-lite"
    assert call.call_type == "llm"


@pytest.mark.asyncio
async def test_pipeline_execution_populates_run_response_metrics():
    """OrchestratedPipeline populates RunResponse.metrics across entire lifecycle."""
    llm = MockLLMProvider()
    settings = Settings(llm_provider="mock")
    pipeline = OrchestratedPipeline(llm_provider=llm, settings=settings)

    response = await pipeline.execute("Compare battery chemistry and aerodynamic cooling drag.")
    assert response.metrics is not None
    assert response.metrics["strategy"] == "MULTI_AGENT"
    assert response.metrics["total_llm_calls"] >= 4  # Planner, research, analysis, eval, synth
    assert response.metrics["total_tokens"] > 0
    assert response.metrics["wall_time_ms"] > 0.0
    assert len(response.metrics["calls"]) >= 4

    stages = [c["stage"] for c in response.metrics["calls"]]
    assert any("StrategySelection" in s for s in stages)
    assert any("Planning" in s for s in stages)
    assert any("Analysis" in s for s in stages)
    assert any("Synthesis" in s for s in stages)


# ==============================================================================
# 4. Metrics API Endpoint Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_get_run_metrics_api_endpoint():
    """GET /api/runs/{run_id}/metrics returns aggregated metrics and call records."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Run pipeline via API
        run_res = await client.post(
            "/api/run",
            json={"task": "Hello assistant, give me a quick welcome!"},
        )
        assert run_res.status_code == 200
        run_data = run_res.json()
        run_id = run_data["run_id"]

        # Query metrics endpoint
        metrics_res = await client.get(f"/api/runs/{run_id}/metrics")
        assert metrics_res.status_code == 200
        metrics = metrics_res.json()

        assert metrics["run_id"] == run_id
        assert metrics["strategy"] == "DIRECT"
        assert metrics["total_llm_calls"] >= 1
        assert metrics["total_tokens"] > 0
        assert metrics["wall_time_ms"] > 0
        assert len(metrics["calls"]) >= 1


@pytest.mark.asyncio
async def test_get_run_metrics_nonexistent_returns_404():
    """GET /api/runs/{unknown_id}/metrics returns 404 Not Found."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/runs/non-existent-uuid-12345/metrics")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()
