"""
Comprehensive tests for the execution event system.

Tests cover:
- Event model
- Per-run event bus
- Event bus registry
- Event emission from pipeline and orchestrator
- SSE endpoint
- Late subscriber buffering
- Concurrent task events
- Run isolation
- Subscriber disconnect resilience
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import Settings, settings
from app.events import (
    EventBusRegistry,
    ExecutionEvent,
    ExecutionEventType,
    RunEventBus,
    event_bus_registry,
)
from app.llm.provider import MockLLMProvider
from app.main import app
from app.models.schemas import PlannerOutput
from app.models.task_graph import TaskGraph, TaskNode, TaskStatus, TaskType
from app.orchestration.agents import (
    CapabilityRegistry,
    TaskExecutionAgent,
    TaskExecutionContext,
    TaskExecutionResult,
)
from app.orchestration.orchestrator import GraphExecutionStatus, Orchestrator
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SucceedingAgent(TaskExecutionAgent):
    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
        return TaskExecutionResult(
            task_id=context.task.id, success=True, output="ok"
        )


class FailingAgent(TaskExecutionAgent):
    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
        raise RuntimeError("deliberate failure")


def _make_settings(**overrides) -> Settings:
    defaults = {
        "llm_provider": "mock",
        "llm_api_key": "",
        "llm_model": "mock",
        "evaluator_pass_threshold": 80,
        "cors_origins": "http://localhost:8082",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# 1. Event model basics
# ---------------------------------------------------------------------------


class TestEventModel:
    def test_event_has_required_fields(self):
        event = ExecutionEvent(
            run_id="r1",
            event_type=ExecutionEventType.RUN_STARTED,
            message="hello",
        )
        assert event.run_id == "r1"
        assert event.event_type == ExecutionEventType.RUN_STARTED
        assert event.event_id  # auto-generated
        assert event.timestamp  # auto-generated

    def test_to_sse_format(self):
        event = ExecutionEvent(
            run_id="r1",
            event_type=ExecutionEventType.TASK_STARTED,
            task_id="T1",
        )
        sse = event.to_sse()
        assert sse.startswith("event: TASK_STARTED\n")
        assert "data: " in sse
        assert sse.endswith("\n\n")


# ---------------------------------------------------------------------------
# 2. RunEventBus
# ---------------------------------------------------------------------------


class TestRunEventBus:
    @pytest.mark.asyncio
    async def test_emit_and_subscribe(self):
        bus = RunEventBus("r1")
        queue = await bus.subscribe()
        event = ExecutionEvent(
            run_id="r1", event_type=ExecutionEventType.RUN_STARTED
        )
        await bus.emit(event)
        received = queue.get_nowait()
        assert received.event_type == ExecutionEventType.RUN_STARTED

    @pytest.mark.asyncio
    async def test_finish_sends_sentinel(self):
        bus = RunEventBus("r1")
        queue = await bus.subscribe()
        await bus.finish()
        sentinel = queue.get_nowait()
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_late_subscriber_gets_history(self):
        bus = RunEventBus("r1")
        e1 = ExecutionEvent(
            run_id="r1", event_type=ExecutionEventType.RUN_STARTED
        )
        e2 = ExecutionEvent(
            run_id="r1", event_type=ExecutionEventType.PLANNING_STARTED
        )
        await bus.emit(e1)
        await bus.emit(e2)
        # Subscribe after events already emitted
        queue = await bus.subscribe()
        assert queue.get_nowait().event_type == ExecutionEventType.RUN_STARTED
        assert queue.get_nowait().event_type == ExecutionEventType.PLANNING_STARTED

    @pytest.mark.asyncio
    async def test_late_subscriber_after_finish(self):
        bus = RunEventBus("r1")
        await bus.emit(
            ExecutionEvent(run_id="r1", event_type=ExecutionEventType.RUN_STARTED)
        )
        await bus.finish()
        queue = await bus.subscribe()
        assert queue.get_nowait().event_type == ExecutionEventType.RUN_STARTED
        assert queue.get_nowait() is None  # sentinel

    @pytest.mark.asyncio
    async def test_history_property(self):
        bus = RunEventBus("r1")
        await bus.emit(
            ExecutionEvent(run_id="r1", event_type=ExecutionEventType.RUN_STARTED)
        )
        assert len(bus.history) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = RunEventBus("r1")
        queue = await bus.subscribe()
        await bus.unsubscribe(queue)
        # Double unsubscribe should be idempotent
        await bus.unsubscribe(queue)


# ---------------------------------------------------------------------------
# 3. EventBusRegistry
# ---------------------------------------------------------------------------


class TestEventBusRegistry:
    @pytest.mark.asyncio
    async def test_create_and_get(self):
        registry = EventBusRegistry()
        bus = await registry.create("run-1")
        assert bus is not None
        assert await registry.get("run-1") is bus

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        registry = EventBusRegistry()
        assert await registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_eviction(self):
        registry = EventBusRegistry(max_retained=2)
        b1 = await registry.create("r1")
        b2 = await registry.create("r2")
        await b1.finish()
        await b2.finish()
        await registry.create("r3")
        # Oldest finished bus should be evicted
        assert await registry.get("r3") is not None


# ---------------------------------------------------------------------------
# 4-6. Orchestrator events
# ---------------------------------------------------------------------------


class TestOrchestratorEvents:
    @pytest.mark.asyncio
    async def test_task_started_completed_events(self):
        bus = RunEventBus("r1")
        registry = CapabilityRegistry()
        registry.register("research", SucceedingAgent())
        graph = TaskGraph(
            tasks=[TaskNode(id="T1", description="test", type=TaskType.RESEARCH)]
        )
        orchestrator = Orchestrator(
            registry=registry, max_concurrency=2, event_bus=bus
        )
        result = await orchestrator.execute(graph)
        assert result.status == GraphExecutionStatus.COMPLETED

        types = [e.event_type for e in bus.history]
        assert ExecutionEventType.TASK_READY in types
        assert ExecutionEventType.TASK_STARTED in types
        assert ExecutionEventType.TASK_COMPLETED in types

    @pytest.mark.asyncio
    async def test_task_failed_events(self):
        bus = RunEventBus("r1")
        registry = CapabilityRegistry()
        registry.register("research", FailingAgent())
        graph = TaskGraph(
            tasks=[TaskNode(id="T1", description="test", type=TaskType.RESEARCH)]
        )
        orchestrator = Orchestrator(
            registry=registry, max_concurrency=2, event_bus=bus
        )
        result = await orchestrator.execute(graph)
        assert result.status == GraphExecutionStatus.FAILED

        types = [e.event_type for e in bus.history]
        assert ExecutionEventType.TASK_FAILED in types

    @pytest.mark.asyncio
    async def test_task_blocked_events(self):
        bus = RunEventBus("r1")
        registry = CapabilityRegistry()
        registry.register("research", FailingAgent())
        registry.register("analysis", SucceedingAgent())
        graph = TaskGraph(
            tasks=[
                TaskNode(id="T1", description="research", type=TaskType.RESEARCH),
                TaskNode(
                    id="T2",
                    description="analysis",
                    type=TaskType.ANALYSIS,
                    dependencies=["T1"],
                ),
            ]
        )
        orchestrator = Orchestrator(
            registry=registry, max_concurrency=2, event_bus=bus
        )
        result = await orchestrator.execute(graph)

        types = [e.event_type for e in bus.history]
        assert ExecutionEventType.TASK_FAILED in types
        assert ExecutionEventType.TASK_BLOCKED in types

    @pytest.mark.asyncio
    async def test_concurrent_start_events(self):
        """Independent tasks should produce overlapping TASK_STARTED events."""
        bus = RunEventBus("r1")
        registry = CapabilityRegistry()
        registry.register("research", SucceedingAgent())
        graph = TaskGraph(
            tasks=[
                TaskNode(id="T1", description="r1", type=TaskType.RESEARCH),
                TaskNode(id="T2", description="r2", type=TaskType.RESEARCH),
                TaskNode(id="T3", description="r3", type=TaskType.RESEARCH),
            ]
        )
        orchestrator = Orchestrator(
            registry=registry, max_concurrency=4, event_bus=bus
        )
        result = await orchestrator.execute(graph)
        assert result.status == GraphExecutionStatus.COMPLETED

        started_ids = [
            e.task_id
            for e in bus.history
            if e.event_type == ExecutionEventType.TASK_STARTED
        ]
        assert set(started_ids) == {"T1", "T2", "T3"}

    @pytest.mark.asyncio
    async def test_event_run_id_correct(self):
        bus = RunEventBus("run-42")
        registry = CapabilityRegistry()
        registry.register("research", SucceedingAgent())
        graph = TaskGraph(
            tasks=[TaskNode(id="T1", description="test", type=TaskType.RESEARCH)]
        )
        orchestrator = Orchestrator(
            registry=registry, max_concurrency=2, event_bus=bus
        )
        await orchestrator.execute(graph)
        for event in bus.history:
            assert event.run_id == "run-42"

    @pytest.mark.asyncio
    async def test_no_rag_events_without_retrieval(self):
        """RAG events should only appear when retrieval actually occurs."""
        bus = RunEventBus("r1")
        registry = CapabilityRegistry()
        registry.register("research", SucceedingAgent())
        graph = TaskGraph(
            tasks=[TaskNode(id="T1", description="test", type=TaskType.RESEARCH)]
        )
        orchestrator = Orchestrator(
            registry=registry, max_concurrency=2, event_bus=bus
        )
        await orchestrator.execute(graph)
        types = [e.event_type for e in bus.history]
        assert ExecutionEventType.RAG_STARTED not in types
        assert ExecutionEventType.RAG_COMPLETED not in types

    @pytest.mark.asyncio
    async def test_no_events_without_bus(self):
        """Orchestrator without event_bus should work silently."""
        registry = CapabilityRegistry()
        registry.register("research", SucceedingAgent())
        graph = TaskGraph(
            tasks=[TaskNode(id="T1", description="test", type=TaskType.RESEARCH)]
        )
        orchestrator = Orchestrator(registry=registry, max_concurrency=2)
        result = await orchestrator.execute(graph)
        assert result.status == GraphExecutionStatus.COMPLETED


# ---------------------------------------------------------------------------
# 7-11. Pipeline-level events
# ---------------------------------------------------------------------------


class TestPipelineEvents:
    @pytest.mark.asyncio
    async def test_run_started_emitted(self):
        bus = RunEventBus("pending")
        settings_obj = _make_settings()
        provider = MockLLMProvider()
        pipeline = OrchestratedPipeline(
            llm_provider=provider, settings=settings_obj, event_bus=bus
        )
        await pipeline.execute("Compare X and Y for a student.")
        types = [e.event_type for e in bus.history]
        assert types[0] == ExecutionEventType.RUN_STARTED

    @pytest.mark.asyncio
    async def test_planning_events_emitted(self):
        bus = RunEventBus("pending")
        settings_obj = _make_settings()
        provider = MockLLMProvider()
        pipeline = OrchestratedPipeline(
            llm_provider=provider, settings=settings_obj, event_bus=bus
        )
        await pipeline.execute("Compare X and Y for a student.")
        types = [e.event_type for e in bus.history]
        assert ExecutionEventType.PLANNING_STARTED in types
        assert ExecutionEventType.PLANNING_COMPLETED in types

    @pytest.mark.asyncio
    async def test_evaluation_events_emitted(self):
        bus = RunEventBus("pending")
        settings_obj = _make_settings()
        provider = MockLLMProvider()
        pipeline = OrchestratedPipeline(
            llm_provider=provider, settings=settings_obj, event_bus=bus
        )
        await pipeline.execute("Compare X and Y for a student.")
        types = [e.event_type for e in bus.history]
        assert ExecutionEventType.EVALUATION_STARTED in types
        assert ExecutionEventType.EVALUATION_COMPLETED in types

    @pytest.mark.asyncio
    async def test_synthesis_events_emitted(self):
        bus = RunEventBus("pending")
        settings_obj = _make_settings()
        provider = MockLLMProvider()
        pipeline = OrchestratedPipeline(
            llm_provider=provider, settings=settings_obj, event_bus=bus
        )
        await pipeline.execute("Compare X and Y for a student.")
        types = [e.event_type for e in bus.history]
        assert ExecutionEventType.SYNTHESIS_STARTED in types
        assert ExecutionEventType.SYNTHESIS_COMPLETED in types

    @pytest.mark.asyncio
    async def test_run_completed_emitted(self):
        bus = RunEventBus("pending")
        settings_obj = _make_settings()
        provider = MockLLMProvider()
        pipeline = OrchestratedPipeline(
            llm_provider=provider, settings=settings_obj, event_bus=bus
        )
        await pipeline.execute("Compare X and Y for a student.")
        types = [e.event_type for e in bus.history]
        assert types[-1] == ExecutionEventType.RUN_COMPLETED
        assert bus.finished

    @pytest.mark.asyncio
    async def test_bus_finished_after_run(self):
        bus = RunEventBus("pending")
        settings_obj = _make_settings()
        provider = MockLLMProvider()
        pipeline = OrchestratedPipeline(
            llm_provider=provider, settings=settings_obj, event_bus=bus
        )
        await pipeline.execute("Compare X and Y for a student.")
        assert bus.finished is True

    @pytest.mark.asyncio
    async def test_pipeline_without_bus(self):
        """Pipeline without event_bus should work exactly as before."""
        settings_obj = _make_settings()
        provider = MockLLMProvider()
        pipeline = OrchestratedPipeline(
            llm_provider=provider, settings=settings_obj
        )
        result = await pipeline.execute("Compare X and Y for a student.")
        assert result.run_id
        assert result.final_answer


# ---------------------------------------------------------------------------
# 12-13. Run isolation
# ---------------------------------------------------------------------------


class TestRunIsolation:
    @pytest.mark.asyncio
    async def test_events_do_not_mix(self):
        bus1 = RunEventBus("run-A")
        bus2 = RunEventBus("run-B")
        await bus1.emit(
            ExecutionEvent(run_id="run-A", event_type=ExecutionEventType.RUN_STARTED)
        )
        await bus2.emit(
            ExecutionEvent(run_id="run-B", event_type=ExecutionEventType.RUN_STARTED)
        )
        assert len(bus1.history) == 1
        assert bus1.history[0].run_id == "run-A"
        assert len(bus2.history) == 1
        assert bus2.history[0].run_id == "run-B"


# ---------------------------------------------------------------------------
# 14-16. SSE endpoint
# ---------------------------------------------------------------------------


class TestSSEEndpoint:
    @pytest.mark.asyncio
    async def test_sse_subscriber_receives_events(self):
        """Create a bus, emit events, then verify SSE endpoint streams them."""
        bus = await event_bus_registry.create("test-sse-run")
        await bus.emit(
            ExecutionEvent(
                run_id="test-sse-run",
                event_type=ExecutionEventType.RUN_STARTED,
                message="hello",
            )
        )
        await bus.emit(
            ExecutionEvent(
                run_id="test-sse-run",
                event_type=ExecutionEventType.RUN_COMPLETED,
                message="done",
            )
        )
        await bus.finish()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/runs/test-sse-run/events")
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            body = response.text
            assert "RUN_STARTED" in body
            assert "RUN_COMPLETED" in body

    @pytest.mark.asyncio
    async def test_sse_creates_bus_for_early_subscription(self):
        bus = await event_bus_registry.get_or_create("early-run-id")
        await bus.finish()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Client connects to SSE before or as run starts
            response = await client.get("/api/runs/early-run-id/events")
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_post_run_accepts_client_run_id(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "mock")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            custom_id = "custom-client-run-123"
            response = await client.post(
                "/api/run",
                json={"task": "Explain multi-agent consensus", "run_id": custom_id},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] == custom_id

    @pytest.mark.asyncio
    async def test_late_subscriber_receives_buffer(self):
        bus = await event_bus_registry.create("test-late-sub")
        await bus.emit(
            ExecutionEvent(
                run_id="test-late-sub",
                event_type=ExecutionEventType.RUN_STARTED,
            )
        )
        await bus.emit(
            ExecutionEvent(
                run_id="test-late-sub",
                event_type=ExecutionEventType.PLANNING_STARTED,
            )
        )
        await bus.emit(
            ExecutionEvent(
                run_id="test-late-sub",
                event_type=ExecutionEventType.RUN_COMPLETED,
            )
        )
        await bus.finish()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/runs/test-late-sub/events")
            body = response.text
            # All three events should be in the buffer
            assert "RUN_STARTED" in body
            assert "PLANNING_STARTED" in body
            assert "RUN_COMPLETED" in body

    @pytest.mark.asyncio
    async def test_disconnect_does_not_stop_pipeline(self):
        """Verify that if a subscriber disconnects, the bus keeps working."""
        bus = RunEventBus("r-disconnect")
        queue = await bus.subscribe()
        # Subscriber disconnects immediately
        await bus.unsubscribe(queue)
        # Pipeline continues emitting
        await bus.emit(
            ExecutionEvent(
                run_id="r-disconnect",
                event_type=ExecutionEventType.RUN_COMPLETED,
            )
        )
        await bus.finish()
        # History still recorded
        assert len(bus.history) == 1
        assert bus.finished


# ---------------------------------------------------------------------------
# 17. Existing tests still pass (sanity check via API)
# ---------------------------------------------------------------------------


class TestExistingContractPreserved:
    @pytest.mark.asyncio
    async def test_post_run_still_returns_run_response(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "mock")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/run",
                json={"task": "Compare X and Y for a student."},
                timeout=30.0,
            )
            assert response.status_code == 200
            data = response.json()
            assert "run_id" in data
            assert "final_answer" in data
            assert "execution_trace" in data

    @pytest.mark.asyncio
    async def test_get_status_unchanged(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/status")
            assert response.status_code == 200
            data = response.json()
            assert "configured" in data
            assert "provider" in data
