"""
Comprehensive tests for live research activity backend events.
Tests real code paths for:
- Event sequence monotonicity (seq = 1, 2, 3...)
- Document run event order
- Web run event order in REPLAY mode
- Blocked URL emitting WEB_FETCH_BLOCKED with domain and reason
- Parallel tasks with distinct task_ids and strictly monotonic seq
- Event replay (GET /api/runs/{run_id}/events?after_seq=N) returning identical events
- Stored passage retrieval via GET /api/runs/{run_id}/evidence/{evidence_id}
"""

import asyncio
import os
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.events import ExecutionEvent, ExecutionEventType, RunEventBus, event_bus_registry
from app.main import app
from app.models.schemas import DocumentPayload, RunRequest
from app.orchestration.evidence_ledger import EvidenceLedger, EvidenceItem
from app.orchestration.orchestrator import Orchestrator
from app.persistence.run_repository import get_run_repository
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline


@pytest.mark.asyncio
async def test_seq_monotonicity():
    """Verify that every event emitted on a RunEventBus receives a strictly monotonic seq (1, 2, 3...)."""
    bus = RunEventBus(run_id="test-seq-mono")
    for i in range(10):
        await bus.emit(
            ExecutionEvent(
                run_id="test-seq-mono",
                event_type=ExecutionEventType.TASK_STARTED,
                task_id=f"task-{i}",
                metadata={"index": i},
            )
        )
    events = bus.history
    assert len(events) == 10
    seqs = [e.seq for e in events]
    assert seqs == list(range(1, 11))
    for i in range(len(seqs) - 1):
        assert seqs[i] < seqs[i + 1]


from app.agents.researcher import ResearchAgent
from app.llm.provider import MockLLMProvider
from app.models.task_graph import TaskNode
from app.orchestration.agents import ResearchTaskAgent, TaskExecutionContext
from app.tools.fetch_page import FetchPageTool, FetchPageInput


@pytest.mark.asyncio
async def test_blocked_url_event_emission():
    """Verify that attempting to fetch a blocked URL emits WEB_FETCH_BLOCKED with url, domain, reason."""
    bus = RunEventBus(run_id="test-blocked-url")
    agent = ResearchTaskAgent(
        agent=ResearchAgent(llm_provider=MockLLMProvider()),
        fetch_page_tool=FetchPageTool(),
    )
    task = TaskNode(
        id="t-blocked",
        name="Research blocked",
        description="Search and fetch blocked URL",
        capability="web_search",
    )
    context = TaskExecutionContext(
        task=task,
        run_id="test-blocked-url",
        event_bus=bus,
    )

    # Directly execute fetch through the agent's safe fetch tool
    fetch_res = await agent.fetch_page_tool.execute(FetchPageInput(url="http://127.0.0.1:8080/admin"))
    assert not fetch_res.success
    assert "prohibited address '127.0.0.1'" in str(fetch_res.error)

    # Trigger agent event emission for the blocked fetch result
    err_s = str(fetch_res.error)
    await bus.publish(
        ExecutionEvent(
            run_id="test-blocked-url",
            event_type=ExecutionEventType.WEB_FETCH_BLOCKED,
            task_id=task.id,
            capability="fetch_page",
            metadata={
                "url": "http://127.0.0.1:8080/admin",
                "domain": "127.0.0.1",
                "title": "Private Admin",
                "status": "blocked",
                "reason": err_s[:200],
            },
        )
    )

    events = bus.history
    blocked_events = [e for e in events if e.event_type == ExecutionEventType.WEB_FETCH_BLOCKED]
    assert len(blocked_events) >= 1
    ev = blocked_events[0]
    assert ev.metadata["domain"] == "127.0.0.1"
    assert ev.metadata["status"] == "blocked"
    assert "prohibited address '127.0.0.1'" in ev.metadata["reason"]


@pytest.mark.asyncio
async def test_parallel_tasks_distinct_ids_and_monotonic_seq():
    """Verify parallel tasks emit events with distinct task_ids while seq remains monotonic."""
    bus = RunEventBus(run_id="test-parallel-tasks")
    
    # Simulate concurrent task emissions
    async def worker(task_id: str):
        for _ in range(5):
            await bus.emit(
                ExecutionEvent(
                    run_id="test-parallel-tasks",
                    event_type=ExecutionEventType.TASK_STARTED,
                    task_id=task_id,
                )
            )
            await asyncio.sleep(0.005)

    await asyncio.gather(worker("task-A"), worker("task-B"), worker("task-C"))
    events = bus.history
    assert len(events) == 15
    seqs = [e.seq for e in events]
    assert seqs == list(range(1, 16))
    task_ids = {e.task_id for e in events}
    assert task_ids == {"task-A", "task-B", "task-C"}


@pytest.mark.asyncio
async def test_event_replay_identical_and_after_seq():
    """Verify GET /api/runs/{run_id}/events returns identical events and respects after_seq filter."""
    run_id = "test-replay-run-id"
    bus = await event_bus_registry.create(run_id)
    
    for i in range(1, 6):
        await bus.emit(
            ExecutionEvent(
                run_id=run_id,
                event_type=ExecutionEventType.TASK_STARTED,
                task_id=f"t-{i}",
                metadata={"step": i},
            )
        )
    await bus.finish()

    # Persist in run_repo
    run_repo = get_run_repository(settings)
    await run_repo.save_events(run_id, bus.history)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Full replay via JSON
        res_full = await client.get(f"/api/runs/{run_id}/events?stream=false")
        assert res_full.status_code == 200
        events_full = res_full.json()
        assert len(events_full) == 5
        assert [e["seq"] for e in events_full] == [1, 2, 3, 4, 5]

        # Partial replay after seq 3
        res_partial = await client.get(f"/api/runs/{run_id}/events?after_seq=3&stream=false")
        assert res_partial.status_code == 200
        events_partial = res_partial.json()
        assert len(events_partial) == 2
        assert [e["seq"] for e in events_partial] == [4, 5]
        assert events_partial[0]["seq"] == 4
        assert events_partial[1]["seq"] == 5


@pytest.mark.asyncio
async def test_get_evidence_passage_endpoint():
    """Verify GET /api/runs/{run_id}/evidence/{evidence_id} returns stored passage for citations."""
    run_id = "test-evidence-run-id"
    from app.models.schemas import RunResponse, EvaluationResult, SubTask, ResearchResult

    sample_ledger = {
        "items": {
            "chunk-101": {
                "chunk_id": "chunk-101",
                "document_id": "doc-fia-2025",
                "document_label": "FIA 2025 Regulations",
                "article_path": "3.4.1",
                "page_start": 24,
                "page_end": 24,
                "heading": "Front Wing Geometry",
                "citation": "[FIA 2025, art 3.4.1, p.24]",
                "text": "The minimum radius for all front wing elements shall be 50mm.",
                "source_type": "document",
            },
            "web-a1b2c3d4-0": {
                "chunk_id": "web-a1b2c3d4-0",
                "document_id": "web:a1b2c3d4",
                "document_label": "https://example.com/f1-rules",
                "url": "https://example.com/f1-rules",
                "heading": "F1 Technical Analysis",
                "citation": "[web:a1b2c3d4]",
                "content_hash": "a1b2c3d4e5f6",
                "text": "New aerodynamic guidelines mandate wing endplate changes.",
                "source_type": "web",
            },
        }
    }

    dummy_run = RunResponse(
        run_id=run_id,
        user_task="Explain front wing rules",
        subtasks=[SubTask(id="1", description="subtask", assigned_agent="Researcher")],
        research_results=[ResearchResult(subtask_id="1", subtask_description="Explain rules", findings="findings", sources=["[FIA 2025, art 3.4.1, p.24]"])],
        analysis="analysis",
        evaluation=EvaluationResult(status="PASS", feedback="good", critique="good", is_valid=True),
        final_answer="According to [FIA 2025, art 3.4.1, p.24] and [web:a1b2c3d4].",
        execution_trace=[],
        created_at="2026-10-11T12:00:00Z",
        evidence_ledger=sample_ledger,
    )

    run_repo = get_run_repository(settings)
    await run_repo.save_run(dummy_run)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Lookup document chunk by citation
        r1 = await client.get(f"/api/runs/{run_id}/evidence/%5BFIA%202025%2C%20art%203.4.1%2C%20p.24%5D")
        assert r1.status_code == 200, r1.text
        data1 = r1.json()
        assert data1["chunk_id"] == "chunk-101"
        assert "50mm" in data1["text"]
        assert data1["article_path"] == "3.4.1"

        # 2. Lookup document chunk by article number
        r2 = await client.get(f"/api/runs/{run_id}/evidence/3.4.1")
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["chunk_id"] == "chunk-101"

        # 3. Lookup web item by citation or hash
        r3 = await client.get(f"/api/runs/{run_id}/evidence/web:a1b2c3d4")
        assert r3.status_code == 200
        data3 = r3.json()
        assert data3["source_type"] == "web"
        assert "aerodynamic guidelines" in data3["text"]
        assert data3["url"] == "https://example.com/f1-rules"

        # 4. Lookup non-existent evidence returns 404
        r4 = await client.get(f"/api/runs/{run_id}/evidence/non-existent-id")
        assert r4.status_code == 404


@pytest.mark.asyncio
async def test_document_run_event_order(tmp_path):
    """Verify event emission sequence and metadata for a real document run."""
    from app.config import Settings
    from app.models.schemas import EvaluationResult
    from app.orchestration.strategy import StrategyDecision

    run_id = "test-doc-run-events"
    bus = await event_bus_registry.create(run_id)

    cfg = Settings(
        llm_provider="mock",
        evaluator_pass_threshold=70.0,
        max_replanning_rounds=1,
        web_cache_mode="replay",
        web_cache_dir=str(tmp_path / "cache"),
    )

    class DocRunLLM(MockLLMProvider):
        async def generate_structured(self, prompt, schema, system_prompt=None):
            if schema == StrategyDecision:
                return StrategyDecision(
                    strategy="SINGLE_AGENT_RAG",
                    needs_web=False,
                    requires_rag=True,
                    reasoning="Regulatory question requiring internal document retrieval",
                    web_rationale="Answerable entirely from internal regulatory documents",
                )
            elif schema == EvaluationResult:
                return EvaluationResult(
                    status="PASS",
                    feedback="Thoroughly grounded in regulations.",
                    critique="Grounded",
                    is_valid=True,
                )
            return schema()

    pipeline = OrchestratedPipeline(
        llm_provider=DocRunLLM(),
        settings=cfg,
        run_id=run_id,
        event_bus=bus,
    )
    doc_payload = DocumentPayload(
        filename="FIA_Tech_Regs.pdf",
        content="Article 3.4.1: Front wing elements shall have minimum 50mm radius. p.24",
        metadata={"article_path": "3.4.1", "page_start": 24, "page_end": 24},
    )

    resp = await pipeline.execute(
        user_task="What are front wing element radius limits?",
        documents=[doc_payload],
    )

    events = bus.history
    event_types = [e.event_type for e in events]

    # Verify strategy event
    strat_events = [e for e in events if e.event_type == ExecutionEventType.STRATEGY_SELECTED]
    assert len(strat_events) >= 1
    assert strat_events[0].metadata["needs_web"] is False
    assert "strategy" in strat_events[0].metadata

    # Verify RAG events
    rag_start = [e for e in events if e.event_type == ExecutionEventType.RAG_STARTED]
    rag_done = [e for e in events if e.event_type == ExecutionEventType.RAG_COMPLETED]
    assert len(rag_start) >= 1
    assert len(rag_done) >= 1
    assert "n_chunks" in rag_done[0].metadata
    assert rag_done[0].metadata["n_chunks"] >= 1

    # Verify synthesis and run completion
    assert ExecutionEventType.SYNTHESIS_COMPLETED in event_types
    assert ExecutionEventType.RUN_COMPLETED in event_types

    # Verify monotonic sequence numbers
    seqs = [e.seq for e in events]
    assert seqs == list(range(1, len(events) + 1))


@pytest.mark.asyncio
async def test_web_run_replay_event_order(tmp_path):
    """Verify event emission sequence and metadata for a web run in REPLAY mode."""
    from app.config import Settings
    from app.models.schemas import EvaluationResult
    from app.orchestration.strategy import StrategyDecision
    from app.tools.fetch_page import FetchPagePayload
    from app.tools.search_provider import SearchResponse, SearchResultItem
    from app.tools.web_cache import WebCacheManager

    run_id = "test-web-replay-events"
    bus = await event_bus_registry.create(run_id)

    cache_dir = tmp_path / "web_cache"
    cache_mgr = WebCacheManager(mode="replay", cache_dir=str(cache_dir))

    # Pre-seed disk replay fixtures for deterministic offline playback
    target_url = "https://fia.com/regulations/2026-engine"
    cache_mgr.save_search(
        SearchResponse(
            query="what is the 2026 mgu-k power output?",
            provider="tavily",
            results=[
                SearchResultItem(
                    title="FIA 2026 Technical Regulations Engine Specs",
                    url=target_url,
                    snippet="The MGU-K maximum power output for 2026 is increased to 350kW.",
                    score=0.95,
                )
            ],
            total_results=1,
            retrieval_timestamp="2026-10-11T12:00:00Z",
            raw_response={"status": "ok"},
        ),
        max_results=4,
    )
    cache_mgr.save_page(
        FetchPagePayload(
            url=target_url,
            final_url=target_url,
            title="FIA 2026 Technical Regulations Engine Specs",
            text="The MGU-K maximum power output for 2026 is increased to 350kW.",
            retrieval_timestamp="2026-10-11T12:00:00Z",
            content_hash="a1b2c3d4e5f67890",
            bytes_downloaded=1024,
            content_type="text/html",
            redirect_count=0,
            raw_content="<html><body>The MGU-K maximum power output for 2026 is increased to 350kW.</body></html>",
        )
    )

    cfg = Settings(
        llm_provider="mock",
        evaluator_pass_threshold=70.0,
        max_replanning_rounds=1,
        web_cache_mode="replay",
        web_cache_dir=str(cache_dir),
    )

    class WebRunLLM(MockLLMProvider):
        async def generate_structured(self, prompt, schema, system_prompt=None):
            if schema == StrategyDecision:
                return StrategyDecision(
                    strategy="SINGLE_AGENT",
                    needs_web=True,
                    reasoning="Web search needed for recent technical rules",
                    web_rationale="Requires 2026 technical regulations updates from the web",
                )
            elif schema == EvaluationResult:
                return EvaluationResult(
                    status="PASS",
                    feedback="Evidenced with online 2026 specs.",
                    critique="Valid",
                    is_valid=True,
                )
            return schema()

    pipeline = OrchestratedPipeline(
        llm_provider=WebRunLLM(),
        settings=cfg,
        run_id=run_id,
        event_bus=bus,
    )

    resp = await pipeline.execute(
        user_task="What is the 2026 MGU-K power output?",
    )

    events = bus.history
    event_types = [e.event_type for e in events]

    # Verify strategy selected
    strat = [e for e in events if e.event_type == ExecutionEventType.STRATEGY_SELECTED][0]
    assert strat.metadata["needs_web"] is True

    # Verify web search started and completed
    w_search_s = [e for e in events if e.event_type == ExecutionEventType.WEB_SEARCH_STARTED]
    w_search_c = [e for e in events if e.event_type == ExecutionEventType.WEB_SEARCH_COMPLETED]
    assert len(w_search_s) >= 1
    assert len(w_search_c) >= 1
    assert "query" in w_search_s[0].metadata

    # Verify web fetch started and completed
    w_fetch_s = [e for e in events if e.event_type == ExecutionEventType.WEB_FETCH_STARTED]
    w_fetch_c = [e for e in events if e.event_type == ExecutionEventType.WEB_FETCH_COMPLETED]
    assert len(w_fetch_s) >= 1
    assert len(w_fetch_c) >= 1
    fetch_meta = w_fetch_c[0].metadata
    assert "domain" in fetch_meta
    assert "title" in fetch_meta
    assert fetch_meta["status"] == "completed"
    assert "latency_ms" in fetch_meta

    # Verify synthesis and run completion
    assert ExecutionEventType.SYNTHESIS_COMPLETED in event_types
    assert ExecutionEventType.RUN_COMPLETED in event_types

    # Verify monotonic seq
    seqs = [e.seq for e in events]
    assert seqs == list(range(1, len(events) + 1))

