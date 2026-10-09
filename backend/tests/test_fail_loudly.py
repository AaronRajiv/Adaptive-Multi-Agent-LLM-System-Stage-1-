"""
Tests for Phase 1: Fail Loudly.
Verifies that:
1. GeminiEmbeddingProvider never silently falls back to MockEmbeddingProvider on error.
2. Document ingestion exceptions are never swallowed; pipeline halts and emits RUN_FAILED.
3. Attaching documents when retrieval_service is None fails loudly with RUN_FAILED.
4. Single-agent RAG retrieval failure halts and emits RUN_FAILED.
5. Multi-agent tasks requesting retrieval with retrieval_service=None fail loudly.
6. Document-grounded tasks producing 0 research results fail loudly instead of generating parametric answers.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.config import Settings
from app.events import ExecutionEventType, RunEventBus
from app.knowledge.embeddings import (
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    EmbeddingError,
    get_embedding_provider,
)
from app.knowledge.models import KnowledgeDocument, RetrievedChunk, RetrievedContext
from app.knowledge.repository import InMemoryKnowledgeRepository
from app.knowledge.retrieval import RetrievalService
from app.llm.provider import MockLLMProvider
from app.models.task_graph import TaskGraph, TaskNode, TaskType
from app.orchestration.agents import CapabilityRegistry, ResearchTaskAgent
from app.orchestration.orchestrator import Orchestrator
from app.orchestration.strategy import ExecutionStrategy, StrategyDecision
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline


@pytest.mark.asyncio
async def test_gemini_embedding_provider_fails_loudly_on_http_error():
    """Verify GeminiEmbeddingProvider raises EmbeddingError without fallback on permanent HTTP error."""
    provider = GeminiEmbeddingProvider(api_key="test_key")

    mock_response = httpx.Response(
        status_code=400,
        json={"error": {"message": "Invalid embedding request arguments."}},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.embed("Sample text to embed")

        assert "Gemini embedding API failed (HTTP 400)" in str(exc_info.value)
        assert "Invalid embedding request arguments" in str(exc_info.value)


@pytest.mark.asyncio
async def test_gemini_embedding_provider_fails_loudly_on_retries_exhausted():
    """Verify GeminiEmbeddingProvider raises EmbeddingError after transient retries are exhausted."""
    provider = GeminiEmbeddingProvider(api_key="test_key")

    mock_response = httpx.Response(
        status_code=503,
        json={"error": {"message": "Service unavailable temporarily."}},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
         patch("asyncio.sleep", new_callable=AsyncMock):
        mock_post.return_value = mock_response
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.embed("Sample text to embed")

        assert "HTTP 503" in str(exc_info.value)


@pytest.mark.asyncio
async def test_gemini_embedding_provider_fails_loudly_on_network_error():
    """Verify GeminiEmbeddingProvider raises EmbeddingError on persistent network error."""
    provider = GeminiEmbeddingProvider(api_key="test_key")

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(EmbeddingError) as exc_info:
            await provider.embed("Sample text")

        assert "Network error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_embedding_provider_raises_when_gemini_key_missing_in_production():
    """Verify get_embedding_provider raises when provider is gemini and key is missing (non-mock mode)."""
    cfg = Settings(llm_provider="gemini", embedding_provider="gemini", llm_api_key="", embedding_api_key="")
    with pytest.raises(EmbeddingError, match="Gemini embedding API key is missing"):
        get_embedding_provider(cfg)


@pytest.mark.asyncio
async def test_orchestrated_pipeline_fails_loudly_when_documents_attached_without_retrieval_service():
    """Verify pipeline raises and emits RUN_FAILED when documents are attached but retrieval_service is None."""
    event_bus = RunEventBus("test_run_fail_1")
    mock_llm = MockLLMProvider()
    pipeline = OrchestratedPipeline(
        llm_provider=mock_llm,
        settings=Settings(),
        retrieval_service=None,
        event_bus=event_bus,
    )
    # Ensure retrieval_service is None
    pipeline.retrieval_service = None

    docs = [{"filename": "doc1.txt", "content": "Sample content"}]
    with pytest.raises(RuntimeError) as exc_info:
        await pipeline.execute(user_task="Explain doc1", documents=docs)

    assert "retrieval service is not available" in str(exc_info.value)

    run_failed_events = [e for e in event_bus.history if e.event_type == ExecutionEventType.RUN_FAILED]
    assert len(run_failed_events) >= 1
    assert "retrieval service is not available" in run_failed_events[0].message


@pytest.mark.asyncio
async def test_orchestrated_pipeline_fails_loudly_on_ingestion_failure():
    """Verify pipeline raises and emits RUN_FAILED if document ingestion fails."""
    event_bus = RunEventBus("test_run_fail_2")
    mock_llm = MockLLMProvider()
    repo = InMemoryKnowledgeRepository()
    embedder = MockEmbeddingProvider()
    retrieval = RetrievalService(embedder, repo)

    pipeline = OrchestratedPipeline(
        llm_provider=mock_llm,
        settings=Settings(),
        retrieval_service=retrieval,
        event_bus=event_bus,
    )

    # Patch repo.store to simulate storage/embedding failure during ingestion
    with patch.object(repo, "store", side_effect=RuntimeError("Disk/database write failed")):
        docs = [{"filename": "regulations.txt", "content": "Article 1.1 Technical Regulations"}]
        with pytest.raises(RuntimeError) as exc_info:
            await pipeline.execute(user_task="Check article 1.1", documents=docs)

        assert "Document ingestion failed" in str(exc_info.value)

    run_failed_events = [e for e in event_bus.history if e.event_type == ExecutionEventType.RUN_FAILED]
    assert len(run_failed_events) >= 1
    assert "ingestion failed" in run_failed_events[0].message.lower()


@pytest.mark.asyncio
async def test_single_agent_rag_fails_loudly_on_retrieval_error():
    """Verify single-agent execution fails loudly when requires_rag is True and retrieval fails."""
    event_bus = RunEventBus("test_run_fail_3")
    mock_llm = MockLLMProvider()
    repo = InMemoryKnowledgeRepository()
    embedder = MockEmbeddingProvider()
    retrieval = RetrievalService(embedder, repo)

    pipeline = OrchestratedPipeline(
        llm_provider=mock_llm,
        settings=Settings(),
        retrieval_service=retrieval,
        event_bus=event_bus,
    )

    # Force strategy decision to SINGLE_AGENT with requires_rag=True
    pipeline.strategy_selector.select_strategy = AsyncMock(
        return_value=StrategyDecision(
            strategy=ExecutionStrategy.SINGLE_AGENT,
            reasoning="Testing single agent RAG failure",
            requires_rag=True,
            estimated_complexity="low",
        )
    )

    with patch.object(retrieval, "retrieve", side_effect=RuntimeError("Vector search index corrupted")):
        with pytest.raises(RuntimeError) as exc_info:
            await pipeline.execute(user_task="Lookup section 3.2")

        assert "Knowledge retrieval failed" in str(exc_info.value)

    run_failed_events = [e for e in event_bus.history if e.event_type == ExecutionEventType.RUN_FAILED]
    assert len(run_failed_events) >= 1
    assert "retrieval failed" in run_failed_events[0].message.lower()


@pytest.mark.asyncio
async def test_orchestrator_fails_loudly_when_retrieval_service_is_none_for_query():
    """Verify Orchestrator fails the task if retrieval_query is present but retrieval_service is None."""
    registry = CapabilityRegistry()
    mock_agent = AsyncMock()
    registry.register("research", mock_agent)

    orchestrator = Orchestrator(registry=registry, retrieval_service=None)

    task = TaskNode(
        id="T1",
        description="Find article 3.2",
        type=TaskType.RESEARCH,
        assigned_capability="research",
        metadata={"retrieval_query": "Article 3.2"},
    )
    graph = TaskGraph(tasks=[task])

    result = await orchestrator.execute(graph)
    assert result.task_results["T1"].success is False
    assert "retrieval service is not available" in result.task_results["T1"].error


@pytest.mark.asyncio
async def test_multi_agent_fails_loudly_when_document_research_yields_no_findings():
    """Verify multi-agent execution halts with RUN_FAILED when all research tasks fail on document question."""
    event_bus = RunEventBus("test_run_fail_4")
    mock_llm = MockLLMProvider()
    repo = InMemoryKnowledgeRepository()
    embedder = MockEmbeddingProvider()
    retrieval = RetrievalService(embedder, repo)

    pipeline = OrchestratedPipeline(
        llm_provider=mock_llm,
        settings=Settings(),
        retrieval_service=retrieval,
        event_bus=event_bus,
    )

    # Force strategy decision to MULTI_AGENT with requires_rag=True
    pipeline.strategy_selector.select_strategy = AsyncMock(
        return_value=StrategyDecision(
            strategy=ExecutionStrategy.MULTI_AGENT,
            reasoning="Testing multi-agent research failure",
            requires_rag=True,
            estimated_complexity="high",
        )
    )

    # Force planner to produce one research task
    from app.models.schemas import PlannerOutput, SubTask
    pipeline.planner.plan = AsyncMock(
        return_value=PlannerOutput(
            tasks=[SubTask(id="T1", description="Research aerofoil", type="research")],
            reasoning="Decomposed",
        )
    )

    # Force researcher to fail (e.g. timeout or exception)
    pipeline.researcher.execute_subtask = AsyncMock(
        side_effect=RuntimeError("Retrieval timeout during subtask execution")
    )

    with pytest.raises(RuntimeError) as exc_info:
        await pipeline.execute(
            user_task="Explain aerofoil",
            documents=[{"filename": "doc.txt", "content": "Sample content"}],
        )

    assert "Multi-agent research produced no findings" in str(exc_info.value)
    run_failed_events = [e for e in event_bus.history if e.event_type == ExecutionEventType.RUN_FAILED]
    assert len(run_failed_events) >= 1
    assert "no findings for document-grounded task" in run_failed_events[0].message
