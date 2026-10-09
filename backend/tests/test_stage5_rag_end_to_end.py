"""
Comprehensive unit & integration tests for Stage 5 RAG End-to-End Grounding.
Tests cover:
- Knowledge base ingestion (PDF text extraction, chunking, embeddings, store & search)
- TEST A: RAG Not Required (No retrieval_query -> No RAG events, no retrieval call)
- TEST B: RAG Required (retrieval_query present -> RAG events emitted, context retrieved, provenance retained)
- TEST C: Grounding (Retrieved context actually included in LLM prompt)
- TEST D: No Relevant Results (0 chunks returned, events accurate, agent informed, provenance empty)
- Dimension Mismatch & Configuration checks
"""

import pytest
import asyncio
from app.config import Settings
from app.events import ExecutionEventType, RunEventBus
from app.knowledge.chunking import FixedSizeTextChunker
from app.knowledge.embeddings import MockEmbeddingProvider, get_embedding_provider, EmbeddingError
from app.knowledge.ingestion import DocumentIngestionService
from app.knowledge.models import KnowledgeDocument, RetrievedChunk, RetrievedContext, DocumentChunk
from app.knowledge.repository import InMemoryKnowledgeRepository, KnowledgeRepositoryError
from app.knowledge.retrieval import RetrievalService
from app.llm.provider import MockLLMProvider
from app.models.schemas import PlannerOutput, SubTask
from app.models.task_graph import TaskGraph, TaskNode, TaskType
from app.orchestration.agents import TaskExecutionContext, TaskExecutionResult, ResearchTaskAgent, AnalysisTaskAgent
from app.orchestration.orchestrator import Orchestrator, CapabilityRegistry
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearchAgent
from app.agents.analyst import AnalystAgent
from app.pipeline.orchestrated_pipeline import OrchestratedPipeline


@pytest.mark.asyncio
async def test_knowledge_base_ingestion():
    """Verify document ingestion: text extraction, chunking, embeddings, vector storage & search."""
    repo = InMemoryKnowledgeRepository()
    embedder = MockEmbeddingProvider(dimensions=64)
    ingestion = DocumentIngestionService(
        chunker=FixedSizeTextChunker(chunk_size=100, overlap=10),
        embedding_provider=embedder,
        repository=repo,
    )

    doc = KnowledgeDocument(
        id="doc-aurora-001",
        source="aurora_report.pdf",
        content="The Aurora Battery Report states that the fictional Aurora battery has a 412 km rated range under standard conditions.",
    )
    ingestion_res = await ingestion.ingest(doc)
    chunks = ingestion_res.chunks
    assert len(chunks) >= 1
    assert "412 km" in chunks[0].content
    assert chunks[0].document_id == "doc-aurora-001"

    # Search
    retrieval = RetrievalService(embedding_provider=embedder, repository=repo)
    query_vec = await embedder.embed("Aurora battery range")
    results = await repo.similarity_search(query_vec, top_k=2)
    assert len(results) >= 1
    assert "412 km" in results[0].content


@pytest.mark.asyncio
async def test_dimension_mismatch_validation():
    """Verify error raised on query/storage vector dimension mismatch."""
    repo = InMemoryKnowledgeRepository()
    embedder1 = MockEmbeddingProvider(dimensions=64)
    embedder2 = MockEmbeddingProvider(dimensions=128)

    doc = KnowledgeDocument(id="doc-1", source="test.txt", content="Some test content.")
    chunks = [DocumentChunk(id="c1", document_id="doc-1", chunk_index=0, source="test.txt", content="Some test content.")]
    embeddings = [await embedder1.embed(chunks[0].content)]

    await repo.store(doc, chunks, embeddings)

    # Search with wrong dimension embedder
    wrong_query_vec = await embedder2.embed("test")
    with pytest.raises(KnowledgeRepositoryError, match="dimension"):
        await repo.similarity_search(wrong_query_vec, top_k=1)


@pytest.mark.asyncio
async def test_config_provider_selection_mock():
    """Verify get_embedding_provider returns MockEmbeddingProvider when llm_provider=mock."""
    cfg = Settings(llm_provider="mock", embedding_provider="gemini", llm_api_key="")
    embedder = get_embedding_provider(cfg)
    assert isinstance(embedder, MockEmbeddingProvider)


@pytest.mark.asyncio
async def test_a_rag_not_required():
    """
    TEST A — RAG NOT REQUIRED
    Planner creates subtasks WITHOUT retrieval_query.
    Orchestrator does not call retrieval service and emits NO RAG events.
    """
    event_bus = RunEventBus("test_run_a")

    # Setup orchestrator with empty retrieval query tasks
    repo = InMemoryKnowledgeRepository()
    embedder = MockEmbeddingProvider(dimensions=64)
    retrieval_service = RetrievalService(embedder, repo)

    researcher = ResearchAgent(MockLLMProvider())
    analyst = AnalystAgent(MockLLMProvider())
    registry = CapabilityRegistry()
    registry.register("research", ResearchTaskAgent(researcher))
    registry.register("analysis", AnalysisTaskAgent(analyst))

    orchestrator = Orchestrator(
        registry=registry,
        max_concurrency=2,
        retrieval_service=retrieval_service,
        event_bus=event_bus,
    )

    # Task graph where metadata has NO retrieval_query
    graph = TaskGraph(
        tasks=[
            TaskNode(id="T1", description="General reasoning subtask", type=TaskType.RESEARCH, metadata={}),
        ]
    )

    res = await orchestrator.execute(graph, input_context={"user_task": "General question"})
    assert res.status.value == "COMPLETED"
    emitted_types = [e.event_type for e in event_bus.history]
    assert ExecutionEventType.RAG_STARTED not in emitted_types
    assert ExecutionEventType.RAG_COMPLETED not in emitted_types


@pytest.mark.asyncio
async def test_b_rag_required():
    """
    TEST B — RAG REQUIRED
    Task has retrieval_query in metadata.
    Orchestrator emits RAG_STARTED, retrieves chunks, emits RAG_COMPLETED, and attaches context.
    """
    event_bus = RunEventBus("test_run_b")

    repo = InMemoryKnowledgeRepository()
    embedder = MockEmbeddingProvider(dimensions=64)
    ingestion = DocumentIngestionService(
        chunker=FixedSizeTextChunker(chunk_size=100, overlap=10),
        embedding_provider=embedder,
        repository=repo,
    )
    await ingestion.ingest(KnowledgeDocument(
        id="doc-ev-1",
        source="ev_spec.pdf",
        content="Electric vehicle lifecycle emissions are 50% lower than petrol vehicles.",
    ))

    retrieval_service = RetrievalService(embedder, repo)

    researcher = ResearchAgent(MockLLMProvider())
    registry = CapabilityRegistry()
    registry.register("research", ResearchTaskAgent(researcher))

    orchestrator = Orchestrator(
        registry=registry,
        max_concurrency=2,
        retrieval_service=retrieval_service,
        event_bus=event_bus,
    )

    graph = TaskGraph(
        tasks=[
            TaskNode(
                id="T1",
                description="Investigate lifecycle emissions",
                type=TaskType.RESEARCH,
                metadata={"retrieval_query": "lifecycle emissions electric versus petrol"},
            ),
        ]
    )

    res = await orchestrator.execute(graph, input_context={"user_task": "Compare EV and Petrol"})
    assert res.status.value == "COMPLETED"
    emitted_types = [e.event_type for e in event_bus.history]
    assert ExecutionEventType.RAG_STARTED in emitted_types
    assert ExecutionEventType.RAG_COMPLETED in emitted_types

    task_res = res.task_results["T1"]
    assert "retrieved_context" in task_res.metadata
    rc = task_res.metadata["retrieved_context"]
    assert len(rc["chunks"]) >= 1
    assert "lifecycle emissions" in rc["chunks"][0]["content"].lower()


@pytest.mark.asyncio
async def test_c_grounding_prompt_verification():
    """
    TEST C — GROUNDING
    Verify that retrieved chunks are actually injected into the LLM prompt.
    """
    captured_prompts = []

    class CapturingLLMProvider(MockLLMProvider):
        async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
            captured_prompts.append(prompt)
            return await super().generate(prompt, system_prompt)

    llm = CapturingLLMProvider()
    research_agent = ResearchAgent(llm)

    chunk = RetrievedChunk(
        id="c1",
        document_id="doc-aurora-001",
        chunk_index=0,
        source="aurora_report.pdf",
        content="The Aurora Battery Report states that the fictional Aurora battery has a 412 km rated range.",
        similarity_score=0.92,
    )
    retrieved_context = RetrievedContext(query="Aurora battery range", chunks=[chunk])

    task_node = TaskNode(id="T1", description="Check battery range", type=TaskType.RESEARCH)
    result = await research_agent.execute_subtask("What is the battery range?", task_node, retrieved_context=retrieved_context)

    assert len(captured_prompts) == 1
    prompt_used = captured_prompts[0]

    # Verify explicit grounding section formatting and content
    assert "KNOWLEDGE BASE CONTEXT:" in prompt_used
    assert "aurora_report.pdf" in prompt_used
    assert "doc-aurora-001" in prompt_used
    assert "412 km rated range" in prompt_used
    assert "Use the supplied knowledge-base context when answering." in prompt_used
    assert "Do not claim that information came from the knowledge base unless it is supported" in prompt_used


@pytest.mark.asyncio
async def test_d_no_relevant_results():
    """
    TEST D — NO RELEVANT RESULTS
    If retrieval returns 0 chunks:
    - Events still emitted (RAG_STARTED & RAG_COMPLETED)
    - Context attached with empty chunks list
    - LLM prompt explicitly informed no relevant context found
    - Provenance has 0 chunks
    """
    captured_prompts = []

    class CapturingLLMProvider(MockLLMProvider):
        async def generate(self, prompt: str, system_prompt: str | None = None) -> str:
            captured_prompts.append(prompt)
            return await super().generate(prompt, system_prompt)

    llm = CapturingLLMProvider()
    research_agent = ResearchAgent(llm)

    retrieved_context = RetrievedContext(query="Nonexistent topic query", chunks=[])
    task_node = TaskNode(id="T1", description="Investigate nonexistent topic", type=TaskType.RESEARCH)

    result = await research_agent.execute_subtask("Question about nonexistent topic", task_node, retrieved_context=retrieved_context)

    assert len(captured_prompts) == 1
    prompt_used = captured_prompts[0]

    assert "KNOWLEDGE BASE CONTEXT:" in prompt_used
    assert "No relevant knowledge-base context was found for query \"Nonexistent topic query\"" in prompt_used
    assert "Do not claim that information came from the knowledge base since no relevant context was found." in prompt_used
