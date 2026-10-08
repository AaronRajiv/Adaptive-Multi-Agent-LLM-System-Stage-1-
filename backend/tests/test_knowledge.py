"""Deterministic unit tests for the Stage 5 optional RAG subsystem."""

from pydantic import ValidationError

import pytest

from app.knowledge.chunking import FixedSizeTextChunker
from app.knowledge.embeddings import EmbeddingError, EmbeddingProvider, MockEmbeddingProvider
from app.knowledge.ingestion import DocumentIngestionService
from app.knowledge.models import KnowledgeDocument
from app.knowledge.repository import InMemoryKnowledgeRepository, KnowledgeRepository, KnowledgeRepositoryError
from app.knowledge.retrieval import RetrievalError, RetrievalService
from app.models.task_graph import TaskGraph, TaskNode
from app.orchestration.agents import (
    CapabilityRegistry,
    TaskExecutionAgent,
    TaskExecutionContext,
    TaskExecutionResult,
)
from app.orchestration.orchestrator import GraphExecutionStatus, Orchestrator


class StaticEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    async def embed(self, text: str) -> list[float]:
        try:
            return self.vectors[text]
        except KeyError as exc:
            raise EmbeddingError(f"No static embedding for {text!r}.") from exc


class FailingEmbeddingProvider(EmbeddingProvider):
    async def embed(self, text: str) -> list[float]:
        raise EmbeddingError("intentional embedding failure")


class FailingRepository(KnowledgeRepository):
    async def store(self, document, chunks, embeddings) -> None:
        raise KnowledgeRepositoryError("intentional store failure")

    async def similarity_search(self, query_embedding, top_k):
        raise KnowledgeRepositoryError("intentional search failure")


class ContextCaptureAgent(TaskExecutionAgent):
    def __init__(self):
        self.contexts: list[TaskExecutionContext] = []

    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
        self.contexts.append(context)
        return TaskExecutionResult(task_id=context.task.id, success=True, output="done")


def document(document_id: str = "doc-1", content: str = "solar energy provides renewable electricity") -> KnowledgeDocument:
    return KnowledgeDocument(
        id=document_id,
        source=f"{document_id}.txt",
        content=content,
        metadata={"topic": "energy", "owner": "project"},
    )


@pytest.mark.asyncio
async def test_document_ingestion_chunks_embeddings_and_metadata_are_preserved():
    repository = InMemoryKnowledgeRepository()
    service = DocumentIngestionService(
        FixedSizeTextChunker(chunk_size=12, overlap=2), MockEmbeddingProvider(dimensions=16), repository
    )

    result = await service.ingest(document(content="solar power wind power storage"))

    assert result.document_id == "doc-1"
    assert [chunk.id for chunk in result.chunks] == ["doc-1:chunk:0", "doc-1:chunk:1", "doc-1:chunk:2"]
    stored = repository.chunks["doc-1:chunk:0"]
    assert stored.document_id == "doc-1"
    assert stored.source == "doc-1.txt"
    assert stored.metadata == {"topic": "energy", "owner": "project"}
    assert len(stored.embedding) == 16


def test_fixed_size_chunking_is_deterministic_and_validates_configuration():
    chunker = FixedSizeTextChunker(chunk_size=5, overlap=1)
    chunks = chunker.chunk(document(content="abcdefghij"))

    assert [chunk.content for chunk in chunks] == ["abcde", "efghi", "ij"]
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        FixedSizeTextChunker(chunk_size=5, overlap=5)


def test_malformed_document_data_is_rejected_by_the_model():
    with pytest.raises(ValidationError):
        KnowledgeDocument(id="", source="source.txt", content="content")


@pytest.mark.asyncio
async def test_mock_embeddings_are_deterministic_and_reject_empty_text():
    provider = MockEmbeddingProvider(dimensions=8)

    assert await provider.embed("Solar energy") == await provider.embed("Solar energy")
    with pytest.raises(EmbeddingError, match="empty text"):
        await provider.embed("   ")


@pytest.mark.asyncio
async def test_similarity_top_k_ordering_and_provenance_are_preserved():
    provider = StaticEmbeddingProvider({
        "solar document": [1.0, 0.0],
        "wind document": [0.0, 1.0],
        "solar query": [1.0, 0.0],
    })
    repository = InMemoryKnowledgeRepository()
    ingestion = DocumentIngestionService(FixedSizeTextChunker(chunk_size=100), provider, repository)
    await ingestion.ingest(document("solar", "solar document"))
    await ingestion.ingest(document("wind", "wind document"))

    context = await RetrievalService(provider, repository).retrieve("solar query", top_k=2)

    assert [chunk.document_id for chunk in context.chunks] == ["solar", "wind"]
    assert context.chunks[0].similarity_score == 1.0
    assert context.chunks[0].source == "solar.txt"
    assert context.chunks[0].metadata["topic"] == "energy"


@pytest.mark.asyncio
async def test_empty_knowledge_base_and_no_relevant_results_are_supported():
    provider = StaticEmbeddingProvider({"query": [1.0, 0.0], "other": [0.0, 1.0]})
    repository = InMemoryKnowledgeRepository()
    retrieval = RetrievalService(provider, repository)

    assert (await retrieval.retrieve("query")).chunks == []

    await DocumentIngestionService(FixedSizeTextChunker(100), provider, repository).ingest(
        document(content="other")
    )
    assert (await retrieval.retrieve("query", min_similarity=0.5)).chunks == []


@pytest.mark.asyncio
async def test_retrieval_and_ingestion_failures_are_clear():
    document_service = DocumentIngestionService(
        FixedSizeTextChunker(100), FailingEmbeddingProvider(), InMemoryKnowledgeRepository()
    )
    with pytest.raises(EmbeddingError, match="intentional embedding failure"):
        await document_service.ingest(document())

    retrieval = RetrievalService(MockEmbeddingProvider(), FailingRepository())
    with pytest.raises(RetrievalError, match="intentional search failure"):
        await retrieval.retrieve("valid query")
    with pytest.raises(RetrievalError, match="cannot be empty"):
        await retrieval.retrieve(" ")
    with pytest.raises(RetrievalError, match="top_k"):
        await retrieval.retrieve("valid query", top_k=0)


@pytest.mark.asyncio
async def test_repository_rejects_malformed_store_data():
    repository = InMemoryKnowledgeRepository()
    doc = document()
    chunk = FixedSizeTextChunker(100).chunk(doc)[0]

    with pytest.raises(KnowledgeRepositoryError, match="counts must match"):
        await repository.store(doc, [chunk], [])


@pytest.mark.asyncio
async def test_optional_retrieval_context_is_attached_only_when_task_requests_it():
    provider = StaticEmbeddingProvider({"grounding evidence": [1.0, 0.0], "solar query": [1.0, 0.0]})
    repository = InMemoryKnowledgeRepository()
    await DocumentIngestionService(FixedSizeTextChunker(100), provider, repository).ingest(
        document(content="grounding evidence")
    )
    retrieval = RetrievalService(provider, repository)
    agent = ContextCaptureAgent()
    registry = CapabilityRegistry()
    registry.register("capture", agent)

    requested = TaskNode(
        id="with-rag",
        description="Use supplied evidence",
        assigned_capability="capture",
        metadata={"retrieval_query": "solar query", "retrieval_top_k": 1},
    )
    normal = TaskNode(
        id="without-rag", description="Normal task", assigned_capability="capture"
    )
    result = await Orchestrator(registry, retrieval_service=retrieval).execute(
        TaskGraph(tasks=[requested, normal])
    )

    assert result.status == GraphExecutionStatus.COMPLETED
    contexts = {context.task.id: context for context in agent.contexts}
    assert contexts["with-rag"].retrieved_context is not None
    assert contexts["with-rag"].retrieved_context.chunks[0].source == "doc-1.txt"
    assert contexts["without-rag"].retrieved_context is None
