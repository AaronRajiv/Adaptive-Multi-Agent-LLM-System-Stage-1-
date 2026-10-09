"""
Phase 4 Tests: Hybrid Retrieval Engine, BM25 Keyword Search, Direct Article-Number Lookup,
Reciprocal Rank Fusion (RRF), Multi-Query Deduplication, Retrieval Budget, and Evaluation Harness.
"""

import pytest
from typing import List, Optional

from app.knowledge.bm25 import BM25Okapi, tokenize
from app.knowledge.chunking import StructureAwareChunker
from app.knowledge.document_service import DocumentService, DocumentStatus
from app.knowledge.embeddings import MockEmbeddingProvider
from app.knowledge.models import DocumentChunk, KnowledgeDocument
from app.knowledge.repository import InMemoryKnowledgeRepository
from app.knowledge.retrieval import RetrievalService, extract_article_query
from app.knowledge.retrieval_eval import evaluate_retrieval_dataset


def test_article_query_extraction():
    """Verify that regexes accurately isolate exact article numbers from natural language queries."""
    assert extract_article_query("What does Article 3.2.1 say about car width?") == "3.2.1"
    assert extract_article_query("Check regulation Article C3.2 for the power unit.") == "C3.2"
    assert extract_article_query("Explain section 14.1.2 requirements.") == "14.1.2"
    assert extract_article_query("art. 5.4 minimum engine mass") == "5.4"
    assert extract_article_query("What is the rule in 3.2.1?") == "3.2.1"
    assert extract_article_query("What is the general car mass?") is None


def test_bm25_keyword_scoring():
    """Verify BM25Okapi correctly ranks keyword occurrences and lengths."""
    docs = [
        "The overall width of the car must not exceed 2000mm.",
        "The internal combustion engine minimum mass is 150kg.",
        "Aerodynamic wings and endplates must remain immobile.",
    ]
    corpus = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus)

    results = bm25.search("car width 2000mm", docs, top_k=2)
    assert len(results) >= 1
    assert "2000mm" in results[0][0]
    assert results[0][1] > 0.0


@pytest.mark.asyncio
async def test_direct_article_lookup_priority():
    """Verify exact article-number queries resolve by direct lookup with highest priority."""
    repo = InMemoryKnowledgeRepository(expected_dimensions=64)
    embedder = MockEmbeddingProvider(dimensions=64)
    service = RetrievalService(embedding_provider=embedder, repository=repo)

    doc = KnowledgeDocument(id="doc_regs", source="regs.pdf", content="full text")
    chunk1 = DocumentChunk(
        id="doc_regs:chunk:0",
        document_id="doc_regs",
        chunk_index=0,
        source="regs.pdf",
        content="General aerodynamic components principles.",
        article_path="3.1",
        heading="General Principles",
        metadata={"article_path": "3.1", "heading": "General Principles"},
    )
    chunk2 = DocumentChunk(
        id="doc_regs:chunk:1",
        document_id="doc_regs",
        chunk_index=1,
        source="regs.pdf",
        content="Overall car width must not exceed 2000mm.",
        article_path="3.2.1",
        heading="Overall Width",
        metadata={"article_path": "3.2.1", "heading": "Overall Width"},
    )

    vec1 = await embedder.embed(chunk1.content)
    vec2 = await embedder.embed(chunk2.content)
    await repo.store(doc, [chunk1, chunk2], [vec1, vec2])

    # Query specifically referencing Article 3.2.1
    ctx = await service.retrieve("What does Article 3.2.1 say about dimensions?", top_k=2)
    assert len(ctx.chunks) >= 1
    top_chunk = ctx.chunks[0]
    assert top_chunk.article_path == "3.2.1"
    assert top_chunk.retrieval_method == "direct_article_lookup"
    assert top_chunk.similarity_score > 0.0


@pytest.mark.asyncio
async def test_reciprocal_rank_fusion_and_provenance():
    """Verify RRF fuses dense vector and BM25 streams and preserves full provenance."""
    repo = InMemoryKnowledgeRepository(expected_dimensions=64)
    embedder = MockEmbeddingProvider(dimensions=64)
    service = RetrievalService(embedding_provider=embedder, repository=repo)

    doc = KnowledgeDocument(id="doc_f1", source="f1.pdf", content="f1 content")
    chunks = [
        DocumentChunk(
            id=f"doc_f1:chunk:{i}",
            document_id="doc_f1",
            chunk_index=i,
            source="f1.pdf",
            content=f"Technical regulation rule {i} on powertrain component {i}.",
            page_start=i + 1,
            page_end=i + 1,
            article_path=f"5.{i}",
            heading=f"Component {i}",
            metadata={"article_path": f"5.{i}", "heading": f"Component {i}"},
        )
        for i in range(5)
    ]
    vectors = await embedder.embed_many([c.content for c in chunks])
    await repo.store(doc, chunks, vectors)

    ctx = await service.retrieve("powertrain component 2", top_k=3)
    assert len(ctx.chunks) == 3

    # Every chunk must have complete provenance
    for c in ctx.chunks:
        assert c.document_id == "doc_f1"
        assert c.source == "f1.pdf"
        assert c.page_start is not None
        assert c.page_end is not None
        assert c.article_path is not None
        assert c.retrieval_method in ("hybrid_rrf", "dense", "bm25", "direct_article_lookup")
        assert c.similarity_score > 0.0


@pytest.mark.asyncio
async def test_multi_query_retrieval_and_deduplication():
    """Verify multi-query execution merges and deduplicates chunks across queries."""
    repo = InMemoryKnowledgeRepository(expected_dimensions=64)
    embedder = MockEmbeddingProvider(dimensions=64)
    service = RetrievalService(embedding_provider=embedder, repository=repo)

    doc = KnowledgeDocument(id="doc_mq", source="mq.pdf", content="text")
    chunks = [
        DocumentChunk(
            id=f"doc_mq:chunk:{i}",
            document_id="doc_mq",
            chunk_index=i,
            source="mq.pdf",
            content=f"Rule item {i} regarding braking system and wheels.",
            page_start=1,
            page_end=1,
            article_path=f"11.{i}",
            heading=f"Brakes {i}",
            metadata={"article_path": f"11.{i}"},
        )
        for i in range(4)
    ]
    vectors = await embedder.embed_many([c.content for c in chunks])
    await repo.store(doc, chunks, vectors)

    queries = ["braking system hydraulic rules", "wheels and brakes"]
    ctx = await service.retrieve_multi(queries, top_k=3)

    # Verify deduplication: no duplicate chunk IDs
    chunk_ids = [c.id for c in ctx.chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.asyncio
async def test_retrieval_budget_enforcement():
    """Verify that retrieval budget limits maximum total chunks returned."""
    repo = InMemoryKnowledgeRepository(expected_dimensions=64)
    embedder = MockEmbeddingProvider(dimensions=64)
    service = RetrievalService(embedding_provider=embedder, repository=repo)

    doc = KnowledgeDocument(id="doc_bg", source="bg.pdf", content="text")
    chunks = [
        DocumentChunk(
            id=f"doc_bg:chunk:{i}",
            document_id="doc_bg",
            chunk_index=i,
            source="bg.pdf",
            content=f"Regulation statement {i} on aerodynamic airflow.",
            page_start=1,
            page_end=1,
            article_path=f"3.{i}",
            heading=f"Section {i}",
            metadata={"article_path": f"3.{i}"},
        )
        for i in range(10)
    ]
    vectors = await embedder.embed_many([c.content for c in chunks])
    await repo.store(doc, chunks, vectors)

    # With budget=3, exactly 3 chunks returned even if top_k=10
    ctx = await service.retrieve_multi(["aerodynamic airflow rules"], top_k=10, budget=3)
    assert len(ctx.chunks) == 3


@pytest.mark.asyncio
async def test_retrieval_evaluation_harness_metrics():
    """Verify Recall@5, Recall@10, and MRR computation on a controlled dataset."""
    repo = InMemoryKnowledgeRepository(expected_dimensions=64)
    embedder = MockEmbeddingProvider(dimensions=64)
    service = RetrievalService(embedding_provider=embedder, repository=repo)

    doc = KnowledgeDocument(id="doc_eval", source="eval.pdf", content="text")
    chunks = [
        DocumentChunk(
            id="doc_eval:chunk:0",
            document_id="doc_eval",
            chunk_index=0,
            source="eval.pdf",
            content="Article 3.2.1 Overall Width: Maximum 2000mm allowed across wheels.",
            page_start=2,
            page_end=2,
            article_path="3.2.1",
            heading="Overall Width",
            metadata={"article_path": "3.2.1", "heading": "Overall Width"},
        ),
        DocumentChunk(
            id="doc_eval:chunk:1",
            document_id="doc_eval",
            chunk_index=1,
            source="eval.pdf",
            content="Article 4.1 Minimum Mass: The car weight must not be below 798kg.",
            page_start=5,
            page_end=5,
            article_path="4.1",
            heading="Minimum Mass",
            metadata={"article_path": "4.1", "heading": "Minimum Mass"},
        ),
    ]
    vectors = await embedder.embed_many([c.content for c in chunks])
    await repo.store(doc, chunks, vectors)

    eval_subset = [
        {"id": 1, "question": "What is the car maximum width in Article 3.2.1?", "expected_article": "3.2.1"},
        {"id": 2, "question": "What is the minimum mass of the vehicle in Article 4.1?", "expected_article": "4.1"},
    ]

    metrics = await evaluate_retrieval_dataset(service, eval_subset, top_k=5)
    assert metrics["total_queries"] == 2
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr"] == 1.0
    assert len(metrics["details"]) == 2
