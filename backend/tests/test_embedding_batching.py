"""Tests for Gemini batch embeddings, concurrency, ordering, failure handling, chunk sizes, and observability."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

from app.knowledge.chunking import FixedSizeTextChunker
from app.knowledge.embeddings import (
    EmbeddingError,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
)
from app.knowledge.ingestion import DocumentIngestionService
from app.knowledge.models import KnowledgeDocument
from app.knowledge.repository import InMemoryKnowledgeRepository


@pytest.mark.asyncio
async def test_gemini_embedding_batching_and_concurrency():
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        model="gemini-embedding-001",
        batch_size=10,
        max_concurrency=2,
    )

    batch_calls = []

    async def mock_post(url, json=None, headers=None):
        batch_calls.append(json)
        requests = json.get("requests", [])
        embeddings = []
        for i, req in enumerate(requests):
            embeddings.append({"values": [float(i + 1), 0.0]})
        
        class MockResponse:
            status_code = 200
            def json(self):
                return {"embeddings": embeddings}
        return MockResponse()

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        texts = [f"Text chunk {i}" for i in range(25)]
        embeddings = await provider.embed_many(texts)

        assert len(embeddings) == 25
        # 25 texts with batch_size=10 => 3 batches (10, 10, 5)
        assert len(batch_calls) == 3
        assert len(batch_calls[0]["requests"]) == 10
        assert len(batch_calls[1]["requests"]) == 10
        assert len(batch_calls[2]["requests"]) == 5
        # Check ordering preserved
        assert embeddings[0] == [1.0, 0.0]
        assert embeddings[9] == [10.0, 0.0]


@pytest.mark.asyncio
async def test_mock_embedding_provider_does_not_call_remote_api():
    provider = MockEmbeddingProvider(dimensions=16)
    texts = ["Sample text 1", "Sample text 2"]
    
    res1 = await provider.embed_many(texts)
    res2 = await provider.embed_many(texts)

    assert len(res1) == 2
    assert res1 == res2
    assert len(res1[0]) == 16


@pytest.mark.asyncio
async def test_ingestion_observability_and_chunk_size():
    chunker = FixedSizeTextChunker(chunk_size=1200, overlap=120)
    provider = MockEmbeddingProvider(dimensions=8)
    repository = InMemoryKnowledgeRepository()
    service = DocumentIngestionService(chunker, provider, repository)

    large_text = "Word " * 600  # 3000 chars
    doc = KnowledgeDocument(id="doc-large", source="large.txt", content=large_text)

    result = await service.ingest(doc)

    assert result.document_id == "doc-large"
    assert result.doc_char_count == len(large_text)
    assert result.chunk_count > 0
    assert result.total_duration_ms >= 0.0
    assert result.embedding_duration_ms >= 0.0
    assert result.batch_count >= 1


@pytest.mark.asyncio
async def test_ingestion_failure_does_not_leave_partial_chunks():
    class FailingProvider(MockEmbeddingProvider):
        async def embed_many(self, texts):
            raise EmbeddingError("API key expired or quota exceeded")

    chunker = FixedSizeTextChunker(chunk_size=500, overlap=50)
    provider = FailingProvider()
    repository = InMemoryKnowledgeRepository()
    service = DocumentIngestionService(chunker, provider, repository)

    doc = KnowledgeDocument(id="doc-fail", source="fail.txt", content="Testing failure scenario")

    with pytest.raises(EmbeddingError, match="API key expired or quota exceeded"):
        await service.ingest(doc)

    assert len(repository.documents) == 0
    assert len(repository.chunks) == 0
