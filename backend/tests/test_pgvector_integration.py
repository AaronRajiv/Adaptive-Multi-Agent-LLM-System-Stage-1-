"""
Integration tests for PostgresPgvectorKnowledgeRepository.
Executes against a real PostgreSQL instance when KNOWLEDGE_DATABASE_URL is set,
or gracefully skips when running in unit test mode without a live DB.
"""

import os
import pytest
from app.knowledge.models import DocumentChunk, KnowledgeDocument
from app.knowledge.repository import PostgresPgvectorKnowledgeRepository


@pytest.fixture
def pgvector_url():
    url = os.environ.get("KNOWLEDGE_DATABASE_URL", "").strip()
    if not url:
        pytest.skip("KNOWLEDGE_DATABASE_URL is not set. Skipping pgvector live integration tests.")
    return url


@pytest.mark.asyncio
async def test_pgvector_live_repository_lifecycle(pgvector_url):
    """Verifies store, dense vector similarity, BM25 full-text, article lookup, and delete."""
    repo = PostgresPgvectorKnowledgeRepository(
        database_url=pgvector_url,
        embedding_dimensions=768,
    )

    try:
        conn = await repo._connect()
        await conn.close()
    except Exception as exc:
        pytest.skip(f"PostgreSQL connection to '{pgvector_url}' failed: {exc}. Skipping live integration test.")

    doc = KnowledgeDocument(
        id="doc_pg_test",
        source="pg_test.pdf",
        content="Article 3.2.1 Overall Width: Car width must not exceed 2000mm.",
        metadata={"label": "F1 Live DB Test"},
    )
    chunk = DocumentChunk(
        id="doc_pg_test:chunk:0",
        document_id="doc_pg_test",
        chunk_index=0,
        source="pg_test.pdf",
        content="Article 3.2.1 Overall Width: Car width must not exceed 2000mm.",
        page_start=1,
        page_end=1,
        article_path="3.2.1",
        heading="Overall Width",
        metadata={"article_path": "3.2.1", "heading": "Overall Width"},
    )
    embedding = [0.01] * 768

    try:
        # 1. Store
        await repo.store(doc, [chunk], [embedding])
        assert await repo.has_document("doc_pg_test") is True

        # 2. Dense search
        dense_results = await repo.similarity_search(embedding, top_k=1)
        assert len(dense_results) >= 1
        assert dense_results[0].document_id == "doc_pg_test"

        # 3. BM25 / Full-text search
        bm25_results = await repo.bm25_search("width 2000mm", top_k=1)
        assert len(bm25_results) >= 1
        assert "2000mm" in bm25_results[0].content

        # 4. Article lookup
        art_results = await repo.lookup_by_article("3.2.1")
        assert len(art_results) >= 1
        assert art_results[0].article_path == "3.2.1"

    finally:
        # 5. Cleanup
        await repo.delete_document("doc_pg_test")
        assert await repo.has_document("doc_pg_test") is False
