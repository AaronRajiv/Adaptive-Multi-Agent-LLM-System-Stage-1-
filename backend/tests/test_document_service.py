"""
Tests for Phase 2: Document Service and Decoupled Ingestion.
Verifies:
1. Multipart upload endpoint (POST /api/documents) and background processing.
2. SHA-256 deduplication of uploaded document bytes.
3. Document status tracking (GET /api/documents/{id}/status).
4. Document listing (GET /api/documents) and deletion (DELETE /api/documents/{id}).
5. Rejection of runs when referenced document_ids are not ready or missing.
6. Per-document retrieval filtering (no cross-document contamination).
7. Asynchronous pipeline execution via POST /api/run and result fetching via GET /api/runs/{run_id}.
"""

import asyncio
import io
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.knowledge.document_service import DocumentService, DocumentStatus
from app.knowledge.embeddings import MockEmbeddingProvider
from app.knowledge.repository import InMemoryKnowledgeRepository
from app.api.routes import get_shared_document_service


@pytest.fixture(autouse=True)
def setup_mock_environment(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "embedding_provider", "mock")


@pytest.mark.asyncio
async def test_upload_document_and_deduplication():
    """Verify document upload, background extraction, and SHA-256 deduplication."""
    doc_content = b"Technical Regulations 2026: The minimum weight of the car is 798 kg."

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Upload document
        res = await client.post(
            "/api/documents",
            files={"file": ("fia_2026_regulations.txt", doc_content, "text/plain")},
            data={"label": "FIA 2026"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "document_id" in data
        doc_id = data["document_id"]
        assert data["filename"] == "fia_2026_regulations.txt"
        assert data["label"] == "FIA 2026"
        assert data["sha256"] != ""

        # Wait for background ingestion to complete
        for _ in range(40):
            status_res = await client.get(f"/api/documents/{doc_id}/status")
            assert status_res.status_code == 200
            status_data = status_res.json()
            if status_data["status"] == "ready":
                break
            await asyncio.sleep(0.05)

        status_res = await client.get(f"/api/documents/{doc_id}/status")
        status_data = status_res.json()
        assert status_data["status"] == "ready"
        assert status_data["chunk_count"] >= 1

        # 2. Upload identical document bytes (Deduplication test)
        res_dup = await client.post(
            "/api/documents",
            files={"file": ("fia_duplicate.txt", doc_content, "text/plain")},
            data={"label": "FIA 2026 Duplicate"},
        )
        assert res_dup.status_code == 200
        data_dup = res_dup.json()
        assert data_dup["document_id"] == doc_id
        assert data_dup["sha256"] == data["sha256"]
        assert data_dup["status"] == "ready"
        assert "already ingested" in data_dup["message"].lower()


@pytest.mark.asyncio
async def test_list_and_delete_document():
    """Verify listing documents and deleting a document."""
    content = b"Article 5.1 Power Unit Specifications: ICE displacement 1.6 litres."

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/documents",
            files={"file": ("engine_specs.txt", content, "text/plain")},
            data={"label": "Engine"},
        )
        assert res.status_code == 200
        doc_id = res.json()["document_id"]

        # List documents
        list_res = await client.get("/api/documents")
        assert list_res.status_code == 200
        docs = list_res.json()
        assert any(d["document_id"] == doc_id for d in docs)

        # Delete document
        del_res = await client.delete(f"/api/documents/{doc_id}")
        assert del_res.status_code == 200
        assert del_res.json()["deleted"] is True

        # Check 404 after deletion
        status_res = await client.get(f"/api/documents/{doc_id}/status")
        assert status_res.status_code == 404


@pytest.mark.asyncio
async def test_refuse_run_when_document_not_ready_or_missing():
    """Verify /api/run refuses to execute if referenced document_id is missing or not ready."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Non-existent document ID
        res = await client.post(
            "/api/run",
            json={
                "task": "Explain the minimum weight.",
                "document_ids": ["doc_non_existent_12345"],
            },
        )
        assert res.status_code == 400
        assert "does not exist" in res.json()["detail"]

        # 2. Document not ready
        doc_service = get_shared_document_service()
        content = b"Sample draft text not yet ready."
        upload_res = await client.post(
            "/api/documents",
            files={"file": ("draft.txt", content, "text/plain")},
        )
        doc_id = upload_res.json()["document_id"]

        # Temporarily set status to EXTRACTING to simulate in-flight ingestion
        doc_meta = doc_service.get_document(doc_id)
        doc_meta.status = DocumentStatus.EXTRACTING

        res_not_ready = await client.post(
            "/api/run",
            json={
                "task": "Explain draft.",
                "document_ids": [doc_id],
            },
        )
        assert res_not_ready.status_code == 400
        assert "is not ready" in res_not_ready.json()["detail"]


@pytest.mark.asyncio
async def test_per_document_retrieval_filtering():
    """Verify retrieval is strictly isolated to specified document_ids without cross-contamination."""
    repo = InMemoryKnowledgeRepository()
    embedder = MockEmbeddingProvider(dimensions=64)
    service = DocumentService(repository=repo, embedding_provider=embedder)

    # Ingest Doc A (2025 Regulations: Weight 798kg)
    bytes_a = b"Article 4.1: The 2025 car minimum weight is 798 kg."
    res_a = await service.register_and_start_ingestion("doc_2025.txt", bytes_a, label="FIA 2025")
    doc_a_id = res_a.document_id

    # Ingest Doc B (2026 Regulations: Weight 768kg)
    bytes_b = b"Article 4.1: The 2026 car minimum weight is 768 kg."
    res_b = await service.register_and_start_ingestion("doc_2026.txt", bytes_b, label="FIA 2026")
    doc_b_id = res_b.document_id

    # Wait for both workers to finish
    for _ in range(50):
        doc_a = service.get_document(doc_a_id)
        doc_b = service.get_document(doc_b_id)
        if doc_a and doc_a.status == DocumentStatus.READY and doc_b and doc_b.status == DocumentStatus.READY:
            break
        await asyncio.sleep(0.05)

    assert service.get_document(doc_a_id).status == DocumentStatus.READY
    assert service.get_document(doc_b_id).status == DocumentStatus.READY

    from app.knowledge.retrieval import RetrievalService
    retrieval = RetrievalService(embedder, repo)

    # Search filtered strictly to Doc A
    ctx_a = await retrieval.retrieve("minimum weight", top_k=5, document_ids=[doc_a_id])
    assert len(ctx_a.chunks) > 0
    assert all(c.document_id == doc_a_id for c in ctx_a.chunks)
    assert any("798 kg" in c.content for c in ctx_a.chunks)
    assert not any("768 kg" in c.content for c in ctx_a.chunks)

    # Search filtered strictly to Doc B
    ctx_b = await retrieval.retrieve("minimum weight", top_k=5, document_ids=[doc_b_id])
    assert len(ctx_b.chunks) > 0
    assert all(c.document_id == doc_b_id for c in ctx_b.chunks)
    assert any("768 kg" in c.content for c in ctx_b.chunks)
    assert not any("798 kg" in c.content for c in ctx_b.chunks)


@pytest.mark.asyncio
async def test_async_run_and_get_run_by_id():
    """Verify POST /api/run with async_run=True returns immediately and GET /api/runs/{id} returns result."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "task": "Compare battery safety features.",
            "async_run": True,
        }
        response = await client.post("/api/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert data["status"] in ("started", "running")
        run_id = data["run_id"]

        # Poll /api/runs/{run_id} until completed (or timeout)
        completed = False
        for _ in range(40):
            get_res = await client.get(f"/api/runs/{run_id}")
            assert get_res.status_code == 200
            run_data = get_res.json()
            if run_data.get("status") == "completed":
                completed = True
                assert run_data["result"] is not None
                assert "final_answer" in run_data["result"]
                break
            await asyncio.sleep(0.1)

        assert completed is True
