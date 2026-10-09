"""
Phase 3 Tests: Structure-Aware Chunking, Metadata Provenance, Multi-Page PDF Extraction,
Task Types, and Embedding Dimension Consistency.
"""

import io
import pytest
from typing import List, Optional

from app.knowledge.chunking import StructureAwareChunker
from app.knowledge.document_service import DocumentService, DocumentStatus
from app.knowledge.embeddings import EmbeddingProvider, MockEmbeddingProvider
from app.knowledge.models import KnowledgeDocument
from app.knowledge.pdf_extractor import (
    extract_pages_from_document,
    extract_text_from_document,
)
from app.knowledge.repository import (
    InMemoryKnowledgeRepository,
    KnowledgeRepositoryError,
)
from app.knowledge.retrieval import RetrievalService


def create_binary_multipage_pdf(pages: List[List[str]]) -> bytes:
    """Generates a valid, multi-page binary PDF 1.4 document without external dependencies."""
    num_pages = len(pages)
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")

    offsets = {0: 0}
    curr_id = 1

    # 1 0 obj: Catalog
    offsets[curr_id] = out.tell()
    out.write(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    curr_id += 1

    # Placeholders for page objects (obj 2 is Pages, obj 3 is Font)
    pages_obj_offset_pos = out.tell()
    curr_id += 1  # 2 is Pages

    font_id = 3
    offsets[font_id] = out.tell()
    out.write(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    curr_id += 1  # now curr_id is 4

    page_obj_ids = []
    page_data = []

    for lines in pages:
        p_id = curr_id
        c_id = curr_id + 1
        curr_id += 2
        page_obj_ids.append(p_id)

        stream_lines = ["BT", "/F1 12 Tf", "72 720 Td"]
        first = True
        for line in lines:
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if not first:
                stream_lines.append(f"0 -18 Td ({safe}) Tj")
            else:
                stream_lines.append(f"({safe}) Tj")
                first = False
        stream_lines.append("ET")
        stream_bytes = "\n".join(stream_lines).encode("latin1")

        p_obj = (
            f"{p_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {c_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n"
        ).encode("latin1")
        c_obj = (
            f"{c_id} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin1")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )
        page_data.append((p_id, p_obj, c_id, c_obj))

    # Now write Object 2 (Pages)
    offsets[2] = out.tell()
    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    out.write(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n".encode("latin1"))

    # Write Page and Content objects
    for p_id, p_obj, c_id, c_obj in page_data:
        offsets[p_id] = out.tell()
        out.write(p_obj)
        offsets[c_id] = out.tell()
        out.write(c_obj)

    # Write XREF and Trailer
    xref_offset = out.tell()
    total_objs = curr_id
    out.write(f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode("latin1"))
    for i in range(1, total_objs):
        out.write(f"{offsets[i]:010d} 00000 n \n".encode("latin1"))

    out.write(f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1"))
    return out.getvalue()


class RecordingEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that logs task_types and text arguments for assertion."""

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions
        self.recorded_calls = []
        self._inner = MockEmbeddingProvider(dimensions=dimensions)

    async def embed(self, text: str, task_type: Optional[str] = None) -> List[float]:
        self.recorded_calls.append({"text": text, "task_type": task_type})
        return await self._inner.embed(text, task_type=task_type)

    async def embed_many(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        for t in texts:
            self.recorded_calls.append({"text": t, "task_type": task_type})
        return await self._inner.embed_many(texts, task_type=task_type)


def test_real_multipage_pdf_extraction():
    """Verify that multi-page binary PDFs extract text per page preserving 1-indexed page numbers."""
    pdf_bytes = create_binary_multipage_pdf([
        ["ARTICLE 3: AERODYNAMIC COMPONENTS", "3.1 General Principles: The car must comply with all restrictions."],
        ["3.2 Overall Dimensions", "3.2.1 Overall Width: Must not exceed 2000mm."],
        ["Article C3.2: Power Unit", "C3.2.1 Minimum mass: 150kg."]
    ])

    pages = extract_pages_from_document("fia_test.pdf", pdf_bytes)
    assert len(pages) == 3
    assert pages[0][0] == 1
    assert "ARTICLE 3" in pages[0][1]
    assert pages[1][0] == 2
    assert "3.2.1 Overall Width" in pages[1][1]
    assert pages[2][0] == 3
    assert "C3.2.1" in pages[2][1]

    full_text = extract_text_from_document("fia_test.pdf", pdf_bytes)
    assert "ARTICLE 3" in full_text
    assert "3.2.1 Overall Width" in full_text
    assert "C3.2.1" in full_text


def test_table_structure_extraction():
    """Verify that tabular multi-column lines are detected and formatted with markdown pipes."""
    text_with_table = (
        "SECTION 4: TECHNICAL SPECIFICATIONS\n\n"
        "Parameter\tValue\tTolerance\n"
        "Front Track\t1600mm\t+/- 5mm\n"
        "Rear Track\t1550mm\t+/- 5mm\n"
    )
    pages = extract_pages_from_document("specs.txt", text_with_table)
    assert len(pages) == 1
    page_text = pages[0][1]
    assert "| Parameter | Value | Tolerance |" in page_text
    assert "| Front Track | 1600mm | +/- 5mm |" in page_text


def test_structure_aware_chunker_article_paths_and_metadata():
    """Verify regulation structure identification (articles, headings, page numbers, prepended header)."""
    pages = [
        (1, "ARTICLE 3: AERODYNAMIC COMPONENTS\n\n3.1 General Principles\nAerodynamic components must remain immobile."),
        (2, "3.2 Overall Dimensions\n3.2.1 Overall Width: The overall width of the car must not exceed 2000mm."),
        (3, "Article C3.2: Power Unit Flow\nC3.2.1 Minimum mass: The internal combustion engine mass must be at least 150kg.")
    ]
    doc = KnowledgeDocument(
        id="doc_fia",
        source="fia_regulations.pdf",
        content="\n\n".join(p[1] for p in pages),
        metadata={"label": "FIA Technical Regulations", "pages": pages}
    )

    chunker = StructureAwareChunker(target_tokens=800, max_tokens=1200)
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3

    # Check article path extraction
    article_paths = [c.article_path for c in chunks]
    assert "3.1" in article_paths or "3" in article_paths
    assert "3.2.1" in article_paths
    assert "C3.2.1" in article_paths

    # Check page preservation
    w_chunk = next(c for c in chunks if c.article_path == "3.2.1")
    assert w_chunk.page_start == 2
    assert w_chunk.page_end == 2
    assert w_chunk.heading == "Overall Width"
    assert "2000mm" in w_chunk.content

    # Check prepended embedding header
    assert w_chunk.metadata.get("embedding_text") is not None
    assert "[FIA Technical Regulations | 3.2.1 | Overall Width]" in w_chunk.metadata["embedding_text"]


def test_oversized_article_splits_with_bounded_overlap():
    """Verify that articles exceeding max_tokens are split on paragraph boundaries with small overlap."""
    long_paragraphs = [
        f"Paragraph {i}: This is detailed regulation text for section 5.4.1 covering exhaust dimensions and routing restrictions in extreme detail." * 10
        for i in range(15)
    ]
    pages = [
        (1, "5.4.1 Exhaust Systems\n\n" + "\n\n".join(long_paragraphs))
    ]
    doc = KnowledgeDocument(
        id="doc_large",
        source="large.pdf",
        content="\n\n".join(p[1] for p in pages),
        metadata={"pages": pages}
    )

    chunker = StructureAwareChunker(target_tokens=300, max_tokens=500, overlap_tokens=50)
    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    for c in chunks:
        assert c.article_path == "5.4.1"
        assert c.heading == "Exhaust Systems"
        assert c.page_start == 1


@pytest.mark.asyncio
async def test_embedding_task_types_and_metadata_flow():
    """Verify RETRIEVAL_DOCUMENT used during ingestion and RETRIEVAL_QUERY used during retrieval."""
    repo = InMemoryKnowledgeRepository(expected_dimensions=64)
    embedder = RecordingEmbeddingProvider(dimensions=64)
    service = DocumentService(repository=repo, embedding_provider=embedder)

    pdf_bytes = create_binary_multipage_pdf([
        ["ARTICLE 3: AERODYNAMICS", "3.2.1 Front wing flap angle must be fixed during qualifying."]
    ])

    resp = await service.register_and_start_ingestion("regs.pdf", pdf_bytes, label="F1 Regs")
    doc_id = resp.document_id

    # Wait for ingestion to finish
    import asyncio
    for _ in range(20):
        meta = service.get_document(doc_id)
        if meta and meta.status == DocumentStatus.READY:
            break
        await asyncio.sleep(0.05)

    assert service.get_document(doc_id).status == DocumentStatus.READY

    # Verify that ingestion embedded with RETRIEVAL_DOCUMENT
    ingest_calls = [c for c in embedder.recorded_calls if c["task_type"] == "RETRIEVAL_DOCUMENT"]
    assert len(ingest_calls) >= 1
    # Check that prepended headers were embedded with RETRIEVAL_DOCUMENT
    assert any("[F1 Regs | 3 | AERODYNAMICS]" in c["text"] for c in ingest_calls)
    assert any("[F1 Regs | 3.2.1 |" in c["text"] for c in ingest_calls)

    # Now perform retrieval with RetrievalService
    retriever = RetrievalService(embedding_provider=embedder, repository=repo)
    results = await retriever.retrieve("What is front wing flap rule?", top_k=2)

    query_calls = [c for c in embedder.recorded_calls if c["task_type"] == "RETRIEVAL_QUERY"]
    assert len(query_calls) >= 1
    assert query_calls[0]["text"] == "What is front wing flap rule?"
    assert len(results.chunks) >= 1
    assert results.chunks[0].article_path == "3.2.1"
    assert results.chunks[0].page_start == 1


@pytest.mark.asyncio
async def test_embedding_dimension_consistency_and_mismatch():
    """Verify strict dimension checking: fails on mismatch between repository and embedding."""
    repo = InMemoryKnowledgeRepository(expected_dimensions=768)

    doc = KnowledgeDocument(id="doc_dim", source="doc.txt", content="Sample text")
    from app.knowledge.models import DocumentChunk
    chunk = DocumentChunk(
        id="doc_dim:chunk:0",
        document_id="doc_dim",
        chunk_index=0,
        source="doc.txt",
        content="Sample text",
        metadata={}
    )

    # 64-dimensional vector into 768-dimensional repository must raise KnowledgeRepositoryError
    bad_vectors = [[0.1] * 64]
    with pytest.raises(KnowledgeRepositoryError, match="Expected 768 embedding dimensions, got 64"):
        await repo.store(doc, [chunk], bad_vectors)

    # Correct 768-dimensional vector succeeds
    valid_vectors = [[0.1] * 768]
    await repo.store(doc, [chunk], valid_vectors)

    # Query with mismatched dimensions must fail
    with pytest.raises(KnowledgeRepositoryError, match="Query embedding dimension does not match"):
        await repo.similarity_search([0.1] * 64, top_k=1)
