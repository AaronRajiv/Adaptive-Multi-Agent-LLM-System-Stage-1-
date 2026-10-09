"""Document ingestion service for the Stage 5 knowledge base."""

import math
import time
from pydantic import BaseModel

from app.knowledge.chunking import Chunker
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.models import DocumentChunk, KnowledgeDocument
from app.knowledge.repository import KnowledgeRepository


class IngestionResult(BaseModel):
    """Outcome of ingesting one document into a knowledge repository."""

    document_id: str
    chunks: list[DocumentChunk]
    doc_char_count: int = 0
    chunk_count: int = 0
    batch_count: int = 0
    embedding_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    retries: int = 0
    failed: bool = False
    error_message: str | None = None


class DocumentIngestionService:
    """Coordinates document validation, chunking, embedding generation, and storage."""

    def __init__(
        self, chunker: Chunker, embedding_provider: EmbeddingProvider, repository: KnowledgeRepository
    ) -> None:
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.repository = repository

    async def ingest(self, document: KnowledgeDocument) -> IngestionResult:
        start_time = time.perf_counter()
        doc_char_count = len(document.content or "")
        if not document.content.strip():
            raise ValueError("Document content cannot be empty or whitespace only.")

        # Check whether the document is already ingested to prevent duplicate chunks
        if await self.repository.has_document(document.id):
            total_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return IngestionResult(
                document_id=document.id,
                chunks=[],
                doc_char_count=doc_char_count,
                chunk_count=0,
                batch_count=0,
                embedding_duration_ms=0.0,
                total_duration_ms=total_duration_ms,
                failed=False,
            )

        chunks = self.chunker.chunk(document)
        chunk_count = len(chunks)

        batch_size = getattr(self.embedding_provider, "batch_size", 64)
        batch_count = math.ceil(chunk_count / batch_size) if chunk_count > 0 else 0

        try:
            embed_start = time.perf_counter()
            embeddings = await self.embedding_provider.embed_many([chunk.content for chunk in chunks])
            embed_duration_ms = round((time.perf_counter() - embed_start) * 1000, 2)

            await self.repository.store(document, chunks, embeddings)
            total_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return IngestionResult(
                document_id=document.id,
                chunks=chunks,
                doc_char_count=doc_char_count,
                chunk_count=chunk_count,
                batch_count=batch_count,
                embedding_duration_ms=embed_duration_ms,
                total_duration_ms=total_duration_ms,
                failed=False,
            )
        except Exception as exc:
            # Safe rollback: purge any partially written state so an incomplete index is not exposed
            try:
                await self.repository.delete_document(document.id)
            except Exception:
                pass
            raise

