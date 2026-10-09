"""
Document management service: decoupled background ingestion, deduplication, and SSE progress.
"""

import asyncio
import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

from app.knowledge.chunking import FixedSizeTextChunker
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.models import DocumentChunk, KnowledgeDocument
from app.knowledge.pdf_extractor import extract_text_from_document
from app.knowledge.repository import KnowledgeRepository
from app.logging_config import get_logger

logger = get_logger("app.knowledge.document_service")


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    label: Optional[str] = None
    sha256: str
    file_size_bytes: int
    status: DocumentStatus
    chunk_count: int = 0
    embedded_chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    label: Optional[str] = None
    sha256: str
    status: DocumentStatus
    message: str


class DocumentEvent(BaseModel):
    document_id: str
    status: DocumentStatus
    chunk_count: int = 0
    embedded_chunk_count: int = 0
    progress_percent: float = 0.0
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DocumentEventBus:
    """Pub/sub queue for streaming SSE progress of document ingestion."""

    def __init__(self, maxsize: int = 200):
        self._subscribers: List[asyncio.Queue[Optional[DocumentEvent]]] = []
        self._history: List[DocumentEvent] = []
        self._lock = asyncio.Lock()
        self._maxsize = maxsize

    async def emit(self, event: DocumentEvent) -> None:
        async with self._lock:
            self._history.append(event)
            for queue in self._subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    async def subscribe(self) -> asyncio.Queue[Optional[DocumentEvent]]:
        queue: asyncio.Queue[Optional[DocumentEvent]] = asyncio.Queue(maxsize=self._maxsize)
        async with self._lock:
            for event in self._history:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    break
            self._subscribers.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[Optional[DocumentEvent]]) -> None:
        async with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    async def close(self) -> None:
        async with self._lock:
            for queue in self._subscribers:
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass


class DocumentService:
    """
    Manages document lifecycle: deduplication, non-blocking background ingestion,
    rate-limited embedding, and status/event tracking.
    """

    def __init__(
        self,
        repository: KnowledgeRepository,
        embedding_provider: EmbeddingProvider,
        chunk_size: int = 1200,
        chunk_overlap: int = 120,
        embedding_batch_size: int = 32,
        embedding_rate_limit_delay_seconds: float = 0.0,
    ):
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.chunker = FixedSizeTextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedding_batch_size = max(1, embedding_batch_size)
        self.embedding_rate_limit_delay = embedding_rate_limit_delay_seconds

        self._documents: Dict[str, DocumentMetadata] = {}
        self._sha256_index: Dict[str, str] = {}  # sha256 -> document_id
        self._event_buses: Dict[str, DocumentEventBus] = {}
        self._background_tasks: Dict[str, asyncio.Task] = {}

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        return self._documents.get(document_id)

    def list_documents(self) -> List[DocumentMetadata]:
        return list(self._documents.values())

    async def get_or_create_event_bus(self, document_id: str) -> DocumentEventBus:
        if document_id not in self._event_buses:
            self._event_buses[document_id] = DocumentEventBus()
        return self._event_buses[document_id]

    async def register_and_start_ingestion(
        self,
        filename: str,
        file_bytes: bytes,
        label: Optional[str] = None,
    ) -> DocumentUploadResponse:
        """
        Deduplicates by SHA-256 and kicks off background ingestion in a worker task.
        Returns immediately with document_id and initial status.
        """
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        # Deduplication check
        if sha256_hash in self._sha256_index:
            existing_id = self._sha256_index[sha256_hash]
            existing_doc = self._documents.get(existing_id)
            if existing_doc and existing_doc.status == DocumentStatus.READY:
                logger.info(
                    "Document with SHA-256 '%s' already exists (%s). Skipping ingestion.",
                    sha256_hash[:12],
                    existing_id,
                )
                return DocumentUploadResponse(
                    document_id=existing_id,
                    filename=existing_doc.filename,
                    label=existing_doc.label or label,
                    sha256=sha256_hash,
                    status=DocumentStatus.READY,
                    message="Document already ingested (deduplicated by SHA-256).",
                )

        doc_id = f"doc_{sha256_hash[:16]}"
        metadata = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            label=label,
            sha256=sha256_hash,
            file_size_bytes=len(file_bytes),
            status=DocumentStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )

        self._documents[doc_id] = metadata
        self._sha256_index[sha256_hash] = doc_id

        event_bus = await self.get_or_create_event_bus(doc_id)
        await event_bus.emit(
            DocumentEvent(
                document_id=doc_id,
                status=DocumentStatus.QUEUED,
                message=f"Document '{filename}' queued for ingestion.",
            )
        )

        # Launch non-blocking background ingestion
        task = asyncio.create_task(
            self._ingest_worker(doc_id, filename, file_bytes, label, event_bus)
        )
        self._background_tasks[doc_id] = task

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=filename,
            label=label,
            sha256=sha256_hash,
            status=DocumentStatus.QUEUED,
            message=f"Document uploaded successfully. Ingestion queued in background.",
        )

    async def _ingest_worker(
        self,
        doc_id: str,
        filename: str,
        file_bytes: bytes,
        label: Optional[str],
        event_bus: DocumentEventBus,
    ) -> None:
        """Background worker that extracts text in a thread executor, chunks, and embeds."""
        metadata = self._documents[doc_id]
        try:
            # 1. Extraction (running blocking PDF parsing in thread executor)
            metadata.status = DocumentStatus.EXTRACTING
            metadata.updated_at = datetime.now(timezone.utc).isoformat()
            await event_bus.emit(
                DocumentEvent(
                    document_id=doc_id,
                    status=DocumentStatus.EXTRACTING,
                    message=f"Extracting text from '{filename}'...",
                    progress_percent=10.0,
                )
            )

            clean_text = await asyncio.to_thread(extract_text_from_document, filename, file_bytes)
            if not clean_text or not clean_text.strip():
                raise RuntimeError(f"No extractable text found in '{filename}'.")

            # 2. Chunking
            knowledge_doc = KnowledgeDocument(
                id=doc_id,
                source=filename,
                content=clean_text,
                metadata={"label": label} if label else {},
            )
            chunks = self.chunker.chunk(knowledge_doc)
            total_chunks = len(chunks)
            metadata.chunk_count = total_chunks
            metadata.status = DocumentStatus.EMBEDDING
            metadata.updated_at = datetime.now(timezone.utc).isoformat()

            await event_bus.emit(
                DocumentEvent(
                    document_id=doc_id,
                    status=DocumentStatus.EMBEDDING,
                    chunk_count=total_chunks,
                    embedded_chunk_count=0,
                    progress_percent=20.0,
                    message=f"Chunked into {total_chunks} chunk(s). Starting embeddings...",
                )
            )

            # 3. Rate-limited and batched embedding
            all_embeddings: List[List[float]] = []
            chunk_texts = [c.content for c in chunks]

            for i in range(0, total_chunks, self.embedding_batch_size):
                batch = chunk_texts[i : i + self.embedding_batch_size]
                batch_vectors = await self.embedding_provider.embed_many(batch)
                all_embeddings.extend(batch_vectors)

                metadata.embedded_chunk_count = len(all_embeddings)
                progress = 20.0 + (len(all_embeddings) / total_chunks) * 75.0
                await event_bus.emit(
                    DocumentEvent(
                        document_id=doc_id,
                        status=DocumentStatus.EMBEDDING,
                        chunk_count=total_chunks,
                        embedded_chunk_count=len(all_embeddings),
                        progress_percent=round(progress, 1),
                        message=f"Embedded {len(all_embeddings)} of {total_chunks} chunks...",
                    )
                )

                if self.embedding_rate_limit_delay > 0:
                    await asyncio.sleep(self.embedding_rate_limit_delay)

            # 4. Storage in repository
            knowledge_doc = KnowledgeDocument(
                id=doc_id,
                source=filename,
                content=clean_text,
            )
            await self.repository.store(knowledge_doc, chunks, all_embeddings)

            # 5. Ready state
            metadata.status = DocumentStatus.READY
            metadata.updated_at = datetime.now(timezone.utc).isoformat()
            await event_bus.emit(
                DocumentEvent(
                    document_id=doc_id,
                    status=DocumentStatus.READY,
                    chunk_count=total_chunks,
                    embedded_chunk_count=total_chunks,
                    progress_percent=100.0,
                    message=f"Document '{filename}' successfully ingested and ready ({total_chunks} chunks).",
                )
            )
            logger.info("Document '%s' ready (%d chunks).", doc_id, total_chunks)

        except Exception as exc:
            logger.error("Document ingestion worker failed for '%s': %s", doc_id, exc, exc_info=True)
            metadata.status = DocumentStatus.FAILED
            metadata.error_message = str(exc)
            metadata.updated_at = datetime.now(timezone.utc).isoformat()
            await event_bus.emit(
                DocumentEvent(
                    document_id=doc_id,
                    status=DocumentStatus.FAILED,
                    chunk_count=metadata.chunk_count,
                    embedded_chunk_count=metadata.embedded_chunk_count,
                    progress_percent=0.0,
                    message=f"Ingestion failed: {exc}",
                )
            )

    async def delete_document(self, document_id: str) -> bool:
        """Deletes a document from metadata index and purges its chunks from knowledge repository."""
        if document_id not in self._documents:
            return False

        doc = self._documents.pop(document_id)
        if doc.sha256 in self._sha256_index:
            del self._sha256_index[doc.sha256]

        if document_id in self._background_tasks:
            task = self._background_tasks.pop(document_id)
            if not task.done():
                task.cancel()

        if document_id in self._event_buses:
            bus = self._event_buses.pop(document_id)
            await bus.close()

        await self.repository.delete_document(document_id)
        logger.info("Deleted document '%s' and purged repository vectors.", document_id)
        return True
