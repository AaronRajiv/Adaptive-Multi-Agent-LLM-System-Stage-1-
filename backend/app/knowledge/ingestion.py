"""Document ingestion service for the Stage 5 knowledge base."""

from pydantic import BaseModel

from app.knowledge.chunking import Chunker
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.models import DocumentChunk, KnowledgeDocument
from app.knowledge.repository import KnowledgeRepository


class IngestionResult(BaseModel):
    """Outcome of ingesting one document into a knowledge repository."""

    document_id: str
    chunks: list[DocumentChunk]


class DocumentIngestionService:
    """Coordinates document validation, chunking, embedding generation, and storage."""

    def __init__(
        self, chunker: Chunker, embedding_provider: EmbeddingProvider, repository: KnowledgeRepository
    ) -> None:
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.repository = repository

    async def ingest(self, document: KnowledgeDocument) -> IngestionResult:
        if not document.content.strip():
            raise ValueError("Document content cannot be empty or whitespace only.")
        chunks = self.chunker.chunk(document)
        embeddings = await self.embedding_provider.embed_many([chunk.content for chunk in chunks])
        await self.repository.store(document, chunks, embeddings)
        return IngestionResult(document_id=document.id, chunks=chunks)
