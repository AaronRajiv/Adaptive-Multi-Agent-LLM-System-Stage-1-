"""Structured document, chunk, and provenance models for Stage 5 retrieval."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    """Text document accepted by the knowledge ingestion pipeline."""

    id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Deterministic text segment retaining its document provenance."""

    id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    source: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    page_start: int | None = Field(default=None, description="Starting 1-indexed page number")
    page_end: int | None = Field(default=None, description="Ending 1-indexed page number")
    article_path: str | None = Field(default=None, description="Extracted regulation article path (e.g., '3.2.1')")
    heading: str | None = Field(default=None, description="Section or article heading")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StoredKnowledgeChunk(DocumentChunk):
    """A document chunk paired with the embedding stored by a repository."""

    embedding: List[float] = Field(..., min_length=1)


class RetrievedChunk(DocumentChunk):
    """A retrieved chunk with similarity score and complete provenance."""

    similarity_score: float
    retrieval_method: str = Field(
        default="dense",
        description="Method used to retrieve chunk: 'hybrid_rrf', 'dense', 'bm25', or 'direct_article_lookup'",
    )
    score: float | None = Field(
        default=None,
        description="Fused rank score or raw retrieval score",
    )


class RetrievedContext(BaseModel):
    """Structured context delivered to a task agent only when retrieval is requested."""

    query: str
    chunks: List[RetrievedChunk] = Field(default_factory=list)
