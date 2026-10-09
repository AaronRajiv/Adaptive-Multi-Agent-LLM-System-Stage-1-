"""Deterministic, replaceable text chunking strategies."""

from abc import ABC, abstractmethod
from typing import List

from app.knowledge.models import DocumentChunk, KnowledgeDocument


class Chunker(ABC):
    """Converts one knowledge document into provenance-preserving chunks."""

    @abstractmethod
    def chunk(self, document: KnowledgeDocument) -> List[DocumentChunk]:
        """Create deterministic chunks for a document."""


class FixedSizeTextChunker(Chunker):
    """Character-window chunker with bounded overlap for the Stage 5 baseline."""

    def __init__(self, chunk_size: int = 1200, overlap: int | None = None):
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if overlap is None:
            overlap = min(120, int(chunk_size * 0.1))
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: KnowledgeDocument) -> List[DocumentChunk]:
        content = document.content.strip()
        if not content:
            raise ValueError("Document content cannot be empty or whitespace only.")

        chunks: List[DocumentChunk] = []
        start = 0
        index = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        id=f"{document.id}:chunk:{index}",
                        document_id=document.id,
                        chunk_index=index,
                        source=document.source,
                        content=chunk_text,
                        metadata=dict(document.metadata),
                    )
                )
                index += 1
            if end == len(content):
                break
            start = end - self.overlap
        return chunks
