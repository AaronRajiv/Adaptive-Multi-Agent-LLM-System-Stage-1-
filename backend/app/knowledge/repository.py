"""Knowledge repository abstractions with in-memory and optional pgvector implementations."""

import json
import math
from abc import ABC, abstractmethod
from typing import Dict, List

from app.knowledge.models import (
    DocumentChunk,
    KnowledgeDocument,
    RetrievedChunk,
    StoredKnowledgeChunk,
)


class KnowledgeRepositoryError(RuntimeError):
    """Raised when knowledge storage or similarity search cannot be completed."""


class KnowledgeRepository(ABC):
    """Storage boundary for knowledge documents, vectors, and similarity search."""

    @abstractmethod
    async def store(
        self,
        document: KnowledgeDocument,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """Persist one document's chunks and aligned embeddings."""

    @abstractmethod
    async def similarity_search(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        """Return the top-k most similar chunks with provenance."""

    async def has_document(self, document_id: str) -> bool:
        """Check whether a document is already stored in the repository."""
        return False

    async def delete_document(self, document_id: str) -> None:
        """Remove a document and all its associated chunks from the repository."""
        pass


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """Deterministic in-memory repository for tests and local development."""

    def __init__(self) -> None:
        self.documents: Dict[str, KnowledgeDocument] = {}
        self.chunks: Dict[str, StoredKnowledgeChunk] = {}

    async def has_document(self, document_id: str) -> bool:
        """Check whether a document is already stored in-memory."""
        return document_id in self.documents

    async def delete_document(self, document_id: str) -> None:
        """Delete document and its chunks to avoid partial or dirty states."""
        if document_id in self.documents:
            del self.documents[document_id]
        self.chunks = {
            cid: chunk for cid, chunk in self.chunks.items() if chunk.document_id != document_id
        }

    async def store(
        self,
        document: KnowledgeDocument,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise KnowledgeRepositoryError("Chunk and embedding counts must match.")
        if document.id in self.documents:
            raise KnowledgeRepositoryError(f"Document '{document.id}' is already stored.")
        for chunk, embedding in zip(chunks, embeddings):
            if chunk.document_id != document.id:
                raise KnowledgeRepositoryError("Chunk document ID does not match the stored document.")
            if not embedding:
                raise KnowledgeRepositoryError("Embeddings cannot be empty.")
            if chunk.id in self.chunks:
                raise KnowledgeRepositoryError(f"Chunk '{chunk.id}' is already stored.")

        self.documents[document.id] = document
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks[chunk.id] = StoredKnowledgeChunk(**chunk.model_dump(), embedding=embedding)

    async def similarity_search(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        if top_k < 1:
            raise KnowledgeRepositoryError("top_k must be at least 1.")
        if not query_embedding:
            raise KnowledgeRepositoryError("Query embedding cannot be empty.")
        if not self.chunks:
            return []

        results: List[RetrievedChunk] = []
        for stored in self.chunks.values():
            if len(stored.embedding) != len(query_embedding):
                raise KnowledgeRepositoryError("Query embedding dimension does not match stored embeddings.")
            score = _cosine_similarity(query_embedding, stored.embedding)
            results.append(RetrievedChunk(**stored.model_dump(exclude={"embedding"}), similarity_score=score))
        return sorted(results, key=lambda chunk: chunk.similarity_score, reverse=True)[:top_k]


class PostgresPgvectorKnowledgeRepository(KnowledgeRepository):
    """Optional persistent repository for PostgreSQL with pgvector; not used by normal tests."""

    def __init__(self, database_url: str, embedding_dimensions: int):
        if not database_url:
            raise ValueError("KNOWLEDGE_DATABASE_URL is required for the pgvector repository.")
        if embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be at least 1.")
        self.database_url = database_url
        self.embedding_dimensions = embedding_dimensions

    async def store(
        self,
        document: KnowledgeDocument,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise KnowledgeRepositoryError("Chunk and embedding counts must match.")
        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO knowledge_documents (id, source, content, metadata)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET source = EXCLUDED.source,
                    content = EXCLUDED.content, metadata = EXCLUDED.metadata
                    """,
                    (document.id, document.source, document.content, json.dumps(document.metadata)),
                )
                for chunk, embedding in zip(chunks, embeddings):
                    self._validate_embedding(embedding)
                    await cursor.execute(
                        """
                        INSERT INTO knowledge_chunks
                        (id, document_id, chunk_index, source, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding
                        """,
                        (
                            chunk.id,
                            chunk.document_id,
                            chunk.chunk_index,
                            chunk.source,
                            chunk.content,
                            json.dumps(chunk.metadata),
                            _vector_literal(embedding),
                        ),
                    )

    async def has_document(self, document_id: str) -> bool:
        """Check whether a document is already stored in Postgres."""
        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT 1 FROM knowledge_documents WHERE id = %s LIMIT 1",
                    (document_id,),
                )
                row = await cursor.fetchone()
                return row is not None

    async def delete_document(self, document_id: str) -> None:
        """Delete a document and all associated chunks from Postgres."""
        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DELETE FROM knowledge_chunks WHERE document_id = %s",
                    (document_id,),
                )
                await cursor.execute(
                    "DELETE FROM knowledge_documents WHERE id = %s",
                    (document_id,),
                )

    async def similarity_search(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        if top_k < 1:
            raise KnowledgeRepositoryError("top_k must be at least 1.")
        self._validate_embedding(query_embedding)
        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT id, document_id, chunk_index, source, content, metadata,
                    1 - (embedding <=> %s::vector) AS similarity_score
                    FROM knowledge_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (_vector_literal(query_embedding), _vector_literal(query_embedding), top_k),
                )
                rows = await cursor.fetchall()
        return [
            RetrievedChunk(
                id=row[0], document_id=row[1], chunk_index=row[2], source=row[3],
                content=row[4], metadata=row[5], similarity_score=float(row[6]),
            )
            for row in rows
        ]

    async def _connect(self):
        try:
            from psycopg import AsyncConnection
        except ImportError as exc:
            raise KnowledgeRepositoryError(
                "psycopg is required for PostgreSQL/pgvector. Install backend requirements first."
            ) from exc
        return await AsyncConnection.connect(self.database_url)

    def _validate_embedding(self, embedding: List[float]) -> None:
        if len(embedding) != self.embedding_dimensions:
            raise KnowledgeRepositoryError(
                f"Expected {self.embedding_dimensions} embedding dimensions, got {len(embedding)}."
            )


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))
    if not left_magnitude or not right_magnitude:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_magnitude * right_magnitude)


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"
