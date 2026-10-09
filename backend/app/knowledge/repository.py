"""Knowledge repository abstractions with in-memory and optional pgvector implementations."""

import json
import math
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

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
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Return the top-k most similar chunks with provenance, optionally filtered to document_ids."""

    async def bm25_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Return top-k chunks matching the query using BM25 keyword scoring."""
        return []

    async def lookup_by_article(
        self,
        article_path: str,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Return chunks matching an exact or normalized article path (e.g. '3.2.1')."""
        return []

    async def has_document(self, document_id: str) -> bool:
        """Check whether a document is already stored in the repository."""
        return False

    async def delete_document(self, document_id: str) -> None:
        """Remove a document and all its associated chunks from the repository."""
        pass


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """Deterministic in-memory repository for tests and local development."""

    def __init__(self, expected_dimensions: Optional[int] = None) -> None:
        self.documents: Dict[str, KnowledgeDocument] = {}
        self.chunks: Dict[str, StoredKnowledgeChunk] = {}
        self.expected_dimensions = expected_dimensions

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
            if self.expected_dimensions is not None and len(embedding) != self.expected_dimensions:
                raise KnowledgeRepositoryError(
                    f"Expected {self.expected_dimensions} embedding dimensions, got {len(embedding)}."
                )
            if chunk.id in self.chunks:
                raise KnowledgeRepositoryError(f"Chunk '{chunk.id}' is already stored.")

        self.documents[document.id] = document
        for chunk, embedding in zip(chunks, embeddings):
            self.chunks[chunk.id] = StoredKnowledgeChunk(**chunk.model_dump(), embedding=embedding)

    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        if top_k < 1:
            raise KnowledgeRepositoryError("top_k must be at least 1.")
        if not query_embedding:
            raise KnowledgeRepositoryError("Query embedding cannot be empty.")
        if not self.chunks:
            return []

        results: List[RetrievedChunk] = []
        for stored in self.chunks.values():
            if document_ids is not None and stored.document_id not in document_ids:
                continue
            if len(stored.embedding) != len(query_embedding):
                raise KnowledgeRepositoryError("Query embedding dimension does not match stored embeddings.")
            score = _cosine_similarity(query_embedding, stored.embedding)
            results.append(
                RetrievedChunk(
                    **stored.model_dump(exclude={"embedding"}),
                    similarity_score=score,
                    score=score,
                    retrieval_method="dense",
                )
            )
        return sorted(results, key=lambda chunk: chunk.similarity_score, reverse=True)[:top_k]

    async def bm25_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        if top_k < 1:
            raise KnowledgeRepositoryError("top_k must be at least 1.")
        if not query or not query.strip():
            return []

        candidates = [
            c for c in self.chunks.values()
            if document_ids is None or c.document_id in document_ids
        ]
        if not candidates:
            return []

        from app.knowledge.bm25 import BM25Okapi, tokenize

        corpus = [
            tokenize(f"{c.article_path or ''} {c.heading or ''} {c.content} {c.metadata.get('embedding_text', '')}")
            for c in candidates
        ]
        bm25 = BM25Okapi(corpus)
        scored_candidates = bm25.search(query, candidates, top_k=top_k)

        return [
            RetrievedChunk(
                **chunk.model_dump(exclude={"embedding"}),
                similarity_score=score,
                score=score,
                retrieval_method="bm25",
            )
            for chunk, score in scored_candidates
        ]

    async def lookup_by_article(
        self,
        article_path: str,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        clean_path = article_path.strip().lower()
        if not clean_path:
            return []

        # Remove leading "article " or "section " prefix for normalized comparison
        normalized_target = re.sub(r"^(?:article|section)\s*", "", clean_path).strip()

        results: List[RetrievedChunk] = []
        for stored in self.chunks.values():
            if document_ids is not None and stored.document_id not in document_ids:
                continue
            chunk_art = stored.article_path or stored.metadata.get("article_path", "")
            if not chunk_art:
                continue
            clean_chunk_art = chunk_art.strip().lower()
            normalized_chunk = re.sub(r"^(?:article|section)\s*", "", clean_chunk_art).strip()

            if (
                clean_chunk_art == clean_path
                or normalized_chunk == normalized_target
                or normalized_chunk.startswith(f"{normalized_target}.")
            ):
                results.append(
                    RetrievedChunk(
                        **stored.model_dump(exclude={"embedding"}),
                        similarity_score=1.0,
                        score=1.0,
                        retrieval_method="direct_article_lookup",
                    )
                )

        return results


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

    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        if top_k < 1:
            raise KnowledgeRepositoryError("top_k must be at least 1.")
        self._validate_embedding(query_embedding)
        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                query_sql = """
                    SELECT id, document_id, chunk_index, source, content, metadata,
                    1 - (embedding <=> %s::vector) AS similarity_score
                    FROM knowledge_chunks
                """
                params: list = [_vector_literal(query_embedding)]
                if document_ids is not None:
                    query_sql += " WHERE document_id = ANY(%s)"
                    params.append(document_ids)
                query_sql += """
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                params.extend([_vector_literal(query_embedding), top_k])
                await cursor.execute(query_sql, tuple(params))
                rows = await cursor.fetchall()
        return [
            RetrievedChunk(
                id=row[0], document_id=row[1], chunk_index=row[2], source=row[3],
                content=row[4], metadata=row[5], similarity_score=float(row[6]),
                score=float(row[6]), retrieval_method="dense",
            )
            for row in rows
        ]

    async def bm25_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        if top_k < 1:
            raise KnowledgeRepositoryError("top_k must be at least 1.")
        if not query or not query.strip():
            return []

        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                query_sql = """
                    SELECT id, document_id, chunk_index, source, content, metadata,
                    ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) AS bm25_score
                    FROM knowledge_chunks
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                """
                params: list = [query, query]
                if document_ids is not None:
                    query_sql += " AND document_id = ANY(%s)"
                    params.append(document_ids)
                query_sql += """
                    ORDER BY ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) DESC
                    LIMIT %s
                """
                params.extend([query, top_k])
                await cursor.execute(query_sql, tuple(params))
                rows = await cursor.fetchall()
        return [
            RetrievedChunk(
                id=row[0], document_id=row[1], chunk_index=row[2], source=row[3],
                content=row[4], metadata=row[5], similarity_score=float(row[6]),
                score=float(row[6]), retrieval_method="bm25",
            )
            for row in rows
        ]

    async def lookup_by_article(
        self,
        article_path: str,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        clean_path = article_path.strip()
        if not clean_path:
            return []

        connection = await self._connect()
        async with connection:
            async with connection.cursor() as cursor:
                query_sql = """
                    SELECT id, document_id, chunk_index, source, content, metadata,
                    1.0 AS score
                    FROM knowledge_chunks
                    WHERE (metadata->>'article_path' ILIKE %s OR metadata->>'article_path' = %s)
                """
                params: list = [f"%{clean_path}%", clean_path]
                if document_ids is not None:
                    query_sql += " AND document_id = ANY(%s)"
                    params.append(document_ids)
                query_sql += " ORDER BY chunk_index ASC LIMIT 10"
                await cursor.execute(query_sql, tuple(params))
                rows = await cursor.fetchall()
        return [
            RetrievedChunk(
                id=row[0], document_id=row[1], chunk_index=row[2], source=row[3],
                content=row[4], metadata=row[5], similarity_score=float(row[6]),
                score=float(row[6]), retrieval_method="direct_article_lookup",
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
