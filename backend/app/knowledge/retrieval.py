"""Retrieval service that coordinates query embeddings and provenance-preserving search."""

from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.models import RetrievedContext
from app.knowledge.repository import KnowledgeRepository
from app.logging_config import get_logger

logger = get_logger("app.knowledge.retrieval")


class RetrievalError(RuntimeError):
    """Raised when a retrieval request is invalid or a retrieval dependency fails."""


class RetrievalService:
    """Provider-agnostic retrieval coordinator; it does not own storage or chunking."""

    def __init__(self, embedding_provider: EmbeddingProvider, repository: KnowledgeRepository):
        self.embedding_provider = embedding_provider
        self.repository = repository

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float | None = None,
        document_ids: list[str] | None = None,
    ) -> RetrievedContext:
        if not query or not query.strip():
            raise RetrievalError("Retrieval query cannot be empty.")
        if top_k < 1:
            raise RetrievalError("top_k must be at least 1.")
        if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
            raise RetrievalError("min_similarity must be between -1.0 and 1.0.")

        try:
            try:
                query_embedding = await self.embedding_provider.embed(
                    query, task_type="RETRIEVAL_QUERY"
                )
            except TypeError:
                query_embedding = await self.embedding_provider.embed(query)
            if document_ids is not None:
                chunks = await self.repository.similarity_search(
                    query_embedding, top_k, document_ids=document_ids
                )
            else:
                chunks = await self.repository.similarity_search(query_embedding, top_k)
        except Exception as exc:
            logger.error("Retrieval failed for query '%s': %s", query, str(exc), exc_info=True)
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        if min_similarity is not None:
            chunks = [chunk for chunk in chunks if chunk.similarity_score >= min_similarity]

        logger.info(
            "Retrieved %d chunk(s) for query '%s' (top_k=%d, doc_filter=%s)",
            len(chunks),
            query,
            top_k,
            document_ids,
        )
        return RetrievedContext(query=query, chunks=chunks)
