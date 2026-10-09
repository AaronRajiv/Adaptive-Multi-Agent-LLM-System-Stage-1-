"""
Hybrid Retrieval Engine with Dense Vectors, BM25 Keyword Search,
Direct Article-Number Lookup, and Reciprocal Rank Fusion (RRF).
"""

import re
from typing import Dict, List, Optional, Set

from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.models import RetrievedChunk, RetrievedContext
from app.knowledge.repository import KnowledgeRepository
from app.logging_config import get_logger

logger = get_logger("app.knowledge.retrieval")

# Matches patterns like "Article 3.2.1", "section C3.2", "art 14.1.2", or standalone "3.2.1"
ARTICLE_QUERY_PATTERN = re.compile(
    r"(?:article|section|art\.?)\s*([A-Z0-9]+(?:\.[A-Z0-9]+)*)|\b([A-Z]?\d+(?:\.\d+)+)\b",
    re.IGNORECASE,
)


def extract_article_query(query: str) -> Optional[str]:
    """Detects if a query targets a specific regulation article number."""
    if not query:
        return None
    match = ARTICLE_QUERY_PATTERN.search(query)
    if match:
        return match.group(1) or match.group(2)
    return None


class RetrievalError(RuntimeError):
    """Raised when a retrieval request is invalid or a retrieval dependency fails."""


class RetrievalService:
    """
    Coordinates hybrid retrieval across:
    1. Direct Article Lookup (exact regulation article resolution)
    2. Dense Vector Similarity Search
    3. BM25 Lexical Keyword Search
    Fuses all ranked streams with Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        repository: KnowledgeRepository,
        default_top_k: int = 10,
        rrf_k: int = 60,
    ):
        self.embedding_provider = embedding_provider
        self.repository = repository
        self.default_top_k = max(1, default_top_k)
        self.rrf_k = max(1, rrf_k)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: Optional[float] = None,
        document_ids: Optional[List[str]] = None,
    ) -> RetrievedContext:
        """
        Execute hybrid retrieval for a single query using dense vector, BM25,
        direct article lookup, and Reciprocal Rank Fusion.
        """
        if not query or not query.strip():
            raise RetrievalError("Retrieval query cannot be empty.")
        if top_k < 1:
            raise RetrievalError("top_k must be at least 1.")
        if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
            raise RetrievalError("min_similarity must be between -1.0 and 1.0.")

        clean_query = query.strip()
        fetch_k = max(top_k * 2, 20)

        # 1. Direct Article-Number Lookup
        direct_chunks: List[RetrievedChunk] = []
        target_article = extract_article_query(clean_query)
        if target_article:
            try:
                direct_chunks = await self.repository.lookup_by_article(
                    target_article, document_ids=document_ids
                )
                if direct_chunks:
                    logger.info(
                        "Direct article lookup found %d chunk(s) for article '%s'",
                        len(direct_chunks),
                        target_article,
                    )
            except Exception as exc:
                logger.warning("Direct article lookup failed for '%s': %s", target_article, exc)

        # 2. Dense Vector Similarity Search
        dense_chunks: List[RetrievedChunk] = []
        try:
            import time
            from app.telemetry.metrics import CallMetric, compute_estimated_cost, global_metrics_collector

            t_emb = time.perf_counter()
            try:
                query_embedding = await self.embedding_provider.embed(
                    clean_query, task_type="RETRIEVAL_QUERY"
                )
            except TypeError:
                query_embedding = await self.embedding_provider.embed(clean_query)
            emb_lat = round((time.perf_counter() - t_emb) * 1000, 2)

            emb_tokens = max(1, len(clean_query.split()))
            emb_model = getattr(self.embedding_provider, "model", "gemini-embedding-001")
            run_id = getattr(self, "current_run_id", "retrieval") or "retrieval"
            global_metrics_collector.record_call(
                CallMetric(
                    run_id=run_id,
                    call_type="embedding",
                    stage="RAG",
                    model=emb_model,
                    tokens_in=emb_tokens,
                    tokens_out=0,
                    total_tokens=emb_tokens,
                    latency_ms=emb_lat,
                    estimated_cost=compute_estimated_cost(emb_model, emb_tokens, 0),
                    metadata={"query": clean_query[:100]},
                )
            )

            dense_chunks = await self.repository.similarity_search(
                query_embedding, top_k=fetch_k, document_ids=document_ids
            )
        except Exception as exc:
            logger.error("Dense vector search failed for query '%s': %s", clean_query, exc, exc_info=True)
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        # 3. BM25 Lexical Keyword Search
        bm25_chunks: List[RetrievedChunk] = []
        try:
            bm25_chunks = await self.repository.bm25_search(
                clean_query, top_k=fetch_k, document_ids=document_ids
            )
        except Exception as exc:
            logger.warning("BM25 search failed for query '%s': %s", clean_query, exc)

        # 4. Reciprocal Rank Fusion (RRF)
        fused_chunks = self._fuse_rrf(
            direct_chunks=direct_chunks,
            dense_chunks=dense_chunks,
            bm25_chunks=bm25_chunks,
            k=self.rrf_k,
            top_k=top_k,
        )

        if min_similarity is not None:
            fused_chunks = [c for c in fused_chunks if c.similarity_score >= min_similarity]

        logger.info(
            "Hybrid retrieval returned %d chunk(s) for query '%s' (direct=%d, dense=%d, bm25=%d)",
            len(fused_chunks),
            clean_query,
            len(direct_chunks),
            len(dense_chunks),
            len(bm25_chunks),
        )
        return RetrievedContext(query=clean_query, chunks=fused_chunks)

    async def retrieve_multi(
        self,
        queries: List[str],
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
        budget: Optional[int] = None,
    ) -> RetrievedContext:
        """
        Execute multi-query retrieval with cross-query deduplication and budget enforcement.
        """
        if not queries:
            raise RetrievalError("Query list cannot be empty.")

        merged_chunks_map: Dict[str, RetrievedChunk] = {}
        merged_scores: Dict[str, float] = {}
        all_queries_text = " | ".join(q.strip() for q in queries if q.strip())

        for q in queries:
            if not q.strip():
                continue
            ctx = await self.retrieve(q.strip(), top_k=top_k, document_ids=document_ids)
            for chunk in ctx.chunks:
                cid = chunk.id
                current_score = chunk.score if chunk.score is not None else chunk.similarity_score
                if cid not in merged_chunks_map:
                    merged_chunks_map[cid] = chunk
                    merged_scores[cid] = current_score
                else:
                    # Deduplicate: take max score across queries or boost
                    merged_scores[cid] = max(merged_scores[cid], current_score) + 0.05 * min(
                        merged_scores[cid], current_score
                    )

        # Sort deduplicated chunks by score descending
        sorted_chunks = sorted(
            merged_chunks_map.values(),
            key=lambda c: merged_scores.get(c.id, 0.0),
            reverse=True,
        )

        # Enforce budget cap if specified
        if budget is not None and budget > 0:
            sorted_chunks = sorted_chunks[:budget]

        return RetrievedContext(query=all_queries_text, chunks=sorted_chunks)

    def _fuse_rrf(
        self,
        direct_chunks: List[RetrievedChunk],
        dense_chunks: List[RetrievedChunk],
        bm25_chunks: List[RetrievedChunk],
        k: int,
        top_k: int,
    ) -> List[RetrievedChunk]:
        """
        Fuses ranked candidate lists using Reciprocal Rank Fusion (RRF).
        Assigns weights: direct lookup (2.0), dense (1.0), bm25 (1.0).
        """
        rrf_scores: Dict[str, float] = {}
        chunk_objects: Dict[str, RetrievedChunk] = {}
        retrieval_methods: Dict[str, Set[str]] = {}

        dense_scores: Dict[str, float] = {}

        # 1. Direct Article Lookup Stream (highest priority)
        for rank, chunk in enumerate(direct_chunks):
            cid = chunk.id
            chunk_objects[cid] = chunk
            retrieval_methods.setdefault(cid, set()).add("direct_article_lookup")
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (2.0 / (k + rank + 1))

        # 2. Dense Vector Stream
        for rank, chunk in enumerate(dense_chunks):
            cid = chunk.id
            chunk_objects[cid] = chunk
            dense_scores[cid] = chunk.similarity_score
            retrieval_methods.setdefault(cid, set()).add("dense")
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))

        # 3. BM25 Stream (do not overwrite chunk_objects if already present from dense)
        for rank, chunk in enumerate(bm25_chunks):
            cid = chunk.id
            if cid not in chunk_objects:
                chunk_objects[cid] = chunk
            retrieval_methods.setdefault(cid, set()).add("bm25")
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))

        # Build fused result list
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        fused_results: List[RetrievedChunk] = []
        for cid in sorted_cids:
            original = chunk_objects[cid]
            methods = retrieval_methods[cid]
            if "direct_article_lookup" in methods:
                method_label = "direct_article_lookup"
            elif len(methods) > 1:
                method_label = "hybrid_rrf"
            else:
                method_label = next(iter(methods))

            score_val = round(rrf_scores[cid], 4)
            sim_score = dense_scores.get(cid, original.similarity_score if original.similarity_score is not None else score_val)
            fused_chunk = RetrievedChunk(
                id=original.id,
                document_id=original.document_id,
                chunk_index=original.chunk_index,
                source=original.source,
                content=original.content,
                page_start=original.page_start,
                page_end=original.page_end,
                article_path=original.article_path,
                heading=original.heading,
                similarity_score=sim_score,
                score=score_val,
                retrieval_method=method_label,
                metadata=dict(original.metadata),
            )
            fused_results.append(fused_chunk)

        return fused_results
