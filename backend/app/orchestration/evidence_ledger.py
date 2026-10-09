"""
Shared Evidence Ledger for Phase 5 Multi-Agent Orchestration.

Maintains a canonical index of all retrieved chunks, metadata, and citation strings
for a pipeline run, ensuring Analysts and Synthesizers reason over verbatim grounding
evidence rather than solely lossy downstream paraphrases.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from app.knowledge.models import RetrievedChunk


def format_citation(
    document_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    document_label: Optional[str] = None,
) -> str:
    """
    Format standard citation string: [doc_label or doc_id, art X.Y.Z, p.N].
    Example: [FIA 2025, art 3.4.1, p.24] or [doc-1, art 12.1, p.5-6]
    """
    meta = metadata or {}
    label = document_label or meta.get("label") or meta.get("document_label") or document_id

    # Clean document label if it's a long filename or path
    if "/" in label:
        label = label.split("/")[-1]
    if label.lower().endswith(".pdf"):
        label = label[:-4]

    article_path = meta.get("article_path")
    page_start = meta.get("page_start")
    page_end = meta.get("page_end")
    page = meta.get("page")

    parts = [label]

    if article_path:
        parts.append(f"art {article_path}")

    if page_start is not None and page_end is not None:
        if page_start == page_end:
            parts.append(f"p.{page_start}")
        else:
            parts.append(f"p.{page_start}-{page_end}")
    elif page is not None:
        parts.append(f"p.{page}")

    return f"[{', '.join(parts)}]"


class EvidenceItem(BaseModel):
    """A single canonical evidence entry in the ledger."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Identifier of the source document")
    document_label: str = Field(..., description="Human-friendly document label or filename")
    article_path: Optional[str] = Field(default=None, description="Regulatory article path (e.g. '3.4.1')")
    page_start: Optional[int] = Field(default=None, description="Starting 1-indexed page")
    page_end: Optional[int] = Field(default=None, description="Ending 1-indexed page")
    heading: Optional[str] = Field(default=None, description="Article heading or title")
    citation: str = Field(..., description="Standardized citation string, e.g. [FIA 2025, art 3.4.1, p.24]")
    text: str = Field(..., description="Verbatim chunk content text")
    similarity_score: Optional[float] = Field(default=None, description="Cosine similarity score if vector retrieved")
    score: Optional[float] = Field(default=None, description="RRF or fused retrieval score")
    retrieval_method: Optional[str] = Field(default=None, description="dense, bm25, or direct_article")


class EvidenceLedger(BaseModel):
    """
    Shared repository of verified grounding evidence for an orchestration run.
    Passed directly to Researcher, Analyst, Evaluator, and Synthesizer agents.
    """

    items: Dict[str, EvidenceItem] = Field(
        default_factory=dict,
        description="Map of chunk_id to EvidenceItem"
    )

    def add_chunk(
        self,
        chunk: RetrievedChunk,
        document_label: Optional[str] = None,
    ) -> EvidenceItem:
        """Add a retrieved chunk to the ledger, deduplicating by chunk.id."""
        if chunk.id in self.items:
            existing = self.items[chunk.id]
            # Update score if higher
            if chunk.score is not None and (existing.score is None or chunk.score > existing.score):
                existing.score = chunk.score
            return existing

        meta = chunk.metadata or {}
        label = document_label or meta.get("label") or meta.get("document_label") or chunk.source or chunk.document_id
        citation = format_citation(chunk.document_id, meta, document_label=label)

        item = EvidenceItem(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_label=label,
            article_path=meta.get("article_path"),
            page_start=meta.get("page_start") if isinstance(meta.get("page_start"), int) else None,
            page_end=meta.get("page_end") if isinstance(meta.get("page_end"), int) else None,
            heading=meta.get("heading"),
            citation=citation,
            text=chunk.content.strip(),
            similarity_score=chunk.similarity_score,
            score=chunk.score,
            retrieval_method=meta.get("retrieval_method"),
        )
        self.items[chunk.id] = item
        return item

    def add_chunks(
        self,
        chunks: List[RetrievedChunk],
        document_label: Optional[str] = None,
    ) -> List[EvidenceItem]:
        """Add a batch of retrieved chunks to the ledger."""
        return [self.add_chunk(c, document_label=document_label) for c in chunks]

    def get(self, chunk_id: str) -> Optional[EvidenceItem]:
        """Look up an evidence item by chunk ID."""
        return self.items.get(chunk_id)

    def find_by_article(self, article_path: str) -> List[EvidenceItem]:
        """Find all evidence items matching an article path."""
        norm = article_path.strip().lower()
        return [
            item for item in self.items.values()
            if item.article_path and item.article_path.strip().lower() == norm
        ]

    def format_for_prompt(self, max_chunks: Optional[int] = None) -> str:
        """
        Format the entire ledger into a structured, numbered section suitable for agent prompts.
        """
        if not self.items:
            return "No verified grounding evidence in ledger."

        entries = list(self.items.values())
        if max_chunks is not None:
            entries = entries[:max_chunks]

        blocks = []
        for i, item in enumerate(entries, start=1):
            header = f"[Evidence #{i} | ID: {item.chunk_id} | Citation: {item.citation}]"
            if item.heading:
                header += f" (Heading: {item.heading})"
            blocks.append(f"{header}\nContent:\n{item.text}")

        return "\n\n".join(blocks)

    def all_citations(self) -> List[str]:
        """List all unique formatted citation strings in the ledger."""
        return list({item.citation for item in self.items.values()})

    def __len__(self) -> int:
        return len(self.items)
