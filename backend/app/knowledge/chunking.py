"""Deterministic, replaceable text chunking strategies."""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from app.knowledge.models import DocumentChunk, KnowledgeDocument

ARTICLE_PATTERN = re.compile(
    r"^(?:ARTICLE\s+([A-Z0-9]+(?:\.[A-Z0-9]+)*)|(?:Article\s+)?([A-Z]?\d+(?:\.\d+)+))\b[:\.\-\s]*(.*)$",
    re.IGNORECASE,
)


def _estimate_tokens(text: str) -> int:
    """Heuristic token estimation: ~4 characters per token."""
    return max(1, len(text) // 4)


class Chunker(ABC):
    """Converts one knowledge document into provenance-preserving chunks."""

    @abstractmethod
    def chunk(self, document: KnowledgeDocument) -> List[DocumentChunk]:
        """Create deterministic chunks for a document."""


class FixedSizeTextChunker(Chunker):
    """Character-window chunker with bounded overlap for the baseline."""

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


class StructureAwareChunker(Chunker):
    """
    Regulation structure-aware chunker:
    - Identifies article numbering ('3.2.1', 'ARTICLE 3', 'Article C3.2') and headings
    - Chunks on article/paragraph boundaries targeting 600-1000 tokens
    - Bounded overlap only when an article exceeds max token cap (default 1200 tokens)
    - Records provenance: document_id, document_label, page_start/page_end, article_path, heading, chunk_index
    - Attaches prepended header '[Doc | Article path | Heading]' in metadata['embedding_text']
    """

    def __init__(
        self,
        target_tokens: int = 800,
        max_tokens: int = 1200,
        overlap_tokens: int = 150,
    ):
        if target_tokens < 50:
            raise ValueError("target_tokens must be at least 50.")
        if max_tokens < target_tokens:
            raise ValueError("max_tokens must be greater than or equal to target_tokens.")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be non-negative and less than max_tokens.")

        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, document: KnowledgeDocument) -> List[DocumentChunk]:
        content = document.content.strip()
        if not content:
            raise ValueError("Document content cannot be empty or whitespace only.")

        # Document pages: either passed explicitly in metadata or treated as single/split text
        pages: List[Tuple[int, str]] = document.metadata.get("pages", [])
        if not pages:
            pages = [(1, content)]

        doc_label = document.metadata.get("label") or document.source

        # 1. Parse text stream into structured article sections
        sections = self._parse_article_sections(pages)

        # 2. Convert article sections into bounded chunks
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for sec in sections:
            sec_text = sec["content"].strip()
            if not sec_text:
                continue

            sec_tokens = _estimate_tokens(sec_text)

            if sec_tokens <= self.max_tokens:
                # Article fits within cap: keep as single coherent semantic unit
                chunk = self._create_chunk(
                    document=document,
                    doc_label=doc_label,
                    chunk_index=chunk_idx,
                    content=sec_text,
                    page_start=sec["page_start"],
                    page_end=sec["page_end"],
                    article_path=sec["article_path"],
                    heading=sec["heading"],
                )
                chunks.append(chunk)
                chunk_idx += 1
            else:
                # Article exceeds max cap: split into sub-chunks with bounded overlap
                sub_chunks = self._split_large_section(sec, document, doc_label, chunk_idx)
                for sc in sub_chunks:
                    chunks.append(sc)
                    chunk_idx += 1

        # Fallback if no sections were produced
        if not chunks and content:
            fallback = self._create_chunk(
                document=document,
                doc_label=doc_label,
                chunk_index=0,
                content=content[: self.max_tokens * 4],
                page_start=1,
                page_end=1,
                article_path="General",
                heading="",
            )
            chunks.append(fallback)

        return chunks

    def _parse_article_sections(self, pages: List[Tuple[int, str]]) -> List[Dict[str, Any]]:
        """Parses page text into a sequence of article sections with active headings and page ranges."""
        sections: List[Dict[str, Any]] = []

        curr_article: str = "General"
        curr_heading: str = ""
        curr_page_start: int = pages[0][0] if pages else 1
        curr_page_end: int = curr_page_start
        curr_lines: List[str] = []

        def flush_current():
            nonlocal curr_lines, curr_article, curr_heading, curr_page_start, curr_page_end
            text = "\n".join(curr_lines).strip()
            if text:
                sections.append(
                    {
                        "article_path": curr_article,
                        "heading": curr_heading,
                        "page_start": curr_page_start,
                        "page_end": curr_page_end,
                        "content": text,
                    }
                )
            curr_lines = []

        for page_num, page_text in pages:
            lines = page_text.split("\n")
            for line in lines:
                raw_line = line.strip()
                if not raw_line:
                    if curr_lines and curr_lines[-1] != "":
                        curr_lines.append("")
                    continue

                match = ARTICLE_PATTERN.match(raw_line)
                if match:
                    # New article or sub-article detected
                    flush_current()
                    art_num = match.group(1) or match.group(2)
                    rest = match.group(3).strip()

                    # Separate heading title from rest if separated by colon
                    if ":" in rest:
                        head_part, _ = rest.split(":", 1)
                        curr_heading = head_part.strip()
                    else:
                        curr_heading = rest[:80].strip()

                    curr_article = art_num
                    curr_page_start = page_num
                    curr_page_end = page_num
                    curr_lines = [raw_line]
                else:
                    curr_lines.append(raw_line)
                    curr_page_end = page_num

        flush_current()
        return sections

    def _split_large_section(
        self,
        sec: Dict[str, Any],
        document: KnowledgeDocument,
        doc_label: str,
        start_idx: int,
    ) -> List[DocumentChunk]:
        """Splits an oversized section into bounded chunks with small overlap."""
        text = sec["content"]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        chunks: List[DocumentChunk] = []
        curr_paragraphs: List[str] = []
        curr_tokens = 0
        local_idx = start_idx

        max_char_cap = self.max_tokens * 4
        overlap_char_cap = self.overlap_tokens * 4

        for p in paragraphs:
            p_tokens = _estimate_tokens(p)
            if curr_tokens + p_tokens > self.max_tokens and curr_paragraphs:
                # Flush current sub-chunk
                chunk_body = "\n\n".join(curr_paragraphs).strip()
                chunk = self._create_chunk(
                    document=document,
                    doc_label=doc_label,
                    chunk_index=local_idx,
                    content=chunk_body,
                    page_start=sec["page_start"],
                    page_end=sec["page_end"],
                    article_path=sec["article_path"],
                    heading=sec["heading"],
                )
                chunks.append(chunk)
                local_idx += 1

                # Keep small overlap from tail of previous sub-chunk
                overlap_text = chunk_body[-overlap_char_cap:] if len(chunk_body) > overlap_char_cap else ""
                curr_paragraphs = [overlap_text, p] if overlap_text else [p]
                curr_tokens = _estimate_tokens("\n\n".join(curr_paragraphs))
            else:
                curr_paragraphs.append(p)
                curr_tokens += p_tokens

        if curr_paragraphs:
            chunk_body = "\n\n".join(curr_paragraphs).strip()
            chunk = self._create_chunk(
                document=document,
                doc_label=doc_label,
                chunk_index=local_idx,
                content=chunk_body,
                page_start=sec["page_start"],
                page_end=sec["page_end"],
                article_path=sec["article_path"],
                heading=sec["heading"],
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        document: KnowledgeDocument,
        doc_label: str,
        chunk_index: int,
        content: str,
        page_start: int,
        page_end: int,
        article_path: str,
        heading: str,
    ) -> DocumentChunk:
        """Constructs a DocumentChunk with complete provenance and prepended embedding header."""
        header_tag = f"[{doc_label} | {article_path} | {heading}]" if heading else f"[{doc_label} | {article_path}]"
        embedding_text = f"{header_tag}\n{content}"

        meta = dict(document.metadata)
        meta.update(
            {
                "document_id": document.id,
                "document_label": doc_label,
                "page_start": page_start,
                "page_end": page_end,
                "article_path": article_path,
                "heading": heading,
                "chunk_index": chunk_index,
                "embedding_text": embedding_text,
            }
        )

        return DocumentChunk(
            id=f"{document.id}:chunk:{chunk_index}",
            document_id=document.id,
            chunk_index=chunk_index,
            source=document.source,
            content=content,
            page_start=page_start,
            page_end=page_end,
            article_path=article_path,
            heading=heading,
            metadata=meta,
        )
