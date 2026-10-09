"""
Research Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible for executing a single research subtask using parametric LLM knowledge.
Stage 1 intentionally does NOT perform live web search or RAG.
"""

import re
from typing import List, Optional

from app.knowledge.models import RetrievedContext
from app.llm.base import LLMProvider
from app.models.schemas import SubTask, ResearchResult
from app.orchestration.evidence_ledger import format_citation

RESEARCHER_SYSTEM_PROMPT = """You are the RESEARCH AGENT in an adaptive multi-agent system.
Your responsibility is to investigate a SINGLE specific subtask strictly using supplied grounding evidence and core principles.

Rules:
1. STRICT GROUNDING: When knowledge base context chunks are provided, answer ONLY from the supplied chunks. Do NOT invent regulations, dimensional bounds, or requirements.
2. CITATIONS & LABELS: Every factual claim or assertion MUST carry:
   - A citation in brackets: `[doc, art X.Y.Z, p.N]` matching the source citation given with each chunk (e.g. `[FIA 2025, art 3.4.1, p.24]`).
   - A grounding label:
     - `[EXPLICIT]`: Directly stated verbatim or explicitly codified in the regulatory/source text.
     - `[INFERENCE]`: A logical deduction or engineering implication derived from the cited regulatory text.
     - `[NOT_IN_SOURCE]`: Background engineering knowledge not explicitly found in the supplied chunks.
3. MISSING EVIDENCE: If the supplied chunks are missing or insufficient to address any aspect of the assigned subtask, explicitly output:
   `MISSING EVIDENCE: <describe specifically what regulation, dimension, or evidence is missing>`.
4. FOCUS: Answer only the assigned subtask. Do not drift into unrelated areas.
"""


class ResearchAgent:
    """Agent that performs targeted research on a single subtask with strict citation grounding."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def execute_subtask(
        self,
        user_task: str,
        subtask: SubTask,
        retrieved_context: Optional[RetrievedContext] = None,
    ) -> ResearchResult:
        """
        Execute research for a single assigned subtask.

        Args:
            user_task: The broader context / original user task.
            subtask: The specific SubTask object to investigate.
            retrieved_context: Optional structured RAG context retrieved for this task.

        Returns:
            ResearchResult containing findings, extracted citations, chunk IDs, and missing evidence.
        """
        grounding_section = ""
        available_citations: List[str] = []
        chunk_ids_present: List[str] = []

        if retrieved_context is not None:
            if retrieved_context.chunks:
                chunk_blocks = []
                for c in retrieved_context.chunks:
                    meta = c.metadata or {}
                    cit = format_citation(c.document_id, meta, document_label=c.source or c.document_id)
                    available_citations.append(cit)
                    chunk_ids_present.append(c.id)
                    chunk_blocks.append(
                        f"[Source: {c.source} | Doc: {c.document_id} | Chunk ID: {c.id} | Citation: {cit}]:\n{c.content}"
                    )

                grounding_section = (
                    f"\n\nKNOWLEDGE BASE CONTEXT:\n"
                    f"GROUNDING EVIDENCE CHUNKS:\n"
                    + "\n\n".join(chunk_blocks)
                    + f"\n\nINSTRUCTIONS:\n"
                    f"Use the supplied knowledge-base context when answering.\n"
                    f"Do not claim that information came from the knowledge base unless it is supported by the supplied context.\n"
                    f"Answer ONLY from the supplied evidence chunks above.\n"
                    f"Every claim must carry its citation `[doc, art X.Y.Z, p.N]` and label `[EXPLICIT]`, `[INFERENCE]`, or `[NOT_IN_SOURCE]`.\n"
                    f"If the supplied chunks do not contain enough information, state exactly what is missing under `MISSING EVIDENCE:`.\n"
                )
            else:
                grounding_section = (
                    f"\n\nKNOWLEDGE BASE CONTEXT:\n"
                    f"No relevant knowledge-base context was found for query \"{retrieved_context.query}\".\n\n"
                    f"INSTRUCTIONS:\n"
                    f"Do not claim that information came from the knowledge base since no relevant context was found.\n"
                    f"State clearly under `MISSING EVIDENCE:` that no relevant regulatory chunks were found for this subtask.\n"
                )

        prompt = (
            f"Overall Task Context:\n\"{user_task.strip()}\"\n\n"
            f"Assigned Subtask ID: {subtask.id}\n"
            f"Assigned Subtask Description: {subtask.description}"
            f"{grounding_section}\n\n"
            f"Please provide your focused, cited investigation for subtask {subtask.id}."
        )

        findings = await self.llm.generate(
            prompt=prompt,
            system_prompt=RESEARCHER_SYSTEM_PROMPT
        )

        # Extract citations [doc, art ..., p. ...] from findings
        extracted_citations = re.findall(
            r"\[([^\]]+?,\s*(?:art\s+[^,\]]+|p\.[0-9\-]+)[^\]]*?)\]",
            findings,
            re.IGNORECASE,
        )
        formatted_extracted = [f"[{c.strip()}]" for c in extracted_citations]

        # Check for referenced chunk IDs
        referenced_chunks = [cid for cid in chunk_ids_present if cid in findings]

        # Check for missing evidence section
        missing_match = re.search(r"MISSING EVIDENCE:\s*(.*?)(?:\n\n|\Z)", findings, re.DOTALL | re.IGNORECASE)
        missing_evidence = missing_match.group(1).strip() if missing_match else None

        return ResearchResult(
            subtask_id=subtask.id,
            subtask_description=subtask.description,
            findings=findings.strip(),
            citations=formatted_extracted or available_citations[:3],
            chunk_ids=referenced_chunks or chunk_ids_present,
            missing_evidence=missing_evidence,
        )
