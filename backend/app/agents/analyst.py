"""
Analyst Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible for synthesizing all collected research results with the original user task,
comparing dimensions, evaluating trade-offs, and producing a structured analytical report.
"""

from typing import List, Optional, Union

from app.knowledge.models import RetrievedContext
from app.llm.base import LLMProvider
from app.models.schemas import ResearchResult
from app.orchestration.evidence_ledger import EvidenceLedger

ANALYST_SYSTEM_PROMPT = """You are the ANALYST AGENT in an adaptive multi-agent system.
Your responsibility is rigorous comparative analysis, cross-subsystem reasoning, and trade-off evaluation.

Rules:
1. STRICT REASONING SOURCE: You must ONLY reason from the supplied Evidence Ledger and research findings. Do NOT invent regulatory requirements, specifications, or numbers not present in the evidence.
2. DUAL-SIDED CITATIONS FOR CROSS-SYSTEM INTERACTIONS:
   Whenever evaluating cross-system interactions or trade-offs between two subsystems (e.g., Aerodynamics vs Suspension, Power Unit vs Thermal Packaging, Minimum Mass vs Ballast Distribution):
   YOU MUST CITE BOTH SIDES using their explicit citations (e.g., citing the aerodynamic article `[FIA 2025, art 3.10, p.45]` AND the suspension article `[FIA 2025, art 10.2, p.112]`).
3. GROUNDING & CITATION PRESERVATION:
   Preserve citations on all cited parameters and rules. Distinguish explicit regulatory boundaries from engineering inference.
4. STRUCTURED ANALYSIS:
   Organize your analytical report logically:
   - Key Dimensions & Regulatory Constraints (cited)
   - Cross-System Interactions & Trade-offs (with dual-sided citations)
   - Risk, Packaging & Feasibility Assessment
"""


class AnalystAgent:
    """Agent that performs synthesis and comparative analysis across all research findings and the evidence ledger."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def analyze(
        self,
        user_task: str,
        research_results: List[ResearchResult],
        retrieved_context: Optional[RetrievedContext] = None,
        evidence_ledger: Optional[EvidenceLedger] = None,
    ) -> str:
        """
        Synthesize research findings and evidence ledger into a comprehensive analytical report.

        Args:
            user_task: The original user task.
            research_results: List of ResearchResult objects from the Research Agents.
            retrieved_context: Optional legacy structured RAG context.
            evidence_ledger: Optional shared canonical Evidence Ledger.

        Returns:
            A coherent, well-reasoned analytical report string with preserved dual-sided citations.
        """
        research_context = "\n\n".join(
            f"--- Subtask [{res.subtask_id}]: {res.subtask_description} ---\n{res.findings}"
            for res in research_results
        )

        ledger_section = ""
        if evidence_ledger is not None and len(evidence_ledger) > 0:
            ledger_section = (
                f"\n\nCANONICAL EVIDENCE LEDGER:\n"
                f"{evidence_ledger.format_for_prompt()}\n\n"
                f"INSTRUCTIONS:\n"
                f"Reason ONLY from the Evidence Ledger and Research Findings above.\n"
                f"Every cross-system trade-off MUST cite both interacting regulations/domains.\n"
            )
        elif retrieved_context is not None:
            if retrieved_context.chunks:
                chunk_texts = "\n\n".join(
                    f"[Source: {c.source} | Doc: {c.document_id} | Chunk #{c.chunk_index} (ID: {c.id}, Score: {c.similarity_score:.2f})]:\n"
                    f"Content:\n{c.content}"
                    for c in retrieved_context.chunks
                )
                ledger_section = (
                    f"\n\nKNOWLEDGE BASE CONTEXT:\n{chunk_texts}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"Use the supplied knowledge-base context when synthesizing analysis.\n"
                    f"Cross-system interactions must cite both interacting areas.\n"
                )
            else:
                ledger_section = (
                    f"\n\nKNOWLEDGE BASE CONTEXT:\n"
                    f"No relevant knowledge-base context was found for query \"{retrieved_context.query}\".\n"
                )

        prompt = (
            f"Original User Task:\n\"{user_task.strip()}\"\n\n"
            f"Collected Research Findings from Subtasks:\n"
            f"{research_context}"
            f"{ledger_section}\n\n"
            f"Based on the above findings and evidence ledger, generate a thorough analytical synthesis "
            f"evaluating dimensions, cross-system trade-offs (citing both sides), and conclusions."
        )

        analysis = await self.llm.generate(
            prompt=prompt,
            system_prompt=ANALYST_SYSTEM_PROMPT
        )

        return analysis.strip()
