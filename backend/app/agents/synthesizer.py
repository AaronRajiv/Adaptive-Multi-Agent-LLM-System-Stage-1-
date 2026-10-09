"""
Synthesizer Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible for crafting the final, polished, user-facing answer incorporating
the original task, research findings, analytical synthesis, and evaluation feedback.
"""

from typing import List, Optional
from app.llm.base import LLMProvider
from app.models.schemas import ChatMessagePayload, EvaluationResult, ResearchResult
from app.orchestration.evidence_ledger import EvidenceLedger

SYNTHESIZER_SYSTEM_PROMPT = """You are the SYNTHESIZER AGENT in an adaptive multi-agent system.
Your responsibility is to produce a rigorous, grounded, user-facing answer addressing the user's task.

Rules:
1. CITATION PRESERVATION: You MUST preserve all citations in brackets `[doc, art X.Y.Z, p.N]` for every regulatory requirement and factual claim.
2. MANDATORY SECTIONS FOR REGULATORY / TECHNICAL ANALYSIS:
   When answering questions involving technical regulations or documents, structure your response with the following explicit sections:
   - `### Explicit Regulatory Requirements`: Strict factual requirements, dimensional bounds, and mandatory constraints explicitly stated in the source regulations, cited with `[doc, art X.Y.Z, p.N]`.
   - `### Engineering Interpretation / Inference`: Engineering trade-offs, mechanical/aerodynamic interactions, and packaging implications logically deduced from the regulations, clearly designated as inferences.
   - `### Limitations and Unknowns`: Ambiguities, gaps where regulations do not specify implementation details, or information missing from the supplied evidence.
3. NO UNGROUNDED QUANTIFICATION OF LAP-TIME GAINS:
   You MUST NOT quantify lap-time gains (such as "gains 0.3 seconds per lap" or "worth half a second") UNLESS the supplied source text explicitly specifies that numerical figure. If the source does not provide a lap-time number, discuss performance effects strictly qualitatively.
4. NO AUTHORITATIVE / DEFINITIVE WORDING:
   Do NOT use words like "definitive", "authoritative", "indisputable", or "absolute". Frame findings with technical objectivity and precision based on the supplied evidence.
"""


class SynthesizerAgent:
    """Agent that creates the final response presented directly to the user."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def synthesize(
        self,
        user_task: str,
        research_results: List[ResearchResult],
        analysis: str,
        evaluation: EvaluationResult,
        evidence_ledger: Optional[EvidenceLedger] = None,
        history: Optional[List[ChatMessagePayload]] = None,
    ) -> str:
        """
        Synthesize all intermediate agent outputs into the final user-facing response.

        Args:
            user_task: The original user task.
            research_results: List of ResearchResult objects.
            analysis: The analytical synthesis from AnalystAgent.
            evaluation: The EvaluationResult from EvaluatorAgent.
            evidence_ledger: Optional shared canonical Evidence Ledger.
            history: Optional conversation history.

        Returns:
            The final comprehensive user response string.
        """
        research_summary = "\n\n".join(
            f"--- [{r.subtask_id}] {r.subtask_description} ---\n{r.findings}"
            for r in research_results
        )

        history_text = ""
        if history:
            turns = [f"{msg.role.capitalize()}: {msg.content}" for msg in history]
            history_text = "Prior Conversation History:\n" + "\n".join(turns) + "\n\n"

        ledger_text = ""
        if evidence_ledger is not None and len(evidence_ledger) > 0:
            ledger_text = f"\n\nGrounding Evidence Ledger Citations:\n{', '.join(evidence_ledger.all_citations())}\n"

        prompt = (
            f"{history_text}"
            f"Current User Task:\n\"{user_task.strip()}\"\n\n"
            f"Collected Research Subtasks & Findings:\n{research_summary}\n\n"
            f"Analyst Agent Findings:\n{analysis}\n"
            f"{ledger_text}\n"
            f"Evaluator Score: {evaluation.score}/100 (Status: {evaluation.status})\n"
            f"Evaluator Feedback:\n{evaluation.feedback}\n\n"
            f"Please generate the grounded, user-facing response following the mandatory section structure: "
            f"'### Explicit Regulatory Requirements', '### Engineering Interpretation / Inference', and '### Limitations and Unknowns'. "
            f"Preserve all citations and do not quantify lap-time gains unless the text explicitly states them."
        )

        final_response = await self.llm.generate(
            prompt=prompt,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT
        )

        return final_response.strip()

