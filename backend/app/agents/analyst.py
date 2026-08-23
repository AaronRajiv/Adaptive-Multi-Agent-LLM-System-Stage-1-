"""
Analyst Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible for synthesizing all collected research results with the original user task,
comparing dimensions, evaluating trade-offs, and producing a structured analytical report.
"""

from typing import List
from app.llm.base import LLMProvider
from app.models.schemas import ResearchResult

ANALYST_SYSTEM_PROMPT = """You are the ANALYST AGENT in a baseline multi-agent system.
Your responsibility is deep synthesis, logical reasoning, and cross-comparison of research findings.

Rules:
1. Review the original user task and all provided research findings from the Research Agents.
2. DO NOT simply copy-paste research points.
3. Compare perspectives, identify trade-offs, evaluate nuances, and highlight logical implications.
4. Produce a rigorous, structured analytical assessment with clear headers and bullet points.
5. Structure your analysis to cover:
   - Key Dimensions & Comparative Breakdown
   - Core Trade-offs & Practical Implications
   - Risk & Feasibility Assessment
"""


class AnalystAgent:
    """Agent that performs synthesis and comparative analysis across all research findings."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def analyze(self, user_task: str, research_results: List[ResearchResult]) -> str:
        """
        Synthesize research findings into a comprehensive analytical report.

        Args:
            user_task: The original user task.
            research_results: List of ResearchResult objects from the Research Agents.

        Returns:
            A coherent, well-reasoned analytical report string.
        """
        research_context = "\n\n".join(
            f"--- Subtask [{res.subtask_id}]: {res.subtask_description} ---\n{res.findings}"
            for res in research_results
        )

        prompt = (
            f"Original User Task:\n\"{user_task.strip()}\"\n\n"
            f"Collected Research Findings from Subtasks:\n"
            f"{research_context}\n\n"
            f"Based on the above findings, generate a thorough, cohesive analytical synthesis "
            f"evaluating the primary dimensions, trade-offs, and strategic conclusions."
        )

        analysis = await self.llm.generate(
            prompt=prompt,
            system_prompt=ANALYST_SYSTEM_PROMPT
        )

        return analysis.strip()
