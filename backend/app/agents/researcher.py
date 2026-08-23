"""
Research Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible for executing a single research subtask using parametric LLM knowledge.
Stage 1 intentionally does NOT perform live web search or RAG.
"""

from app.llm.base import LLMProvider
from app.models.schemas import SubTask, ResearchResult

RESEARCHER_SYSTEM_PROMPT = """You are the RESEARCH AGENT in a baseline multi-agent system.
Your responsibility is to investigate a SINGLE specific subtask using your core parametric knowledge.

Rules:
1. Focus ONLY on the assigned subtask. Do not drift into other areas.
2. Provide factual, balanced, and clear contextual insights.
3. Explicitly state uncertainty or trade-offs where appropriate.
4. IMPORTANT: Stage 1 operates on parametric knowledge. Do NOT fabricate live web citations, URLs, or claim to have browsed live external databases.
5. Return clear, concise markdown text organized with bullet points or brief sections.
"""


class ResearchAgent:
    """Agent that performs targeted parametric research on a single subtask."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def execute_subtask(self, user_task: str, subtask: SubTask) -> ResearchResult:
        """
        Execute research for a single assigned subtask.

        Args:
            user_task: The broader context / original user task.
            subtask: The specific SubTask object to investigate.

        Returns:
            ResearchResult containing the subtask ID, description, and generated findings.
        """
        prompt = (
            f"Overall Task Context:\n\"{user_task.strip()}\"\n\n"
            f"Assigned Subtask ID: {subtask.id}\n"
            f"Assigned Subtask Description: {subtask.description}\n\n"
            f"Please provide a focused, fact-based investigation covering key factors, data points, "
            f"and principles relevant to this subtask."
        )

        findings = await self.llm.generate(
            prompt=prompt,
            system_prompt=RESEARCHER_SYSTEM_PROMPT
        )

        return ResearchResult(
            subtask_id=subtask.id,
            subtask_description=subtask.description,
            findings=findings.strip()
        )
