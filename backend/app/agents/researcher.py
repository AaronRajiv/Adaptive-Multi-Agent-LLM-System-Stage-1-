"""
Research Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible for executing a single research subtask using parametric LLM knowledge.
Stage 1 intentionally does NOT perform live web search or RAG.
"""

from typing import Optional

from app.knowledge.models import RetrievedContext
from app.llm.base import LLMProvider
from app.models.schemas import SubTask, ResearchResult

RESEARCHER_SYSTEM_PROMPT = """You are the RESEARCH AGENT in a multi-agent system.
Your responsibility is to investigate a SINGLE specific subtask using core domain knowledge and any supplied grounding evidence.

Rules:
1. Focus ONLY on the assigned subtask. Do not drift into other areas.
2. Provide factual, balanced, and clear contextual insights.
3. Explicitly state uncertainty or trade-offs where appropriate.
4. Ground your findings in any provided retrieved knowledge context when available.
5. Return clear, concise markdown text organized with bullet points or brief sections.
"""


class ResearchAgent:
    """Agent that performs targeted research on a single subtask with optional RAG grounding."""

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
            ResearchResult containing the subtask ID, description, and generated findings.
        """
        grounding_section = ""
        if retrieved_context and retrieved_context.chunks:
            chunk_texts = "\n\n".join(
                f"[Source: {c.source} | Doc: {c.document_id} | Chunk #{c.chunk_index} (Score: {c.similarity_score:.2f})]:\n{c.content}"
                for c in retrieved_context.chunks
            )
            grounding_section = (
                f"\n\nRetrieved Knowledge Context (Grounding Evidence):\n{chunk_texts}\n\n"
                f"Please ground your investigation in the retrieved evidence above where applicable."
            )

        prompt = (
            f"Overall Task Context:\n\"{user_task.strip()}\"\n\n"
            f"Assigned Subtask ID: {subtask.id}\n"
            f"Assigned Subtask Description: {subtask.description}"
            f"{grounding_section}\n\n"
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
