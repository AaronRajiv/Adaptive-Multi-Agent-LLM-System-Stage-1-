"""
Synthesizer Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible for crafting the final, polished, user-facing answer incorporating
the original task, research findings, analytical synthesis, and evaluation feedback.
"""

from typing import List
from app.llm.base import LLMProvider
from app.models.schemas import EvaluationResult, ResearchResult

SYNTHESIZER_SYSTEM_PROMPT = """You are the FINAL SYNTHESIZER AGENT in a baseline multi-agent system.
Your responsibility is to produce the final, definitive, user-facing answer to the user's initial inquiry.

Inputs available to you:
1. Original User Task
2. Key Findings from Research Agents
3. In-depth Analysis from the Analyst Agent
4. Critique and Quality Score from the Evaluator Agent

Rules:
1. Deliver a polished, authoritative, well-structured, and complete response.
2. Incorporate the strengths and address any nuances raised by the Evaluator's critique.
3. Structure your response with clear headings, executive summary, comprehensive breakdown, practical implications, and a definitive conclusion/recommendation.
4. Keep the tone professional, objective, and directly tailored to the user's intent.
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
    ) -> str:
        """
        Synthesize all intermediate agent outputs into the final user-facing response.

        Args:
            user_task: The original user task.
            research_results: List of ResearchResult objects.
            analysis: The analytical synthesis from AnalystAgent.
            evaluation: The EvaluationResult from EvaluatorAgent.

        Returns:
            The final comprehensive user response string.
        """
        research_summary = "\n".join(
            f"- [{r.subtask_id}] {r.subtask_description}: {r.findings[:200]}..."
            for r in research_results
        )

        prompt = (
            f"Original User Task:\n\"{user_task.strip()}\"\n\n"
            f"Summary of Research Subtasks:\n{research_summary}\n\n"
            f"Analyst Agent Findings:\n{analysis}\n\n"
            f"Evaluator Score: {evaluation.score}/100 (Status: {evaluation.status})\n"
            f"Evaluator Feedback:\n{evaluation.feedback}\n\n"
            f"Please generate the definitive, final user-facing response."
        )

        final_response = await self.llm.generate(
            prompt=prompt,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT
        )

        return final_response.strip()
