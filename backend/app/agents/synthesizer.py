"""
Synthesizer Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible for crafting the final, polished, user-facing answer incorporating
the original task, research findings, analytical synthesis, and evaluation feedback.
"""

from typing import List, Optional
from app.llm.base import LLMProvider
from app.models.schemas import ChatMessagePayload, EvaluationResult, ResearchResult

SYNTHESIZER_SYSTEM_PROMPT = """You are the FINAL SYNTHESIZER AGENT in an adaptive multi-agent system.
Your responsibility is to produce the definitive, user-facing answer addressing the user's task.

Inputs available to you:
1. Original User Task & Conversation History
2. Key Findings from Research Agents
3. Analysis from the Analyst Agent
4. Critique and Quality Score from the Evaluator Agent

Rules:
1. Deliver a clear, polished, and task-appropriate response.
2. Address the user's intent directly.
3. Format your output naturally according to what the task warrants. Do NOT force corporate report headings (such as 'Executive Summary', 'Findings', or 'Conclusion') onto direct answers, simple explanations, content drafting, or conversational questions. Use Markdown structure only when requested or when presenting complex multi-part analyses.
4. Keep the tone helpful, professional, and aligned with conversation history.
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
        history: Optional[List[ChatMessagePayload]] = None,
    ) -> str:
        """
        Synthesize all intermediate agent outputs into the final user-facing response.

        Args:
            user_task: The original user task.
            research_results: List of ResearchResult objects.
            analysis: The analytical synthesis from AnalystAgent.
            evaluation: The EvaluationResult from EvaluatorAgent.
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

        prompt = (
            f"{history_text}"
            f"Current User Task:\n\"{user_task.strip()}\"\n\n"
            f"Collected Research Subtasks & Findings:\n{research_summary}\n\n"
            f"Analyst Agent Findings:\n{analysis}\n\n"
            f"Evaluator Score: {evaluation.score}/100 (Status: {evaluation.status})\n"
            f"Evaluator Feedback:\n{evaluation.feedback}\n\n"
            f"Please generate the definitive, final user-facing response addressing the current task in light of all evidence and history."
        )

        final_response = await self.llm.generate(
            prompt=prompt,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT
        )

        return final_response.strip()

