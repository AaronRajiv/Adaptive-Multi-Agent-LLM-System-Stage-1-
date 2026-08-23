"""
Planner Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible strictly for Task Decomposition: breaking a complex user task
into discrete, typed subtasks. Does NOT answer the user task itself.
"""

from typing import Optional
from app.llm.base import LLMProvider, LLMStructuredOutputError
from app.models.schemas import PlannerOutput, SubTask

PLANNER_SYSTEM_PROMPT = """You are the PLANNER AGENT in a multi-agent problem-solving system.
Your SOLE responsibility is TASK DECOMPOSITION.

Rules:
1. Break the user's high-level analytical task into 3 to 5 discrete, concrete subtasks.
2. Assign each subtask an ID (e.g. 'T1', 'T2', 'T3', 'T4').
3. For information gathering subtasks, set type="research".
4. For synthesis / comparative evaluation subtasks, set type="analysis".
5. DO NOT solve or answer the task yourself. Only decompose it into subtasks.
6. Return your output strictly matching the required schema.
"""


class PlannerAgent:
    """Agent that decomposes user tasks into structured subtasks."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def plan(self, user_task: str) -> PlannerOutput:
        """
        Decompose the given user task into structured subtasks.

        Args:
            user_task: The complex analytical task entered by the user.

        Returns:
            PlannerOutput containing the validated list of SubTasks.

        Raises:
            LLMStructuredOutputError: If the model fails to return valid structured output.
        """
        prompt = (
            f"Please decompose the following complex task into structured subtasks:\n\n"
            f"User Task:\n\"{user_task.strip()}\"\n\n"
            f"Decompose this into 3-5 clear, distinct subtasks with appropriate types ('research' or 'analysis')."
        )

        try:
            output = await self.llm.generate_structured(
                prompt=prompt,
                schema=PlannerOutput,
                system_prompt=PLANNER_SYSTEM_PROMPT
            )
            # Ensure each task has valid non-empty values
            if not output.tasks:
                raise LLMStructuredOutputError("Planner returned an empty list of subtasks.")
            return output
        except Exception as e:
            if isinstance(e, LLMStructuredOutputError):
                raise
            raise LLMStructuredOutputError(f"Planner Agent execution failed: {str(e)}") from e
