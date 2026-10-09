"""
Planner Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible strictly for Task Decomposition: breaking a complex user task
into discrete, typed subtasks. Does NOT answer the user task itself.
"""

from typing import List, Optional
from app.llm.base import LLMProvider, LLMStructuredOutputError
from app.models.schemas import ChatMessagePayload, PlannerOutput, SubTask

PLANNER_SYSTEM_PROMPT = """You are the PLANNER AGENT in a multi-agent problem-solving system.
Your SOLE responsibility is TASK DECOMPOSITION.

Rules:
1. Break the user's high-level analytical task into 3 to 5 discrete, concrete subtasks.
2. Assign each subtask an ID (e.g. 'T1', 'T2', 'T3', 'T4').
3. For information gathering subtasks, set type="research".
4. For synthesis / comparative evaluation subtasks, set type="analysis".
5. If a subtask requires searching or retrieving facts from attached documents or external knowledge base, include a `retrieval_query` string key in the task's `metadata` dictionary (e.g., `metadata={"retrieval_query": "electric versus petrol vehicle lifecycle emissions..."}`).
6. If a subtask does NOT require knowledge-base document retrieval (e.g., general reasoning or synthesis), leave `metadata` as null or do NOT include `retrieval_query`. DO NOT force retrieval_query on every task.
7. DO NOT solve or answer the task yourself. Only decompose it into subtasks.
8. Take into account any prior conversation context and attached knowledge documents when breaking down tasks.
9. Return your output strictly matching the required schema.
"""


class PlannerAgent:
    """Agent that decomposes user tasks into structured subtasks."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def plan(
        self,
        user_task: str,
        history: Optional[List[ChatMessagePayload]] = None,
        attached_documents: Optional[List[str]] = None,
    ) -> PlannerOutput:
        """
        Decompose the given user task into structured subtasks.

        Args:
            user_task: The complex analytical task entered by the user.
            history: Optional conversation history for context aware follow-up planning.
            attached_documents: Optional list of document names attached to the request.

        Returns:
            PlannerOutput containing the validated list of SubTasks.

        Raises:
            LLMStructuredOutputError: If the model fails to return valid structured output.
        """
        history_text = ""
        if history:
            turns = [f"{msg.role.capitalize()}: {msg.content}" for msg in history]
            history_text = "Prior Conversation History:\n" + "\n".join(turns) + "\n\n"

        doc_text = ""
        if attached_documents:
            doc_text = f"Attached Knowledge Documents: {', '.join(attached_documents)}\n\n"

        prompt = (
            f"{history_text}"
            f"{doc_text}"
            f"Current User Task:\n\"{user_task.strip()}\"\n\n"
            f"Please decompose the current user task into 3-5 clear, distinct subtasks with appropriate types ('research' or 'analysis'). "
            f"For subtasks that require knowledge-base retrieval, provide a concise `retrieval_query` in `metadata`. For subtasks that do not, leave `retrieval_query` out."
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

