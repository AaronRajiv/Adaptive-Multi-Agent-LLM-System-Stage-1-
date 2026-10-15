"""
Planner Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible strictly for Task Decomposition: breaking a complex user task
into discrete, typed subtasks. Does NOT answer the user task itself.
"""

from typing import List, Optional
from app.llm.base import LLMProvider, LLMStructuredOutputError
from app.models.schemas import ChatMessagePayload, PlannerOutput, SubTask

PLANNER_SYSTEM_PROMPT = """You are the PLANNER AGENT in a multi-agent problem-solving system.
Your SOLE responsibility is TASK DECOMPOSITION: breaking down a user request into an optimal dependency graph of discrete, typed subtasks.

Rules:
1. DECOMPOSITION DEPTH: Do NOT artificially restrict subtask count to 3-5. Decompose the task into as many concrete, actionable subtasks as needed to cover all implicated technical and regulatory dimensions.
2. DOMAIN-SPECIFIC SUBTASKS: For questions referencing documents or technical regulations:
   - Create one dedicated research subtask per distinct domain or topic named or implicated in the question (e.g., aerodynamics, minimum mass, power unit, suspension, safety structures).
   - Create cross-system interaction subtasks (type="analysis") where multiple subsystems interact (e.g., aerodynamic load vs suspension travel, power unit cooling vs sidepod aerodynamic packaging).
3. RETRIEVAL QUERIES (1 TO 4 PER RESEARCH SUBTASK):
   - For every subtask requiring document or regulation knowledge, include `metadata={"domain": "<domain_name>", "retrieval_queries": ["query 1", "query 2", ...], "retrieval_query": "primary query"}`.
   - Provide between 1 and 4 specific, targeted queries per subtask (such as exact Article references like "Article 3.4.1 overall width" or specific keywords).
   - If a subtask does not require document retrieval (e.g., pure logic or synthesis), omit `retrieval_queries`.
4. TASK TYPING & DEPENDENCIES:
   - For factual investigations across distinct topics or regulations, set type="research" and dependencies=[] so they execute concurrently in parallel.
   - For comparative evaluations, trade-off syntheses, or cross-system interactions, set type="analysis" and explicitly list prerequisite research task IDs in dependencies (e.g. dependencies=["T1", "T2"]).
5. NO DIRECT ANSWERS: Do NOT answer or solve the user's prompt yourself. Only construct the plan.
6. Return output strictly matching the PlannerOutput schema.
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
            f"Please decompose the current user task into clear, distinct subtasks with appropriate types ('research' or 'analysis') and dependencies. "
            f"For document questions, create one research subtask per distinct domain or topic named, plus cross-system interaction subtasks where applicable. "
            f"Provide 1-4 targeted retrieval queries in metadata['retrieval_queries'] for each research subtask."
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

