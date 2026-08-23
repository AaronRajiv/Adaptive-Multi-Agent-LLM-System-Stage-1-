"""
Evaluator Agent for Stage 1 Fixed Baseline Multi-Agent System.

Responsible for critically evaluating the Analyst Agent's output against the user's task
across 4 dimensions: Correctness, Completeness, Relevance, and Clarity.
Emits a structured score, PASS/FAIL status, and detailed constructive feedback.
"""

from app.llm.base import LLMProvider, LLMStructuredOutputError
from app.models.schemas import EvaluationResult

EVALUATOR_SYSTEM_PROMPT = """You are the EVALUATOR AGENT in a baseline multi-agent system.
Your responsibility is to perform an objective, critical quality audit of the Analyst Agent's work.

Evaluation Criteria (100 total points):
1. Correctness & Logical Consistency (25 pts): Are the arguments coherent, sound, and free of contradictions?
2. Completeness & Depth (25 pts): Did the analysis thoroughly address all core aspects of the user's prompt?
3. Relevance & Focus (25 pts): Is the analysis directly relevant to the user query without fluff?
4. Clarity & Structure (25 pts): Is the output well-organized, readable, and clearly communicated?

Rules:
1. Provide an integer score between 0 and 100.
2. If score >= {threshold}, status MUST be "PASS". Otherwise status MUST be "FAIL".
3. Provide constructive, rigorous feedback explaining the score breakdown and noting specific strengths/weaknesses.
4. Return your evaluation strictly in the required JSON schema.
"""


class EvaluatorAgent:
    """Agent that evaluates analytical quality and outputs structured score and feedback."""

    def __init__(self, llm_provider: LLMProvider, pass_threshold: int = 80):
        self.llm = llm_provider
        self.pass_threshold = pass_threshold

    async def evaluate(self, user_task: str, analysis: str) -> EvaluationResult:
        """
        Evaluate the analyst's output against the original user task.

        Args:
            user_task: The original user task.
            analysis: The analytical synthesis produced by the Analyst Agent.

        Returns:
            EvaluationResult with score, status, and feedback.
        """
        system_prompt = EVALUATOR_SYSTEM_PROMPT.format(threshold=self.pass_threshold)

        prompt = (
            f"Original User Task:\n\"{user_task.strip()}\"\n\n"
            f"Analyst Agent Output to Evaluate:\n{analysis}\n\n"
            f"Pass Threshold: {self.pass_threshold} / 100\n\n"
            f"Please score this output and provide constructive evaluation feedback."
        )

        try:
            result = await self.llm.generate_structured(
                prompt=prompt,
                schema=EvaluationResult,
                system_prompt=system_prompt
            )

            # Enforce pass threshold status consistency
            calculated_status = "PASS" if result.score >= self.pass_threshold else "FAIL"
            if result.status != calculated_status:
                result.status = calculated_status

            return result
        except Exception as e:
            if isinstance(e, LLMStructuredOutputError):
                raise
            raise LLMStructuredOutputError(f"Evaluator Agent execution failed: {str(e)}") from e
