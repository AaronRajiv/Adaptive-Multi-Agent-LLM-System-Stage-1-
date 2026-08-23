"""
Fixed Baseline Pipeline (Baseline Coordinator) for Stage 1.

Executes a fixed, sequential multi-agent pipeline:
User Task -> Planner -> Research Agents -> Analyst -> Evaluator -> Synthesizer -> Final Answer

This baseline deliberately executes sequentially with in-memory execution trace collection.
Dynamic routing, task graphs, parallel execution, replanning loops, and persistent databases
are intentionally deferred to later project stages.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from app.config import Settings
from app.llm.base import LLMProvider
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearchAgent
from app.agents.analyst import AnalystAgent
from app.agents.evaluator import EvaluatorAgent
from app.agents.synthesizer import SynthesizerAgent
from app.models.schemas import (
    RunResponse,
    StageTrace,
    ResearchResult,
)


class BaselinePipeline:
    """
    Coordinates the fixed sequential execution of Stage 1 agents.
    Records per-stage execution durations and status for observability.
    """

    def __init__(self, llm_provider: LLMProvider, settings: Settings):
        self.llm = llm_provider
        self.settings = settings
        self.planner = PlannerAgent(self.llm)
        self.researcher = ResearchAgent(self.llm)
        self.analyst = AnalystAgent(self.llm)
        self.evaluator = EvaluatorAgent(self.llm, pass_threshold=settings.evaluator_pass_threshold)
        self.synthesizer = SynthesizerAgent(self.llm)

    async def execute(self, user_task: str) -> RunResponse:
        """
        Execute the complete fixed baseline pipeline sequentially.

        Args:
            user_task: Raw task prompt from the user.

        Returns:
            RunResponse containing all intermediate agent outputs, final answer,
            and measured execution trace.
        """
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        trace: List[StageTrace] = []

        # -------------------------------------------------------------
        # Stage 1: Planner Agent (Task Decomposition)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            planner_output = await self.planner.plan(user_task)
            dur_ms = (time.perf_counter() - t0) * 1000
            trace.append(StageTrace(
                stage_name="Planner",
                agent_name="PlannerAgent",
                duration_ms=round(dur_ms, 2),
                status="SUCCESS",
                summary=f"Decomposed task into {len(planner_output.tasks)} subtasks."
            ))
        except Exception as e:
            dur_ms = (time.perf_counter() - t0) * 1000
            trace.append(StageTrace(
                stage_name="Planner",
                agent_name="PlannerAgent",
                duration_ms=round(dur_ms, 2),
                status="FAILED",
                summary=f"Planner failure: {str(e)}"
            ))
            raise

        # -------------------------------------------------------------
        # Stage 2: Research Agents (Parametric Knowledge Execution)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        research_results: List[ResearchResult] = []
        research_tasks = [t for t in planner_output.tasks if t.type == "research"]

        # If planner didn't tag any as research, treat all non-analysis as research
        if not research_tasks:
            research_tasks = planner_output.tasks

        try:
            for subtask in research_tasks:
                sub_res = await self.researcher.execute_subtask(user_task, subtask)
                research_results.append(sub_res)

            dur_ms = (time.perf_counter() - t0) * 1000
            trace.append(StageTrace(
                stage_name="Research",
                agent_name="ResearchAgent",
                duration_ms=round(dur_ms, 2),
                status="SUCCESS",
                summary=f"Completed {len(research_results)} research subtask(s)."
            ))
        except Exception as e:
            dur_ms = (time.perf_counter() - t0) * 1000
            trace.append(StageTrace(
                stage_name="Research",
                agent_name="ResearchAgent",
                duration_ms=round(dur_ms, 2),
                status="FAILED",
                summary=f"Research failure: {str(e)}"
            ))
            raise

        # -------------------------------------------------------------
        # Stage 3: Analyst Agent (Synthesis & Reasoning)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            analysis_output = await self.analyst.analyze(user_task, research_results)
            dur_ms = (time.perf_counter() - t0) * 1000
            trace.append(StageTrace(
                stage_name="Analysis",
                agent_name="AnalystAgent",
                duration_ms=round(dur_ms, 2),
                status="SUCCESS",
                summary="Completed analytical synthesis and cross-comparison."
            ))
        except Exception as e:
            dur_ms = (time.perf_counter() - t0) * 1000
            trace.append(StageTrace(
                stage_name="Analysis",
                agent_name="AnalystAgent",
                duration_ms=round(dur_ms, 2),
                status="FAILED",
                summary=f"Analyst failure: {str(e)}"
            ))
            raise

        # -------------------------------------------------------------
        # Stage 4: Evaluator Agent (Quality Audit & Scoring)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            evaluation_output = await self.evaluator.evaluate(user_task, analysis_output)
            dur_ms = (time.perf_counter() - t0) * 1000
            trace.append(StageTrace(
                stage_name="Evaluation",
                agent_name="EvaluatorAgent",
                duration_ms=round(dur_ms, 2),
                status="SUCCESS",
                summary=f"Evaluation complete: Score {evaluation_output.score}/100 ({evaluation_output.status})."
            ))
        except Exception as e:
            dur_ms = (time.perf_counter() - t0) * 1000
            trace.append(StageTrace(
                stage_name="Evaluation",
                agent_name="EvaluatorAgent",
                duration_ms=round(dur_ms, 2),
                status="FAILED",
                summary=f"Evaluator failure: {str(e)}"
            ))
            raise

        # -------------------------------------------------------------
        # Stage 5: Synthesizer Agent (Final User Response)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            final_answer = await self.synthesizer.synthesize(
                user_task, research_results, analysis_output, evaluation_output
            )
            dur_ms = (time.perf_counter() - t0) * 1000
            trace.append(StageTrace(
                stage_name="Synthesis",
                agent_name="SynthesizerAgent",
                duration_ms=round(dur_ms, 2),
                status="SUCCESS",
                summary="Generated final user-facing response."
            ))
        except Exception as e:
            dur_ms = (time.perf_counter() - t0) * 1000
            trace.append(StageTrace(
                stage_name="Synthesis",
                agent_name="SynthesizerAgent",
                duration_ms=round(dur_ms, 2),
                status="FAILED",
                summary=f"Synthesizer failure: {str(e)}"
            ))
            raise

        return RunResponse(
            run_id=run_id,
            user_task=user_task,
            subtasks=planner_output.tasks,
            research_results=research_results,
            analysis=analysis_output,
            evaluation=evaluation_output,
            final_answer=final_answer,
            execution_trace=trace,
            created_at=created_at,
        )


# In-memory execution storage for observability (max 50 runs)
RUNS_IN_MEMORY: List[RunResponse] = []


def record_run(run: RunResponse) -> None:
    """Store run response in in-memory list (retaining up to the most recent 50 runs)."""
    RUNS_IN_MEMORY.append(run)
    if len(RUNS_IN_MEMORY) > 50:
        RUNS_IN_MEMORY.pop(0)


def get_recent_runs() -> List[RunResponse]:
    """Retrieve all recent in-memory runs in reverse chronological order."""
    return list(reversed(RUNS_IN_MEMORY))
