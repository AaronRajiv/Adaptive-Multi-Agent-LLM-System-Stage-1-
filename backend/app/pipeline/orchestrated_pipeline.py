"""
Orchestrated Multi-Agent Pipeline for Stages 2–5.

Executes a dynamic, dependency-aware task graph with bounded parallel concurrency,
capability-based agent routing, optional RAG context grounding, evaluator auditing,
and final synthesis.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.agents.analyst import AnalystAgent
from app.agents.evaluator import EvaluatorAgent
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearchAgent
from app.agents.synthesizer import SynthesizerAgent
from app.config import Settings
from app.knowledge.embeddings import get_embedding_provider
from app.knowledge.models import RetrievedChunk
from app.knowledge.repository import InMemoryKnowledgeRepository, KnowledgeRepository
from app.knowledge.retrieval import RetrievalService
from app.llm.base import LLMProvider
from app.models.schemas import (
    EvaluationResult,
    ResearchResult,
    RunResponse,
    StageTrace,
    TaskExecutionTrace,
)
from app.models.task_graph import TaskGraph, TaskNode, TaskStatus, TaskType
from app.orchestration.agents import (
    AnalysisTaskAgent,
    CapabilityRegistry,
    ResearchTaskAgent,
)
from app.orchestration.orchestrator import GraphExecutionStatus, Orchestrator
from app.pipeline.baseline_pipeline import record_run


class OrchestratedPipeline:
    """
    Coordinates the dynamic, dependency-aware execution of Stage 2–5 multi-agent graphs.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        settings: Settings,
        knowledge_repository: Optional[KnowledgeRepository] = None,
        retrieval_service: Optional[RetrievalService] = None,
        max_concurrency: int = 4,
    ):
        self.llm = llm_provider
        self.settings = settings
        self.max_concurrency = max_concurrency

        # Domain Agents
        self.planner = PlannerAgent(self.llm)
        self.researcher = ResearchAgent(self.llm)
        self.analyst = AnalystAgent(self.llm)
        self.evaluator = EvaluatorAgent(self.llm, pass_threshold=settings.evaluator_pass_threshold)
        self.synthesizer = SynthesizerAgent(self.llm)

        # Retrieval Service Setup (Stage 5)
        if retrieval_service is not None:
            self.retrieval_service = retrieval_service
        else:
            try:
                embedding_provider = get_embedding_provider(settings)
                repo = knowledge_repository or InMemoryKnowledgeRepository()
                self.retrieval_service = RetrievalService(embedding_provider, repo)
            except Exception:
                self.retrieval_service = None

        # Capability Registry Setup (Stage 3/4)
        self.registry = CapabilityRegistry()
        self.registry.register("research", ResearchTaskAgent(self.researcher))
        self.registry.register("analysis", AnalysisTaskAgent(self.analyst))

    async def execute(self, user_task: str) -> RunResponse:
        """
        Execute the complete dynamic graph pipeline:
        Planner -> TaskGraph -> Orchestrator (Parallel + RAG) -> Evaluator -> Synthesizer.
        """
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        traces: List[StageTrace] = []

        # -------------------------------------------------------------
        # Stage 1: Planning & Task Decomposition (Dynamic TaskGraph)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        planner_output = await self.planner.plan(user_task)
        plan_dur = round((time.perf_counter() - t0) * 1000, 2)
        traces.append(
            StageTrace(
                stage_name="Planner",
                agent_name="PlannerAgent",
                duration_ms=plan_dur,
                status="SUCCESS",
                summary=f"Decomposed task into {len(planner_output.tasks)} graph tasks.",
            )
        )

        # Build and validate TaskGraph
        graph = planner_output.to_task_graph()
        self._ensure_default_dependencies(graph)

        # -------------------------------------------------------------
        # Stage 2-4: Orchestrated Graph Execution (Parallel Concurrency + Capabilities + RAG)
        # -------------------------------------------------------------
        orchestrator = Orchestrator(
            registry=self.registry,
            max_concurrency=self.max_concurrency,
            retrieval_service=self.retrieval_service,
        )

        t_orch = time.perf_counter()
        exec_result = await orchestrator.execute(graph, input_context={"user_task": user_task})
        orch_dur = round((time.perf_counter() - t_orch) * 1000, 2)

        # Collect research results, analysis outputs, and RAG chunks from executed graph tasks
        research_results: List[ResearchResult] = []
        analysis_outputs: List[str] = []
        retrieved_chunks_collected: List[RetrievedChunk] = []

        for task in graph.tasks:
            result = exec_result.task_results.get(task.id)
            if not result:
                continue

            # Collect retrieved context from task result metadata
            if result.metadata and "retrieved_context" in result.metadata:
                rc_data = result.metadata["retrieved_context"]
                if isinstance(rc_data, dict) and "chunks" in rc_data:
                    for chunk_dict in rc_data["chunks"]:
                        retrieved_chunks_collected.append(RetrievedChunk(**chunk_dict))

            if task.type == TaskType.RESEARCH and result.success:
                research_results.append(
                    ResearchResult(
                        subtask_id=task.id,
                        subtask_description=task.description,
                        findings=str(result.output or ""),
                    )
                )
            elif task.type == TaskType.ANALYSIS and result.success:
                analysis_outputs.append(str(result.output or ""))

        # Record StageTrace for Research
        research_dur = sum(
            t.duration_ms for t in exec_result.execution_trace if t.capability == "research"
        )
        traces.append(
            StageTrace(
                stage_name="Research",
                agent_name="ResearchAgent",
                duration_ms=round(research_dur, 2) if research_dur else orch_dur,
                status="SUCCESS" if research_results else ("FAILED" if exec_result.status == GraphExecutionStatus.FAILED else "SUCCESS"),
                summary=f"Orchestrated {len(research_results)} research subtask(s) with concurrency limit {self.max_concurrency}.",
            )
        )

        # Resolve analysis
        t_ana = time.perf_counter()
        if analysis_outputs:
            analysis_text = "\n\n".join(analysis_outputs)
            analysis_dur = sum(
                t.duration_ms for t in exec_result.execution_trace if t.capability == "analysis"
            )
        else:
            analysis_text = await self.analyst.analyze(user_task, research_results)
            analysis_dur = round((time.perf_counter() - t_ana) * 1000, 2)

        traces.append(
            StageTrace(
                stage_name="Analysis",
                agent_name="AnalystAgent",
                duration_ms=round(analysis_dur, 2),
                status="SUCCESS" if analysis_text else "FAILED",
                summary="Completed analytical synthesis and cross-comparison.",
            )
        )

        # -------------------------------------------------------------
        # Stage 5: Evaluation Agent (Quality Audit & Scoring)
        # -------------------------------------------------------------
        t_eval = time.perf_counter()
        evaluation_output = await self.evaluator.evaluate(user_task, analysis_text)
        eval_dur = round((time.perf_counter() - t_eval) * 1000, 2)
        traces.append(
            StageTrace(
                stage_name="Evaluation",
                agent_name="EvaluatorAgent",
                duration_ms=eval_dur,
                status="SUCCESS",
                summary=f"Evaluation complete: Score {evaluation_output.score}/100 ({evaluation_output.status}).",
            )
        )

        # -------------------------------------------------------------
        # Stage 6: Synthesizer Agent (Final User Response)
        # -------------------------------------------------------------
        t_synth = time.perf_counter()
        final_answer = await self.synthesizer.synthesize(
            user_task, research_results, analysis_text, evaluation_output
        )
        synth_dur = round((time.perf_counter() - t_synth) * 1000, 2)
        traces.append(
            StageTrace(
                stage_name="Synthesis",
                agent_name="SynthesizerAgent",
                duration_ms=synth_dur,
                status="SUCCESS",
                summary="Generated final user-facing response.",
            )
        )

        response = RunResponse(
            run_id=run_id,
            user_task=user_task,
            subtasks=graph.tasks,
            research_results=research_results,
            analysis=analysis_text,
            evaluation=evaluation_output,
            final_answer=final_answer,
            execution_trace=traces,
            created_at=created_at,
            task_graph=graph,
            retrieved_chunks=retrieved_chunks_collected,
            task_execution_traces=exec_result.execution_trace,
        )

        record_run(response)
        return response

    def _ensure_default_dependencies(self, graph: TaskGraph) -> None:
        """If an analysis task has no explicit dependencies, link it to all preceding research tasks."""
        research_ids = [t.id for t in graph.tasks if t.type == TaskType.RESEARCH]
        if not research_ids:
            return
        for task in graph.tasks:
            if task.type == TaskType.ANALYSIS and not task.dependencies:
                task.dependencies = list(research_ids)
        graph.validate_graph()
