"""
Orchestrated Multi-Agent Pipeline with Adaptive Strategy Selection & Resource-Aware Orchestration.

Executes user requests using task-appropriate strategies:
1. DIRECT: Natural conversational response without graph orchestration (Evaluation NOT_APPLICABLE).
2. SINGLE_AGENT: Direct execution for single-step questions, math, or content drafting (Evaluation SKIPPED).
3. MULTI_AGENT: Dynamic task-graph decomposition, parallel execution, RAG grounding, evaluator critique, targeted replanning, and synthesis.
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
from app.events import ExecutionEvent, ExecutionEventType, RunEventBus
from app.knowledge.embeddings import get_embedding_provider
from app.knowledge.models import RetrievedChunk, RetrievedContext
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
from app.orchestration.strategy import ExecutionStrategy, StrategySelector
from app.pipeline.baseline_pipeline import record_run


class OrchestratedPipeline:
    """
    Coordinates adaptive, resource-aware execution across DIRECT, SINGLE_AGENT, and MULTI_AGENT strategies.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        settings: Settings,
        knowledge_repository: Optional[KnowledgeRepository] = None,
        retrieval_service: Optional[RetrievalService] = None,
        max_concurrency: int = 4,
        event_bus: Optional[RunEventBus] = None,
        run_id: Optional[str] = None,
    ):
        self.llm = llm_provider
        self.settings = settings
        self.max_concurrency = max_concurrency
        self.event_bus = event_bus
        self.run_id = run_id

        # Strategy Selector
        self.strategy_selector = StrategySelector(self.llm)

        # Domain Agents
        self.planner = PlannerAgent(self.llm)
        self.researcher = ResearchAgent(self.llm)
        self.analyst = AnalystAgent(self.llm)
        self.evaluator = EvaluatorAgent(self.llm, pass_threshold=settings.evaluator_pass_threshold)
        self.synthesizer = SynthesizerAgent(self.llm)

        # Retrieval Service Setup
        if retrieval_service is not None:
            self.retrieval_service = retrieval_service
        else:
            try:
                embedding_provider = get_embedding_provider(settings)
                repo = knowledge_repository or InMemoryKnowledgeRepository()
                self.retrieval_service = RetrievalService(embedding_provider, repo)
            except Exception:
                self.retrieval_service = None

        # Capability Registry Setup
        self.registry = CapabilityRegistry()
        self.registry.register("research", ResearchTaskAgent(self.researcher))
        self.registry.register("analysis", AnalysisTaskAgent(self.analyst))

    async def execute(
        self,
        user_task: str,
        documents: Optional[List[Any]] = None,
        history: Optional[List[Any]] = None,
    ) -> RunResponse:
        """
        Execute the adaptive multi-agent pipeline using the selected execution strategy.
        """
        run_id = self.run_id or str(uuid.uuid4())
        self.run_id = run_id
        created_at = datetime.now(timezone.utc).isoformat()
        traces: List[StageTrace] = []

        if self.event_bus is not None:
            self.event_bus.run_id = run_id

        await self._emit(ExecutionEventType.RUN_STARTED, message=f"Pipeline started for task: {user_task[:100]}")

        # Ingest attached documents if present and RAG retrieval service is enabled
        if documents and self.retrieval_service:
            try:
                import hashlib
                from app.knowledge.chunking import FixedSizeTextChunker
                from app.knowledge.ingestion import DocumentIngestionService
                from app.knowledge.models import KnowledgeDocument
                from app.knowledge.pdf_extractor import extract_text_from_document

                chunk_size = getattr(self.settings, "knowledge_chunk_size", 1200)
                overlap = getattr(self.settings, "knowledge_chunk_overlap", 120)

                ingestion_service = DocumentIngestionService(
                    chunker=FixedSizeTextChunker(chunk_size=chunk_size, overlap=overlap),
                    embedding_provider=self.retrieval_service.embedding_provider,
                    repository=self.retrieval_service.repository,
                )

                for doc_payload in documents:
                    fname = getattr(doc_payload, "filename", None) or doc_payload.get("filename", "attached_doc")
                    fcontent = getattr(doc_payload, "content", None) or doc_payload.get("content", "")
                    clean_text = extract_text_from_document(fname, fcontent)
                    if clean_text and clean_text.strip():
                        # Deterministic document fingerprint prevents duplicate chunks on re-ingestion
                        fingerprint = hashlib.sha256(f"{fname}:{clean_text[:5000]}:{len(clean_text)}".encode("utf-8")).hexdigest()[:12]
                        doc_id = f"doc-{fingerprint}"
                        knowledge_doc = KnowledgeDocument(
                            id=doc_id,
                            source=fname,
                            content=clean_text,
                        )
                        res = await ingestion_service.ingest(knowledge_doc)
                        if res.failed:
                            await self._emit(
                                ExecutionEventType.RAG_FAILED,
                                message=f"Document '{fname}' ingestion failed: {res.error_message}",
                            )
            except Exception as e:
                await self._emit(
                    ExecutionEventType.RAG_FAILED,
                    message=f"Document ingestion error: {str(e)}",
                )

        try:
            return await self._execute_inner(
                user_task, run_id, created_at, traces, has_documents=bool(documents), documents=documents, history=history
            )
        except Exception as exc:
            await self._emit(
                ExecutionEventType.RUN_FAILED,
                message=f"Pipeline failed: {exc}",
            )
            raise
        finally:
            if self.event_bus is not None:
                await self.event_bus.finish()

    async def _execute_inner(
        self,
        user_task: str,
        run_id: str,
        created_at: str,
        traces: List[StageTrace],
        has_documents: bool = False,
        documents: Optional[List[Any]] = None,
        history: Optional[List[Any]] = None,
    ) -> RunResponse:
        """Inner execution logic supporting DIRECT, SINGLE_AGENT, and MULTI_AGENT strategies."""

        # -------------------------------------------------------------
        # Stage 0: Adaptive Strategy Selection
        # -------------------------------------------------------------
        decision = await self.strategy_selector.select_strategy(
            user_task, history=history, has_documents=has_documents
        )

        await self._emit(
            ExecutionEventType.STRATEGY_SELECTED,
            message=f"Strategy selected: {decision.strategy.value} ({decision.reasoning})",
            metadata={
                "strategy": decision.strategy.value,
                "reasoning": decision.reasoning,
                "estimated_complexity": decision.estimated_complexity,
                "requires_rag": decision.requires_rag,
            },
        )

        # =============================================================
        # STRATEGY 1: DIRECT Conversational Response
        # =============================================================
        if decision.strategy == ExecutionStrategy.DIRECT:
            t0 = time.perf_counter()
            await self._emit(ExecutionEventType.SYNTHESIS_STARTED, message="Generating direct response.")

            history_text = ""
            if history:
                turns = [f"{msg.role.capitalize()}: {msg.content}" for msg in history[-5:]]
                history_text = "Prior Conversation History:\n" + "\n".join(turns) + "\n\n"

            direct_prompt = f"{history_text}User: {user_task.strip()}\nAssistant:"
            direct_answer = await self.llm.generate(
                prompt=direct_prompt,
                system_prompt="You are a helpful, friendly AI assistant. Provide a concise, natural response without unnecessary markdown report headers or meta-analysis."
            )
            dur = round((time.perf_counter() - t0) * 1000, 2)
            await self._emit(ExecutionEventType.SYNTHESIS_COMPLETED, message="Direct response completed.")

            eval_res = EvaluationResult(
                score=None,
                status="NOT_APPLICABLE",
                feedback="Formal quality evaluation not applicable for direct conversational response."
            )
            traces.append(
                StageTrace(
                    stage_name="DirectResponse",
                    agent_name="LLMProvider",
                    duration_ms=dur,
                    status="SUCCESS",
                    summary="Direct conversational response."
                )
            )

            response = RunResponse(
                run_id=run_id,
                user_task=user_task,
                execution_strategy="DIRECT",
                replanning_count=0,
                subtasks=[],
                research_results=[],
                analysis="",
                evaluation=eval_res,
                final_answer=direct_answer.strip(),
                execution_trace=traces,
                created_at=created_at,
                task_graph=TaskGraph(tasks=[]),
                retrieved_chunks=[],
                task_execution_traces=[],
            )
            record_run(response)
            await self._emit(ExecutionEventType.RUN_COMPLETED, message=f"Pipeline completed (DIRECT). Run ID: {run_id}")
            return response

        # =============================================================
        # STRATEGY 2: SINGLE_AGENT Execution
        # =============================================================
        elif decision.strategy == ExecutionStrategy.SINGLE_AGENT:
            t0 = time.perf_counter()
            await self._emit(ExecutionEventType.PLANNING_STARTED, message="Single-agent strategy selected.")

            task_node = TaskNode(
                id="T1",
                description=user_task,
                type=TaskType.RESEARCH,
                assigned_capability="research",
                status=TaskStatus.COMPLETED,
            )
            single_graph = TaskGraph(tasks=[task_node])
            await self._emit(
                ExecutionEventType.PLANNING_COMPLETED,
                message="Single-agent task initialized.",
                metadata={"task_ids": ["T1"]}
            )

            retrieved_chunks_collected: List[RetrievedChunk] = []
            retrieval_context_prompt = ""
            if decision.requires_rag and self.retrieval_service:
                await self._emit(ExecutionEventType.RAG_STARTED, message="Retrieving knowledge context...")
                try:
                    ctx = await self.retrieval_service.retrieve(user_task, top_k=3)
                    retrieved_chunks_collected = ctx.chunks
                    if ctx.chunks:
                        snippets = [f"[{c.source}]: {c.content}" for c in ctx.chunks]
                        retrieval_context_prompt = "\n\nRetrieved Knowledge Base Context:\n" + "\n\n".join(snippets) + "\n\n"
                    await self._emit(ExecutionEventType.RAG_COMPLETED, message=f"Retrieved {len(ctx.chunks)} chunks.")
                except Exception:
                    pass

            await self._emit(ExecutionEventType.SYNTHESIS_STARTED, message="Generating single-agent response.")
            history_text = ""
            if history:
                turns = [f"{msg.role.capitalize()}: {msg.content}" for msg in history[-5:]]
                history_text = "Prior Conversation History:\n" + "\n".join(turns) + "\n\n"

            single_prompt = f"{history_text}{retrieval_context_prompt}User Task:\n\"{user_task.strip()}\"\n\nPlease provide a clear, accurate, direct response."
            single_answer = await self.llm.generate(
                prompt=single_prompt,
                system_prompt="You are a helpful, expert AI assistant. Deliver an accurate, well-structured response directly matching the user's intent."
            )
            dur = round((time.perf_counter() - t0) * 1000, 2)
            await self._emit(ExecutionEventType.SYNTHESIS_COMPLETED, message="Single-agent response completed.")

            eval_res = EvaluationResult(
                score=None,
                status="SKIPPED",
                feedback="Formal quality evaluation skipped for single-agent execution."
            )
            traces.append(
                StageTrace(
                    stage_name="SingleAgentExecution",
                    agent_name="TaskExecutionAgent",
                    duration_ms=dur,
                    status="SUCCESS",
                    summary="Executed single-agent strategy."
                )
            )

            response = RunResponse(
                run_id=run_id,
                user_task=user_task,
                execution_strategy="SINGLE_AGENT",
                replanning_count=0,
                subtasks=single_graph.tasks,
                research_results=[
                    ResearchResult(subtask_id="T1", subtask_description=user_task, findings=single_answer.strip())
                ],
                analysis=single_answer.strip(),
                evaluation=eval_res,
                final_answer=single_answer.strip(),
                execution_trace=traces,
                created_at=created_at,
                task_graph=single_graph,
                retrieved_chunks=retrieved_chunks_collected,
                task_execution_traces=[],
            )
            record_run(response)
            await self._emit(ExecutionEventType.RUN_COMPLETED, message=f"Pipeline completed (SINGLE_AGENT). Run ID: {run_id}")
            return response

        # =============================================================
        # STRATEGY 3: MULTI_AGENT Orchestrated Decomposition & Replanning
        # =============================================================
        await self._emit(ExecutionEventType.PLANNING_STARTED, message="Multi-agent planning started.")
        t0 = time.perf_counter()

        attached_doc_names = []
        if documents:
            for doc in documents:
                fname = getattr(doc, "filename", None) or (doc.get("filename") if isinstance(doc, dict) else None)
                if fname:
                    attached_doc_names.append(fname)

        planner_output = await self.planner.plan(user_task, history=history, attached_documents=attached_doc_names)
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

        graph = planner_output.to_task_graph()
        self._ensure_default_dependencies(graph)

        has_any_retrieval_query = any(
            t.metadata and t.metadata.get("retrieval_query") for t in graph.tasks
        )

        if (has_documents and decision.requires_rag) and not has_any_retrieval_query:
            for task_node in graph.tasks:
                if task_node.type == TaskType.RESEARCH:
                    if task_node.metadata is None:
                        task_node.metadata = {}
                    task_node.metadata["retrieval_query"] = task_node.description or user_task

        task_ids = [t.id for t in graph.tasks]
        await self._emit(
            ExecutionEventType.PLANNING_COMPLETED,
            message=f"Planning completed: {len(graph.tasks)} tasks.",
            metadata={"task_ids": task_ids},
        )

        orchestrator = Orchestrator(
            registry=self.registry,
            max_concurrency=self.max_concurrency,
            retrieval_service=self.retrieval_service,
            event_bus=self.event_bus,
        )

        t_orch = time.perf_counter()
        exec_result = await orchestrator.execute(graph, input_context={"user_task": user_task})
        orch_dur = round((time.perf_counter() - t_orch) * 1000, 2)

        research_results: List[ResearchResult] = []
        analysis_outputs: List[str] = []
        retrieved_chunks_collected: List[RetrievedChunk] = []

        for task in graph.tasks:
            result = exec_result.task_results.get(task.id)
            if not result:
                continue

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
                summary="Completed analytical synthesis.",
            )
        )

        # Stage 5: Evaluator Agent (Genuine 4-dimension audit scoring)
        await self._emit(ExecutionEventType.EVALUATION_STARTED, message="Evaluation started.")
        t_eval = time.perf_counter()
        evaluation_output = await self.evaluator.evaluate(user_task, analysis_text)
        eval_dur = round((time.perf_counter() - t_eval) * 1000, 2)
        traces.append(
            StageTrace(
                stage_name="Evaluation",
                agent_name="EvaluatorAgent",
                duration_ms=eval_dur,
                status="SUCCESS",
                summary=f"Evaluation score {evaluation_output.score}/100 ({evaluation_output.status}).",
            )
        )
        await self._emit(
            ExecutionEventType.EVALUATION_COMPLETED,
            message=f"Evaluation completed: {evaluation_output.score}/100 ({evaluation_output.status}).",
            metadata={"score": evaluation_output.score, "status": evaluation_output.status},
        )

        # Targeted Iterative Replanning with Feedback Injection
        replanning_count = 0
        max_replans = 1
        if evaluation_output.status == "FAIL" and replanning_count < max_replans:
            replanning_count += 1
            await self._emit(
                ExecutionEventType.REPLANNING_STARTED,
                message=f"Targeted replanning triggered (Iter {replanning_count}): Score {evaluation_output.score}/100."
            )
            t_replan = time.perf_counter()
            for r in research_results:
                task_res = exec_result.task_results.get(r.subtask_id)
                rc: Optional[RetrievedContext] = None
                if task_res and task_res.metadata and "retrieved_context" in task_res.metadata:
                    try:
                        rc = RetrievedContext.model_validate(task_res.metadata["retrieved_context"])
                    except Exception:
                        rc = None

                revised_prompt = (
                    f"{r.subtask_description}\n\n"
                    f"Evaluator Critique to Address:\n{evaluation_output.feedback}"
                )
                revised_subtask = TaskNode(
                    id=r.subtask_id,
                    description=revised_prompt,
                    type=TaskType.RESEARCH,
                )
                re_research = await self.researcher.execute_subtask(
                    user_task=user_task,
                    subtask=revised_subtask,
                    retrieved_context=rc,
                )
                r.findings = re_research.findings

            analysis_text = await self.analyst.analyze(user_task, research_results)
            evaluation_output = await self.evaluator.evaluate(user_task, analysis_text)
            replan_dur = round((time.perf_counter() - t_replan) * 1000, 2)

            traces.append(
                StageTrace(
                    stage_name="TargetedReplanning",
                    agent_name="Evaluator+Researcher",
                    duration_ms=replan_dur,
                    status="SUCCESS",
                    summary=f"Replanning completed. Revised evaluation score: {evaluation_output.score}/100."
                )
            )
            await self._emit(
                ExecutionEventType.REPLANNING_COMPLETED,
                message=f"Replanning completed: Revised score {evaluation_output.score}/100.",
                metadata={"score": evaluation_output.score, "status": evaluation_output.status},
            )

        # Stage 6: Synthesizer Agent
        await self._emit(ExecutionEventType.SYNTHESIS_STARTED, message="Synthesis started.")
        t_synth = time.perf_counter()
        final_answer = await self.synthesizer.synthesize(
            user_task, research_results, analysis_text, evaluation_output, history=history
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
        await self._emit(ExecutionEventType.SYNTHESIS_COMPLETED, message="Synthesis completed.")

        response = RunResponse(
            run_id=run_id,
            user_task=user_task,
            execution_strategy="MULTI_AGENT",
            replanning_count=replanning_count,
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
        await self._emit(
            ExecutionEventType.RUN_COMPLETED,
            message=f"Pipeline completed (MULTI_AGENT). Run ID: {run_id}",
            metadata={"run_id": run_id},
        )
        return response

    def _ensure_default_dependencies(self, graph: TaskGraph) -> None:
        """Sanitize dependencies and ensure DAG graph validity."""
        all_ids = {t.id for t in graph.tasks}
        for task in graph.tasks:
            task.dependencies = [d for d in task.dependencies if d != task.id and d in all_ids]

        try:
            graph.validate_graph()
        except ValueError:
            for task in graph.tasks:
                if task.type == TaskType.RESEARCH:
                    task.dependencies = []

        research_ids = [t.id for t in graph.tasks if t.type == TaskType.RESEARCH]
        if research_ids:
            for task in graph.tasks:
                if task.type == TaskType.ANALYSIS and not task.dependencies:
                    task.dependencies = list(research_ids)

        try:
            graph.validate_graph()
        except ValueError:
            for task in graph.tasks:
                task.dependencies = []
            graph.validate_graph()

    async def _emit(
        self,
        event_type: ExecutionEventType,
        message: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Emit a pipeline-level event if an event bus is attached."""
        if self.event_bus is None:
            return
        await self.event_bus.emit(
            ExecutionEvent(
                run_id=self.event_bus.run_id,
                event_type=event_type,
                message=message,
                metadata=metadata,
            )
        )
