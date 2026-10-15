"""
Orchestrated Multi-Agent Pipeline with Adaptive Strategy Selection & Resource-Aware Orchestration.

Executes user requests using task-appropriate strategies:
1. DIRECT: Natural conversational response without graph orchestration (Evaluation NOT_APPLICABLE).
2. SINGLE_AGENT: Direct execution for single-step questions, math, or content drafting (Evaluation SKIPPED).
3. MULTI_AGENT: Dynamic task-graph decomposition, parallel execution, RAG grounding, evaluator critique, targeted replanning, and synthesis.
"""

import asyncio
import hashlib
import re
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
from app.orchestration.evidence_ledger import EvidenceLedger
from app.orchestration.grounding_verifier import GroundingVerifier, GroundingVerificationResult
from app.orchestration.orchestrator import GraphExecutionStatus, Orchestrator
from app.logging_config import get_logger
from app.orchestration.strategy import ExecutionStrategy, StrategySelector
from app.orchestration.resource_budget import BudgetTracker, ResourceBudgetExceededError, get_budget_for_strategy
from app.pipeline.baseline_pipeline import record_run
from app.telemetry.instrumented_provider import InstrumentedLLMProvider
from app.telemetry.metrics import global_metrics_collector
from app.tools.fetch_page import FetchPageInput, FetchPageTool
from app.tools.search_provider import CreditTracker
from app.tools.web_cache import WebCacheManager
from app.tools.web_search import WebSearchInput, WebSearchTool

logger = get_logger("app.pipeline.orchestrated_pipeline")


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
        document_service: Optional[Any] = None,
        max_concurrency: int = 4,
        event_bus: Optional[RunEventBus] = None,
        run_id: Optional[str] = None,
    ):
        self.llm = llm_provider
        self.settings = settings
        self.document_service = document_service
        self.max_concurrency = max_concurrency
        if getattr(settings, "llm_provider", None) == "gemini" and self.max_concurrency > 2:
            self.max_concurrency = 2
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
        self._retrieval_init_error: Optional[str] = None
        if retrieval_service is not None:
            self.retrieval_service = retrieval_service
        else:
            try:
                embedding_provider = get_embedding_provider(settings)
                repo = knowledge_repository or InMemoryKnowledgeRepository()
                self.retrieval_service = RetrievalService(embedding_provider, repo)
            except Exception as exc:
                self._retrieval_init_error = str(exc)
                logger.warning("Failed to initialize default RetrievalService: %s", exc)
                self.retrieval_service = None

        # Web Tools & Cache Setup
        self.credit_tracker = CreditTracker()
        self.cache_manager = WebCacheManager(custom_settings=self.settings)
        self.web_search_tool = WebSearchTool(
            custom_settings=self.settings,
            credit_tracker=self.credit_tracker,
            cache_manager=self.cache_manager,
        )
        self.fetch_page_tool = FetchPageTool(
            custom_settings=self.settings,
            cache_manager=self.cache_manager,
        )

        # Capability Registry Setup
        self.registry = CapabilityRegistry()
        research_task_agent = ResearchTaskAgent(
            self.researcher,
            web_search_tool=self.web_search_tool,
            fetch_page_tool=self.fetch_page_tool,
        )
        self.registry.register("research", research_task_agent)
        self.registry.register("web_research", research_task_agent)
        self.registry.register("analysis", AnalysisTaskAgent(self.analyst))

    async def execute(
        self,
        user_task: str,
        documents: Optional[List[Any]] = None,
        document_ids: Optional[List[str]] = None,
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

        effective_doc_ids: List[str] = list(document_ids or [])

        # Validate pre-ingested document_ids if provided
        if document_ids:
            if not self.retrieval_service:
                reason = self._retrieval_init_error or "Retrieval service is not configured or failed to initialize"
                err_msg = f"Document IDs {document_ids} were provided but retrieval service is not available ({reason})."
                logger.error(err_msg)
                await self._emit(ExecutionEventType.RUN_FAILED, message=err_msg)
                raise RuntimeError(err_msg)

            if getattr(self, "document_service", None) is not None:
                for doc_id in document_ids:
                    meta = self.document_service.get_document(doc_id)
                    if not meta:
                        err_msg = f"Referenced document '{doc_id}' does not exist."
                        logger.error(err_msg)
                        await self._emit(ExecutionEventType.RUN_FAILED, message=err_msg)
                        raise RuntimeError(err_msg)
                    if getattr(meta, "status", None) and str(meta.status.value if hasattr(meta.status, "value") else meta.status) != "ready":
                        status_str = meta.status.value if hasattr(meta.status, "value") else str(meta.status)
                        err_msg = f"Referenced document '{doc_id}' is not ready (status: '{status_str}'). Refusing to run."
                        logger.error(err_msg)
                        await self._emit(ExecutionEventType.RUN_FAILED, message=err_msg)
                        raise RuntimeError(err_msg)

        # Validate and ingest attached legacy documents if present
        if documents:
            if not self.retrieval_service:
                reason = self._retrieval_init_error or "Retrieval service is not configured or failed to initialize"
                err_msg = f"Documents were provided but retrieval service is not available ({reason})."
                logger.error(err_msg)
                await self._emit(ExecutionEventType.RUN_FAILED, message=err_msg)
                raise RuntimeError(err_msg)

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
                        fingerprint = hashlib.sha256(f"{fname}:{clean_text[:5000]}:{len(clean_text)}".encode("utf-8")).hexdigest()[:12]
                        doc_id = f"doc-{fingerprint}"
                        if doc_id not in effective_doc_ids:
                            effective_doc_ids.append(doc_id)
                        knowledge_doc = KnowledgeDocument(
                            id=doc_id,
                            source=fname,
                            content=clean_text,
                        )
                        res = await ingestion_service.ingest(knowledge_doc)
                        if res.failed:
                            err_msg = f"Document '{fname}' ingestion failed: {res.error_message}"
                            logger.error(err_msg)
                            await self._emit(ExecutionEventType.RUN_FAILED, message=err_msg)
                            raise RuntimeError(err_msg)
            except Exception as e:
                err_msg = f"Document ingestion failed: {str(e)}"
                logger.error(err_msg, exc_info=True)
                await self._emit(ExecutionEventType.RUN_FAILED, message=err_msg)
                raise RuntimeError(err_msg) from e

        try:
            return await self._execute_inner(
                user_task,
                run_id,
                created_at,
                traces,
                has_documents=bool(documents or effective_doc_ids),
                documents=documents,
                document_ids=effective_doc_ids or None,
                history=history,
            )
        except ResourceBudgetExceededError as rbe:
            logger.error("Resource budget exceeded for run_id=%s: %s", run_id, str(rbe))
            await self._emit(
                ExecutionEventType.RESOURCE_BUDGET_EXCEEDED,
                message=str(rbe),
                metadata={
                    "limit_type": rbe.limit_type,
                    "current_value": rbe.current_value,
                    "limit_value": rbe.limit_value,
                },
            )
            await self._emit(
                ExecutionEventType.RUN_FAILED,
                message=f"Pipeline failed: {rbe}",
            )
            raise
        except Exception as exc:
            logger.error("Pipeline execution failed for run_id=%s: %s", run_id, str(exc), exc_info=True)
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
        document_ids: Optional[List[str]] = None,
        history: Optional[List[Any]] = None,
    ) -> RunResponse:
        """Inner execution logic supporting DIRECT, SINGLE_AGENT, SINGLE_AGENT_RAG, and MULTI_AGENT strategies."""
        t_run_start = time.perf_counter()

        # Wrap LLM provider with telemetry instrumentation & budget tracker for this run
        instrumented_llm = InstrumentedLLMProvider(
            base_provider=self.llm,
            run_id=run_id,
            metrics_collector=global_metrics_collector,
            model_name=getattr(self.settings, "llm_model", "gemini-3.1-flash-lite"),
        )
        self.strategy_selector.llm = instrumented_llm
        self.planner.llm = instrumented_llm
        self.researcher.llm = instrumented_llm
        self.analyst.llm = instrumented_llm
        self.evaluator.llm = instrumented_llm
        self.synthesizer.llm = instrumented_llm

        if self.retrieval_service:
            self.retrieval_service.current_run_id = run_id

        # -------------------------------------------------------------
        # Stage 0: Adaptive Strategy Selection
        # -------------------------------------------------------------
        instrumented_llm.set_stage("StrategySelection")
        decision = await self.strategy_selector.select_strategy(
            user_task, history=history, has_documents=has_documents
        )

        # Allocate and enforce strategy resource budget
        budget = get_budget_for_strategy(decision.strategy, self.settings)
        budget_tracker = BudgetTracker(budget)
        instrumented_llm.budget_tracker = budget_tracker
        self.web_search_tool.budget_tracker = budget_tracker.web_tracker
        self.fetch_page_tool.budget_tracker = budget_tracker.web_tracker
        decision.budget = budget.model_dump()

        strategy_meta = {
            "strategy": decision.strategy.value,
            "reasoning": decision.reasoning,
            "estimated_complexity": decision.estimated_complexity,
            "requires_rag": decision.requires_rag,
            "domains": decision.domains,
            "cross_document_reasoning": decision.cross_document_reasoning,
            "budget": decision.budget,
            "needs_web": decision.needs_web,
            "web_rationale": decision.web_rationale,
        }
        logger.info(
            "Strategy decided for run_id=%s: strategy=%s, needs_web=%s, reasoning=%s, web_rationale=%s",
            run_id, decision.strategy.value, decision.needs_web, decision.reasoning, decision.web_rationale,
        )
        await self._emit(
            ExecutionEventType.STRATEGY_DECIDED,
            message=f"Strategy decided: {decision.strategy.value} ({decision.reasoning})",
            metadata=strategy_meta,
        )
        await self._emit(
            ExecutionEventType.STRATEGY_SELECTED,
            message=f"Strategy selected: {decision.strategy.value} ({decision.reasoning})",
            metadata=strategy_meta,
        )

        # =============================================================
        # STRATEGY 1: DIRECT Conversational Response
        # =============================================================
        if decision.strategy == ExecutionStrategy.DIRECT:
            instrumented_llm.set_stage("Synthesis")
            t0 = time.perf_counter()
            await self._emit(ExecutionEventType.SYNTHESIS_STARTED, message="Generating direct response.")

            history_text = ""
            if history:
                turns = [f"{msg.role.capitalize()}: {msg.content}" for msg in history[-5:]]
                history_text = "Prior Conversation History:\n" + "\n".join(turns) + "\n\n"

            direct_prompt = f"{history_text}User: {user_task.strip()}\nAssistant:"
            direct_answer = await instrumented_llm.generate(
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

            total_wall_ms = round((time.perf_counter() - t_run_start) * 1000, 2)
            run_metrics = global_metrics_collector.finalize_run(run_id, wall_time_ms=total_wall_ms, strategy="DIRECT")

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
                metrics=run_metrics.model_dump(),
            )
            record_run(response)
            await self._emit(ExecutionEventType.RUN_COMPLETED, message=f"Pipeline completed (DIRECT). Run ID: {run_id}")
            return response

        # =============================================================
        # STRATEGY 2: SINGLE_AGENT & SINGLE_AGENT_RAG Execution
        # =============================================================
        elif decision.strategy in (ExecutionStrategy.SINGLE_AGENT, ExecutionStrategy.SINGLE_AGENT_RAG):
            strategy_name = decision.strategy.value
            is_rag = decision.strategy == ExecutionStrategy.SINGLE_AGENT_RAG or decision.requires_rag
            t0 = time.perf_counter()
            await self._emit(ExecutionEventType.PLANNING_STARTED, message=f"{strategy_name} strategy selected.")

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
            if is_rag:
                instrumented_llm.set_stage("RAG")
                if not self.retrieval_service:
                    err_msg = "Task requires RAG retrieval but retrieval service is not available."
                    logger.error(err_msg)
                    await self._emit(ExecutionEventType.RUN_FAILED, message=err_msg)
                    raise RuntimeError(err_msg)
                await self._emit(ExecutionEventType.RAG_STARTED, message="Retrieving knowledge context...", metadata={"queries": [user_task[:150]]})
                try:
                    ctx = await self.retrieval_service.retrieve(user_task, top_k=3, document_ids=document_ids)
                    retrieved_chunks_collected = ctx.chunks
                    if ctx.chunks:
                        snippets = [f"[{c.source}]: {c.content}" for c in ctx.chunks]
                        retrieval_context_prompt = "\n\nRetrieved Knowledge Base Context:\n" + "\n\n".join(snippets) + "\n\n"
                    rag_articles = sorted(list({
                        c.metadata.get("article_path") or c.metadata.get("article")
                        for c in ctx.chunks if getattr(c, "metadata", None)
                    } - {None, ""}))
                    rag_pages = sorted(list(set([
                        c.metadata["page_start"]
                        for c in ctx.chunks
                        if getattr(c, "metadata", None) and isinstance(c.metadata.get("page_start"), int)
                    ])))
                    await self._emit(
                        ExecutionEventType.RAG_COMPLETED,
                        message=f"Retrieved {len(ctx.chunks)} chunks.",
                        metadata={
                            "queries": [user_task[:150]],
                            "n_chunks": len(ctx.chunks),
                            "articles": rag_articles,
                            "pages": rag_pages,
                            "document_ids": list({c.document_id for c in ctx.chunks if c.document_id}),
                        }
                    )
                except Exception as exc:
                    err_msg = f"Knowledge retrieval failed: {str(exc)}"
                    logger.error(err_msg, exc_info=True)
                    await self._emit(ExecutionEventType.RUN_FAILED, message=err_msg)
                    raise RuntimeError(err_msg) from exc

            evidence_ledger = EvidenceLedger()
            web_sources_list: List[Dict[str, Any]] = []
            if decision.needs_web and self.web_search_tool:
                instrumented_llm.set_stage("WebSearch")
                search_q = user_task.strip()
                await self._emit(
                    ExecutionEventType.WEB_SEARCH_STARTED,
                    message=f"Searching the web for: \"{search_q[:150]}\"",
                    metadata={"query": search_q[:150], "provider": "tavily"}
                )
                try:
                    search_res = await self.web_search_tool.execute(WebSearchInput(query=search_q, max_results=4))
                    if search_res.success and search_res.output:
                        results = search_res.output.get("results", [])
                        urls = [r.get("url") for r in results if r.get("url")]
                        await self._emit(
                            ExecutionEventType.WEB_SEARCH_COMPLETED,
                            message=f"Found {len(results)} web results for \"{search_q[:150]}\"",
                            metadata={"query": search_q[:150], "urls": urls, "results": results}
                        )
                        import urllib.parse
                        for hit in results[:3]:
                            hit_url = hit.get("url")
                            hit_title = hit.get("title", "")
                            hit_snippet = hit.get("snippet", "")
                            if hit_url and self.fetch_page_tool:
                                hit_domain = urllib.parse.urlparse(hit_url).netloc.replace("www.", "")
                                t_fstart = time.perf_counter()
                                await self._emit(
                                    ExecutionEventType.WEB_FETCH_STARTED,
                                    message=f"Viewing website: {hit_title or hit_domain}",
                                    metadata={"url": hit_url, "domain": hit_domain, "title": hit_title[:150]}
                                )
                                fetch_res = await self.fetch_page_tool.execute(FetchPageInput(url=hit_url))
                                f_dur_ms = round((time.perf_counter() - t_fstart) * 1000, 2)
                                if fetch_res.success and fetch_res.output:
                                    out = fetch_res.output
                                    final_u = out.get("final_url", hit_url)
                                    final_d = urllib.parse.urlparse(final_u).netloc.replace("www.", "")
                                    evidence_ledger.add_web_page(
                                        url=final_u,
                                        title=out.get("title", hit_title),
                                        text=out.get("text", ""),
                                        retrieval_timestamp=out.get("retrieval_timestamp", ""),
                                        content_hash=out.get("content_hash", ""),
                                        raw_response=fetch_res.metadata,
                                    )
                                    await self._emit(
                                        ExecutionEventType.WEB_FETCH_COMPLETED,
                                        message=f"Read {out.get('title') or hit_domain}",
                                        metadata={
                                            "url": final_u,
                                            "domain": final_d,
                                            "title": (out.get("title") or hit_title)[:150],
                                            "status": "completed",
                                            "bytes": out.get("bytes_downloaded") or fetch_res.metadata.get("bytes", 0),
                                            "latency_ms": f_dur_ms,
                                            "content_hash": (out.get("content_hash") or "")[:8],
                                        }
                                    )
                                    web_sources_list.append({
                                        "id": (out.get("content_hash") or "")[:8],
                                        "title": out.get("title", hit_title),
                                        "url": final_u,
                                        "domain": final_d,
                                        "status": "completed",
                                        "citation": f"[web:{(out.get('content_hash') or '')[:8]}]",
                                        "snippet": out.get("text", "")[:250],
                                    })
                                else:
                                    err_s = str(fetch_res.error or "Fetch failed")
                                    is_bl = (
                                        (fetch_res.metadata and fetch_res.metadata.get("violation_type") == "security")
                                        or "robots.txt" in err_s.lower()
                                        or "disallowed" in err_s.lower()
                                        or "security" in err_s.lower()
                                    )
                                    ev_tp = ExecutionEventType.WEB_FETCH_BLOCKED if is_bl else ExecutionEventType.WEB_FETCH_FAILED
                                    st_nm = "blocked" if is_bl else "failed"
                                    await self._emit(
                                        ev_tp,
                                        message=f"{'Blocked from' if is_bl else 'Failed to read'} {hit_domain}: {err_s[:120]}",
                                        metadata={
                                            "url": hit_url,
                                            "domain": hit_domain,
                                            "title": hit_title[:150],
                                            "status": st_nm,
                                            "reason": err_s[:200],
                                            "latency_ms": f_dur_ms,
                                        }
                                    )
                                    web_sources_list.append({
                                        "id": hashlib.sha256(hit_url.encode()).hexdigest()[:8],
                                        "title": hit_title,
                                        "url": hit_url,
                                        "domain": hit_domain,
                                        "status": st_nm,
                                        "reason": err_s[:200],
                                        "citation": f"[web:{hashlib.sha256(hit_url.encode()).hexdigest()[:8]}]",
                                        "snippet": hit_snippet[:250],
                                    })
                        formatted_web = evidence_ledger.format_for_prompt(max_chunks=6)
                        if formatted_web:
                            retrieval_context_prompt += "\n\nRetrieved Web Evidence:\n" + formatted_web + "\n\n"
                except Exception as exc:
                    logger.warning("Single-agent web search execution failed: %s", exc)

            instrumented_llm.set_stage("Synthesis")
            await self._emit(ExecutionEventType.SYNTHESIS_STARTED, message="Generating single-agent response.")
            history_text = ""
            if history:
                turns = [f"{msg.role.capitalize()}: {msg.content}" for msg in history[-5:]]
                history_text = "Prior Conversation History:\n" + "\n".join(turns) + "\n\n"

            sys_prompt = (
                "You are a helpful, expert AI assistant. Deliver an accurate, well-structured response directly matching the user's intent. "
                "If web evidence is provided, cite sources with their [web:<hash>] tags."
            )
            single_prompt = f"{history_text}{retrieval_context_prompt}User Task:\n\"{user_task.strip()}\"\n\nPlease provide a clear, accurate, direct response."
            single_answer = await instrumented_llm.generate(
                prompt=single_prompt,
                system_prompt=sys_prompt
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
                    summary=f"Executed {strategy_name} strategy."
                )
            )

            total_wall_ms = round((time.perf_counter() - t_run_start) * 1000, 2)
            run_metrics = global_metrics_collector.finalize_run(run_id, wall_time_ms=total_wall_ms, strategy=strategy_name)

            response = RunResponse(
                run_id=run_id,
                user_task=user_task,
                execution_strategy=strategy_name,
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
                evidence_ledger=evidence_ledger.model_dump() if evidence_ledger.items else None,
                task_execution_traces=[],
                metrics=run_metrics.model_dump(),
            )
            record_run(response)
            await self._emit(ExecutionEventType.RUN_COMPLETED, message=f"Pipeline completed ({strategy_name}). Run ID: {run_id}")
            return response

        # =============================================================
        # STRATEGY 3: MULTI_AGENT Orchestrated Decomposition & Replanning
        # =============================================================
        instrumented_llm.set_stage("Planning")
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

        evidence_ledger = EvidenceLedger()
        orchestrator = Orchestrator(
            registry=self.registry,
            max_concurrency=self.max_concurrency,
            retrieval_service=self.retrieval_service,
            event_bus=self.event_bus,
            evidence_ledger=evidence_ledger,
        )

        instrumented_llm.set_stage("Research")
        t_orch = time.perf_counter()
        exec_result = await orchestrator.execute(
            graph,
            input_context={
                "user_task": user_task,
                "document_ids": document_ids,
                "needs_web": decision.needs_web,
                "run_id": run_id,
            },
        )
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
                res_metadata = result.metadata.get("research_result", {}) if result.metadata else {}
                research_results.append(
                    ResearchResult(
                        subtask_id=task.id,
                        subtask_description=task.description,
                        findings=str(result.output or ""),
                        citations=res_metadata.get("citations", []),
                        chunk_ids=res_metadata.get("chunk_ids", []),
                        missing_evidence=res_metadata.get("missing_evidence"),
                    )
                )
            elif task.type == TaskType.ANALYSIS and result.success:
                analysis_outputs.append(str(result.output or ""))

        if (has_documents or decision.requires_rag) and not research_results:
            err_msg = f"Multi-agent research produced no findings for document-grounded task (Graph status: {exec_result.status.value})."
            logger.error(err_msg)
            await self._emit(ExecutionEventType.RUN_FAILED, message=err_msg)
            raise RuntimeError(err_msg)

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

        instrumented_llm.set_stage("Analysis")
        t_ana = time.perf_counter()
        if analysis_outputs:
            analysis_text = "\n\n".join(analysis_outputs)
            analysis_dur = sum(
                t.duration_ms for t in exec_result.execution_trace if t.capability == "analysis"
            )
        else:
            analysis_text = await self.analyst.analyze(
                user_task, research_results, evidence_ledger=evidence_ledger
            )
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

        # Stage 5: Evaluator Agent Audit
        instrumented_llm.set_stage("Evaluation")
        await self._emit(ExecutionEventType.EVALUATION_STARTED, message="Evaluation started.")
        t_eval = time.perf_counter()
        evaluation_output = await self.evaluator.evaluate(
            user_task=user_task,
            analysis=analysis_text,
            evidence_ledger=evidence_ledger,
        )
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

        # Stage 6: Synthesis Agent (incorporates analysis + evaluation feedback)
        instrumented_llm.set_stage("Synthesis")
        await self._emit(ExecutionEventType.SYNTHESIS_STARTED, message="Synthesis started.")
        t_synth = time.perf_counter()
        final_answer = await self.synthesizer.synthesize(
            user_task,
            research_results,
            analysis_text,
            evaluation=evaluation_output,
            evidence_ledger=evidence_ledger,
            history=history,
        )
        synth_dur = round((time.perf_counter() - t_synth) * 1000, 2)
        traces.append(
            StageTrace(
                stage_name="Synthesis",
                agent_name="SynthesizerAgent",
                duration_ms=synth_dur,
                status="SUCCESS",
                summary="Generated user-facing response.",
            )
        )
        await self._emit(ExecutionEventType.SYNTHESIS_COMPLETED, message="Synthesis completed.")

        # Stage 7: Programmatic Grounding Verifier (No LLM)
        verifier_result = GroundingVerifier.verify(
            text=final_answer,
            evidence_ledger=evidence_ledger,
            min_validity_threshold=80.0,
        )
        if verifier_result:
            evaluation_output.verification_result = verifier_result.model_dump()

        # Stage 8: Targeted Iterative Replanning with Feedback & Query Refinement
        replanning_count = 0
        max_replans = getattr(self.settings, "max_replanning_rounds", 1)

        # Replanning is only warranted if the evaluation failed or citation validity fell below threshold
        needs_replan = (
            evaluation_output.status == "FAIL"
            or (verifier_result and verifier_result.citation_validity < float(self.evaluator.pass_threshold))
        )

        while needs_replan and replanning_count < max_replans:
            replanning_count += 1
            round_start_score = evaluation_output.score or 0

            # Identify weak / uncovered domains
            target_domains = list(evaluation_output.uncovered_domains or [])
            target_tasks = []
            for r in research_results:
                task_node = graph.get_task(r.subtask_id) if graph else None
                task_domain = str(task_node.metadata.get("domain") or "").lower() if task_node and task_node.metadata else ""
                is_targeted = False
                if target_domains:
                    is_targeted = any(d.lower() in task_domain or task_domain in d.lower() for d in target_domains)
                if not is_targeted:
                    is_targeted = bool(r.missing_evidence) or (
                        bool(evaluation_output.feedback) and (
                            r.subtask_id in evaluation_output.feedback
                            or (task_domain and task_domain in evaluation_output.feedback.lower())
                        )
                    )
                if is_targeted:
                    target_tasks.append(r)

            if not target_tasks:
                target_tasks = list(research_results)[:2]
            else:
                target_tasks = target_tasks[:2]

            replan_msg = (
                f"Targeted replanning round {replanning_count}/{max_replans} triggered: "
                f"Score {round_start_score}/100. Targeting {len(target_tasks)} subtask(s)."
            )
            replan_start_meta = {
                "round": replanning_count,
                "initial_score": round_start_score,
                "target_subtask_ids": [t.subtask_id for t in target_tasks],
            }
            await self._emit(ExecutionEventType.REPLANNING_STARTED, message=replan_msg, metadata=replan_start_meta)
            await self._emit(ExecutionEventType.REPLAN_STARTED, message=replan_msg, metadata=replan_start_meta)

            instrumented_llm.set_stage("Replanning")
            t_replan = time.perf_counter()

            # Execute targeted retrieval query refinement and re-run targeted subtasks concurrently
            async def _re_research_single(r: ResearchResult) -> None:
                task_res = exec_result.task_results.get(r.subtask_id)
                rc: Optional[RetrievedContext] = None
                if task_res and task_res.metadata and "retrieved_context" in task_res.metadata:
                    try:
                        rc = RetrievedContext.model_validate(task_res.metadata["retrieved_context"])
                    except Exception:
                        rc = None

                task_node = graph.get_task(r.subtask_id) if graph else None
                if self.retrieval_service and task_node and task_node.metadata:
                    meta = task_node.metadata
                    queries = meta.get("retrieval_queries") or ([meta.get("retrieval_query")] if meta.get("retrieval_query") else [])
                    for q in queries:
                        if q:
                            try:
                                targeted_ctx = await self.retrieval_service.retrieve(
                                    query=str(q),
                                    top_k=meta.get("retrieval_top_k", 5),
                                    document_ids=document_ids or None,
                                )
                                if targeted_ctx and targeted_ctx.chunks:
                                    if evidence_ledger:
                                        evidence_ledger.add_chunks(targeted_ctx.chunks)
                                    if rc is None:
                                        rc = targeted_ctx
                                    else:
                                        existing_ids = {c.id for c in rc.chunks}
                                        new_chunks = [c for c in targeted_ctx.chunks if c.id not in existing_ids]
                                        rc.chunks.extend(new_chunks)
                            except Exception as exc:
                                logger.warning("Targeted retrieval during replanning failed for '%s': %s", q, exc)

                # Emit targeted web queries for missing evidence gaps (ONLY if web is enabled for run: decision.needs_web)
                has_missing_evidence = bool(r.missing_evidence)
                needs_web_expansion = decision.needs_web and (has_missing_evidence or (verifier_result and not verifier_result.passed))
                if self.web_search_tool and needs_web_expansion:
                    gap_query = r.missing_evidence or f"{user_task} {r.subtask_description}"
                    clean_gap_query = re.sub(r"[^\w\s\-\.]", " ", str(gap_query)).strip()
                    words = clean_gap_query.split()
                    if len(words) > 12:
                        clean_gap_query = " ".join(words[:12])
                    if clean_gap_query:
                        try:
                            search_res = await self.web_search_tool.execute(
                                WebSearchInput(query=clean_gap_query, max_results=3)
                            )
                            if search_res.success and search_res.output:
                                results = search_res.output.get("results", [])
                                existing_urls = {
                                    item.url for item in evidence_ledger.items.values() if item.source_type == "web" and item.url
                                }
                                new_hits = [h for h in results if h.get("url") and h["url"] not in existing_urls]
                                if new_hits and self.fetch_page_tool:
                                    import urllib.parse
                                    for hit in new_hits[:2]:
                                        h_u = hit["url"]
                                        h_t = hit.get("title", "")
                                        h_dom = urllib.parse.urlparse(h_u).netloc.replace("www.", "")
                                        t_fstart = time.perf_counter()
                                        await self._emit(
                                            ExecutionEventType.WEB_FETCH_STARTED,
                                            message=f"Viewing website: {h_t or h_dom}",
                                            metadata={"url": h_u, "domain": h_dom, "title": h_t[:150]},
                                        )
                                        fetch_res = await self.fetch_page_tool.execute(FetchPageInput(url=h_u))
                                        f_dur_ms = round((time.perf_counter() - t_fstart) * 1000, 2)
                                        if fetch_res.success and fetch_res.output:
                                            out = fetch_res.output
                                            evidence_ledger.add_web_page(
                                                url=out.get("final_url", h_u),
                                                title=out.get("title", h_t),
                                                text=out.get("text", ""),
                                                retrieval_timestamp=out.get("retrieval_timestamp", ""),
                                                content_hash=out.get("content_hash", ""),
                                                raw_response=fetch_res.metadata,
                                            )
                                            await self._emit(
                                                ExecutionEventType.WEB_FETCH_COMPLETED,
                                                message=f"Read {out.get('title') or h_dom}",
                                                metadata={
                                                    "url": out.get("final_url", h_u),
                                                    "domain": h_dom,
                                                    "title": (out.get("title") or h_t)[:150],
                                                    "status": "completed",
                                                    "bytes": out.get("bytes_downloaded") or fetch_res.metadata.get("bytes", 0),
                                                    "latency_ms": f_dur_ms,
                                                    "content_hash": (out.get("content_hash") or "")[:8],
                                                },
                                            )
                                        else:
                                            err_s = str(fetch_res.error or "Fetch failed")
                                            is_bl = (
                                                (fetch_res.metadata and fetch_res.metadata.get("violation_type") == "security")
                                                or "robots.txt" in err_s.lower()
                                                or "disallowed" in err_s.lower()
                                                or "security" in err_s.lower()
                                            )
                                            ev_tp = ExecutionEventType.WEB_FETCH_BLOCKED if is_bl else ExecutionEventType.WEB_FETCH_FAILED
                                            st_nm = "blocked" if is_bl else "failed"
                                            await self._emit(
                                                ev_tp,
                                                message=f"{'Blocked from' if is_bl else 'Failed to read'} {h_dom}: {err_s[:120]}",
                                                metadata={
                                                    "url": h_u,
                                                    "domain": h_dom,
                                                    "title": h_t[:150],
                                                    "status": st_nm,
                                                    "reason": err_s[:200],
                                                    "latency_ms": f_dur_ms,
                                                },
                                            )
                                elif not new_hits:
                                    logger.info("Replanner targeted web query '%s' yielded no new sources.", clean_gap_query)
                            elif search_res.error and "budget exceeded" in search_res.error.lower():
                                logger.warning("Replanner web search budget exhausted: %s", search_res.error)
                        except Exception as exc:
                            logger.warning("Replanner targeted web search failed for '%s': %s", clean_gap_query, exc)

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
                    evidence_ledger=evidence_ledger,
                )
                r.findings = re_research.findings
                if re_research.citations:
                    r.citations = re_research.citations
                if re_research.missing_evidence:
                    r.missing_evidence = re_research.missing_evidence

            await asyncio.gather(*[_re_research_single(r) for r in target_tasks])

            # Re-run analysis with revised research findings and updated ledger
            analysis_text = await self.analyst.analyze(
                user_task, research_results, evidence_ledger=evidence_ledger
            )

            # Re-synthesize final answer
            final_answer = await self.synthesizer.synthesize(
                user_task,
                research_results,
                analysis_text,
                evaluation=evaluation_output,
                evidence_ledger=evidence_ledger,
                history=history,
            )

            # Re-verify with programmatic verifier
            verifier_result = GroundingVerifier.verify(
                text=final_answer,
                evidence_ledger=evidence_ledger,
                min_validity_threshold=80.0,
            )

            # Re-evaluate
            evaluation_output = await self.evaluator.evaluate(
                user_task=user_task,
                analysis=analysis_text,
                final_answer=final_answer,
                evidence_ledger=evidence_ledger,
                verification_result=verifier_result,
            )
            replan_dur = round((time.perf_counter() - t_replan) * 1000, 2)
            revised_score = evaluation_output.score or 0
            score_improved = revised_score > round_start_score

            traces.append(
                StageTrace(
                    stage_name="TargetedReplanning",
                    agent_name="Evaluator+Researcher",
                    duration_ms=replan_dur,
                    status="SUCCESS",
                    summary=f"Replanning completed. Revised evaluation score: {revised_score}/100.",
                )
            )

            replan_comp_msg = (
                f"Replanning round {replanning_count} completed: Revised score {revised_score}/100 "
                f"(improved: {score_improved})."
            )
            replan_comp_meta = {
                "round": replanning_count,
                "initial_score": round_start_score,
                "revised_score": revised_score,
                "score_improved": score_improved,
                "status": evaluation_output.status,
            }
            await self._emit(ExecutionEventType.REPLANNING_COMPLETED, message=replan_comp_msg, metadata=replan_comp_meta)
            await self._emit(ExecutionEventType.REPLAN_COMPLETED, message=replan_comp_msg, metadata=replan_comp_meta)

            if evaluation_output.status == "PASS" and (
                not verifier_result
                or verifier_result.citation_validity >= float(self.evaluator.pass_threshold)
            ):
                break

        if getattr(evaluation_output, "score_normalized", False):
            run_m = global_metrics_collector.get_or_create(run_id, strategy="MULTI_AGENT")
            run_m.score_normalizations_count += 1

        total_wall_ms = round((time.perf_counter() - t_run_start) * 1000, 2)
        run_metrics = global_metrics_collector.finalize_run(run_id, wall_time_ms=total_wall_ms, strategy="MULTI_AGENT")

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
            evidence_ledger=evidence_ledger.model_dump(),
            verification_result=verifier_result.model_dump() if verifier_result else None,
            metrics=run_metrics.model_dump(),
        )

        record_run(response)
        await self._emit(
            ExecutionEventType.RUN_COMPLETED,
            message=f"Pipeline completed (MULTI_AGENT). Run ID: {run_id}",
            metadata={"run_id": run_id},
        )
        return response

    def _ensure_default_dependencies(self, graph: TaskGraph) -> None:
        """Sanitize dependencies and ensure DAG graph validity with parallel research execution."""
        all_ids = {t.id for t in graph.tasks}
        research_ids = [t.id for t in graph.tasks if t.type == TaskType.RESEARCH]

        for task in graph.tasks:
            # Filter invalid or self references
            task.dependencies = [d for d in task.dependencies if d != task.id and d in all_ids]
            # Research subtasks explore distinct domains and must execute concurrently in parallel
            if task.type == TaskType.RESEARCH:
                task.dependencies = []

        if research_ids:
            for task in graph.tasks:
                if task.type == TaskType.ANALYSIS and not task.dependencies:
                    task.dependencies = list(research_ids)

        try:
            graph.validate_graph()
        except ValueError:
            for task in graph.tasks:
                if task.type == TaskType.RESEARCH:
                    task.dependencies = []
                elif task.type == TaskType.ANALYSIS:
                    task.dependencies = list(research_ids)
            graph.validate_graph()

    async def _emit(
        self,
        event_type: ExecutionEventType,
        message: str | None = None,
        metadata: dict | None = None,
        task_id: str | None = None,
        capability: str | None = None,
        status: str | None = None,
    ) -> None:
        """Emit a pipeline-level event if an event bus is attached."""
        if self.event_bus is None:
            return
        await self.event_bus.emit(
            ExecutionEvent(
                run_id=self.event_bus.run_id,
                event_type=event_type,
                task_id=task_id,
                capability=capability,
                status=status,
                message=message,
                metadata=metadata,
            )
        )
