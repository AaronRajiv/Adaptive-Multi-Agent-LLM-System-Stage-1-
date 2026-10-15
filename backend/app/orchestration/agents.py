"""Task execution contracts and adapters for the Stage 3 orchestrator."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel, Field

from app.agents.analyst import AnalystAgent
from app.agents.researcher import ResearchAgent
from app.knowledge.models import RetrievedContext
from app.models.schemas import ResearchResult
from app.models.task_graph import TaskNode
from app.orchestration.evidence_ledger import EvidenceLedger


class TaskExecutionResult(BaseModel):
    """Structured result returned by an agent after attempting one task."""

    task_id: str
    success: bool
    output: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class TaskExecutionContext(BaseModel):
    """The bounded inputs made available to one task execution."""

    task: TaskNode
    input_context: Dict[str, Any] = Field(default_factory=dict)
    dependency_results: Dict[str, TaskExecutionResult] = Field(default_factory=dict)
    retrieved_context: RetrievedContext | None = None
    evidence_ledger: EvidenceLedger | None = None
    event_bus: Any | None = None
    run_id: str | None = None


class TaskExecutionAgent(ABC):
    """Provider-agnostic contract for a capability that performs one graph task."""

    @abstractmethod
    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
        """Execute the supplied task and return its structured result."""


class CapabilityRegistry:
    """In-memory mapping from capability names to task execution agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, TaskExecutionAgent] = {}

    def register(self, capability: str, agent: TaskExecutionAgent) -> None:
        """Register an agent for a capability name."""
        normalized = capability.strip().lower()
        if not normalized:
            raise ValueError("Capability name cannot be empty.")
        self._agents[normalized] = agent

    def resolve(self, capability: str) -> TaskExecutionAgent:
        """Resolve a registered agent or raise a clear lookup error."""
        normalized = capability.strip().lower()
        try:
            return self._agents[normalized]
        except KeyError as exc:
            raise KeyError(f"No agent is registered for capability '{capability}'.") from exc


from app.tools.fetch_page import FetchPageInput, FetchPageTool
from app.tools.web_search import WebSearchInput, WebSearchTool


class ResearchTaskAgent(TaskExecutionAgent):
    """Adapter that exposes the existing ResearchAgent as a graph capability with optional RAG and web research."""

    def __init__(
        self,
        agent: ResearchAgent,
        web_search_tool: Optional[WebSearchTool] = None,
        fetch_page_tool: Optional[FetchPageTool] = None,
    ):
        self.agent = agent
        self.web_search_tool = web_search_tool
        self.fetch_page_tool = fetch_page_tool

    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
        if hasattr(self.agent.llm, "set_stage"):
            self.agent.llm.set_stage(f"Research:{context.task.id}")
        user_task = str(context.input_context.get("user_task", context.task.description))

        task_meta = context.task.metadata or {}
        has_doc_evidence = bool(context.retrieved_context and context.retrieved_context.chunks)
        needs_web = bool(
            task_meta.get("needs_web")
            or task_meta.get("web_search_query")
            or (context.input_context.get("needs_web") and not has_doc_evidence)
            or context.task.assigned_capability == "web_research"
        )

        web_metadata: Dict[str, Any] = {}
        if needs_web and self.web_search_tool:
            search_query = str(task_meta.get("web_search_query") or context.task.description or user_task)
            active_run_id = context.run_id or context.input_context.get("run_id") or "run-web"
            if context.event_bus:
                from app.events import ExecutionEvent, ExecutionEventType
                await context.event_bus.publish(
                    ExecutionEvent(
                        run_id=active_run_id,
                        event_type=ExecutionEventType.WEB_SEARCH_STARTED,
                        task_id=context.task.id,
                        capability="web_search",
                        message=f"Searching the web for: \"{search_query}\"",
                        metadata={"query": search_query},
                    )
                )
            try:
                search_res = await self.web_search_tool.execute(
                    WebSearchInput(query=search_query, max_results=task_meta.get("web_max_results", 4))
                )
                web_metadata["search_result"] = {
                    "success": search_res.success,
                    "metadata": search_res.metadata,
                    "error": search_res.error,
                }
                if search_res.success and search_res.output:
                    results = search_res.output.get("results", [])
                    if context.event_bus:
                        from app.events import ExecutionEvent, ExecutionEventType
                        await context.event_bus.publish(
                            ExecutionEvent(
                                run_id=active_run_id,
                                event_type=ExecutionEventType.WEB_SEARCH_COMPLETED,
                                task_id=context.task.id,
                                capability="web_search",
                                message=f"Found {len(results)} web results for \"{search_query}\"",
                                metadata={"query": search_query, "results": results},
                            )
                        )
                    fetched_urls = []
                    # Fetch top pages if fetch_page_tool is available (cap at 2 to stay within budget)
                    if self.fetch_page_tool and results:
                        import urllib.parse
                        from time import perf_counter

                        for hit in results[:2]:
                            hit_url = hit.get("url")
                            hit_title = hit.get("title", "")
                            if hit_url:
                                hit_domain = urllib.parse.urlparse(hit_url).netloc.replace("www.", "")
                                t_fstart = perf_counter()
                                if context.event_bus:
                                    from app.events import ExecutionEvent, ExecutionEventType
                                    await context.event_bus.publish(
                                        ExecutionEvent(
                                            run_id=active_run_id,
                                            event_type=ExecutionEventType.WEB_FETCH_STARTED,
                                            task_id=context.task.id,
                                            capability="fetch_page",
                                            message=f"Viewing website: {hit_title or hit_domain}",
                                            metadata={"url": hit_url, "domain": hit_domain, "title": hit_title[:150]},
                                        )
                                    )
                                fetch_res = await self.fetch_page_tool.execute(FetchPageInput(url=hit_url))
                                f_dur_ms = round((perf_counter() - t_fstart) * 1000, 2)
                                if fetch_res.success and fetch_res.output:
                                    out = fetch_res.output
                                    final_u = out.get("final_url", hit_url)
                                    final_d = urllib.parse.urlparse(final_u).netloc.replace("www.", "")
                                    fetched_urls.append(final_u)
                                    if context.evidence_ledger is not None:
                                        context.evidence_ledger.add_web_page(
                                            url=final_u,
                                            title=out.get("title", hit_title),
                                            text=out.get("text", ""),
                                            retrieval_timestamp=out.get("retrieval_timestamp", ""),
                                            content_hash=out.get("content_hash", ""),
                                            raw_response=fetch_res.metadata,
                                        )
                                    if context.event_bus:
                                        from app.events import ExecutionEvent, ExecutionEventType
                                        await context.event_bus.publish(
                                            ExecutionEvent(
                                                run_id=active_run_id,
                                                event_type=ExecutionEventType.WEB_FETCH_COMPLETED,
                                                task_id=context.task.id,
                                                capability="fetch_page",
                                                message=f"Read {out.get('title') or hit_domain}",
                                                metadata={
                                                    "url": final_u,
                                                    "domain": final_d,
                                                    "title": (out.get("title") or hit_title)[:150],
                                                    "status": "completed",
                                                    "bytes": out.get("bytes_downloaded") or fetch_res.metadata.get("bytes", 0),
                                                    "latency_ms": f_dur_ms,
                                                    "content_hash": (out.get("content_hash") or "")[:8],
                                                },
                                            )
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
                                    if context.event_bus:
                                        from app.events import ExecutionEvent, ExecutionEventType
                                        await context.event_bus.publish(
                                            ExecutionEvent(
                                                run_id=active_run_id,
                                                event_type=ev_tp,
                                                task_id=context.task.id,
                                                capability="fetch_page",
                                                message=f"{'Blocked from' if is_bl else 'Failed to read'} {hit_domain}: {err_s[:120]}",
                                                metadata={
                                                    "url": hit_url,
                                                    "domain": hit_domain,
                                                    "title": hit_title[:150],
                                                    "status": st_nm,
                                                    "reason": err_s[:200],
                                                    "latency_ms": f_dur_ms,
                                                },
                                            )
                                        )
                    web_metadata["fetched_urls"] = fetched_urls
            except Exception as exc:
                web_metadata["error"] = str(exc)

        # Ingest local RAG context if present
        if context.retrieved_context and context.evidence_ledger is not None:
            context.evidence_ledger.add_chunks(context.retrieved_context.chunks)

        result = await self.agent.execute_subtask(
            user_task,
            context.task,
            retrieved_context=context.retrieved_context,
            evidence_ledger=context.evidence_ledger,
        )

        metadata: Dict[str, Any] = {"research_result": result.model_dump()}
        if context.retrieved_context:
            metadata["retrieved_context"] = context.retrieved_context.model_dump()
        if web_metadata:
            metadata["web_research"] = web_metadata

        return TaskExecutionResult(
            task_id=context.task.id,
            success=True,
            output=result.findings,
            metadata=metadata,
        )


WebResearchTaskAgent = ResearchTaskAgent


class AnalysisTaskAgent(TaskExecutionAgent):
    """Adapter that gives the existing AnalystAgent completed dependency findings, evidence ledger, and optional RAG."""

    def __init__(self, agent: AnalystAgent):
        self.agent = agent

    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
        if hasattr(self.agent.llm, "set_stage"):
            self.agent.llm.set_stage("Analysis")
        user_task = str(context.input_context.get("user_task", context.task.description))
        research_results = [
            ResearchResult(
                subtask_id=task_id,
                subtask_description=f"Dependency task {task_id}",
                findings=str(result.output),
            )
            for task_id, result in context.dependency_results.items()
        ]
        analysis = await self.agent.analyze(
            user_task,
            research_results,
            retrieved_context=context.retrieved_context,
            evidence_ledger=context.evidence_ledger,
        )
        metadata: Dict[str, Any] = {}
        if context.retrieved_context:
            metadata["retrieved_context"] = context.retrieved_context.model_dump()

        return TaskExecutionResult(
            task_id=context.task.id,
            success=True,
            output=analysis,
            metadata=metadata,
        )
