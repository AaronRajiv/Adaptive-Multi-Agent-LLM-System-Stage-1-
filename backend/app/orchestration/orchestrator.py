"""Bounded, dependency-aware asynchronous execution for Stage 4 task graphs."""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from app.events import ExecutionEvent, ExecutionEventType, RunEventBus
from app.knowledge.retrieval import RetrievalService
from app.models.schemas import TaskExecutionTrace
from app.models.task_graph import TaskGraph, TaskNode, TaskStatus
from app.orchestration.agents import (
    CapabilityRegistry,
    TaskExecutionAgent,
    TaskExecutionContext,
    TaskExecutionResult,
)


class GraphExecutionStatus(str, Enum):
    """Terminal outcomes for one orchestrated graph run."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INVALID = "INVALID"


class GraphExecutionResult(BaseModel):
    """Final in-memory result of executing a task graph."""

    status: GraphExecutionStatus
    graph: TaskGraph | None
    task_results: Dict[str, TaskExecutionResult] = Field(default_factory=dict)
    execution_trace: List[TaskExecutionTrace] = Field(default_factory=list)
    error: str | None = None


class Orchestrator:
    """Executes a graph with bounded concurrency while enforcing dependencies and states."""

    _ALLOWED_TRANSITIONS = {
        TaskStatus.PENDING: {TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.SKIPPED},
        TaskStatus.READY: {TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.SKIPPED},
        TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED},
        TaskStatus.COMPLETED: set(),
        TaskStatus.FAILED: set(),
        TaskStatus.BLOCKED: set(),
        TaskStatus.SKIPPED: set(),
    }

    def __init__(
        self,
        registry: CapabilityRegistry,
        max_concurrency: int = 4,
        retrieval_service: RetrievalService | None = None,
        event_bus: Optional[RunEventBus] = None,
    ):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        self.registry = registry
        self.max_concurrency = max_concurrency
        self.retrieval_service = retrieval_service
        self.event_bus = event_bus

    async def execute(
        self, graph: TaskGraph, input_context: Dict[str, Any] | None = None
    ) -> GraphExecutionResult:
        """Run ready tasks concurrently up to the configured limit until terminal state."""
        context = input_context or {}
        results: Dict[str, TaskExecutionResult] = {}
        trace: List[TaskExecutionTrace] = []

        try:
            graph.validate_graph()
        except ValueError as exc:
            return GraphExecutionResult(
                status=GraphExecutionStatus.INVALID,
                graph=None,
                execution_trace=trace,
                error=str(exc),
            )

        if not graph.tasks:
            return GraphExecutionResult(
                status=GraphExecutionStatus.INVALID,
                graph=None,
                execution_trace=trace,
                error="Task graph cannot be empty.",
            )

        semaphore = asyncio.Semaphore(self.max_concurrency)
        active: Set[asyncio.Task[None]] = set()

        while True:
            await self._mark_tasks_blocked_by_failed_dependencies(graph, trace)
            capacity = self.max_concurrency - len(active)
            if capacity > 0:
                for task in graph.runnable_tasks()[:capacity]:
                    await self._prepare_task_for_execution(task)
                    active.add(
                        asyncio.create_task(
                            self._execute_running_task(
                                task, context, results, trace, semaphore
                            )
                        )
                    )

            if active:
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for completed_task in done:
                    await completed_task
                continue

            return GraphExecutionResult(
                status=self._terminal_status(graph),
                graph=graph,
                task_results=results,
                execution_trace=trace,
            )

    async def _prepare_task_for_execution(self, task: TaskNode) -> None:
        """Reserve a runnable task before creating its asynchronous execution task."""
        if task.status == TaskStatus.PENDING:
            self._transition(task, TaskStatus.READY)
            await self._emit_task_event(ExecutionEventType.TASK_READY, task)
        await self._emit_task_event(ExecutionEventType.TASK_STARTED, task)
        self._transition(task, TaskStatus.RUNNING)

    async def _execute_running_task(
        self,
        task: TaskNode,
        input_context: Dict[str, Any],
        results: Dict[str, TaskExecutionResult],
        trace: List[TaskExecutionTrace],
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a reserved task and commit its result without touching other task states."""
        capability = task.assigned_capability or task.type.value
        async with semaphore:
            started_at = self._timestamp()
            started = perf_counter()
            try:
                agent = self.registry.resolve(capability)
                dependency_results = {
                    dependency_id: results[dependency_id] for dependency_id in task.dependencies
                }
                retrieved_context = await self._retrieve_context_for_task(task)
                execution_context = TaskExecutionContext(
                    task=task,
                    input_context=input_context,
                    dependency_results=dependency_results,
                    retrieved_context=retrieved_context,
                )
                result = await agent.execute(execution_context)
                if result.task_id != task.id:
                    raise ValueError(
                        f"Agent returned result for '{result.task_id}', expected task '{task.id}'."
                    )
                if not result.success:
                    raise RuntimeError(result.error or "Agent reported unsuccessful execution.")

                self._transition(task, TaskStatus.COMPLETED)
                results[task.id] = result
                self._append_trace(trace, task, capability, agent, started_at, started, result=result)
                await self._emit_task_event(
                    ExecutionEventType.TASK_COMPLETED, task, capability=capability,
                    message=f"Task '{task.id}' completed successfully.",
                )
            except Exception as exc:
                self._transition(task, TaskStatus.FAILED)
                failure = TaskExecutionResult(task_id=task.id, success=False, error=str(exc))
                results[task.id] = failure
                self._append_trace(trace, task, capability, None, started_at, started, result=failure)
                await self._emit_task_event(
                    ExecutionEventType.TASK_FAILED, task, capability=capability,
                    message=f"Task '{task.id}' failed: {exc}",
                )

    async def _retrieve_context_for_task(self, task: TaskNode):
        """Retrieve structured context only when the task explicitly requests it."""
        metadata = task.metadata or {}
        query = metadata.get("retrieval_query")
        if query is None or not str(query).strip():
            return None
        if self.retrieval_service is None:
            return None
        top_k = metadata.get("retrieval_top_k", 3)
        if not isinstance(top_k, int):
            raise ValueError("retrieval_top_k must be an integer.")

        await self._emit_task_event(
            ExecutionEventType.RAG_STARTED, task,
            message=f"Retrieving context for task '{task.id}': {query}",
        )
        result = await self.retrieval_service.retrieve(str(query), top_k=top_k)
        await self._emit_task_event(
            ExecutionEventType.RAG_COMPLETED, task,
            message=f"Retrieved {len(result.chunks)} chunk(s) for task '{task.id}'.",
        )
        return result

    async def _mark_tasks_blocked_by_failed_dependencies(
        self, graph: TaskGraph, trace: List[TaskExecutionTrace]
    ) -> None:
        blocking_states = {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.SKIPPED}
        for task in graph.tasks:
            if task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
                continue
            failed_dependencies = [
                dependency_id
                for dependency_id in task.dependencies
                if graph.get_task(dependency_id).status in blocking_states
            ]
            if failed_dependencies:
                self._transition(task, TaskStatus.BLOCKED)
                now = self._timestamp()
                error = f"Blocked by unsuccessful prerequisite(s): {', '.join(failed_dependencies)}."
                self._append_trace(
                    trace,
                    task,
                    task.assigned_capability or task.type.value,
                    None,
                    now,
                    perf_counter(),
                    result=TaskExecutionResult(task_id=task.id, success=False, error=error),
                )
                await self._emit_task_event(
                    ExecutionEventType.TASK_BLOCKED, task,
                    message=error,
                )

    def _terminal_status(self, graph: TaskGraph) -> GraphExecutionStatus:
        statuses = {task.status for task in graph.tasks}
        if statuses == {TaskStatus.COMPLETED}:
            return GraphExecutionStatus.COMPLETED
        if TaskStatus.FAILED in statuses:
            return GraphExecutionStatus.FAILED
        return GraphExecutionStatus.BLOCKED

    def _transition(self, task: TaskNode, target: TaskStatus) -> None:
        if target not in self._ALLOWED_TRANSITIONS[task.status]:
            raise ValueError(f"Invalid state transition for task '{task.id}': {task.status} -> {target}.")
        task.status = target

    def _append_trace(
        self,
        trace: List[TaskExecutionTrace],
        task: TaskNode,
        capability: str,
        agent: TaskExecutionAgent | None,
        started_at: str,
        started: float,
        result: TaskExecutionResult,
    ) -> None:
        trace.append(
            TaskExecutionTrace(
                task_id=task.id,
                capability=capability,
                agent_name=agent.__class__.__name__ if agent else None,
                started_at=started_at,
                ended_at=self._timestamp(),
                duration_ms=round((perf_counter() - started) * 1000, 2),
                status=task.status,
                success=result.success,
                output=result.output,
                error=result.error,
            )
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    async def _emit_task_event(
        self,
        event_type: ExecutionEventType,
        task: TaskNode,
        capability: str | None = None,
        message: str | None = None,
    ) -> None:
        """Emit a task-level event if an event bus is attached."""
        if self.event_bus is None:
            return
        await self.event_bus.emit(
            ExecutionEvent(
                run_id=self.event_bus.run_id,
                event_type=event_type,
                task_id=task.id,
                capability=capability or task.assigned_capability or task.type.value,
                status=task.status.value,
                message=message,
            )
        )
