"""Deterministic tests for the Stage 3 dependency-aware orchestrator."""

import pytest

from app.agents.analyst import AnalystAgent
from app.agents.researcher import ResearchAgent
from app.llm.provider import MockLLMProvider
from app.models.task_graph import TaskGraph, TaskNode, TaskStatus
from app.orchestration.agents import (
    AnalysisTaskAgent,
    CapabilityRegistry,
    ResearchTaskAgent,
    TaskExecutionAgent,
    TaskExecutionContext,
    TaskExecutionResult,
)
from app.orchestration.orchestrator import GraphExecutionStatus, Orchestrator


class RecordingAgent(TaskExecutionAgent):
    def __init__(self, calls: list[str], fail_task_id: str | None = None):
        self.calls = calls
        self.fail_task_id = fail_task_id
        self.contexts: dict[str, TaskExecutionContext] = {}
        self.observed_statuses: dict[str, TaskStatus] = {}

    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
        task_id = context.task.id
        self.calls.append(task_id)
        self.contexts[task_id] = context
        self.observed_statuses[task_id] = context.task.status
        if task_id == self.fail_task_id:
            return TaskExecutionResult(task_id=task_id, success=False, error="planned failure")
        return TaskExecutionResult(task_id=task_id, success=True, output=f"output:{task_id}")


def make_task(task_id: str, **kwargs) -> TaskNode:
    return TaskNode(id=task_id, description=f"Task {task_id}", **kwargs)


def make_orchestrator(agent: RecordingAgent, capability: str = "test") -> Orchestrator:
    registry = CapabilityRegistry()
    registry.register(capability, agent)
    return Orchestrator(registry)


@pytest.mark.asyncio
async def test_single_task_execution_and_trace_generation():
    calls: list[str] = []
    agent = RecordingAgent(calls)
    result = await make_orchestrator(agent).execute(
        TaskGraph(tasks=[make_task("A", assigned_capability="test")])
    )

    assert result.status == GraphExecutionStatus.COMPLETED
    assert calls == ["A"]
    assert agent.observed_statuses["A"] == TaskStatus.RUNNING
    assert result.graph.get_task("A").status == TaskStatus.COMPLETED
    assert result.task_results["A"].output == "output:A"
    assert result.execution_trace[0].task_id == "A"
    assert result.execution_trace[0].success is True


@pytest.mark.asyncio
async def test_sequential_dependency_chain():
    calls: list[str] = []
    result = await make_orchestrator(RecordingAgent(calls)).execute(TaskGraph(tasks=[
        make_task("A", assigned_capability="test"),
        make_task("B", assigned_capability="test", dependencies=["A"]),
        make_task("C", assigned_capability="test", dependencies=["B"]),
    ]))

    assert result.status == GraphExecutionStatus.COMPLETED
    assert calls == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_independent_tasks_execute_sequentially_in_graph_order():
    calls: list[str] = []
    result = await make_orchestrator(RecordingAgent(calls)).execute(TaskGraph(tasks=[
        make_task("A", assigned_capability="test"),
        make_task("B", assigned_capability="test"),
    ]))

    assert result.status == GraphExecutionStatus.COMPLETED
    assert calls == ["A", "B"]


@pytest.mark.asyncio
async def test_dependency_outputs_are_propagated_after_prerequisites_complete():
    calls: list[str] = []
    agent = RecordingAgent(calls)
    await make_orchestrator(agent).execute(TaskGraph(tasks=[
        make_task("A", assigned_capability="test"),
        make_task("B", assigned_capability="test", dependencies=["A"]),
    ]), input_context={"user_task": "demo"})

    assert list(agent.contexts["B"].dependency_results) == ["A"]
    assert agent.contexts["B"].dependency_results["A"].output == "output:A"
    assert agent.contexts["B"].input_context["user_task"] == "demo"


@pytest.mark.asyncio
async def test_capability_resolution_uses_assigned_capability():
    calls: list[str] = []
    result = await make_orchestrator(RecordingAgent(calls), capability="custom").execute(
        TaskGraph(tasks=[make_task("A", assigned_capability="custom")])
    )

    assert result.status == GraphExecutionStatus.COMPLETED
    assert result.execution_trace[0].capability == "custom"


@pytest.mark.asyncio
async def test_failed_task_marks_dependents_blocked_and_reports_failure():
    calls: list[str] = []
    result = await make_orchestrator(RecordingAgent(calls, fail_task_id="A")).execute(TaskGraph(tasks=[
        make_task("A", assigned_capability="test"),
        make_task("B", assigned_capability="test", dependencies=["A"]),
    ]))

    assert result.status == GraphExecutionStatus.FAILED
    assert calls == ["A"]
    assert result.graph.get_task("A").status == TaskStatus.FAILED
    assert result.graph.get_task("B").status == TaskStatus.BLOCKED
    assert result.execution_trace[-1].task_id == "B"
    assert result.execution_trace[-1].status == TaskStatus.BLOCKED


@pytest.mark.asyncio
async def test_missing_capability_fails_task_without_executing_it():
    result = await Orchestrator(CapabilityRegistry()).execute(
        TaskGraph(tasks=[make_task("A", assigned_capability="missing")])
    )

    assert result.status == GraphExecutionStatus.FAILED
    assert result.graph.get_task("A").status == TaskStatus.FAILED
    assert "No agent is registered" in result.task_results["A"].error


@pytest.mark.asyncio
async def test_empty_graph_is_invalid():
    result = await make_orchestrator(RecordingAgent([])).execute(TaskGraph())

    assert result.status == GraphExecutionStatus.INVALID
    assert result.error == "Task graph cannot be empty."


@pytest.mark.asyncio
async def test_mutated_invalid_graph_is_reported_without_execution():
    calls: list[str] = []
    graph = TaskGraph(tasks=[make_task("A", assigned_capability="test")])
    graph.tasks.append(make_task("A", assigned_capability="test"))

    result = await make_orchestrator(RecordingAgent(calls)).execute(graph)

    assert result.status == GraphExecutionStatus.INVALID
    assert calls == []
    assert result.graph is None
    assert "Duplicate task ID" in result.error


@pytest.mark.asyncio
async def test_existing_research_and_analysis_adapters_execute_through_registry():
    provider = MockLLMProvider()
    registry = CapabilityRegistry()
    registry.register("research", ResearchTaskAgent(ResearchAgent(provider)))
    registry.register("analysis", AnalysisTaskAgent(AnalystAgent(provider)))

    result = await Orchestrator(registry).execute(TaskGraph(tasks=[
        make_task("A"),
        make_task("B", type="analysis", dependencies=["A"]),
    ]), input_context={"user_task": "Compare two options."})

    assert result.status == GraphExecutionStatus.COMPLETED
    assert result.task_results["A"].output
    assert result.task_results["B"].output
    assert [item.agent_name for item in result.execution_trace] == ["ResearchTaskAgent", "AnalysisTaskAgent"]
