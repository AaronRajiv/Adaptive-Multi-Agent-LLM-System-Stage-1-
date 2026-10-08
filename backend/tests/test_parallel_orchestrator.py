"""Deterministic Stage 4 tests for bounded parallel orchestration and tools."""

import asyncio

import pytest
from pydantic import BaseModel

from app.models.task_graph import TaskGraph, TaskNode, TaskStatus
from app.orchestration.agents import (
    CapabilityRegistry,
    TaskExecutionAgent,
    TaskExecutionContext,
    TaskExecutionResult,
)
from app.orchestration.orchestrator import GraphExecutionStatus, Orchestrator
from app.orchestration.tools import (
    ControlledTool,
    ToolInputError,
    ToolRegistry,
    ToolRequest,
    ToolResult,
)


class GateAgent(TaskExecutionAgent):
    def __init__(self, expected_starts: int = 1, fail_task_id: str | None = None):
        self.expected_starts = expected_starts
        self.fail_task_id = fail_task_id
        self.started: list[str] = []
        self.completed: list[str] = []
        self.active = 0
        self.peak_active = 0
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()
        self.fast_task_finished = asyncio.Event()
        self.contexts: dict[str, TaskExecutionContext] = {}

    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
        task_id = context.task.id
        self.contexts[task_id] = context
        self.started.append(task_id)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        if len(self.started) >= self.expected_starts:
            self.all_started.set()
        if task_id == "fast":
            self.fast_task_finished.set()
            self.active -= 1
            self.completed.append(task_id)
            return TaskExecutionResult(task_id=task_id, success=True, output=f"output:{task_id}")

        await self.release.wait()
        self.active -= 1
        self.completed.append(task_id)
        if task_id == self.fail_task_id:
            return TaskExecutionResult(task_id=task_id, success=False, error="planned failure")
        return TaskExecutionResult(task_id=task_id, success=True, output=f"output:{task_id}")


def task(task_id: str, **kwargs) -> TaskNode:
    return TaskNode(id=task_id, description=f"Task {task_id}", assigned_capability="test", **kwargs)


def orchestrator(agent: TaskExecutionAgent, max_concurrency: int = 4) -> Orchestrator:
    registry = CapabilityRegistry()
    registry.register("test", agent)
    return Orchestrator(registry, max_concurrency=max_concurrency)


@pytest.mark.asyncio
async def test_independent_tasks_overlap_and_traces_are_retained():
    agent = GateAgent(expected_starts=2)
    run = asyncio.create_task(orchestrator(agent).execute(TaskGraph(tasks=[task("A"), task("B")])))

    await asyncio.wait_for(agent.all_started.wait(), timeout=1)
    assert set(agent.started) == {"A", "B"}
    assert agent.peak_active == 2
    agent.release.set()
    result = await run

    assert result.status == GraphExecutionStatus.COMPLETED
    assert set(result.task_results) == {"A", "B"}
    assert {trace.task_id for trace in result.execution_trace} == {"A", "B"}
    assert all(trace.started_at <= trace.ended_at for trace in result.execution_trace)


@pytest.mark.asyncio
async def test_dependent_task_waits_for_all_prerequisites_and_receives_outputs():
    agent = GateAgent(expected_starts=2)
    graph = TaskGraph(tasks=[task("A"), task("B"), task("C", dependencies=["A", "B"])])
    run = asyncio.create_task(orchestrator(agent).execute(graph))

    await asyncio.wait_for(agent.all_started.wait(), timeout=1)
    assert "C" not in agent.started
    agent.release.set()
    result = await run

    assert result.status == GraphExecutionStatus.COMPLETED
    assert agent.started[-1] == "C"
    assert set(agent.contexts["C"].dependency_results) == {"A", "B"}


@pytest.mark.asyncio
async def test_dependency_chain_remains_sequential():
    agent = GateAgent(expected_starts=1)
    graph = TaskGraph(tasks=[task("A"), task("B", dependencies=["A"]), task("C", dependencies=["B"])])
    run = asyncio.create_task(orchestrator(agent).execute(graph))

    await asyncio.wait_for(agent.all_started.wait(), timeout=1)
    assert agent.started == ["A"]
    agent.release.set()
    result = await run

    assert result.status == GraphExecutionStatus.COMPLETED
    assert agent.started == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_maximum_concurrency_is_respected():
    agent = GateAgent(expected_starts=2)
    graph = TaskGraph(tasks=[task("A"), task("B"), task("C")])
    run = asyncio.create_task(orchestrator(agent, max_concurrency=2).execute(graph))

    await asyncio.wait_for(agent.all_started.wait(), timeout=1)
    assert len(agent.started) == 2
    assert agent.peak_active == 2
    agent.release.set()
    result = await run

    assert result.status == GraphExecutionStatus.COMPLETED
    assert agent.peak_active == 2


@pytest.mark.asyncio
async def test_slow_task_does_not_block_independent_fast_task():
    agent = GateAgent(expected_starts=2)
    run = asyncio.create_task(orchestrator(agent).execute(TaskGraph(tasks=[task("slow"), task("fast")])))

    await asyncio.wait_for(agent.fast_task_finished.wait(), timeout=1)
    assert "slow" in agent.started
    assert "fast" in agent.completed
    assert not run.done()
    agent.release.set()

    assert (await run).status == GraphExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_failed_prerequisite_blocks_dependent_task_after_concurrent_batch():
    agent = GateAgent(expected_starts=2, fail_task_id="A")
    graph = TaskGraph(tasks=[task("A"), task("B"), task("C", dependencies=["A", "B"])])
    run = asyncio.create_task(orchestrator(agent).execute(graph))

    await asyncio.wait_for(agent.all_started.wait(), timeout=1)
    agent.release.set()
    result = await run

    assert result.status == GraphExecutionStatus.FAILED
    assert graph.get_task("A").status == TaskStatus.FAILED
    assert graph.get_task("B").status == TaskStatus.COMPLETED
    assert graph.get_task("C").status == TaskStatus.BLOCKED
    assert "C" not in agent.started


class AdditionInput(BaseModel):
    left: int
    right: int


class AdditionTool(ControlledTool):
    name = "add"
    description = "Adds two integer values deterministically."
    input_model = AdditionInput

    async def execute(self, tool_input: AdditionInput) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=tool_input.left + tool_input.right,
        )


@pytest.mark.asyncio
async def test_tool_registry_registration_resolution_and_controlled_invocation():
    registry = ToolRegistry()
    registry.register(AdditionTool())

    assert registry.resolve("ADD").definition.input_schema["properties"].keys() == {"left", "right"}
    result = await registry.invoke(ToolRequest(name="add", arguments={"left": 2, "right": 5}))

    assert result.success is True
    assert result.output == 7


@pytest.mark.asyncio
async def test_invalid_tool_requests_fail_clearly():
    registry = ToolRegistry()
    registry.register(AdditionTool())

    with pytest.raises(KeyError, match="No tool is registered"):
        await registry.invoke(ToolRequest(name="missing"))
    with pytest.raises(ToolInputError, match="Invalid input"):
        await registry.invoke(ToolRequest(name="add", arguments={"left": "not-an-int", "right": 1}))
