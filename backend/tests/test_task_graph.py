"""Deterministic tests for the Stage 2 planning-only task graph."""

import pytest
from pydantic import ValidationError

from app.models.schemas import PlannerOutput
from app.models.task_graph import TaskGraph, TaskGraphError, TaskNode, TaskStatus, TaskType


def task(task_id: str, **kwargs) -> TaskNode:
    return TaskNode(id=task_id, description=f"Task {task_id}", **kwargs)


def test_create_valid_graph_and_retrieve_tasks():
    graph = TaskGraph(tasks=[task("T1"), task("T2", dependencies=["T1"])])

    assert graph.get_task("T1").id == "T1"
    assert graph.get_task("T2").dependencies == ["T1"]


def test_add_task_and_dependency():
    graph = TaskGraph()
    graph.add_task(task("T1"))
    graph.add_task(task("T2"))
    graph.add_dependency("T2", "T1")

    assert graph.get_task("T2").dependencies == ["T1"]


def test_reject_duplicate_task_ids():
    with pytest.raises(ValidationError, match="Duplicate task ID"):
        TaskGraph(tasks=[task("T1"), task("T1")])


def test_reject_nonexistent_task_references():
    graph = TaskGraph(tasks=[task("T1")])

    with pytest.raises(TaskGraphError, match="does not exist"):
        graph.get_task("T9")
    with pytest.raises(TaskGraphError, match="does not exist"):
        graph.add_dependency("T1", "T9")
    with pytest.raises(ValidationError, match="nonexistent task ID"):
        TaskGraph(tasks=[task("T2", dependencies=["T9"])])


def test_reject_self_dependency():
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        task("T1", dependencies=["T1"])


def test_reject_circular_dependencies():
    with pytest.raises(ValidationError, match="Circular dependency detected"):
        TaskGraph(tasks=[task("T1", dependencies=["T2"]), task("T2", dependencies=["T1"])])


def test_runnable_tasks_require_completed_dependencies():
    graph = TaskGraph(tasks=[
        task("T1", status=TaskStatus.COMPLETED),
        task("T2", dependencies=["T1"], status=TaskStatus.READY),
        task("T3", dependencies=["T2"]),
        task("T4", status=TaskStatus.RUNNING),
        task("T5", status=TaskStatus.BLOCKED),
    ])

    assert [node.id for node in graph.runnable_tasks()] == ["T2"]


def test_task_node_defaults_and_enum_validation():
    node = task("T1", type=TaskType.ANALYSIS, assigned_capability="AnalystAgent")

    assert node.status == TaskStatus.PENDING
    assert node.dependencies == []
    assert node.type == TaskType.ANALYSIS
    assert node.assigned_capability == "AnalystAgent"

    with pytest.raises(ValidationError):
        task("T2", status="unknown")


def test_planner_output_converts_to_validated_task_graph():
    output = PlannerOutput(tasks=[
        task("T1", assigned_capability="ResearchAgent"),
        task("T2", type=TaskType.ANALYSIS, dependencies=["T1"], assigned_capability="AnalystAgent"),
    ])

    graph = output.to_task_graph()

    assert isinstance(graph, TaskGraph)
    assert graph.get_task("T2").dependencies == ["T1"]
