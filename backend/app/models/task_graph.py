"""Stage 2 planning-only task graph models and validation."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    """The kind of work a task node represents."""

    RESEARCH = "research"
    ANALYSIS = "analysis"


class TaskStatus(str, Enum):
    """Lifecycle states recorded on a task node; Stage 2 does not execute transitions."""

    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class TaskGraphError(ValueError):
    """Raised when a task graph operation violates structural constraints."""


class TaskNode(BaseModel):
    """A graph node that describes one planned unit of work and its prerequisites."""

    id: str = Field(..., description="Unique task identifier, e.g., 'T1', 'T2'")
    description: str = Field(..., description="Concrete, actionable task description")
    type: TaskType = Field(default=TaskType.RESEARCH, description="The planned work category")
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current lifecycle state; execution is deferred to a future orchestrator",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of tasks that must complete successfully before this task can run",
    )
    assigned_capability: Optional[str] = Field(
        default=None,
        description="Named capability or agent intended to perform this task",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional planner metadata that does not affect graph semantics",
    )

    @model_validator(mode="after")
    def validate_no_self_dependency(self) -> "TaskNode":
        if self.id in self.dependencies:
            raise ValueError(f"Task '{self.id}' cannot depend on itself.")
        return self


class TaskGraph(BaseModel):
    """Validated, non-executing dependency graph for a planned task set."""

    tasks: List[TaskNode] = Field(default_factory=list, description="Task nodes in the planned graph")

    @model_validator(mode="after")
    def validate_graph(self) -> "TaskGraph":
        task_ids = [task.id for task in self.tasks]
        duplicate_ids = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
        if duplicate_ids:
            raise ValueError(f"Duplicate task ID(s): {', '.join(duplicate_ids)}.")

        known_ids = set(task_ids)
        for task in self.tasks:
            missing = sorted(set(task.dependencies) - known_ids)
            if missing:
                raise ValueError(
                    f"Task '{task.id}' depends on nonexistent task ID(s): {', '.join(missing)}."
                )

        cycle = self._find_cycle()
        if cycle:
            raise ValueError(f"Circular dependency detected: {' -> '.join(cycle)}.")
        return self

    def add_task(self, task: TaskNode) -> None:
        """Add one task and reject duplicate IDs or invalid references immediately."""
        self.tasks.append(task)
        try:
            self.validate_graph()
        except ValueError:
            self.tasks.pop()
            raise

    def get_task(self, task_id: str) -> TaskNode:
        """Retrieve a task by ID, raising an explicit error when it does not exist."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise TaskGraphError(f"Task ID '{task_id}' does not exist in this graph.")

    def add_dependency(self, task_id: str, dependency_id: str) -> None:
        """Add a prerequisite edge from ``dependency_id`` to ``task_id``."""
        task = self.get_task(task_id)
        self.get_task(dependency_id)
        if task_id == dependency_id:
            raise TaskGraphError(f"Task '{task_id}' cannot depend on itself.")
        if dependency_id in task.dependencies:
            return

        task.dependencies.append(dependency_id)
        try:
            self.validate_graph()
        except ValueError:
            task.dependencies.pop()
            raise

    def runnable_tasks(self) -> List[TaskNode]:
        """Return pending or ready tasks whose dependencies all completed successfully."""
        runnable_states = {TaskStatus.PENDING, TaskStatus.READY}
        return [
            task
            for task in self.tasks
            if task.status in runnable_states
            and all(
                self.get_task(dependency_id).status == TaskStatus.COMPLETED
                for dependency_id in task.dependencies
            )
        ]

    def _find_cycle(self) -> Optional[List[str]]:
        """Return one dependency cycle, if present, using depth-first traversal."""
        nodes = {task.id: task for task in self.tasks}
        visited: set[str] = set()
        active_path: List[str] = []

        def visit(task_id: str) -> Optional[List[str]]:
            if task_id in active_path:
                start = active_path.index(task_id)
                return active_path[start:] + [task_id]
            if task_id in visited:
                return None

            visited.add(task_id)
            active_path.append(task_id)
            for dependency_id in nodes[task_id].dependencies:
                cycle = visit(dependency_id)
                if cycle:
                    return cycle
            active_path.pop()
            return None

        for task_id in nodes:
            cycle = visit(task_id)
            if cycle:
                return cycle
        return None
