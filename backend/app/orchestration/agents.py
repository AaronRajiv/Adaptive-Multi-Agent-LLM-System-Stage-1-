"""Task execution contracts and adapters for the Stage 3 orchestrator."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel, Field

from app.agents.analyst import AnalystAgent
from app.agents.researcher import ResearchAgent
from app.knowledge.models import RetrievedContext
from app.models.schemas import ResearchResult
from app.models.task_graph import TaskNode


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


class ResearchTaskAgent(TaskExecutionAgent):
    """Adapter that exposes the existing ResearchAgent as a graph capability with optional RAG."""

    def __init__(self, agent: ResearchAgent):
        self.agent = agent

    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
        user_task = str(context.input_context.get("user_task", context.task.description))
        result = await self.agent.execute_subtask(
            user_task, context.task, retrieved_context=context.retrieved_context
        )
        metadata: Dict[str, Any] = {"research_result": result.model_dump()}
        if context.retrieved_context:
            metadata["retrieved_context"] = context.retrieved_context.model_dump()

        return TaskExecutionResult(
            task_id=context.task.id,
            success=True,
            output=result.findings,
            metadata=metadata,
        )


class AnalysisTaskAgent(TaskExecutionAgent):
    """Adapter that gives the existing AnalystAgent completed dependency findings and optional RAG."""

    def __init__(self, agent: AnalystAgent):
        self.agent = agent

    async def execute(self, context: TaskExecutionContext) -> TaskExecutionResult:
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
            user_task, research_results, retrieved_context=context.retrieved_context
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
